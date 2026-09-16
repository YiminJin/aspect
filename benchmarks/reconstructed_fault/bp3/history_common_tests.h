// Common smooth virtual velocities: integrate the same test gradient against
// domain-P0 particle history and constrained FE history. No force is applied.
#include <aspect/particle/particle_domain.h>
namespace aspect
{
  namespace BP3Benchmark
  {
    template <int dim>
    void common_history_tests(const SimulatorAccess<dim> &sim,const LinearAlgebra::BlockVector &bulk)
    {
      if constexpr (dim==2)
        {
          const auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
          const auto &intro=sim.introspection();
          const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());
          const auto position=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          std::array<unsigned int,3> fields;
          for (const auto &m:sim.get_parameters().mapped_particle_properties)
            if (m.second.first=="maxwell stress") fields[m.second.second]=m.first;
          Tensor<1,dim> t,n; t[0]=-.5;t[1]=-BP3::sine;n[0]=BP3::sine;n[1]=-.5;
          const double centers[]={25000.,35000.,39900.,115470.05383792514};
          auto gradient=[&](const Point<dim> &p,unsigned int region,unsigned int direction)
          {
            const double s=(BP3::down_dip(p[0],p[1])-centers[region])/1000.;
            const double z=((BP3::trace_x-p[0])*BP3::sine-(BP3::box_size-p[1])*.5)/800.;
            if (std::abs(s)>=1 || std::abs(z)>=1) return SymmetricTensor<2,dim>();
            const double bs=std::pow(1-s*s,3),bn=std::pow(1-z*z,3);
            const Tensor<1,dim> grad=(-6*s*std::pow(1-s*s,2)*bn/1000.)*t
                                    +(6*z*std::pow(1-z*z,2)*bs/800.)*n;
            return symmetrize(outer_product(direction==0 ? n : t,grad));
          };
          std::ofstream out;
          if (sim.get_pcout().is_active())
            {out.open(sim.get_output_directory()+"common_history_tests.csv"); out<<std::setprecision(17)<<"order,center,direction,particle_load,fe_load\n";}
          for (unsigned int order:{4u,6u})
            {
              std::array<double,16> local{};
              FEValues<dim> values(sim.get_mapping(),sim.get_fe(),QGauss<dim>(order),update_values|update_quadrature_points|update_JxW_values);
              const QGauss<2> triangle(order);
              for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    const auto center=cell->center(); const double diameter=cell->diameter();
                    bool relevant=false;
                    for (double s:centers) relevant=relevant || (std::abs(BP3::down_dip(center[0],center[1])-s)<1000.+diameter
                      && BP3::normal_distance(center[0],center[1])<800.+diameter);
                    if (!relevant) continue;
                    values.reinit(cell);
                    std::vector<std::vector<double>> compositions(intro.n_compositional_fields,std::vector<double>(values.n_quadrature_points));
                    std::vector<double> temperatures(values.n_quadrature_points);
                    values[intro.extractors.temperature].get_function_values(bulk,temperatures);
                    for (unsigned int c=0;c<compositions.size();++c)
                      values[intro.extractors.compositional_fields[c]].get_function_values(bulk,compositions[c]);
                    std::vector<double> composition(compositions.size());
                    for (unsigned int q=0;q<values.n_quadrature_points;++q)
                      {
                        for (unsigned int c=0;c<composition.size();++c) composition[c]=compositions[c][q];
                        SymmetricTensor<2,dim> stress;
                        for (unsigned int c=0;c<3;++c) stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=compositions[fields[c]][q];
                        const auto frozen=model.evaluate_frozen_maxwell_stress(temperatures[q],composition,stress);
                        for (unsigned int r=0;r<4;++r) for (unsigned int d=0;d<2;++d)
                          local[8+2*r+d]-=values.JxW(q)*(frozen*gradient(values.quadrature_point(q),r,d));
                      }
                    // BP3 beta is spatially constant (uniform G, eta). Verify
                    // that coefficient from the production constitutive helper.
                    SymmetricTensor<2,dim> unit;unit[0][0]=1.;
                    const double beta=model.evaluate_frozen_maxwell_stress(temperatures[0],composition,unit)[0][0];
                    AssertThrow(std::abs(beta-std::exp(-sim.get_timestep()*BP3::G/1e26))<1e-14,ExcMessage("Common-test audit requires uniform BP3 beta."));
                    for (const auto &particle:pm.get_particle_handler().particles_in_cell(cell))
                      {
                        const auto domain=pm.get_particle_domain_handler().get_particle_domain(particle.get_local_index());
                        const auto &polygon=domain.vertices();
                        SymmetricTensor<2,dim> stress;
                        for (unsigned int c=0;c<3;++c) stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=beta*particle.get_properties()[position+c];
                        Point<dim> centroid;
                        for (const auto &v:polygon) centroid+=v;
                        centroid/=polygon.size();
                        for (unsigned int e=0;e<polygon.size();++e)
                          {
                            const auto a=polygon[e]-centroid,b=polygon[(e+1)%polygon.size()]-centroid;
                            const double determinant=std::abs(a[0]*b[1]-a[1]*b[0]);
                            for (unsigned int q=0;q<triangle.size();++q)
                              {
                                const double u=triangle.point(q)[0],v=triangle.point(q)[1];
                                const auto point=centroid+u*a+(1-u)*v*b;
                                const double weight=triangle.weight(q)*(1-u)*determinant;
                                for (unsigned int r=0;r<4;++r) for (unsigned int d=0;d<2;++d)
                                  local[2*r+d]-=weight*(stress*gradient(point,r,d));
                              }
                          }
                      }
                  }
              for (unsigned int r=0;r<4;++r) for (unsigned int d=0;d<2;++d)
                {
                  const double particle=Utilities::MPI::sum(local[2*r+d],sim.get_mpi_communicator());
                  const double fe=Utilities::MPI::sum(local[8+2*r+d],sim.get_mpi_communicator());
                  if (out) out<<order<<','<<centers[r]<<','<<(d==0 ? "normal":"tangent")<<','<<particle<<','<<fe<<'\n';
                }
            }
        }
    }
  }
}
