// Archived K5 probe; not included by the maintained BP3 plugin.
// Frozen-data audit only: ordinary step-12 transfer, no Newton or history update.
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <deal.II/base/quadrature_lib.h>

namespace aspect
{
  namespace BP3Benchmark
  {
    bool history_load_ready=false;

    template <int dim>
    void audit_history_load(const SimulatorAccess<dim> &sim, AffineConstraints<double> &pending)
    {
      if (!std::getenv("ASPECT_BP3_HISTORY_LOAD_DIAGNOSTIC") || !history_load_ready)
        return;
      AssertThrow(dim==2 && sim.get_timestep_number()==12,ExcMessage("History-load audit requires step 12."));
      history_load_ready=false;
      const auto comm=sim.get_mpi_communicator();
      const auto &intro=sim.introspection();
      const auto &fe=sim.get_fe();
      const auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
      const auto &particles=pm.get_particle_handler();
      const auto position=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
      std::array<unsigned int,3> fields;
      fields.fill(numbers::invalid_unsigned_int);
      for (const auto &m:sim.get_parameters().mapped_particle_properties)
        if (m.second.first=="maxwell stress") fields[m.second.second]=m.first;
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model());

      // Reproduce the solver's private physical lift, including hanging-node
      // history constraints. Never republish it as the particle/FE history.
      AffineConstraints<double> physical(pending);
      physical.close();
      AffineConstraints<double> homogeneous(physical);
      for (const auto &line:homogeneous.get_lines()) homogeneous.set_inhomogeneity(line.index,0.);
      LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,comm);
      owned=sim.get_solution();
      physical.distribute(owned);
      auto &working=const_cast<LinearAlgebra::BlockVector &>(sim.get_current_linearization_point());
      const LinearAlgebra::BlockVector saved(working);
      struct Restore
      {
        LinearAlgebra::BlockVector &target;
        const LinearAlgebra::BlockVector &saved;
        ~Restore() { target=saved; }
      } restore{working,saved};
      working=owned;

      auto csv=[&](const std::string &name,const std::string &header)
      {
        std::ofstream out(sim.get_output_directory()+"history_load_"+name+"_rank"+
                          std::to_string(Utilities::MPI::this_mpi_process(comm))+".csv");
        out.exceptions(std::ios::failbit|std::ios::badbit);
        out<<std::setprecision(17)<<header<<'\n';
        return out;
      };
      auto parent_out=csv("parents","id,cell,x,y,xd,normal,xx_particle,yy_particle,xy_particle,xx_published,yy_published,xy_published,xx_working,yy_working,xy_working");
      auto qp_out=csv("qp","cell,q,x,y,xd,normal,weight,xx,yy,xy,beta_tau_N");
      auto cell_out=csv("cells","cell,x,y,xd,volume,particle_count,local_load_norm,local_normal_load_norm,local_q4_difference");
      auto dof_out=csv("loads","dof,all,normal,reference_q4,near35,near40,transition,bottom,other");

      Tensor<1,dim> normal;
      normal[0]=BP3::sine; normal[1]=-BP3::cosine;
      const auto N=symmetrize(outer_product(normal,normal));
      const auto xd=[](const Point<dim> &p){return BP3::down_dip(p[0],p[1]);};
      const auto nd=[](const Point<dim> &p){return (BP3::trace_x-p[0])*BP3::sine-(BP3::box_size-p[1])*BP3::cosine;};
      // Region loads partition every QP once. Their assembled vectors, not
      // sums of cell norms, give the constrained global weak-load comparison.
      const auto region=[&](const Point<dim> &p)
      {
        const double d=xd(p);
        if (d>=34000 && d<=36000) return 3u;
        if (d>=39000 && d<=41000) return 4u;
        if (d>=13000 && d<=20000) return 5u;
        if (p[1]<1000) return 6u;
        return 7u;
      };
      std::vector<LinearAlgebra::BlockVector> loads(8);
      for (auto &v:loads) v.reinit(intro.index_sets.system_partitioning,comm);
      std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
      const unsigned int nc=dim*fe.base_element(intro.base_elements.velocities).dofs_per_cell
                            +fe.base_element(intro.base_elements.pressure).dofs_per_cell;
      internal::Assembly::Scratch::StokesSystem<dim> scratch(
        fe,sim.get_mapping(),intro.quadratures.velocities,intro.face_quadratures.velocities,
        update_values|update_gradients|update_quadrature_points|update_JxW_values,
        update_default,intro.n_compositional_fields,nc,false,false,false,false,false,false);
      internal::Assembly::CopyData::StokesSystem<dim> data(nc,false);
      auto &assembler=sim.get_reconstructed_fault_stokes_coupling();
      FEValues<dim> high(sim.get_mapping(),fe,QGauss<dim>(4),
                         update_values|update_gradients|update_quadrature_points|update_JxW_values);
      double max_production_error=0.;
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(indices);
            std::vector<Vector<double>> local(8,Vector<double>(indices.size()));
            std::vector<Point<dim>> points;
            for (const auto &p:particles.particles_in_cell(cell)) points.push_back(p.get_reference_location());
            if (!points.empty() && std::abs(nd(cell->center()))<3500.)
              {
                FEValues<dim> values(sim.get_mapping(),fe,Quadrature<dim>(points),update_values);
                values.reinit(cell);
                std::array<std::vector<double>,3> pub,work;
                for (unsigned int c=0;c<3;++c)
                  {
                    pub[c].resize(points.size()); work[c].resize(points.size());
                    values[intro.extractors.compositional_fields[fields[c]]].get_function_values(sim.get_solution(),pub[c]);
                    values[intro.extractors.compositional_fields[fields[c]]].get_function_values(working,work[c]);
                  }
                unsigned int q=0;
                for (const auto &p:particles.particles_in_cell(cell))
                  {
                    parent_out<<p.get_id()<<','<<cell->id().to_string()<<','<<p.get_location()[0]<<','<<p.get_location()[1]
                              <<','<<xd(p.get_location())<<','<<nd(p.get_location());
                    for (unsigned int c=0;c<3;++c) parent_out<<','<<p.get_properties()[position+c];
                    for (unsigned int c=0;c<3;++c) parent_out<<','<<pub[c][q];
                    for (unsigned int c=0;c<3;++c) parent_out<<','<<work[c][q];
                    parent_out<<'\n'; ++q;
                  }
              }

            // Use the actual assembler and independently integrate its frozen
            // Maxwell RHS. The fixed profile makes its history-localization
            // term zero; agreement checks that invariant rather than assuming it.
            scratch.cell=cell;
            scratch.finite_element_values.reinit(cell);
            data.local_rhs=0.; data.local_frozen_fault_rhs=0.;
            assembler.execute(scratch,data);
            high.reinit(cell);
            for (unsigned int pass=0;pass<2;++pass)
              {
                const FEValues<dim> &v=pass==0 ? scratch.finite_element_values : high;
                std::vector<double> temp(v.n_quadrature_points);
                std::vector<std::vector<double>> comp(intro.n_compositional_fields,std::vector<double>(v.n_quadrature_points));
                v[intro.extractors.temperature].get_function_values(working,temp);
                for (unsigned int c=0;c<comp.size();++c)
                  v[intro.extractors.compositional_fields[c]].get_function_values(working,comp[c]);
                for (unsigned int q=0;q<v.n_quadrature_points;++q)
                  {
                    SymmetricTensor<2,dim> stress;
                    for (unsigned int c=0;c<3;++c)
                      stress[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)]=comp[fields[c]][q];
                    std::vector<double> composition(comp.size());
                    for (unsigned int c=0;c<comp.size();++c) composition[c]=comp[c][q];
                    const auto frozen=model.evaluate_frozen_maxwell_stress(temp[q],composition,stress);
                    const auto p=v.quadrature_point(q);
                    if (pass==0 && std::abs(nd(p))<3500.)
                      qp_out<<cell->id().to_string()<<','<<q<<','<<p[0]<<','<<p[1]<<','<<xd(p)<<','<<nd(p)<<','
                            <<v.JxW(q)<<','<<stress[0][0]<<','<<stress[1][1]<<','<<stress[0][1]<<','<<frozen*N<<'\n';
                    for (unsigned int i=0;i<indices.size();++i)
                      if (intro.component_masks.velocities[fe.system_to_component_index(i).first])
                        {
                          const auto gradient=v[intro.extractors.velocities].symmetric_gradient(i,q);
                          const double load=-frozen*gradient*v.JxW(q);
                          if (pass) local[2][i]+=load;
                          else
                            {
                              local[0][i]+=load;
                              local[1][i]-=(frozen*N)*(N*gradient)*v.JxW(q);
                              local[region(p)][i]+=load;
                            }
                        }
                  }
              }
            for (unsigned int i=0,j=0;i<indices.size();++i)
              if (intro.is_stokes_component(fe.system_to_component_index(i).first))
                { max_production_error=std::max(max_production_error,std::abs(local[0][i]-data.local_frozen_fault_rhs[j])); ++j; }
            Vector<double> error(local[2]); error-=local[0];
            cell_out<<cell->id().to_string()<<','<<cell->center()[0]<<','<<cell->center()[1]<<','<<xd(cell->center())<<','
                    <<cell->measure()<<','<<points.size()<<','<<local[0].l2_norm()<<','<<local[1].l2_norm()<<','<<error.l2_norm()<<'\n';
            for (unsigned int k=0;k<loads.size();++k)
              homogeneous.distribute_local_to_global(local[k],indices,loads[k]);
          }
      for (auto &v:loads) v.compress(VectorOperation::add);
      for (const auto i:sim.get_dof_handler().locally_owned_dofs())
        if (loads[0][i]!=0. || loads[2][i]!=0.)
          { dof_out<<i; for (const auto &v:loads) dof_out<<','<<v[i]; dof_out<<'\n'; }
      const double production_error=Utilities::MPI::max(max_production_error,comm);
      sim.get_pcout()<<"Frozen history audit: step="<<sim.get_timestep_number()<<", time="<<std::setprecision(17)
                     <<sim.get_time()<<", dt="<<sim.get_timestep()<<", max actual-assembler error="<<production_error<<std::endl;
      for (unsigned int k=0;k<loads.size();++k)
        sim.get_pcout()<<"Frozen history load "<<k<<" norm="<<loads[k].l2_norm()<<std::endl;
      AssertThrow(production_error<1e-4,ExcMessage("Independent frozen load disagrees with production assembler."));
      parent_out.close(); qp_out.close(); cell_out.close(); dof_out.close();
      sim.get_pcout()<<"Frozen history extraction completed before Newton; no mechanical solve or history publication."<<std::endl;
      throw std::runtime_error("BP3 frozen history audit completed; intentional pre-Newton stop.");
    }
  }
}
