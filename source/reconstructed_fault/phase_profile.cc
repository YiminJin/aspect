/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/reconstructed_fault/phase_profile.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/fe/fe_values.h>

namespace aspect
{
  namespace
  {
    // Empirical lower undershoot limit of the mature donor. The upper limit
    // is singular in h. Never clip a supercritical or nonfinite phase.
    bool admissible_phase(const double phase)
    {
      return std::isfinite(phase) && phase >= -1.e-4 && phase < 1.;
    }
  }

  template <int dim>
  double FaultPhaseProfile<dim>::weight(const double phase) const
  {
    // Mature AT1, degradation curvature one. The lower tail of a DG
    // interpolant may undershoot at the compact profile edge.
    AssertThrow(admissible_phase(phase), ExcMessage("Fault phase is outside the mature degradation domain [-1e-4,1)."));
    const double d = std::max(0.,phase);
    return magnitude*d*(1.+d)/((1.-d)*(1.-d));
  }

  template <int dim>
  void FaultPhaseProfile<dim>::initialize(const SimulatorAccess<dim> &sim, const unsigned int field,
                                          const LinearAlgebra::BlockVector &fields, const double m)
  {
    magnitude = m;
    const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    const auto upper = box.get_origin()+box.get_extents();
    const auto &labels = geometry.labels();
    const unsigned int n = labels.size();
    const QGauss<1> stations(3), ray(12);
    std::vector<double> rays((n-1)*stations.size());
    const auto &system = sim.get_fe();
    const auto component = sim.introspection().component_indices.compositional_fields[field];
    const auto &dg = system.base_element(system.component_to_base_index(component).first);
    const auto block = sim.introspection().block_indices.compositional_fields[field];
    std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
    std::vector<double> phase(dg.n_dofs_per_cell());
    // Check the same static FE field at bulk assembly and section quadrature
    // before any rank can enter the collective Picard solve.
    FEValues<dim> stokes(sim.get_mapping(),system,QGauss<dim>(3),update_values);
    FEValues<dim> section(sim.get_mapping(),system,QGauss<dim>(8),update_values);
    bool locally_admissible = true;
    for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          for (auto *fe :
               {
                 &stokes,&section
               })
            {
              fe->reinit(cell);
              std::vector<double> values(fe->n_quadrature_points);
              (*fe)[sim.introspection().extractors.compositional_fields[field]].get_function_values(fields,values);
              locally_admissible &= std::all_of(values.begin(),values.end(),admissible_phase);
            }
          const auto lo = cell->vertex(0);
          const auto hi = cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
          cell->get_dof_indices(indices);
          for (unsigned int i=0; i<phase.size(); ++i)
            phase[i] = fields.block(block)[indices[system.component_to_system_index(component,i)]-fields.get_block_indices().block_start(block)];
          for (unsigned int j=0; j+1<n; ++j)
            for (unsigned int k=0; k<stations.size(); ++k)
              {
                const auto &seg = geometry.segment(j);
                const auto center = labels[j].position+stations.point(k)[0]*seg.length*seg.tangent;
                double a = -std::numeric_limits<double>::max(), b = -a;
                for (unsigned int d=0; d<dim; ++d)
                  {
                    if (std::abs(seg.normal[d]) < 16.*std::numeric_limits<double>::epsilon())
                      {
                        // A face-coincident ray samples the higher-coordinate DG
                        // trace, except at the upper physical boundary.
                        if (center[d]<lo[d] || center[d]>hi[d] || (center[d]==hi[d] && hi[d]!=upper[d]))
                          {
                            b=a;
                            break;
                          }
                      }
                    else
                      {
                        const double x=(lo[d]-center[d])/seg.normal[d], y=(hi[d]-center[d])/seg.normal[d];
                        a=std::max(a,std::min(x,y));
                        b=std::min(b,std::max(x,y));
                      }
                  }
                if (b<=a) continue;
                for (unsigned int q=0; q<ray.size(); ++q)
                  {
                    const auto x=center+(a+(b-a)*ray.point(q)[0])*seg.normal;
                    Point<dim> unit;
                    for (unsigned int d=0; d<dim; ++d) unit[d]=(x[d]-lo[d])/(hi[d]-lo[d]);
                    double value=0.;
                    for (unsigned int i=0; i<phase.size(); ++i) value+=dg.shape_value(i,unit)*phase[i];
                    if (admissible_phase(value))
                      rays[j*stations.size()+k] += (b-a)*ray.weight(q)*weight(value);
                    else
                      locally_admissible = false;
                  }
              }
        }
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(locally_admissible),sim.get_mpi_communicator()),
                ExcMessage("The initialized FE phase is outside the mature degradation domain [-1e-4,1)."));
    Utilities::MPI::sum(rays,sim.get_mpi_communicator(),rays);
    FullMatrix<double> mass(n,n);
    Vector<double> rhs(n), solution(n);
    for (unsigned int j=0; j+1<n; ++j)
      for (unsigned int k=0; k<stations.size(); ++k)
        {
          const double xi=stations.point(k)[0], N[2]= {1.-xi,xi};
          const double ds=geometry.segment(j).length*stations.weight(k);
          for (unsigned int i=0; i<2; ++i)
            {
              rhs[j+i]+=ds*N[i]*rays[j*stations.size()+k];
              for (unsigned int l=0; l<2; ++l) mass(j+i,j+l)+=ds*N[i]*N[l];
            }
        }
    mass.gauss_jordan();
    mass.vmult(solution,rhs);
    integrals.assign(solution.begin(),solution.end());
  }

  template <int dim>
  typename FaultPhaseProfile<dim>::Sample FaultPhaseProfile<dim>::sample(const Point<dim> &x,const double phase) const
  {
    const auto foot=geometry.project(x);
    const double h=weight(phase);
    if (!foot.active || h==0.) return {foot,0.};
    const double I=foot.interpolate(integrals[foot.left],integrals[foot.left+1]);
    AssertThrow(I>0. && std::isfinite(I),ExcMessage("Active fault weight has invalid Q1 normal integral."));
    return {foot,h/I};
  }

  template <int dim>
  double FaultPhaseProfile<dim>::Sample::rate(const std::vector<double> &rates) const
  {
    return foot.interpolate(rates[foot.left],rates[foot.left+1]);
  }

  template <int dim>
  void FaultPhaseProfile<dim>::Sample::add_load(const double integrated_traction,std::vector<double> &rows) const
  {
    const double load=chi*integrated_traction;
    rows[foot.left]+=(1.-foot.xi)*load;
    rows[foot.left+1]+=foot.xi*load;
  }

  template <int dim>
  Tensor<2,dim> FaultPhaseProfile<dim>::slip_gradient(const Sample &sample,const double rate) const
  {
    return (sample.chi*rate)*geometry.segment(sample.foot.left).dyad;
  }
#define INSTANTIATE(dim) template class FaultPhaseProfile<dim>;
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
