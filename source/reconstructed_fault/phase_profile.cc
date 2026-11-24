/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/reconstructed_fault/phase_profile.h>
#include <aspect/geometry_model/box.h>
#include <aspect/phase_field/mature_degradation.h>
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
    return PhaseField::excess_compliance(std::max(0.,phase),magnitude);
  }

  template <int dim>
  void FaultPhaseProfile<dim>::initialize(const SimulatorAccess<dim> &sim, const unsigned int field,
                                          const LinearAlgebra::BlockVector &fields, const double m, const unsigned int station_subdivisions)
  {
    magnitude = m;
    const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    const auto upper = box.get_origin()+box.get_extents();
    const auto &labels = geometry.labels();
    const unsigned int n = labels.size();
    const QIterated<1> stations(QGauss<1>(3),station_subdivisions);
    const QGauss<1> ray(12);
    std::vector<double> rays((n-1)*stations.size());
    std::vector<double> omitted(rays.size());
    bool resolved = true;
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
                const auto integrand = [&](const double t)
                {
                  const auto x = center+t*seg.normal;
                  Point<dim> unit;
                  for (unsigned int d=0; d<dim; ++d) unit[d]=(x[d]-lo[d])/(hi[d]-lo[d]);
                  double value=0.;
                  for (unsigned int i=0; i<phase.size(); ++i) value+=dg.shape_value(i,unit)*phase[i];
                  if (admissible_phase(value)) return weight(value);
                  locally_admissible = false;
                  return 0.;
                };
                const auto panel = [&](const double left, const double right)
                {
                  double sum = 0.;
                  for (unsigned int q=0; q<ray.size(); ++q)
                    sum += ray.weight(q)*integrand(left+(right-left)*ray.point(q)[0]);
                  return (right-left)*sum;
                };
                const auto integrate = [&](const auto &self, const double left, const double right,
                                           const double coarse, const unsigned int depth) -> double
                {
                  const double mid = .5*(left+right);
                  const double l=panel(left,mid), r=panel(mid,right);
                  if (std::abs(l+r-coarse) <= 2.e-12*(l+r)+
                  32.*std::numeric_limits<double>::epsilon()*m*(right-left))
                    return l+r;
                  if (depth == 16)
                    {
                      resolved = false;
                      return l+r;
                    }
                  return self(self,left,mid,l,depth+1)+self(self,mid,right,r,depth+1);
                };
                const double width=labels[j].half_width+stations.point(k)[0]*(labels[j+1].half_width-labels[j].half_width);
                const double cuts[4]= {a,std::clamp(-width,a,b),std::clamp(width,a,b),b};
                const unsigned int index=j*stations.size()+k;
                for (unsigned int side=0; side<3; ++side)
                  if (cuts[side+1]>cuts[side])
                    {
                      const double l=cuts[side], r=cuts[side+1];
                      integrand(l);
                      integrand(.5*(l+r));
                      integrand(r);
                      const double integral=integrate(integrate,l,r,panel(l,r),0);
                      rays[index]+=integral;
                      if (side!=1) omitted[index]+=integral;
                    }
              }
        }
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(locally_admissible),sim.get_mpi_communicator()),
                ExcMessage("The initialized FE phase is outside the mature degradation domain [-1e-4,1)."));
    AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(resolved),sim.get_mpi_communicator()),
                ExcMessage("Fault normal integral did not converge within 16 subdivisions."));
    Utilities::MPI::sum(rays,sim.get_mpi_communicator(),rays);
    Utilities::MPI::sum(omitted,sim.get_mpi_communicator(),omitted);
    // The historical finite-width Q1 reference omits at most 2.034 ppm
    // of its full normal weight on an independent cell-split integration.
    constexpr double relative_tail_budget=2.1e-6;
    double maximum_tail=0.;
    std::vector<double> diagonal(n), off_diagonal(n-1);
    integrals.assign(n,0.);
    for (unsigned int j=0; j+1<n; ++j)
      for (unsigned int k=0; k<stations.size(); ++k)
        {
          const unsigned int index=j*stations.size()+k;
          // Both sums consist of nonnegative weights; accumulate the omitted
          // part directly to avoid subtracting nearly equal full rays. An
          // exactly zero ray carries no slip, however small positive m is.
          const double fraction=rays[index]>0. ? omitted[index]/rays[index] : 0.;
          maximum_tail=std::max(maximum_tail,fraction);
          AssertThrow(fraction<=relative_tail_budget,
                      ExcMessage("Profile half width discards significant normal weight on segment "+
                                 std::to_string(j)+" at station "+std::to_string(k)+
                                 " (omitted fraction "+std::to_string(fraction)+")."));
          const double xi=stations.point(k)[0], left=1.-xi;
          const double ds=geometry.segment(j).length*stations.weight(k);
          integrals[j]+=ds*left*rays[index];
          integrals[j+1]+=ds*xi*rays[index];
          diagonal[j]+=ds*left*left;
          diagonal[j+1]+=ds*xi*xi;
          off_diagonal[j]+=ds*left*xi;
        }
    sim.get_pcout()<<"Maximum discarded fault ray weight fraction: "<<maximum_tail<<std::endl;
    // Eliminate the consistent Q1 mass in place, including on unequal segments.
    for (unsigned int j=1; j<n; ++j)
      {
        const double factor=off_diagonal[j-1]/diagonal[j-1];
        diagonal[j]-=factor*off_diagonal[j-1];
        integrals[j]-=factor*integrals[j-1];
      }
    integrals.back()/=diagonal.back();
    for (unsigned int j=n-1; j>0; --j)
      integrals[j-1]=(integrals[j-1]-off_diagonal[j-1]*integrals[j])/diagonal[j-1];
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
