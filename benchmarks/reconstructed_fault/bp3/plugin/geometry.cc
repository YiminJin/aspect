#include "geometry.h"
#include <aspect/geometry_model/box.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/utilities.h>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

namespace BP3
{
  namespace
  {
    std::unique_ptr<const Geometry> configured_geometry;
    std::string configured_path;
  }

  double Geometry::down_dip(const double x, const double y) const
  { return (x-upper[0])*tangent[0] + (y-upper[1])*tangent[1]; }

  double Geometry::signed_normal(const double x, const double y) const
  { return (x-upper[0])*normal[0] + (y-upper[1])*normal[1]; }

  double Geometry::minimum_cell_distance(const dealii::Point<2> &center,
                                         const dealii::Point<2> &half_width) const
  {
    return std::max(0., std::abs(signed_normal(center[0],center[1]))
                       -std::abs(normal[0])*half_width[0]-std::abs(normal[1])*half_width[1]);
  }

  std::vector<double> Geometry::stations() const
  {
    // Retain official distances where they fit; include both physical endpoints.
    std::vector<double> result{0.};
    for (const double s : {2500.,5000.,7500.,10000.,12500.,15000.,17500.,20000.,25000.,30000.,35000.})
      if (s<length) result.push_back(s);
    result.push_back(length);
    return result;
  }

  Geometry make_geometry(const std::vector<aspect::PrescribedInitialFault<2>> &faults,
                         const dealii::Point<2> &origin, const dealii::Point<2> &extent,
                         const double activation, const double upper_phase,
                         const double weakening, const bool allow_truncated_transition)
  {
    using namespace dealii;
    AssertThrow(faults.size()==1, ExcMessage("BP3 requires one straight 2D top-to-bottom fault; curved/multiple faults are unsupported."));
    const auto &fault=faults.front();
    AssertThrow(fault.vertices.size()>=2 && fault.vertices.size()==fault.core_phase_field_values.size(),
                ExcMessage("BP3 fault.txt requires at least two vertices with peak phase values."));
    for (unsigned d=0;d<2;++d)
      AssertThrow(std::isfinite(origin[d]) && std::isfinite(extent[d]) && extent[d]>0.,
                  ExcMessage("BP3 requires finite positive Box extents."));
    Geometry g;
    g.origin=origin;g.extent=extent;g.prescribed=fault;g.weakening_length=weakening;
    g.upper=fault.vertices.front();g.lower=fault.vertices.back();
    if (g.upper[1]<g.lower[1]) std::swap(g.upper,g.lower);
    const double tolerance=1e-10*std::max(extent[0],extent[1]);
    AssertThrow(std::abs(g.upper[1]-origin[1]-extent[1])<=tolerance
                && std::abs(g.lower[1]-origin[1])<=tolerance,
                ExcMessage("BP3 fault endpoints must lie on the Box top and bottom; extend fault.txt to both faces."));
    for (const auto &end : {g.upper,g.lower})
      AssertThrow(end[0]>origin[0]+tolerance && end[0]<origin[0]+extent[0]-tolerance,
                  ExcMessage("BP3 top/bottom intersections must be inside their faces; corners and side contacts are unsupported."));
    g.length=g.upper.distance(g.lower);
    g.tangent=(g.lower-g.upper)/g.length;
    g.sine=-g.tangent[1];g.cosine=std::abs(g.tangent[0]);
    AssertThrow(g.sine>1e-10,ExcMessage("BP3 does not support a fault tangent to the top/bottom boundary."));
    const double direction=g.tangent[0]<0. ? -1. : 1.;
    g.normal[0]=direction*g.sine;g.normal[1]=g.cosine;
    g.dip=std::atan2(g.sine,g.cosine);
    // Both the native tangent and its rotated normal reverse with input order,
    // leaving their symmetric product unchanged. Thrust sense depends on dip side.
    g.shear_sense=direction>0. ? -1 : 1;
    g.peak_phase=fault.core_phase_field_values.front();
    for (unsigned i=0;i<fault.vertices.size();++i)
      {
        const auto &p=fault.vertices[i];
        AssertThrow(std::isfinite(p[0]) && std::isfinite(p[1])
                    && std::abs(g.signed_normal(p[0],p[1]))<=tolerance,
                    ExcMessage("BP3 fault.txt vertices must be finite and collinear; curved faults are unsupported."));
        AssertThrow(p[0]>=origin[0]-tolerance && p[0]<=origin[0]+extent[0]+tolerance
                    && p[1]>=origin[1]-tolerance && p[1]<=origin[1]+extent[1]+tolerance,
                    ExcMessage("BP3 fault.txt vertices must remain inside the Box."));
        const double peak=fault.core_phase_field_values[i];
        AssertThrow(std::isfinite(peak) && peak>=activation && peak<=upper_phase && peak<1.
                    && peak==g.peak_phase,
                    ExcMessage("BP3 requires one constant admissible peak phase in fault.txt for its stationary loading profile."));
        if (i)
          {
            const auto segment=p-fault.vertices[i-1];
            const double orientation=(fault.vertices.back()[1]-fault.vertices.front()[1])>0. ? -1. : 1.;
            AssertThrow(segment.norm()>tolerance && orientation*(segment*g.tangent)>tolerance,
                        ExcMessage("BP3 fault.txt must be strictly ordered with nonzero segments; remove duplicate/backtracking vertices."));
          }
      }
    AssertThrow(allow_truncated_transition || g.length>=weakening+3000.,
                ExcMessage("BP3 Box/fault must contain the full weakening region and 3 km transition. For a labeled small functional fixture only, set Postprocess/BP3/Allow truncated transition=true."));
    std::ostringstream out;out<<std::setprecision(17)<<"BP3 straight geometry v1 box "<<origin<<' '<<extent<<" anchors";
    for(unsigned i=0;i<fault.vertices.size();++i)out<<' '<<fault.vertices[i]<<' '<<fault.core_phase_field_values[i];
    out<<" thrust "<<g.shear_sense<<" weakening "<<weakening<<" width 3000 local "<<allow_truncated_transition;
    g.identity=out.str();
    return g;
  }

  const Geometry &geometry()
  {
    AssertThrow(configured_geometry, dealii::ExcMessage("BP3 geometry must be configured during the consuming plugin's parameter parsing."));
    return *configured_geometry;
  }

  template <int dim>
  void configure_geometry(const aspect::SimulatorAccess<dim> &sim, dealii::ParameterHandler &prm)
  {
    using namespace aspect;
    AssertThrow(dim==2,ExcMessage("BP3 stationary geometry currently supports only 2D."));
    const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    prm.enter_subsection("Fault reconstruction");
    const std::string path=aspect::Utilities::expand_ASPECT_SOURCE_DIR(prm.get("Prescribed faults file"));
    prm.leave_subsection();
    prm.enter_subsection("Postprocess");prm.enter_subsection("BP3");
    const double weakening=prm.get_double("Weakening region length");
    bool local=prm.get_bool("Allow truncated transition");
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
    local=true; // Historical explicitly compiled small bottom-boundary fixture.
#endif
    prm.leave_subsection();prm.leave_subsection();
    if (configured_geometry)
      {
        AssertThrow(path==configured_path && weakening==geometry().weakening_length
                    && box.get_origin()[0]==geometry().origin[0] && box.get_origin()[1]==geometry().origin[1]
                    && box.get_extents()[0]==geometry().extent[0] && box.get_extents()[1]==geometry().extent[1],
                    ExcMessage("BP3 consumers requested inconsistent geometry inputs."));
        return;
      }
    const auto faults=ReconstructedFaultUtilities::parse_prescribed_faults<2>(
      aspect::Utilities::read_and_distribute_file_content(path,sim.get_mpi_communicator()),path);
    const auto &model=dynamic_cast<const MaterialModel::PhaseFieldModel<dim> &>(sim.get_material_model());
    configured_geometry=std::make_unique<const Geometry>(make_geometry(faults,
      Point<2>(box.get_origin()[0],box.get_origin()[1]),Point<2>(box.get_extents()[0],box.get_extents()[1]),
      model.get_phase_field_activation_threshold(),model.get_phase_field_upper_admissibility_threshold(),weakening,local));
    configured_path=path;
    sim.get_pcout()<<geometry().identity<<"\nBP3 dip="<<geometry().dip*180./std::acos(-1.)
                   <<", length="<<geometry().length<<", tangent="<<geometry().tangent
                   <<", normal="<<geometry().normal<<std::endl;
  }
  template void configure_geometry(const aspect::SimulatorAccess<2> &, dealii::ParameterHandler &);
  template void configure_geometry(const aspect::SimulatorAccess<3> &, dealii::ParameterHandler &);
}
