#include "configuration.h"
#include "bp3_model.h"
#include <boost/property_tree/json_parser.hpp>
#include <sstream>

namespace BP3
{
  Configuration read_configuration(const dealii::ParameterHandler &prm)
  {
    // Use deal.II's resolved/defaulted values, without copying material defaults
    // into a second parser. Exact parameter strings deliberately give a strict
    // restart identity; paths, stop times and output schedules are excluded.
    std::stringstream encoded;
    prm.print_parameters(encoded,dealii::ParameterHandler::ShortJSON);
    boost::property_tree::ptree all,model,report;
    boost::property_tree::read_json(encoded,all);
    for(const auto *path : {"Material model.Phase field fault", "Phase field model",
                            "Compositional fields", "Discretization", "Geometry model.Box",
                            "Particles.Initial composition", "Particles.Generator.Reference cell",
                            "Particles.Generator.Random uniform", "Particles.Interpolator.Linear least squares"})
      model.put_child(path,all.get_child(path));
    for(const auto *path : {"Material model.Model name", "Geometry model.Model name",
                            "Formulation.Enable phase field", "Formulation.Reconstruct faults from phase field",
                            "Mesh refinement.Strategy", "Mesh refinement.Initial global refinement",
                            "Mesh refinement.Minimum refinement level", "Mesh refinement.Initial adaptive refinement",
                            "Mesh refinement.Time steps between mesh refinement", "Mesh refinement.Additional refinement times",
                            "Fault reconstruction.Structural point spacing", "Fault reconstruction.Fit prescribed geometry to phase field",
                            "Fault reconstruction.Boundary completion", "Particles.Particle generator name",
                            "Particles.Interpolation scheme", "Particles.Integration scheme", "Particles.List of particle properties",
                            "Particles.Minimum particles per cell", "Particles.Maximum particles per cell",
                            "Particles.Load balancing strategy", "Particles.Particle addition algorithm", "Particles.Particle removal algorithm",
                            "Particles.Generate particle domains", "Particles.Generate CPDI data for particle domains"})
      model.put(path,all.get<std::string>(path));
    // Native population/kernel controls are scalar entries in Particles.
    for(const auto &entry : all.get_child("Particles"))
      if(entry.second.empty()) model.put("Particles."+entry.first,entry.second.data());
    // The BP3 adapter overrides these raw PRM entries by property name before
    // native LLS parses them. Record the effective policy, not the unused mask.
    if(all.get<std::string>("Particles.Interpolation scheme")=="BP3 history linear least squares")
      {
        model.put("Particles.Interpolator.Linear least squares.Use linear least squares limiter",
                  "runtime fields: crack_driving_force, maxwell stress");
        model.put("Particles.Interpolator.Linear least squares.Use boundary extrapolation","false");
      }
    model.put("BP3.loading","stationary symmetric primitive; physical top-to-bottom thrust; full Q1 completion remains core-owned");
    model.put("BP3.loading_rate_m_per_s",Vp);
    model.put("BP3.initial_slip_rate_m_per_s",Vinit);
    model.put("BP3.background_normal_stress_Pa",sigma0);
    model.put("BP3.refinement_policy","support band max(2 ell,R+2 h_fine); exterior min(h_coarse,2 h_fine+(d-band)/4); native startup tagging v2; endpoint strip (R+h_coarse*(abs(nx)+abs(ny)))/sin(dip), plus h_fine margin, depth 2 h_fine");
#ifdef ASPECT_BP3_LOCAL_OSCILLATION_TEST
    model.put("BP3.local_comparison_policy",all.get<std::string>("Mesh refinement.BP3 local comparison.Policy"));
    model.put("BP3.local_comparison_buffer","R+coarse projected width; same maintained boundary strip on A/B and production");
#endif
    report=model;
    report.put_child("Time stepping",all.get_child("Time stepping"));
    report.put_child("Solver parameters",all.get_child("Solver parameters"));
    report.put_child("Postprocess.BP3",all.get_child("Postprocess.BP3"));
    report.put_child("Postprocess.BP3 restored monitor",all.get_child("Postprocess.BP3 restored monitor"));
    for(const auto *path : {"Maximum time step", "Maximum first time step", "Maximum relative increase in time step",
                            "Nonlinear solver tolerance", "Max nonlinear iterations", "Nonlinear solver failure strategy"})
      report.put(path,all.get<std::string>(path));
    std::ostringstream identity,settings;
    boost::property_tree::write_json(identity,model,false);
    boost::property_tree::write_json(settings,report,true);
    return {identity.str(),settings.str()};
  }
}
