#ifndef ASPECT_BENCHMARK_BP3_RUNTIME_H
#define ASPECT_BENCHMARK_BP3_RUNTIME_H

#include <aspect/simulator_access.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/particles/particle.h>
#include <string>
#include <vector>
#include <map>

namespace aspect
{
  struct ReconstructedFaultSurfaceResidual;
  namespace BP3Benchmark
  {
    // Observations shared by initialization callbacks and accepted-state output.
    // Constitutive history remains simulator/manager owned and is never stored here.
    extern bool converged, long_run_stop, restored_history, detailed_diagnostics;
    extern std::string mature_prestress_file;
    extern unsigned int newton_updates, krylov_iterations;
    extern double minimum_alpha, accepted_nonlinear_residual;
    extern std::vector<std::vector<bool>> final_active;
    // Owned/ghost particle audit, transported by the native particle handler.
    extern std::map<types::particle_index,double> work_initial_H;
    extern std::map<types::particle_index,double> checkpoint_particle_H;
    void restore_particle_audit(types::particle_index next_id);
    template <int dim>
    void connect_particle_history_audit(const SimulatorAccess<dim> &sim);
    extern std::vector<Point<2>> work_initial_geometry;
    extern std::vector<double> work_initial_I;

    template <int dim>
    void verify_accepted_work(const SimulatorAccess<dim> &sim,
                              const ReconstructedFaultSurfaceResidual &weak,
                              bool write_diagnostics);

    template <int dim>
    void verify_paired_mesh(const SimulatorAccess<dim> &sim);
  }

  namespace BP3Restore
  {
    extern std::string bottom_velocity_constraint;
    template <int dim>
    void constrain_bottom(const SimulatorAccess<dim> &, AffineConstraints<double> &);
    // The live-profile primitive is prepared lazily before its first consumer;
    // boundary evaluations then use interpolation only.
    template <int dim>
    Tensor<1,dim> loading(const SimulatorAccess<dim> &sim, const Point<dim> &point);
  }
}
#endif
