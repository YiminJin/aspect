/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef aspect_cohesive_system_h
#define aspect_cohesive_system_h
#include <aspect/simulator_access.h>
#include <functional>
#include <memory>

namespace aspect
{
  namespace CohesiveInterface
  {
    /** Default-off 2D fixed-mesh native exact-jump/cohesive–RSF system.
     *
     * The Simulator supplies its actual distributed Stokes inverse with
     * homogeneous constraints. Only the small surface operators are reduced;
     * bulk histories and native solution vectors remain on their MPI owners.
     * Geometry and material history publish as one immutable accepted state.
     */
    class System : public SimulatorAccess<2>
    {
    public:
      using Field = LinearAlgebra::BlockVector;
      using BulkSolve = std::function<std::shared_ptr<Field> (const Field &)>;
      struct Trial;
      struct StepMetrics
      {
        bool pending = false;
        double dt = 0., speed = 0., slip_Dc = 0., log_state = 0., log_speed = 0.;
        double damage_increment = 0.;
        double resolved_log_speed=0.,continuous_error=0.,numerical_resolution=0.;
        double displacement_resolution=0.,resolution_slip_Dc=0.,resolution_damage=0.;
        unsigned int rate_point = 0;
        double rate_x = 0., rate_y = 0., old_damage = 0., new_damage = 0.;
        double old_rate = 0., new_rate = 0., old_jump = 0., new_jump = 0.;
      };
      /** Candidate metrics are read-only and never mutate accepted histories. */
      StepMetrics step_metrics () const;

      explicit System (const Simulator<2> &);
      ~System () override;
      static void declare_parameters (ParameterHandler &);
      void parse_parameters (ParameterHandler &);
      void assemble_bulk_matrix (LinearAlgebra::BlockSparseMatrix &, double pressure_scaling);
      bool bulk_matrix_ready () const;
      bool initialized () const;
      bool has_restart_record () const;
      bool at_end_time () const;
      std::shared_ptr<Field> initialize (const Field &particular_velocity, const BulkSolve &);
      std::shared_ptr<Trial> solve (const Field &particular_increment,
                                    const Field &incoming_solution, const BulkSolve &);
      const Field &solution (const Trial &) const;
      void publish (const std::shared_ptr<Trial> &);
      void write_output () const;

      /** Native checkpoint payload, called collectively only after outer
       * acceptance. Loading holds physical CellId records until ASPECT has
       * restored its mesh, DoFs and solution vectors. */
      std::string checkpoint () const;
      void load_checkpoint (const std::string &);
      void restore ();

    private:
      struct Implementation;
      std::unique_ptr<Implementation> data;
    };
  }
}
#endif
