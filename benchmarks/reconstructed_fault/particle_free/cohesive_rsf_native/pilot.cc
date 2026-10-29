/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/time_stepping/interface.h>
#include <deal.II/grid/grid_tools.h>
#include <fstream>
#include <iomanip>
namespace aspect
{
  namespace GeometryModel
  {
    template <int dim> class CohesivePilotBox : public Box<dim>
    {
    public:
      void
      create_coarse_mesh (parallel::distributed::Triangulation<dim> &grid) const override
      {
        Box<dim>::create_coarse_mesh (grid);
        Tensor<1, dim> shift;
        shift[1] = -6.;
        GridTools::shift (shift, grid);
      }
    };
    ASPECT_REGISTER_GEOMETRY_MODEL (
        CohesivePilotBox, "cohesive pilot box",
        "Finite 12 m wall specimen in physical coordinates; no periodic boundary.")
  }
  namespace BoundaryVelocity
  {
    template <int dim>
    class CohesivePilotLoading : public Interface<dim>, public SimulatorAccess<dim>
    {
      static std::array<double, 2>
      imposed (double t)
      {
        const double z = std::clamp ((t - .410) / .030, 0., 1.);
        const double angle = -20 * numbers::PI / 180 * (z * z * z * (10 + z * (-15 + 6 * z)));
        const double gamma = .0003 + .004 * std::min (t, .455);
        return { gamma * std::cos (2 * angle), gamma * std::sin (2 * angle) };
      }

    public:
      Tensor<1, dim>
      boundary_velocity (types::boundary_id, const Point<dim> &x) const override
      {
        double dt = this->get_timestep ();
        const double t = this->get_time ();
        if (dt == 0.)
          dt = this->get_parameters ().maximum_first_time_step;
        Tensor<1, dim> v;
        if (t == 0.)
          return v;
        const auto now = imposed (t), old = imposed (t - dt);
        const double a = (now[0] - old[0]) / dt, b = (now[1] - old[1]) / dt,
                     axial = .41954981558863996;
        v[0] = a * (-axial * x[0] + .5 * x[1]) + b * (-.5 * x[0] - axial * x[1]);
        v[1] = a * (.5 * x[0] + axial * x[1]) + b * (-axial * x[0] + .5 * x[1]);
        return v;
      }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL (
        CohesivePilotLoading, "cohesive pilot loading",
        "Interval-average physical affine wall velocity for the accepted loading turn and hold; no "
        "stress rotation.")
  }
  namespace TimeStepping
  {
    template <int dim>
    class CohesivePilotReject : public Interface<dim>, public SimulatorAccess<dim>
    {
      bool rejected = false;

    public:
      double
      execute () override
      {
        return std::numeric_limits<double>::max ();
      }
      std::pair<Reaction, double>
      determine_reaction (const TimeStepInfo &) override
      {
        const char *value = std::getenv ("ASPECT_COHESIVE_FORCE_REJECT_STEP");
        if (value && !rejected
            && this->get_timestep_number () == static_cast<unsigned> (std::stoul (value)))
          {
            rejected = true;
            if (Utilities::MPI::this_mpi_process (this->get_mpi_communicator ()) == 0)
              {
                std::ofstream f (this->get_output_directory () + "forced-rejection.csv");
                f << std::setprecision (17) << "step,rejected_time_s,rejected_dt_s,retry_dt_s\n"
                  << this->get_timestep_number () << ',' << this->get_time () << ','
                  << this->get_timestep () << ',' << .5 * this->get_timestep () << '\n';
              }
            return { Reaction::repeat_step, .5 * this->get_timestep () };
          }
        return { Reaction::advance, std::numeric_limits<double>::max () };
      }
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL (
        CohesivePilotReject, "cohesive pilot rejection control",
        "Test-only deterministic outer rejection/cutback, enabled only by an environment variable.")
  }
}
