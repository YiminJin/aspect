/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_midpoint_system_h
#define _aspect_midpoint_system_h
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/phase_field/midpoint_fracture.h>
#include <aspect/reconstructed_fault/onset_score.h>

namespace aspect::GrowingFault
{
  /** Data for one attempt only. Shape gradients and weights are evaluated
   * by ASPECT's mapping at the owned native material points, once per attempt.
   * The replicated stage vector is ordered by displacement increment h*v (m),
   * pressure p/G (dimensionless), and midpoint Bernstein phase coefficients.
   * Native history remains with the material until the simulator accepts a step. */
  struct MidpointSystem
  {
    using Values=std::vector<double>;
    using Matrix=PhaseField::FractureRows::Matrix;
    using Index=PhaseField::FractureRows::Index;
    struct Shape
    {
      Stress strain; //!< 3-D deviator of a velocity shape's plane-strain symmetric gradient (1/m).
      double div=0., pressure=0., phase=0.;
    };
    struct Sample
    {
      unsigned int strip;
      Point<2> position;
      double area;      //!< Mapped quadrature weight (m²).
      double old_phase; //!< Phase interpolated from accepted Bernstein coefficients.
      double kernel=0.; //!< Compact direction-search weight frozen from the accepted front.
      std::vector<Shape> shapes;
    };
    struct Cell
    {
      std::vector<Index> indices;
      std::vector<Sample> points;
    };
    struct Evaluation
    {
      Values residual;
      Matrix jacobian; // Local element blocks and borders for the exact strip derivative.
      Balance ledger;
      std::vector<MaterialModel::Rheology::SectionRoot> roots;
      History candidate;
    };

    MidpointSystem(const SimulatorAccess<2>         &simulator,
                   const StripMaterialHistory<2>    &material,
                   const LinearAlgebra::BlockVector &solution);
    /** Collectively assemble the physical residual and optional exact tangent
     * at (h v, p/G, b). With publish enabled, recover a candidate endpoint
     * from these same stages. The material keeps its accepted parent. */
    Evaluation assemble_stage(const Values &stage_fields, bool tangent, bool publish) const;
    /** Solve a bounded linear correction. Rank zero factors the assembled
     * sparse matrix with one algebraic border per eliminated strip force. */
    Values solve_linearized_obstacle(const Evaluation &evaluation,
                                     const Values     &stage_fields) const;
    /** Project the incoming wall increments onto the discrete continuity
     * constraint using the uncoupled Maxwell Stokes metric. */
    void restore_continuity(Values &stage_fields) const;
    /** Substitute a positive phase search metric while retaining the exact
     * mechanics response to the resulting phase correction. */
    void use_phase_search_metric(Evaluation &evaluation) const;
    /** Recover endpoint fields into the attempt's ASPECT solution vector, after
     * collectively checking the bounds of the extrapolated phase polynomial. */
    void write_candidate_fields(const Values               &stage_fields,
                                LinearAlgebra::BlockVector &solution) const;
    const SimulatorAccess<2> &sim;
    const StripMaterialHistory<2> &material;
    MPI_Comm comm;
    const double h;  //!< Half the attempted timestep (s).
    const double G;  //!< Shear modulus (Pa).
    const double B;  //!< Maxwell relaxation over h.
    const double Kf; //!< Fracture density scale Gc/(cw*ell) (Pa).
    const unsigned int field,phase_block;
    const Index nv,phase_begin,n; //!< Start of pressure, start of phase, and end of physical rows.
    PhaseField::FractureRows fracture;
    const SectionPressures friction_pressures;
    std::map<CellId,Cell> cells;
    Values initial,lower,upper,mass,angles;
    std::map<Index,double> fixed;
    std::vector<unsigned int> formed,eligible;
    /** Collective, once per attempt. Track the connected accepted graph and
     * freeze the material-point weights in the forward disk. */
    void freeze_direction_neighborhood();
    /** Local score and resistance at one prospective angle, using globally
     * reduced moments and the section compression frozen for this attempt. */
    OnsetScore score_plane(unsigned int section,
                           const double *moments,
                           const double *centered_moments,
                           double       angle) const;
    /** Collective, after a converged stage. Compare candidate planes at the
     * returned fields and request a fresh assembly when a trial angle turns. */
    bool select_prospective_planes(const Values &stage_fields);
  };
}
#endif
