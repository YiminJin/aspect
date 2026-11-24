/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_midpoint_system_h
#define _aspect_midpoint_system_h
#include <aspect/simulator/solver/midpoint_fault.h>
#include <aspect/phase_field/midpoint_fracture.h>

namespace aspect::GrowingFault
{
  /** Data for one attempt only. Shape gradients and weights are evaluated
   * by ASPECT's mapping at the native material points, once per attempt. */
  struct MidpointSystem
  {
    using Values=std::vector<double>;
    using Matrix=PhaseField::FractureRows::Matrix;
    using Index=PhaseField::FractureRows::Index;
    struct Shape
    {
      Stress strain;
      double div=0., pressure=0., phase=0.;
    };
    struct Sample
    {
      unsigned int strip;
      Point<2> position;
      double area, old_phase, kernel=0.;
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
      std::vector<SectionRoot> roots;
      History candidate;
    };

    MidpointSystem(const Material<2> &, const LinearAlgebra::BlockVector &);
    /** Collectively assemble the physical residual and optional exact tangent
     * at (h v, p/G, b). With publish enabled, recover a candidate endpoint
     * from these same stages. The material keeps its accepted parent. */
    Evaluation assemble_stage(const Values &,bool tangent,bool publish) const;
    /** Solve a bounded linear correction. Rank zero factors the assembled
     * sparse matrix with one algebraic border per eliminated strip force. */
    Values solve_linearized_obstacle(const Evaluation &,const Values &) const;
    /** Project the incoming wall increments onto the discrete continuity
     * constraint using the uncoupled Maxwell Stokes metric. */
    void restore_continuity(Values &) const;
    /** Substitute a positive phase search metric while retaining the exact
     * mechanics response to the resulting phase correction. */
    void use_phase_search_metric(Evaluation &) const;
    void write_candidate_fields(const Values &,LinearAlgebra::BlockVector &) const;
    const Material<2> &material;
    MPI_Comm comm;
    const double h,G,B,Kf;
    const unsigned int field,phase_block;
    const Index nv,phase_begin,n;
    PhaseField::FractureRows fracture;
    std::map<CellId,Cell> cells;
    Values initial,lower,upper,mass,angles;
    std::map<Index,double> fixed;
    std::vector<unsigned int> formed,eligible;
    /** Collective, once per attempt. Track the connected accepted graph and
     * freeze the material-point weights in the forward disk. */
    void freeze_direction_neighborhood();
    /** Collective, after a converged stage. Compare candidate planes at the
     * returned fields and request a fresh assembly when a trial angle turns. */
    bool select_prospective_planes(const Values &);
  };
}
#endif
