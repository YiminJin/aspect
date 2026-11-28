/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_mesh_refinement_growing_fault_h
#define _aspect_mesh_refinement_growing_fault_h

#include <aspect/mesh_refinement/interface.h>
#include <aspect/mesh_refinement/refinement_target.h>
#include <aspect/time_stepping/interface.h>
#include <optional>

namespace aspect::MeshRefinement
{
  /**
   * Resolve a formed 2-D material fault band and a physical region ahead of
   * its forward tip. A request contains only geometry in metres. It survives
   * refinement of a rejected trial's parent; no constitutive history is
   * published by this plugin. Completed requests are implicit in the mesh.
   */
  template <int dim>
  class GrowingFault : public Interface<dim>, public SimulatorAccess<dim>, public RefinementTarget
  {
    public:
      static void declare_parameters(ParameterHandler &prm);
      void parse_parameters(ParameterHandler &prm) override;
      void initialize() override;
      void tag_additional_cells() const override;
      bool needs_initial_refinement() const override;
      bool needs_further_refinement() const override;

      /** Collective at the return of the midpoint solve, before acceptance.
       * Keep the physical region to mark if the timestep manager refines. */
      TimeStepping::Reaction assess_endpoint() const;

    private:
      struct Region
      {
        std::vector<Point<2>> band;
        std::vector<Point<2>> sector;
        // Coarse cell footprints intersecting connected phase. Physical boxes
        // remain valid through all levels of a retry of the accepted parent.
        std::vector<BoundingBox<2>> damage;
        Point<2> tip;
        double tip_radius = 0.;
      };
      Region fault_region(bool candidate) const;
      bool mark_region(const Region &region, bool mark) const;
      bool needs_coverage(const Region &region) const;
      double bulk_target;
      double profile_half_width;
      double trigger_radius;
      double restore_radius;
      bool initialized_phase = false;

      enum class Coverage
      {
        sector, disk, reactive
      };
      Coverage coverage;
      mutable std::optional<Region> requested;
  };
}
#endif
