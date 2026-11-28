/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_mesh_refinement_refinement_target_h
#define _aspect_mesh_refinement_refinement_target_h

namespace aspect::MeshRefinement
{
  /**
   * Optional contract for a marker that requests several levels in one
   * timestep reaction. After each mesh change the simulator queries the
   * same physical target before deciding whether to start another solve.
   */
  class RefinementTarget
  {
    public:
      virtual ~RefinementTarget() = default;

      /** Collective, after the last scheduled initialization pass. A target
       * may discover underresolved support in the actual initialized field
       * that was unavailable during early mesh marking. */
      virtual bool needs_initial_refinement() const
      {
        return false;
      }

      /** Collective. Do owned active cells still need refinement for this request? */
      virtual bool needs_further_refinement() const = 0;
  };
}

#endif
