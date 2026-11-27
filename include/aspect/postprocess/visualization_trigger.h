/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_postprocess_visualization_trigger_h
#define _aspect_postprocess_visualization_trigger_h

namespace aspect::Postprocess
{
  /** Optional interface for active postprocessors that schedule additional
   * native graphical frames. All ranks must agree on the request. The writer
   * acknowledges every actual frame, including ordinary time/step output. */
  class VisualizationTrigger
  {
    public:
      virtual ~VisualizationTrigger() = default;
      virtual bool requests_graphical_output() const = 0;
      virtual void record_graphical_output() = 0;
  };
}
#endif
