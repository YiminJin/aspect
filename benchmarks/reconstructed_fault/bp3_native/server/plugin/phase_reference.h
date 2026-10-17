#ifndef ASPECT_BP3_PHASE_REFERENCE_H
#define ASPECT_BP3_PHASE_REFERENCE_H
#include <deal.II/base/point.h>
#include <string>
namespace BP3
{
  // Optional immutable physical Q1 snapshot for a mesh-resolution diagnostic.
  // No constitutive histories or normalization values are imported.
  void read_phase_reference(const std::string &path);
  bool has_phase_reference();
  double phase_reference_value(double x, double y);
  double phase_reference_spacing();
}
#endif
