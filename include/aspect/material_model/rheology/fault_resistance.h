/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_material_model_rheology_fault_resistance_h
#define _aspect_material_model_rheology_fault_resistance_h

#include <aspect/global.h>
#include <optional>
#include <deal.II/base/parameter_handler.h>

namespace aspect::MaterialModel::Rheology
{
  /** Traction in Pa and its derivative with respect to positive slip speed
   * (Pa s/m), at fixed incoming constitutive state and compression. */
  struct FaultTraction
  {
    double traction, speed_derivative;
  };

  /** Signed rate (m/s), resistance (Pa) and condensed derivative df/dVfree
   * (Pa s/m) at fixed strip mobility and accepted parent. */
  struct SectionRoot
  {
    double rate, resistance, slope;
  };

  /** Dimensionless direct and evolution effects and slip distance (m) used to
   * resolve slip at a rate-weakening or neutral material identity. */
  struct SlipResolutionParameters
  {
    double direct_effect, evolution_effect, slip_distance;
  };

  /** Material-owned surface law. Index i follows the material, independent of
   * mesh ownership. Evaluations are local and read only throughout an attempt.
   * The default law history is empty. Only acceptance advances physical state. */
  class FaultResistance
  {
    public:
      virtual ~FaultResistance() = default;
      enum class Route
      {
        mature, growing
      };
      using Declare = void (*)(ParameterHandler &, Route);
      using Create = std::unique_ptr<FaultResistance> (*)(const ParameterHandler &, Route);

      /** Register a surface law once for both the positive-speed and signed
       * contracts. Parameters are relative to the active material subsection. */
      static void register_law(const std::string &name, Declare declare, Create create);
      static void declare_parameters(ParameterHandler &prm, Route route);
      static std::unique_ptr<FaultResistance> create(const ParameterHandler &prm, Route route);
      /** Optional initial ages in seconds, ordered by material identity. */
      virtual void set_initial_ages(std::vector<double> ages)
      {
        AssertThrow(ages.empty(),ExcMessage("The selected fault resistance law does not accept initial ages."));
      }
      virtual const std::vector<double> &ages_in_seconds() const
      {
        static const std::vector<double> empty;
        return empty;
      }
      /** Stable checkpoint identifier of the law. */
      virtual std::string name() const = 0;
      /** Evaluate at Q1 coordinate xi between vertices i and i+1. */
      virtual FaultTraction at_speed(unsigned int i,
                                     double       xi,
                                     double       speed,
                                     double       compression) const = 0;
      /** Eliminate V = Vfree - M f on one entire strip at half-duration h.
       * sign(V) sets the shear sense. This is distinct from the mature
       * positive-speed response evaluated at fixed incoming state. */
      virtual SectionRoot solve_strip(unsigned int i,
                                      double       free_rate,
                                      double       mobility,
                                      double       h,
                                      double       compression) const = 0;
      /** Validate the material's signed midpoint law and its attempt duration. */
      virtual void validate_midpoint() const {}
      virtual bool admits_duration(double /*half_duration*/, double /*compression*/) const
      {
        return true;
      }
      /** Characteristic slip distance (m), or infinity if the law has no slip scale. */
      virtual double characteristic_slip_distance() const
      {
        return std::numeric_limits<double>::infinity();
      }
      /** Local RSF parameters, if the law supports a weakening-dependent slip budget. */
      virtual std::optional<SlipResolutionParameters> slip_resolution_parameters(unsigned int /*id*/) const
      {
        return std::nullopt;
      }
      /** Predict the endpoint age in seconds, starting from parent_age (s)
       * at a constant signed slip rate (m/s) for duration (s). */
      virtual double predict_age(double /*rate*/, double /*parent_age*/, double /*duration*/) const
      {
        AssertThrow(false, ExcMessage("This resistance law does not supply an aging prediction."));
        return 0.;
      }
      /** Longest interval (s) at the given signed rate (m/s) for which the
       * absolute change in log age from parent_age (s) fits the dimensionless
       * budget. A law without age imposes no state restriction. */
      virtual double state_time_limit(double /*rate*/, double /*parent_age*/, double /*log_budget*/) const
      {
        return std::numeric_limits<double>::infinity();
      }
      /** Physical parameters whose values must agree on checkpoint restore. */
      virtual std::vector<double> parameters() const = 0;
      /** Seed optional history once for the ordered vertices or strips. */
      virtual void initialize(unsigned int /*number_of_material_ids*/) {}
      /** Publish one material identity at the accepted signed rate in m/s. */
      virtual void accept_rate(unsigned int /*material_id*/, double /*rate*/, double /*timestep*/) {}
      /** Optional scalar history for checkpoints. A stateful law supplies one
       * value per material identity; a state-free law supplies none. */
      virtual const std::vector<double> &state() const
      {
        static const std::vector<double> empty;
        return empty;
      }
      virtual void restore_state(std::vector<double> values)
      {
        AssertThrow(values.empty(),ExcMessage("A state-free fault law cannot restore constitutive history."));
      }
  };
}
#endif
