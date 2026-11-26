/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/rheology/fault_resistance.h>
#include <boost/algorithm/string/join.hpp>
#include <numeric>

namespace aspect::MaterialModel::Rheology
{
  namespace
  {
    // A stateless SDK law with an observable impedance on either fault route.
    // Its parameter names, types and declaration context exercise the reader.
    class WeightedDashpot : public FaultResistance
    {
      public:
        explicit WeightedDashpot(const double impedance) : impedance(impedance) {}
        std::string name() const override
        {
          return "weighted dashpot";
        }
        std::vector<double> parameters() const override
        {
          return {impedance};
        }

        FaultTraction at_speed(unsigned int, double, const double speed, double) const override
        {
          return {impedance * speed, impedance};
        }

        SectionRoot solve_strip(unsigned int, const double free_rate, const double mobility,
                                double, double) const override
        {
          const double slope = impedance / (1. + mobility * impedance);
          return {free_rate / (1. + mobility * impedance), slope * free_rate, slope};
        }

      private:
        const double impedance;
    };

    void declare_parameters(ParameterHandler &prm, const FaultResistance::Route route)
    {
      prm.leave_subsection();
      prm.leave_subsection();
      const unsigned int dimension = prm.get_integer("Dimension");
      prm.enter_subsection("Material model");
      prm.enter_subsection(route == FaultResistance::Route::mature ? "Phase field fault" : "Growing fault");

      AssertThrow(!prm.subsection_path_exists({"Nested law"}),
                  ExcMessage("Duplicate external resistance-law declaration in the same material subsection."));
      prm.declare_entry("Dashpot impedance (Pa.s/m)", "1000", Patterns::Double(0.),
                        "Impedance multiplying the mean weight, in Pa s/m.");
      prm.enter_subsection("Nested law");
      const std::vector<std::string> weights(dimension, "1");
      prm.declare_entry("Weights", boost::algorithm::join(weights, ","),
                        Patterns::List(Patterns::Double(0.), dimension, dimension),
                        "One dimensionless weight per spatial direction.");
      prm.leave_subsection();
    }

    std::unique_ptr<FaultResistance> create(const ParameterHandler &prm, FaultResistance::Route)
    {
      const auto weights = Utilities::string_to_double(Utilities::split_string_list(prm.get({"Nested law"},"Weights")));
      const double mean = std::accumulate(weights.begin(), weights.end(), 0.) / weights.size();
      return std::make_unique<WeightedDashpot>(prm.get_double("Dashpot impedance (Pa.s/m)") * mean);
    }

    const auto registered = (FaultResistance::register_law("weighted dashpot", declare_parameters, create), true);
  }
}
