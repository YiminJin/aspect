/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/rheology/fault_resistance.h>

namespace aspect::MaterialModel
{
  namespace
  {
    /** A dashpot with impedance in Pa s/m. It carries no material history. */
    class LinearResistance : public Rheology::FaultResistance
    {
      public:
        explicit LinearResistance(double impedance) : impedance(impedance) {}
        std::string name() const override
        {
          return "linear dashpot";
        }
        std::vector<double> parameters() const override
        {
          return {impedance};
        }
        Rheology::FaultTraction at_speed(unsigned int,double,double speed,double) const override
        {
          return {impedance*speed,impedance};
        }
        Rheology::SectionRoot solve_strip(unsigned int,double free,double mobility,double,double) const override
        {
          const double rate=free/(1.+mobility*impedance);
          return {rate,impedance*rate,impedance/(1.+mobility*impedance)};
        }
      private:
        const double impedance;
    };
  }

  namespace
  {
    using Law=Rheology::FaultResistance;
    void declare_dashpot(ParameterHandler &prm, const Law::Route route)
    {
      prm.declare_entry("Dashpot impedance",route==Law::Route::growing?"1000":"20",Patterns::Double(0.),"Fault impedance in Pa s/m.");
    }
    std::unique_ptr<Law> read_dashpot(const ParameterHandler &prm,Law::Route)
    {
      return std::make_unique<LinearResistance>(prm.get_double("Dashpot impedance"));
    }
    const auto registered=(Law::register_law("linear dashpot",declare_dashpot,read_dashpot),true);
  }
}
