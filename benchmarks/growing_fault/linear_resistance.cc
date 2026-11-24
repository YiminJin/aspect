/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/growing_fault.h>
#include <aspect/material_model/phase_field_fault.h>

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

  template <int dim>
  class LinearStripMaterial : public GrowingFault::Material<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &) {}
      void save(std::map<std::string,std::string> &status) const override
      {
        GrowingFault::Material<dim>::save(status);
        const auto entry=status.find("Native growing fault v2");
        if (entry==status.end()) return;
        std::istringstream bytes(entry->second);
        aspect::iarchive archive(bytes);
        std::vector<double> signature, edges, state;
        unsigned int order, step;
        double angle, time;
        std::string name;
        std::vector<GrowingFault::SectionHistory> strips;
        archive >> signature >> order >> angle >> edges >> step >> time >> name >> strips >> state;
        AssertThrow(name=="linear dashpot" && strips.size()+1==edges.size() && state.empty()
                    && this->friction().state().empty(), ExcMessage("State-free strip archive contains constitutive history."));
      }
    protected:
      std::unique_ptr<Rheology::FaultResistance> make_resistance(ParameterHandler &) const override
      {
        return std::make_unique<LinearResistance>(1000.);
      }
  };

  template <int dim>
  class LinearTraceMaterial : public PhaseFieldFault<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &) {}
      void save(std::map<std::string,std::string> &status) const override
      {
        PhaseFieldFault<dim>::save(status);
        const auto entry=status.find("Material fault labels linear dashpot");
        if (entry==status.end()) return;
        std::istringstream bytes(entry->second);
        unsigned int count;
        bytes >> count;
        AssertThrow(count==this->labels().size() && this->label_ages().empty(),ExcInternalError());
        std::string row;
        std::getline(bytes,row);
        for (unsigned int i=0; i<count; ++i)
          {
            std::getline(bytes,row);
            std::istringstream fields(row);
            std::vector<double> values {std::istream_iterator<double>(fields),std::istream_iterator<double>()};
            // x,y,width,background shear,rate,slip,Maxwell memory.
            AssertThrow(values.size()==7 && values[4]==this->labels()[i].rate &&
                        values[6]==this->label_memory()[i],ExcMessage("State-free vertex archive contains additional history."));
          }
      }
    protected:
      std::unique_ptr<Rheology::FaultResistance> make_resistance(ParameterHandler &) const override
      {
        return std::make_unique<LinearResistance>(20.);
      }
  };
  ASPECT_REGISTER_MATERIAL_MODEL(LinearStripMaterial,"linear strip resistance","State-free signed strip law.")
  ASPECT_REGISTER_MATERIAL_MODEL(LinearTraceMaterial,"linear trace resistance","State-free positive-speed vertex law.")
}
