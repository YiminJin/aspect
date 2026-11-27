/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/geometry_model/box.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/reconstructed_fault/fault_vertex_table.h>
#include <aspect/time_stepping/interface.h>

namespace aspect::MaterialModel::Rheology
{
  namespace
  {
    // The mature section passes the material segment and its local Q1 coordinate.
    // Only a varies with distance down dip. RateStateHistory owns and restores age.
    class BP3Resistance : public RateStateHistory
    {
      public:
        BP3Resistance(const RateAndStateFriction &law,
                      const double                initial_age,
                      const std::vector<Point<2>> &vertices)
          : RateStateHistory(law, initial_age), shallow_law(law)
        {
          AssertThrow(vertices.size() >= 2, ExcMessage("BP3 requires ordered material vertices."));
          for (const auto &point : vertices)
            stations.push_back(point.distance(vertices.front()));
          for (unsigned int i=0; i<vertices.size(); ++i)
            {
              const double s = stations[i];
              const Point<2> expected(0.5*s, std::sqrt(0.75)*(40000.-s));
              AssertThrow(vertices[i].distance(expected) < 1.e-6 && (i == 0 || s > stations[i-1]),
                          ExcMessage("BP3 expects the ordered 60-degree trace from surface to base."));
            }
          AssertThrow(std::abs(stations.back()-40000.) < 1.e-6,
                      ExcMessage("BP3 requires 40 km of frictional fault."));
        }

        std::string name() const override
        {
          return "BP3 depth dependent RSF";
        }

        std::vector<double> parameters() const override
        {
          auto signature = RateStateHistory::parameters();
          signature.insert(signature.end(), {15000., 3000., 0.025});
          return signature;
        }

        FaultTraction at_speed(const unsigned int segment,
                               const double       xi,
                               const double       speed,
                               const double       compression) const override
        {
          const double s = stations[segment] + xi*(stations[segment+1]-stations[segment]);
          auto law = shallow_law;
          law.direct_effect += (0.025-law.direct_effect)*std::clamp((s-15000.)/3000.,0.,1.);
          const auto &ages = ages_in_seconds();
          return law.resistance(speed, (1.-xi)*ages[segment]+xi*ages[segment+1], compression);
        }

      private:
        const RateAndStateFriction shallow_law;
        std::vector<double> stations;
    };

    std::unique_ptr<FaultResistance> make_resistance(const ParameterHandler &prm,
                                                     const FaultResistance::Route route)
    {
      AssertThrow(route == FaultResistance::Route::mature && prm.get("Fault geometry") == "boundary spanning" &&
                  !prm.get_bool("Fit trace to phase") && !prm.get("Fault vertex file").empty(),
                  ExcMessage("BP3 resistance requires a prescribed boundary-spanning vertex table."));
      const RateAndStateFriction law = {prm.get_double("Reference friction coefficient"), prm.get_double("Direct effect"),
                                        prm.get_double("Evolution effect"), prm.get_double("Reference speed"),
                                        prm.get_double("Slip distance"), prm.get_double("Radiation damping")
                                       };
      // Resistance factories receive no simulation communicator. Read this small
      // static table once per process, including in standalone parameter validation.
      const auto table = FaultVertexTable<2>::read(prm.get("Fault vertex file"), MPI_COMM_SELF);
      AssertThrow(table.initial_ages.size() == table.positions.size(),
                  ExcMessage("BP3 requires an initial RSF age for every station."));
      return std::make_unique<BP3Resistance>(law, prm.get_double("Initial age"), table.positions);
    }
    const bool registered = (FaultResistance::register_law("BP3 depth dependent RSF",
    [](ParameterHandler &, FaultResistance::Route) {}, make_resistance), true);
  }
}

namespace aspect::TimeStepping
{
  // Bound slip in units of Dc and aging in units of the parent age. The
  // timestep manager runs before publication; recheck the actual attempted
  // duration against its own accepted parent before allowing acceptance.
  template <int dim>
  class BP3TimeStep : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Material model");
        prm.enter_subsection("Phase field fault");
        slip_distance = prm.get_double("Slip distance");
        prm.leave_subsection();
        prm.leave_subsection();
      }

      double execute() override
      {
        const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
        const auto &labels = model.labels();
        const auto &ages = model.label_ages();
        AssertThrow(!labels.empty() && ages.size() == labels.size(), ExcMessage("BP3 requires mature RSF history."));
        double limit = std::numeric_limits<double>::max();
        for (unsigned int i=0; i<labels.size(); ++i)
          {
            AssertThrow(labels[i].rate > 0. && std::isfinite(labels[i].rate) &&
                        ages[i] > 0. && std::isfinite(ages[i]), ExcMessage("Invalid accepted BP3 rate or age."));
            limit = std::min({limit, 0.1*slip_distance/labels[i].rate, 0.1*ages[i]});
          }
        return limit;
      }

      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &info) override
      {
        const double bound = execute();
        if (this->get_timestep() > bound*(1.+1.e-12))
          return {Reaction::repeat_step, bound};
        return Interface<dim>::determine_reaction(info);
      }

    private:
      double slip_distance;
  };

  ASPECT_REGISTER_TIME_STEPPING_MODEL(BP3TimeStep, "BP3 slip and state",
                                      "Limit the interval using the replicated accepted mature fault rates and ages.")
}

namespace aspect::GeometryModel
{
  // Build a stationary Cartesian mesh before initialization. The mature phase
  // is never transferred between meshes. Refine every cell touching the normal
  // band, including where it crosses a face obliquely.
  template <int dim>
  class BP3Box : public Box<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &) {}

      void create_coarse_mesh(parallel::distributed::Triangulation<dim> &mesh) const override
      {
        AssertThrow(dim == 2, ExcMessage("The BP3 thrust box requires a 2-D calculation."));
        Box<dim>::create_coarse_mesh(mesh);
        for (unsigned int level=0; level<4; ++level)
          {
            for (const auto &cell : mesh.active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  double low = std::numeric_limits<double>::max();
                  double high = -low;
                  for (const unsigned int i : cell->vertex_indices())
                    {
                      const auto &p = cell->vertex(i);
                      const double r = std::sqrt(0.75)*p[0]+0.5*(p[1]-std::sqrt(0.75)*40000.);
                      low = std::min(low,r);
                      high = std::max(high,r);
                    }
                  if (low <= 1500. && high >= -1500.)
                    cell->set_refine_flag();
                }
            mesh.execute_coarsening_and_refinement();
          }
      }
  };

  ASPECT_REGISTER_GEOMETRY_MODEL(BP3Box, "BP3 fixed box",
                                 "A Cartesian box pre-refined along the stationary 60-degree BP3 thrust.")
}

namespace aspect::Postprocess
{
  // The mature model replicates its accepted material labels. Save compact
  // station observations in ASPECT's restartable statistics table.
  template <int dim>
  class BP3Stations : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &statistics) override
      {
        const auto &model = Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
        const auto &labels = model.labels();
        const auto &ages = model.label_ages();
        const auto column = [&](const std::string &name, const double value)
        {
          statistics.add_value(name,value);
          statistics.set_scientific(name,true);
          statistics.set_precision(name,17);
        };
        const auto fastest = std::max_element(labels.begin(), labels.end(),
                                              [](const auto &a, const auto &b)
        {
          return a.rate < b.rate;
        });
        column("BP3 clock (s)", this->get_time());
        column("BP3 interval (s)", this->get_timestep());
        column("BP3 peak speed (m/s)", fastest->rate);
        column("BP3 peak distance (m)", fastest->position.distance(labels.front().position));
        unsigned int stations_found = 0;
        for (unsigned int i=0; i<labels.size(); ++i)
          {
            const double s = labels[i].position.distance(labels.front().position);
            const unsigned int station = std::lround(s/2500.);
            if (std::abs(s-2500.*station) < 1.e-6)
              {
                const std::string prefix = "BP3 " + std::to_string(station*2500) + " m ";
                column(prefix+"slip (m)", labels[i].slip);
                column(prefix+"rate (m/s)", labels[i].rate);
                column(prefix+"age (s)", ages.at(i));
                ++stations_found;
              }
          }
        AssertThrow(stations_found == 17, ExcMessage("BP3 samples require a material label every 2.5 km."));
        return {"Maximum fault speed", Utilities::to_string(fastest->rate)+" m/s"};
      }
  };

  ASPECT_REGISTER_POSTPROCESSOR(BP3Stations, "BP3 stations",
                                "Accepted maximum speed and down-dip position, slip, speed and age at fixed BP3 stations.")
}
