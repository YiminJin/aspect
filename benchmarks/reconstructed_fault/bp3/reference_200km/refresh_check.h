#include <aspect/particle/property/initial_composition.h>

namespace aspect::BP3Benchmark
{
  template<int dim> void check_strengthening_refresh(const SimulatorAccess<dim> &sim)
  {
    if (!std::getenv("ASPECT_BP3_REFRESH_TEST")) return;
    auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
    const auto &property=pm.get_property_manager().template get_matching_active_plugin<Particle::Property::InitialComposition<dim>>();
    auto &handler=pm.get_particle_handler();
    auto cell=handler.begin()->get_surrounding_cell();
    auto range=handler.particles_in_cell(cell);
    const auto slot=pm.get_property_manager().get_data_info().get_position_by_field_name("initial strengthening");
    std::vector<std::vector<double>> saved;
    for (const auto &p:range) saved.emplace_back(p.get_properties().begin(),p.get_properties().end());
    const auto position=range.begin()->get_location();
    struct Restore
    {
      decltype(range) &particles;
      const std::vector<std::vector<double>> &saved;
      Point<dim> position;
      ~Restore()
      {
        particles.begin()->set_location(position);
        unsigned int i=0;
        for (auto &p:particles)
          { std::copy(saved[i].begin(),saved[i].end(),p.get_properties().begin()); ++i; }
      }
    } restore{range,saved,position};
    std::vector<double> initialized;
    property.initialize_one_particle_property(position,initialized);
    AssertThrow(initialized.size()==2 && initialized[1]==BP3::depth_fraction(position[1]),
                ExcMessage("Selected initial strengthening differs from the former spatial rule."));
    for (const double fraction:{.25,.75})
      {
        Point<dim> moved=position;
        moved[1]=BP3::box_size-(15000.+3000.*fraction)*BP3::sine;
        range.begin()->set_location(moved);
        Particle::Property::ParticleUpdateInputs<dim> inputs;
        property.update_particle_properties(inputs,range);
        AssertThrow(std::abs(range.begin()->get_properties()[slot]-fraction)<1e-14,
                    ExcMessage("Displaced strengthening failed the spatial refresh test."));
        unsigned int i=0;
        for (const auto &p:range)
          {
            for (unsigned int c=0;c<saved[i].size();++c)
              if (c!=slot) AssertThrow(p.get_properties()[c]==saved[i][c],
                                      ExcMessage("Spatial refresh changed unrelated particle history."));
            ++i;
          }
      }
    sim.get_pcout()<<"Selected-field strengthening: initialization and displaced 0.25/0.75 transition tests passed; unrelated fields retained."<<std::endl;
  }
}
