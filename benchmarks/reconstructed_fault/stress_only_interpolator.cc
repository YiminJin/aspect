// Shared benchmark routing: change Maxwell transfer without changing any other
// particle property. Both numerical interpolators are the production plugins.
#include <aspect/particle/interpolator/distance_weighted_average.h>
#include <aspect/particle/interpolator/linear_least_squares.h>
#include <aspect/particle/manager.h>

namespace aspect::Particle::Interpolator
{
  template <int dim>
  class StressOnlyLeastSquares : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void parse_parameters(ParameterHandler &prm) override
      {
        dwa.initialize_simulator(this->get_simulator());
        ls.initialize_simulator(this->get_simulator());
        dwa.set_particle_manager_index(this->get_particle_manager_index());
        ls.set_particle_manager_index(this->get_particle_manager_index());
        dwa.parse_parameters(prm);ls.parse_parameters(prm);
        const auto &info=this->get_particle_manager(this->get_particle_manager_index()).get_property_manager().get_data_info();
        stress=info.get_position_by_field_name("maxwell stress");
        components=info.n_components();
        AssertThrow(info.get_components_by_field_name("maxwell stress")==3,
                    ExcMessage("Stress-only comparison supports the 2-D tensor."));
        prm.enter_subsection("Interpolator");prm.enter_subsection("Linear least squares");
        AssertThrow(prm.get("Use linear least squares limiter")=="false"
                    && prm.get("Use boundary extrapolation")=="false",
                    ExcMessage("Use unlimited least squares for the routed stress components."));
        prm.leave_subsection();prm.leave_subsection();
      }
      void initialize() override {dwa.initialize();ls.initialize();}
      void update() override {dwa.update();ls.update();}
      using Interface<dim>::properties_at_points;
      std::vector<std::vector<double>> properties_at_points(
        const ParticleHandler<dim> &particles,const std::vector<Point<dim>> &points,
        const ComponentMask &selected,
        const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const override
      {
        ComponentMask stress_mask(components,false),other_mask(components,false);
        for (unsigned int c=0;c<components;++c)
          if (selected[c])
            (c>=stress && c<stress+3 ? stress_mask : other_mask).set(c,true);
        if (!stress_mask.n_selected_components()) return dwa.properties_at_points(particles,points,other_mask,cell);
        auto result=ls.properties_at_points(particles,points,stress_mask,cell);
        if (other_mask.n_selected_components())
          {
            const auto other=dwa.properties_at_points(particles,points,other_mask,cell);
            for (unsigned int q=0;q<points.size();++q)
              for (unsigned int c=0;c<components;++c)
                if (other_mask[c]) result[q][c]=other[q][c];
          }
        return result;
      }
    private:
      DistanceWeightedAverage<dim> dwa;
      LinearLeastSquares<dim> ls;
      unsigned int stress=0,components=0;
  };
  ASPECT_REGISTER_PARTICLE_INTERPOLATOR(StressOnlyLeastSquares,"stress only linear least squares",
    "Benchmark routing: unlimited native linear least squares for Maxwell stress; unchanged DWA for every other property.")
}
