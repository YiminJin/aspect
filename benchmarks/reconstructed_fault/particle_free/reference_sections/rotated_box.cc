/* A rigidly rotated Box for the bulk-tip covariance control. Boundary ids and
 * cell reference coordinates are inherited from the unrotated specimen. */
#include <aspect/geometry_model/box.h>
#include <deal.II/grid/grid_tools.h>

namespace aspect::GeometryModel
{
  template <int dim>
  class SectionTestBox : public Box<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Geometry model");
        prm.declare_entry("Section test box rotation","0",Patterns::Double(-180.,180.),
                          "Rigid specimen rotation in degrees for the covariance benchmark.");
        prm.leave_subsection();
      }

      void parse_parameters(ParameterHandler &prm) override
      {
        Box<dim>::parse_parameters(prm);
        prm.enter_subsection("Geometry model");
        angle=prm.get_double("Section test box rotation")*numbers::PI/180.;
        prm.leave_subsection();
      }

      void create_coarse_mesh(parallel::distributed::Triangulation<dim> &mesh) const override
      {
        Box<dim>::create_coarse_mesh(mesh);
        GridTools::transform([&](const Point<dim> &input)
        {
          auto p=input;
          p[0]=std::cos(angle)*input[0]-std::sin(angle)*input[1];
          p[1]=std::sin(angle)*input[0]+std::cos(angle)*input[1];
          return p;
        },mesh);
      }
    private:
      double angle=0.;
  };

  ASPECT_REGISTER_GEOMETRY_MODEL(SectionTestBox,"section test box","Rigidly rotated fixed box for the section benchmark.")
}
