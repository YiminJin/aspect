/* P5c: fresh continuum seed and small distributed tensor controls only. */
#include "propagation.cc"
namespace aspect
{
  namespace BoundaryVelocity
  {
    template<int dim> class P5CLoading:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x)const override
        {
          const double t=this->get_time();Tensor<1,dim> v;
          const bool intact=std::getenv("ASPECT_P5C_INTACT");
          if(t<=.3+1e-12) v[0]=.015*(x[1]-.125);
          else if(!intact && t<=.6+1e-12){v[0]=.015*(x[0]-.125);v[1]=-.015*(x[1]-.125);}
          return v;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(P5CLoading,"p5c loading","Small affine intact/non-proportional tensor ownership controls, not specimen loading.")
    template<int dim> class P5CSpecimenLoading:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x)const override
        {const double t=this->get_time();Tensor<1,dim> v;v[0]=(t>.05+1e-12 && t<=.140+1e-12?.018:0.)*(x[1]-.5);return v;}
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(P5CSpecimenLoading,"p5c specimen loading","P4 symmetric shear history truncated to a declared short hold for the bounded P5c comparison.")
  }
  namespace Postprocess
  {
    template<int dim> class P5CChecks:public BulkPropagationChecks<dim>
    {
      double initial_phase(const Point<dim> &x,const PhaseField::PhaseFieldProfile &profile,const double tip)const override
      {
        if(std::getenv("ASPECT_P5C_INTACT"))return 0.;
        if(std::getenv("ASPECT_P5C_PATCH"))return .2+.03*std::sin(2.*numbers::PI*x[0]/.25)*std::sin(2.*numbers::PI*x[1]/.25);
        return profile.value(std::hypot(std::max(0.,x[0]-tip),x[1]-.5));
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(P5CChecks,"p5c checks","Fresh seed or small uniform intact/nonuniform-history MPI checks using native owners.")
  }
}
