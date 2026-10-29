// Fixed-phase confined block: reuse native ownership and lifecycle diagnostics.
#define ASPECT_COULOMB_BLOCK_FIXTURE
#include "parallel.cc"
namespace aspect { namespace BoundaryVelocity
{
  template<int dim> class CoulombBlockLoading:public Interface<dim>,public SimulatorAccess<dim>
  {
    public:
      Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x)const override
      {
        const double time=this->get_time(),L=.25,rate=.02;
        Tensor<1,dim> v;
        const char *hold=std::getenv("ASPECT_COULOMB_BLOCK_HOLD_TIME");
        const double stop=hold?std::stod(hold):.25;
        AssertThrow(stop>=.1 && stop<=.25,ExcMessage("Bounded block hold must follow the original shear interval."));
        if(time>stop+1e-12)return v;
        if(time<=.1+1e-12)v[0]=rate*(x[1]-.5*L);
        else {v[0]=rate*(x[0]-.5*L);v[1]=-rate*(x[1]-.5*L);}
        if(std::getenv("ASPECT_COULOMB_BLOCK_NONUNIFORM"))
          {
            v[0]+=.006*L*std::sin(numbers::PI*x[0]/L)*std::cos(numbers::PI*x[1]/L);
            v[1]-=.006*L*std::cos(numbers::PI*x[0]/L)*std::sin(numbers::PI*x[1]/L);
          }
        return v;
      }
  };
  ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(CoulombBlockLoading,"confined Coulomb block",
    "Volume-preserving affine/nonuniform load, change of direction and hold for the fixed-phase Coulomb block.")
}}
