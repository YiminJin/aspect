/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Reuse the qualified seed/history observer and alter only the selected loading
// endpoint. Environment control belongs to this isolated benchmark, not physics.
#include "p5c_native.cc"

namespace aspect
{
  namespace BoundaryVelocity
  {
    template<int dim> class P5CCampaignLoading:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x)const override
        {
          const char *value=std::getenv("ASPECT_P5C_LOADING_END");
          const double end=value?std::stod(value):.200;
          AssertThrow(end>=.13875 && end<=.200+1e-12,
                      ExcMessage("P5c campaign loading endpoint is outside its selected interval."));
          Tensor<1,dim> v;
          v[0]=(this->get_time()>.05+1e-12 && this->get_time()<=end+1e-12?.018:0.)*(x[1]-.5);
          return v;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(P5CCampaignLoading,"p5c campaign loading",
      "Selected finite-specimen extension of the same shear rate, with a prescribed endpoint and hold.")
  }
}
