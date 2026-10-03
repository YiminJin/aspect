#include "../../bp3/plugin/profile.h"
#include "../../bp3/plugin/geometry.h"
#include <aspect/postprocess/interface.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess
{
  template <int dim>
  class BP3ProfileCheck : public Interface<dim>, public SimulatorAccess<dim>
  {
    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      const auto &cache=BP3::loading_profile(*this);
      const auto &handler=this->get_phase_field_handler();
      const auto profiles=handler.get_phase_field_profiles(BP3::geometry().peak_phase);
      const auto &x=profiles[0]->get_coordinate_values();
      std::vector<double> fractions(profiles.size(),0.);fractions[0]=1.;
      const QGauss<1> quad(12); // Independent higher-order direct integral.
      const auto integrate=[&](double a,double b)
        {
          double value=0.;
          for(unsigned int q=0;q<quad.size();++q)
            value+=(b-a)*quad.weight(q)*(1./handler.energetic_degradation(fractions,profiles[0]->value(a+(b-a)*quad.point(q)[0]))-1.);
          return value;
        };
      std::vector<double> integral(x.size(),0.);
      for(unsigned int i=1;i<x.size();++i) integral[i]=integral[i-1]+integrate(x[i-1],x[i]);
      double max_error=0.,previous=0.;
      std::ofstream out;
      if(this->get_pcout().is_active())
        {out.open(this->get_output_directory()+"loading_profile.csv");out<<std::setprecision(17)<<"r,phi,C,reference\n";}
      for(unsigned int i=0;i<=2000;++i)
        {
          const double r=cache.support*(-1.1+2.2*i/2000.);
          const double a=std::abs(r);
          double reference=r<0 ? 0. : 1.;
          if(a<cache.support)
            {
              const unsigned int j=std::upper_bound(x.begin(),x.end(),a)-x.begin()-1;
              reference=.5+std::copysign(.5,r)*(integral[j]+integrate(x[j],a))/integral.back();
            }
          const double value=cache.cumulative(r);
          max_error=std::max(max_error,std::abs(value-reference));
          AssertThrow(value>=previous && value>=0. && value<=1.,ExcMessage("Nonmonotone loading primitive."));
          AssertThrow(std::abs(value+cache.cumulative(-r)-1.)<3e-16,ExcMessage("Asymmetric loading primitive."));
          previous=value;
          if(out) out<<r<<','<<profiles[0]->value(a)<<','<<value<<','<<reference<<'\n';
        }
      AssertThrow(max_error<2e-13 && cache.cumulative(0.)==.5
                  && cache.cumulative(-cache.support)==0. && cache.cumulative(cache.support)==1.,
                  ExcMessage("Loading primitive fails independent quadrature/endpoints."));
      const auto &manager=this->get_reconstructed_fault_manager();
      AssertThrow(manager.get_boundary_contacts().size()==2,ExcMessage("Missing automatic contacts."));
      for(const auto &c:manager.get_boundary_contacts())
        {
          const double sign=std::copysign(1.,c.normal*c.inward_boundary_normal);
          const auto exterior=c.position-1e-4*c.inward_tangent;
          AssertThrow(!manager.project_to_bulk_source(exterior+sign*(c.transverse_extent+1.)*c.normal).active,
                      ExcMessage("Endpoint handoff enlarged transverse support."));
          AssertThrow(!manager.project_to_bulk_source(exterior-sign*c.transverse_extent*.5*c.normal).active,
                      ExcMessage("Endpoint handoff crossed physical boundary."));
          const auto interior=c.position+1e-3*c.inward_tangent+sign*.99*c.transverse_extent*c.normal;
          AssertThrow(!manager.project_to_normal_profiles(interior).active
                      && !manager.project_to_bulk_source(interior,true).active
                      && !manager.project_to_bulk_source(interior).active,
                      ExcMessage("Endpoint handoff grew beyond endpoint roundoff."));
        }
      this->get_pcout()<<"BP3 PROFILE/ENDPOINT PASS: maximum primitive error="<<max_error<<std::endl;
      return {};
    }
  };
  ASPECT_REGISTER_POSTPROCESSOR(BP3ProfileCheck,"BP3 profile check","Independent local profile and endpoint checks.")
}}
