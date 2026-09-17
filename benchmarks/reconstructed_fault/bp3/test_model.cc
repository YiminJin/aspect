#include "reference_200km/bp3_model.h" // Airy diagnostic, never production prestress.
#include <iostream>
#include <iomanip>

int main()
{
  using namespace BP3;
  auto require=[](bool condition, const char *message)
  { if (!condition) throw std::runtime_error(message); };
  require(std::abs(theta0(0)-8000.)<1e-8, "Theta0 VW");
  require(std::abs(theta0(Wf)-8e6)<1e-7, "Theta0 VS");
  double max_div=0, max_surface=0, max_fault=0;
  for (double C : {0., 1e6})
    {
      for (double r : {100., 400., 2500., 15000., 40000., 100000.})
        {
          const auto q=stress(trace_x-r*cosine, box_size-r*sine,C);
          max_fault=std::max(max_fault,std::abs(shear(q)-tau0-C));
          max_fault=std::max(max_fault,std::abs(normal(q)-sigma0));
        }
      for (double offset : {-20000.,-100.,100.,20000.})
        {
          const auto q=stress(trace_x+offset,box_size,C);
          max_surface=std::max({max_surface,std::abs(q[1]),std::abs(q[2])});
          const double x=trace_x+offset, y=box_size-5000, h=.01;
          const auto xp=stress(x+h,y,C), xm=stress(x-h,y,C);
          const auto yp=stress(x,y+h,C), ym=stress(x,y-h,C);
          max_div=std::max({max_div,std::abs((xp[0]-xm[0]+yp[2]-ym[2])/(2*h)),
                           std::abs((xp[2]-xm[2]+yp[1]-ym[1])/(2*h))});
        }
    }
  require(max_fault<1e-6,"Fault prestress/sign mismatch");
  require(max_surface<1e-6,"Top traction mismatch");
  require(max_div<1e-4,"Airy equilibrium mismatch");
  // The cohesive augmentation varies down dip. Check the radial potential,
  // including its derivatives, rather than testing only constant prestress.
  const auto variable_stress=[](double x,double y)
  {
    const double xb=trace_x-x,z=box_size-y,r=std::hypot(xb,z);
    const bool plus=xb*sine-z*cosine>=0;
    const int side=plus ? 1 : -1;
    const double angle=plus ? dip : std::acos(-1.)-dip;
    const double theta=std::atan2(z,side*xb);
    const double slope=10., q=tau0+1e6+slope*r;
    auto result=Airy(angle,-sigma0,0).stress(theta,side);
    const auto addition=Airy(angle,0,side).radial_stress(theta,side,
      tau0+1e6+.5*slope*r,q,r*slope);
    for (unsigned int j=0;j<3;++j) result[j]+=addition[j];
    return result;
  };
  double variable_div=0,variable_fault=0,variable_top=0;
  for (double r:{100.,2500.,18000.,100000.})
    {
      const auto s=variable_stress(trace_x-r*cosine,box_size-r*sine);
      variable_fault=std::max({variable_fault,std::abs(shear(s)-tau0-1e6-10*r),
                               std::abs(normal(s)-sigma0)});
    }
  for (double offset:{-20000.,-100.,100.,20000.})
    {
      const auto s=variable_stress(trace_x+offset,box_size);
      variable_top=std::max({variable_top,std::abs(s[1]),std::abs(s[2])});
      const double x=trace_x+offset,y=box_size-5000,h=.01;
      const auto xp=variable_stress(x+h,y),xm=variable_stress(x-h,y);
      const auto yp=variable_stress(x,y+h),ym=variable_stress(x,y-h);
      variable_div=std::max({variable_div,std::abs((xp[0]-xm[0]+yp[2]-ym[2])/(2*h)),
                            std::abs((xp[2]-xm[2]+yp[1]-ym[1])/(2*h))});
    }
  require(variable_fault<1e-6,"Variable cohesive augmentation fault traction");
  require(variable_top<1e-6,"Variable cohesive augmentation top traction");
  require(variable_div<1e-4,"Variable cohesive augmentation equilibrium");
  // The manager's CCW normal and tangent give exactly official thrust q
  // in this coordinate chart. Reversing both does not reverse the slip sense.
  const auto q=stress(trace_x-1000*cosine,box_size-1000*sine,0);
  const double manager_q=cosine*(-sine*q[0]+cosine*q[2])
                        +sine*(-sine*q[2]+cosine*q[1]);
  require(std::abs(manager_q-tau0)<1e-6,"Manager thrust convention");
  std::cout << std::setprecision(17) << "G=" << G << " damping=" << damping
            << " tau0=" << tau0 << " max_fault_error_Pa=" << max_fault
            << " max_top_traction_Pa=" << max_surface
            << " max_divergence_Pa_per_m=" << max_div
            << " variable_fault_error_Pa="<<variable_fault
            << " variable_top_error_Pa="<<variable_top
            << " variable_divergence_Pa_per_m="<<variable_div<<'\n';
}
