#include "bp3_model.h"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iomanip>
#include <iostream>
#include <limits>

int main()
{
  using Reference=boost::multiprecision::cpp_dec_float_50;
  const double old=1855415838.0411193, dt=376359254.47685081;
  unsigned int tests=0;
  for (const double Dc : {.008,.024})
  for (const double V : {3.0611065791756763e-17,1e-18,1e-19,1e-20,1e-9})
    {
      // The independent 50-digit solution checks the benchmark helper;
      // a double-precision sample represents the published history.
      const Reference x=Reference(V)*Reference(dt)/Reference(Dc);
      const Reference exact=Reference(old)*exp(-x)
                            -Reference(Dc)/Reference(V)*expm1(-x);
      const long double expected=BP3::aging_state_reference(V,old,dt,Dc);
      const double actual=static_cast<double>(exact);
      const double error=std::abs(actual/static_cast<double>(expected)-1.);
      if (!(error<1e-12)
          || !(abs(Reference(expected)/exact-1)<Reference(32)*std::numeric_limits<long double>::epsilon()))
        throw std::runtime_error("Stable BP3 Theta audit disagrees with independent reference.");
      // Preserve the threshold: a genuinely wrong committed state must fail.
      if (std::abs((actual*(1+1e-8))/static_cast<double>(expected)-1.)<1e-12)
        throw std::runtime_error("BP3 Theta audit accepted an incorrect committed state.");
      std::cout<<std::setprecision(17)<<"Dc="<<Dc<<", V="<<V<<", relative_error="<<error
               <<", incorrect_state_rejected=1\n";
      ++tests;
    }
  if (BP3::aging_state_reference(1e-20,old,0.,.024)!=old)
    throw std::runtime_error("Zero interval changed Theta.");
  std::cout<<"BP3 Theta audit: "<<tests
           <<" stable-reference cases and incorrect-state rejections; zero interval passed.\n";
}
