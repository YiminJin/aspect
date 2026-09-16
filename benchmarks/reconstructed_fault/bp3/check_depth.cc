#include "bp3_model.h"
#include <iostream>
#include <iomanip>

int main()
{
  double error=0., state_error=0.;
  for (unsigned int i=0; i<=10000; ++i)
    {
      const double s=40000.*i/10000.;
      const double x=BP3::trace_x-BP3::cosine*s, y=BP3::box_size-BP3::sine*s;
      error=std::max(error,std::abs(BP3::depth_fraction(y)-BP3::fraction(BP3::down_dip(x,y))));
      state_error=std::max(state_error,std::abs(
        BP3::theta0((BP3::box_size-y)/BP3::sine)/BP3::theta0(BP3::down_dip(x,y))-1.));
    }
  if (error>1e-14 || state_error>1e-12) return 1;
  std::cout << std::setprecision(17) << "Sharp-fault fraction maximum error=" << error
            << "; Theta relative error=" << state_error
            << "; bulk interfaces y=" << BP3::box_size-15000.*BP3::sine
            << ", " << BP3::box_size-18000.*BP3::sine
            << "; vertical transition width=" << 3000.*BP3::sine << '\n';
}
