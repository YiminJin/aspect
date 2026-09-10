// Probe the installed Voro++ API on temporary prisms only. No live CPDI data.
#include <voro++.hh>
#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
  voro::voronoicell original;
  original.init(-1,1,-1,1,-.5,.5);
  bool passed=true;
  for (const double offset : {-.3,0.,.3,1.-1e-8,1.-1e-13,1.,1.1})
    {
      voro::voronoicell left,right;
      left=original;
      right=original;
      const bool has_left=left.plane(1,0,0,2*offset);
      const bool has_right=right.plane(-1,0,0,-2*offset);
      const double a=has_left ? left.volume() : 0;
      const double b=has_right ? right.volume() : 0;
      const double exact=2*std::max(0.,std::min(2.,1+offset));
      std::cout << std::setprecision(17) << offset << ' ' << a << ' ' << b
                << ' ' << a+b-4 << ' ' << a-exact << '\n';
      passed &= std::abs(a-exact)<1e-10 && std::abs(a+b-4)<1e-10;
    }
  voro::voronoicell diagonal;
  diagonal=original;
  diagonal.plane(1,1,0,1.4);
  std::cout << "oblique " << diagonal.volume() << " original " << original.volume() << '\n';
  passed &= std::abs(diagonal.volume()-3.155)<1e-12 && original.volume()==4;
  return passed ? 0 : 1;
}
