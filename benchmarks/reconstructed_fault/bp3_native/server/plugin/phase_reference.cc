#include "phase_reference.h"
#include "geometry.h"
#include <deal.II/base/exceptions.h>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>

namespace BP3
{
  namespace
  {
    struct Cell
    {
      double x, y, h;
      std::array<double,4> phi;
    };
    std::map<std::array<long,3>,Cell> cells;
    std::string reference_path;
    double finest=std::numeric_limits<double>::infinity();
    double base=0.;
    unsigned int maximum_level=0;
  }
  void read_phase_reference(const std::string &path)
  {
    using namespace dealii;
    if(path.empty()) return;
    if(!reference_path.empty())
      { AssertThrow(path==reference_path,ExcMessage("Phase reference changed.")); return; }
    std::ifstream in(path);
    AssertThrow(in,ExcMessage("Cannot read fixed Q1 phase reference."));
    std::string line;
    std::getline(in,line);
    AssertThrow(line=="x,y,h,phi00,phi10,phi01,phi11",ExcMessage("Invalid phase reference columns."));
    // The fixture's 4x2 Box base mesh consists of 500 m squares.
    base=geometry().extent[0]/4.;
    AssertThrow(base==geometry().extent[1]/2.,ExcMessage("Phase reference requires the retained 4x2 Box."));
    double area=0.;
    while(std::getline(in,line))
      {
        for(auto &c:line) if(c==',') c=' ';
        std::istringstream row(line);
        Cell c;
        row>>c.x>>c.y>>c.h;
        for(auto &v:c.phi) row>>v;
        AssertThrow(row && c.h>0.,ExcMessage("Malformed phase reference cell."));
        const long level=std::lround(std::log2(base/c.h));
        AssertThrow(level>=0 && level<25 && std::ldexp(c.h,level)==base,
                    ExcMessage("Phase reference must be dyadic."));
        const long i=std::lround((c.x-geometry().origin[0])/c.h);
        const long j=std::lround((c.y-geometry().origin[1])/c.h);
        AssertThrow(std::abs(c.x-geometry().origin[0]-i*c.h)<1e-10
                    && std::abs(c.y-geometry().origin[1]-j*c.h)<1e-10,
                    ExcMessage("Misaligned phase reference cell."));
        for(const auto v:c.phi)
          AssertThrow(std::isfinite(v) && v>=0. && v<1.,ExcMessage("Inadmissible reference phase."));
        AssertThrow(cells.emplace(std::array<long,3>{{level,i,j}},c).second,
                    ExcMessage("Duplicate reference cell."));
        maximum_level=std::max(maximum_level,static_cast<unsigned int>(level));
        finest=std::min(finest,c.h);area+=c.h*c.h;
      }
    AssertThrow(!cells.empty() && area==geometry().extent[0]*geometry().extent[1],
                ExcMessage("Phase reference does not cover the Box."));
    // Reject nested overlap, not merely incorrect total area.
    for(const auto &entry:cells)
      for(long level=entry.first[0]-1;level>=0;--level)
        {
          const double h=std::ldexp(base,-level);
          const auto &c=entry.second;
          AssertThrow(cells.count({{level,long(std::floor((c.x+c.h/2-geometry().origin[0])/h)),
                                           long(std::floor((c.y+c.h/2-geometry().origin[1])/h))}})==0,
                      ExcMessage("Overlapping phase reference cells."));
        }
    reference_path=path;
  }
  bool has_phase_reference() { return !reference_path.empty(); }
  double phase_reference_spacing() { return finest; }
  double phase_reference_value(const double x,const double y)
  {
    using namespace dealii;
    const auto &g=geometry();
    AssertThrow(has_phase_reference() && x>=g.origin[0]-1e-10 && y>=g.origin[1]-1e-10
                && x<=g.origin[0]+g.extent[0]+1e-10 && y<=g.origin[1]+g.extent[1]+1e-10,
                ExcMessage("Point outside fixed phase reference."));
    for(int level=maximum_level;level>=0;--level)
      {
        const double h=std::ldexp(base,-level);
        const long i=std::min(long(std::floor((x-g.origin[0])/h)),long(std::lround(g.extent[0]/h))-1);
        const long j=std::min(long(std::floor((y-g.origin[1])/h)),long(std::lround(g.extent[1]/h))-1);
        const auto found=cells.find({{level,i,j}});
        if(found==cells.end()) continue;
        const auto &c=found->second;
        const double a=(x-c.x)/h,b=(y-c.y)/h;
        return (1-a)*(1-b)*c.phi[0]+a*(1-b)*c.phi[1]+(1-a)*b*c.phi[2]+a*b*c.phi[3];
      }
    AssertThrow(false,ExcMessage("Uncovered fixed phase reference point."));
    return 0.;
  }
}
