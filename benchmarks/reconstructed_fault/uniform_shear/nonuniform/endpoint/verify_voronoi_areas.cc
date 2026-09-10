// Independent cross-check of manufactured cells against ASPECT's Voro++ library.
// This does not change ParticleDomainHandler or enable periodic construction.
#include <voro++.hh>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char **argv)
{
  if (argc != 2)
    return 2;
  constexpr double width = .3088215939070757;
  constexpr double thickness = .02;
  voro::container container(0,.25,-width,width,-thickness/2,thickness/2,
                           8,32,1,false,false,false,8);
  std::ifstream input(argv[1]);
  std::vector<double> expected;
  unsigned int id;
  double x,y,area;
  while (input >> id >> x >> y >> area)
    {
      if (id != expected.size())
        return 3;
      expected.push_back(area);
      container.put(id,x,y,0);
    }
  if (!input.eof() || expected.empty())
    return 4;
  voro::c_loop_all loop(container);
  voro::voronoicell cell;
  double max_relative_error=0, total_area=0;
  unsigned int count=0;
  if (loop.start())
    do
      {
        if (!container.compute_cell(cell,loop))
          return 5;
        const double actual=cell.volume()/thickness;
        max_relative_error=std::max(max_relative_error,std::abs(actual/expected.at(loop.pid())-1));
        total_area+=actual;
        ++count;
      }
    while (loop.inc());
  std::cout << std::setprecision(17)
            << "{\"particles\":" << count << ",\"max_relative_area_error\":"
            << max_relative_error << ",\"total_area_m2\":" << total_area << "}\n";
  return count==expected.size() && max_relative_error<1e-10
         && std::abs(total_area-.25*2*width)<1e-11 ? 0 : 6;
}
