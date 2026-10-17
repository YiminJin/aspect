/* Independent closed-neighborhood insertion control against the actual Voro library. */
#include <voro++.hh>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <vector>
#include <array>

int main()
{
  const double lower[2]={.25,.3125},upper[2]={.75,.4375};
  const double length=.0625,origin[2]={.5,.375};
  std::vector<std::array<double,2>> points;
  for(unsigned int i=0;i<6;++i)for(unsigned int j=0;j<24;++j)
    {
      const double x=(i+.5)/6,y=(j+.5)/24;
      if(x>=lower[0] && x<=upper[0] && y>=lower[1] && y<=upper[1])points.push_back({{x,y}});
    }
  std::cout<<std::setprecision(17)<<"{\"intended\":"<<points.size();
  unsigned int counts[2]={};
  double areas[2]={};
  for(unsigned int closed=0;closed<2;++closed)
    {
      double a[2],b[2];
      for(unsigned int d=0;d<2;++d)
        {
          a[d]=(lower[d]-origin[d])/length;
          b[d]=(upper[d]-origin[d])/length;
          if(closed)
            {
              const double u=std::numeric_limits<double>::epsilon()/2;
              const double gamma=8*u/(1-8*u);
              const double pad=gamma*(std::abs(a[d])+std::abs(b[d])+std::abs(b[d]-a[d]));
              b[d]=std::nextafter(b[d]+pad,std::numeric_limits<double>::infinity());
            }
        }
      voro::container container(a[0],b[0],a[1],b[1],-1,1,2,2,1,false,false,false,8);
      for(unsigned int i=0;i<points.size();++i)
        container.put(i,(points[i][0]-origin[0])/length,(points[i][1]-origin[1])/length,0);
      voro::c_loop_all loop(container);
      voro::voronoicell cell;
      if(loop.start())do
        {
          ++counts[closed];
          const auto &p=points[loop.pid()];
          if(std::abs(p[0]-(3.5/6))<1e-15 && std::abs(p[1]-(9.5/24))<1e-15)
            {
              if(!container.compute_cell(cell,loop))return 2;
              areas[closed]=cell.volume()*length*length/2;
            }
        }while(loop.inc());
    }
  std::cout<<",\"original_inserted\":"<<counts[0]<<",\"closed_inserted\":"<<counts[1]
           <<",\"original_target_area\":"<<areas[0]<<",\"closed_target_area\":"<<areas[1]
           <<",\"independent_bisector_area\":"<<1./144<<"}\n";
  return counts[0]<points.size() && counts[1]==points.size() && std::abs(areas[1]-1./144)<1e-15 ? 0:1;
}
