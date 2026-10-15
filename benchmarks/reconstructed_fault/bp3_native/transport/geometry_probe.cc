/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */

#include "geometry_reference.h"
#include <voro++.hh>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main(int argc, char **argv)
{
  using namespace TransportGeometry;
  if (argc!=3) throw std::runtime_error("Usage: geometry_probe failed_domain.csv output.csv");
  std::ifstream input(argv[1]);
  std::string line;
  Point parent, lower, upper;
  bool patch_bounds=false;
  Polygon neighbors;
  std::getline(input,line);
  while (std::getline(input,line))
    {
      std::replace(line.begin(),line.end(),',',' ');
      std::istringstream row(line);
      std::string kind;
      Point p;
      row>>kind>>p[0]>>p[1];
      if (kind=="parent") parent=p;
      if (kind=="lower")
        {
          lower=p;
          patch_bounds=true;
        }
      if (kind=="upper") upper=p;
      if (kind=="neighbor") neighbors.push_back(p);
    }
  const double h=1./128;
  if (!patch_bounds)
    {
      lower= {{parent[0]-3*h,parent[1]-3*h}};
      upper= {{parent[0]+3*h,parent[1]+3*h}};
    }
  const double length=std::sqrt((upper[0]-lower[0])*(upper[1]-lower[1])/neighbors.size());
  const double blocks=std::cbrt(neighbors.size()/(3.4*(upper[0]-lower[0])*(upper[1]-lower[1])));
  const int nx=int((upper[0]-lower[0])*blocks+1), ny=int((upper[1]-lower[1])*blocks+1);
  Polygon reference= {{{-3*h,-3*h}},{{3*h,-3*h}},{{3*h,3*h}},{{-3*h,3*h}}};
  for (const auto &p : neighbors)
    {
      const Point d= {{p[0]-parent[0],p[1]-parent[1]}};
      if (std::hypot(d[0],d[1])>h/10)
        reference=clip(reference,d,.5*(d[0]*d[0]+d[1]*d[1]));
    }
  const auto exact=moments(reference);
  const auto exact_op=cpdi(reference,parent,8);
  std::ofstream output(argv[2]);
  output<<std::setprecision(17)<<"conditioned,scale,shift_x,shift_y,area_error,centroid_error,second_moment_error,weight_error,gradient_error,weak_error,weak_gradient_error,symmetric_difference,vertices\n";
  const std::array<double,8> scales= {{.001,.01,.1,1.,10.,100.,128.,1000.}};
  const std::array<bool,2> modes= {{false,true}};
  for (const bool conditioned : modes)
    for (const double scale : scales)
      {
        const std::array<Point,3> shifts= {{{{0,0}},{{-parent[0] *scale,-parent[1] *scale}},{{1000*scale,-2000*scale}}}};
        for (const Point shift : shifts)
          {
            const auto transform=[&](const Point &p)
            {
              return Point {{scale *p[0]+shift[0],scale *p[1]+shift[1]}};
            };
            const Point origin=conditioned ? transform({{.5*(lower[0]+upper[0]),.5*(lower[1]+upper[1])}}) : Point {{0,0}};
            const double unit=conditioned ? scale*length : 1.;
            const auto chart=[&](const Point &p)
            {
              const auto q=transform(p);
              return Point {{(q[0]-origin[0])/unit,(q[1]-origin[1])/unit}};
            };
            const Point low=chart(lower), high=chart(upper);
            const double slab=.01*std::hypot(high[0]-low[0],high[1]-low[1]);
            voro::container container(low[0],high[0],low[1],high[1],-slab,slab,nx,ny,1,false,false,false,8);
            for (unsigned int i=0; i<neighbors.size(); ++i)
              {
                const auto p=chart(neighbors[i]);
                container.put(i,p[0],p[1],0);
              }
            voro::c_loop_all loop(container);
            if (!loop.start()) throw std::runtime_error("Empty container");
            bool found=false;
            do
              {
                const auto id=loop.pid();
                if (std::hypot(neighbors[id][0]-parent[0],neighbors[id][1]-parent[1])>h/10) continue;
                voro::voronoicell_neighbor cell;
                if (!container.compute_cell(cell,loop)) throw std::runtime_error("Voro++ failed");
                double x,y,z;
                loop.pos(x,y,z);
                std::vector<double> vertices;
                cell.vertices(x,y,z,vertices);
                Polygon polygon;
                for (unsigned int v=0; v<vertices.size(); v+=3)
                  if (vertices[v+2]>0)
                    polygon.push_back({{(origin[0]+unit *vertices[v]-shift[0])/scale-parent[0],(origin[1]+unit *vertices[v+1]-shift[1])/scale-parent[1]}});
                std::sort(polygon.begin(),polygon.end(),[](const Point &a,const Point &b)
                {
                  return std::atan2(a[1],a[0])<std::atan2(b[1],b[0]);
                });
                const auto m=moments(polygon);
                const auto op=cpdi(polygon,parent,8);
                double second=0, weights=0, gradients=0, weak=0, weak_gradient=0;
                for (unsigned int k=3; k<6; ++k)
                  second=std::max(second,std::abs(m[k]/m[0]-exact[k]/exact[0])/(h*h));
                auto delta=op;
                for (const auto &entry : exact_op)
                  for (unsigned int k=0; k<3; ++k) delta[entry.first][k]-=entry.second[k];
                for (const auto &entry : delta)
                  {
                    weak+=std::abs(entry.second[0])/exact[0];
                    weak_gradient+=std::hypot(entry.second[1],entry.second[2])/(8*exact[0]);
                  }
                auto normalized=op;
                for (auto &entry : normalized)
                  for (auto &value : entry.second) value/=m[0];
                for (const auto &entry : exact_op)
                  for (unsigned int k=0; k<3; ++k) normalized[entry.first][k]-=entry.second[k]/exact[0];
                for (const auto &entry : normalized)
                  {
                    weights+=std::abs(entry.second[0]);
                    gradients+=std::hypot(entry.second[1],entry.second[2])/8;
                  }
                const double centroid=std::hypot(m[1]/m[0]-exact[1]/exact[0],m[2]/m[0]-exact[2]/exact[0])/h;
                output<<conditioned<<','<<scale<<','<<shift[0]<<','<<shift[1]<<','<<(m[0]-exact[0])/exact[0]<<','<<centroid<<','<<second<<','<<weights<<','<<gradients<<','<<weak<<','<<weak_gradient<<','<<symmetric_difference(polygon,reference)/exact[0]<<','<<polygon.size()<<'\n';
                found=true;
              }
            while (loop.inc());
            if (!found) throw std::runtime_error("Parent not found");
          }
      }
}
