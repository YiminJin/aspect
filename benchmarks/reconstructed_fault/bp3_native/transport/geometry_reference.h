/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */

#ifndef ASPECT_TRANSPORT_GEOMETRY_REFERENCE_H
#define ASPECT_TRANSPORT_GEOMETRY_REFERENCE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

// Independent, test-only convex clipping and CPDI centroid-fan integration.
// Coordinates supplied to clipping and moment evaluation are parent-relative.
namespace TransportGeometry
{
  using Point = std::array<double,2>;
  using Polygon = std::vector<Point>;
  using Operator = std::map<unsigned int,std::array<double,3>>;

  inline double cross(const Point &a, const Point &b)
  {
    return a[0]*b[1]-a[1]*b[0];
  }

  inline Polygon clip(const Polygon &polygon, const Point &normal, const double rhs)
  {
    Polygon result;
    for (unsigned int i=0; i<polygon.size(); ++i)
      {
        const auto &a=polygon[i], &b=polygon[(i+1)%polygon.size()];
        const double va=a[0]*normal[0]+a[1]*normal[1]-rhs;
        const double vb=b[0]*normal[0]+b[1]*normal[1]-rhs;
        if (va<=0) result.push_back(a);
        if ((va<0 && vb>0) || (vb<0 && va>0))
          {
            const double f=va/(va-vb);
            result.push_back({{a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1])}});
          }
      }
    return result;
  }

  // Integral of (1,x,y,x*x,x*y,y*y). Fan triangles are exact to degree two.
  inline std::array<double,6> moments(Polygon polygon)
  {
    std::array<double,6> result = {{0,0,0,0,0,0}};
    if (polygon.size()<3) return result;
    for (unsigned int i=1; i+1<polygon.size(); ++i)
      {
        const auto &a=polygon[0], &b=polygon[i], &c=polygon[i+1];
        const double area=.5*((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]));
        const double sx=a[0]+b[0]+c[0], sy=a[1]+b[1]+c[1];
        result[0]+=area;
        result[1]+=area*sx/3;
        result[2]+=area*sy/3;
        result[3]+=area*(sx*sx+a[0]*a[0]+b[0]*b[0]+c[0]*c[0])/12;
        result[4]+=area*(sx*sy+a[0]*a[1]+b[0]*b[1]+c[0]*c[1])/12;
        result[5]+=area*(sy*sy+a[1]*a[1]+b[1]*b[1]+c[1]*c[1])/12;
      }
    if (result[0]<0)
      for (auto &value : result) value=-value;
    return result;
  }

  inline Polygon orient(Polygon polygon)
  {
    double area=0;
    for (unsigned int i=0; i<polygon.size(); ++i)
      area+=cross(polygon[i],polygon[(i+1)%polygon.size()]);
    if (area<0) std::reverse(polygon.begin(),polygon.end());
    return polygon;
  }

  inline double symmetric_difference(const Polygon &a, const Polygon &b)
  {
    auto intersection=a;
    const auto boundary=orient(b);
    for (unsigned int i=0; i<boundary.size(); ++i)
      {
        const auto &x=boundary[i], &y=boundary[(i+1)%boundary.size()];
        const Point n= {{y[1]-x[1],x[0]-y[0]}};
        intersection=clip(intersection,n,n[0]*x[0]+n[1]*x[1]);
      }
    return std::max(0.,moments(a)[0]+moments(b)[0]-2*moments(intersection)[0]);
  }

  // Same mathematical CPDI fan as production, independently evaluated in a
  // parent-relative chart. Returned entries are unnormalized integrals of
  // (Nbar, grad_x Nbar, grad_y Nbar), with physical gradients.
  inline Operator cpdi(Polygon polygon, const Point &parent, const unsigned int n)
  {
    polygon=orient(polygon);
    const auto m=moments(polygon);
    const Point center= {{m[1]/m[0],m[2]/m[0]}};
    const auto samples=[&](const Point &relative)
    {
      double x=parent[0]+relative[0], y=parent[1]+relative[1];
      x-=std::floor(x);
      const int ix=std::min(int(n)-1,int(std::floor(n*x)));
      const int iy=std::max(0,std::min(int(n)-1,int(std::floor(n*y))));
      const double u=n*x-ix, v=n*y-iy;
      std::map<unsigned int,double> values;
      for (unsigned int j=0; j<2; ++j)
        for (unsigned int i=0; i<2; ++i)
          values[(iy+j)*n+(ix+i)%n]=(i ? u : 1-u)*(j ? v : 1-v);
      return values;
    };
    const auto nc=samples(center);
    Operator result;
    for (unsigned int i=0; i<polygon.size(); ++i)
      {
        const auto &a=polygon[i], &b=polygon[(i+1)%polygon.size()];
        const auto na=samples(a), nb=samples(b);
        const double twice_area=(b[0]-a[0])*(center[1]-a[1])-(b[1]-a[1])*(center[0]-a[0]);
        // Integrated barycentric gradients equal half the outward edge normals.
        const std::array<Point,3> gradients = {{{{b[1]-center[1],center[0]-b[0]}},
            {{center[1]-a[1],a[0]-center[0]}},
            {{a[1]-b[1],b[0]-a[0]}}
          }
        };
        const std::array<std::map<unsigned int,double>,3> values= {{na,nb,nc}};
        for (unsigned int k=0; k<3; ++k)
          for (const auto &entry : values[k])
            {
              auto &target=result[entry.first];
              target[0]+=entry.second*twice_area/6;
              target[1]+=entry.second*gradients[k][0]/2;
              target[2]+=entry.second*gradients[k][1]/2;
            }
      }
    return result;
  }
}
#endif
