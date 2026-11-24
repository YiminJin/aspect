/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/material_sections.h>
#include <deal.II/base/quadrature_lib.h>
#include <numeric>

namespace aspect::GrowingFault
{
  double MaterialSections::coordinate(const Point<2> &p) const
  {
    return tangent[0]*p[0]+tangent[1]*p[1];
  }

  double MaterialSections::width(const unsigned int k) const
  {
    return edges.at(k+1)-edges[k];
  }

  double MaterialSections::length(const unsigned int k, const double beta) const
  {
    return width(k)/std::cos(beta);
  }

  SymmetricTensor<2,3> MaterialSections::shear(const double beta) const
  {
    const double x=tangent[0]*std::cos(beta)-tangent[1]*std::sin(beta);
    const double y=tangent[1]*std::cos(beta)+tangent[0]*std::sin(beta);
    SymmetricTensor<2,3> S;
    S[0][0]=-x*y;
    S[1][1]=x*y;
    S[0][1]=.5*(x*x-y*y);
    return S;
  }

  MaterialSections::MaterialSections(const Triangulation<2> &mesh, const MPI_Comm comm,
                                     const double degrees, const unsigned int order)
  {
    tangent[0]=std::cos(degrees*numbers::PI/180.);
    tangent[1]=std::sin(degrees*numbers::PI/180.);
    if (std::remainder(degrees,180.)==0.) tangent[1]=0.;
    if (std::remainder(degrees-90.,180.)==0.) tangent[0]=0.;
    double lo=std::numeric_limits<double>::max(), hi=-lo, altitude=lo;
    unsigned int invalid=0;
    for (const auto &cell:mesh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto a=cell->vertex(1)-cell->vertex(0), b=cell->vertex(2)-cell->vertex(0);
          invalid |= a[1]!=0. || b[0]!=0. || a[0]<=0. || b[1]<=0.
                     || (cell->vertex(3)-cell->vertex(2)-a).norm()>128.*std::numeric_limits<double>::epsilon()*cell->diameter();
          altitude=std::min({altitude,a[0],b[1]});
          for (const auto v:cell->vertex_indices())
            {
              const double xi=coordinate(cell->vertex(v));
              lo=std::min(lo,xi);
              hi=std::max(hi,xi);
            }
        }
    AssertThrow(!Utilities::MPI::max(invalid,comm),ExcMessage("Material sections require Cartesian cells."));
    lo=Utilities::MPI::min(lo,comm);
    hi=Utilities::MPI::max(hi,comm);
    const unsigned int count=std::ceil((hi-lo)/Utilities::MPI::min(altitude,comm));
    for (unsigned int k=0; k<=count; ++k) edges.push_back(lo+(hi-lo)*k/count);
    const QGauss<1> line(order);
    const double tolerance=64.*std::numeric_limits<double>::epsilon()*std::max({std::abs(lo),std::abs(hi),hi-lo});
    for (const auto &cell:mesh.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          std::array<double,4> vertex;
          for (unsigned int i=0; i<4; ++i) vertex[i]=coordinate(cell->vertex(i));
          const auto bounds=std::minmax_element(vertex.begin(),vertex.end());
          const auto a=cell->vertex(1)-cell->vertex(0), b=cell->vertex(2)-cell->vertex(0);
          std::vector<Point<2>> points;
          std::vector<double> weights;
          std::vector<unsigned int> ids;
          const unsigned int first=std::max<std::ptrdiff_t>(0,std::upper_bound(edges.begin(),edges.end(),*bounds.first)-edges.begin()-1);
          for (unsigned int k=first; k<count && edges[k]<*bounds.second; ++k)
            {
              // Between successive projected corners both chord ends are affine
              // in xi. The outer Gauss integration is in metres along the seed.
              std::array<double,6> cuts= {{std::max(edges[k],*bounds.first),std::min(edges[k+1],*bounds.second)}};
              for (unsigned int v=0; v<4; ++v) cuts[v+2]=std::clamp(vertex[v],cuts[0],cuts[1]);
              std::sort(cuts.begin(),cuts.end());
              for (unsigned int p=1; p<cuts.size(); ++p)
                {
                  const double span=cuts[p]-cuts[p-1];
                  if (span<=tolerance) continue;
                  for (unsigned int i=0; i<order; ++i)
                    {
                      const double xi=cuts[p-1]+span*line.point(i)[0];
                      std::vector<Point<2>> ends;
                      for (const auto edge:
                      {
                        std::pair<unsigned int,unsigned int> {0,1}, {1,3}, {3,2}, {2,0}
                      })
                      {
                        const auto [v,w]=edge;
                        if (xi<=std::min(vertex[v],vertex[w]) || xi>=std::max(vertex[v],vertex[w])) continue;
                        const auto unit=GeometryInfo<2>::unit_cell_vertex(v);
                        ends.push_back(unit+(xi-vertex[v])/(vertex[w]-vertex[v])*(GeometryInfo<2>::unit_cell_vertex(w)-unit));
                      }
                      AssertThrow(ends.size()==2,ExcMessage("A material chord needs two distinct edge intersections."));
                      const auto delta=ends[1]-ends[0];
                      const double chord=(delta[0]*a+delta[1]*b).norm();
                      for (unsigned int j=0; j<order; ++j)
                        {
                          points.push_back(ends[0]+line.point(j)[0]*delta);
                          weights.push_back(span*line.weight(i)*chord*line.weight(j)/(a[0]*b[1]));
                          ids.push_back(k);
                        }
                    }
                }
            }
          AssertThrow(std::abs(std::accumulate(weights.begin(),weights.end(),0.)-1.)<1e-12,
                      ExcMessage("The material strips must cover each physical cell."));
          cells.emplace(cell->id(),CellRule {Quadrature<2>(points,weights),std::move(ids)});
        }
  }
}
