/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_TIP_SECTION_GEOMETRY_H
#define ASPECT_TIP_SECTION_GEOMETRY_H

#include <aspect/global.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/cell_id.h>
#include <set>
#include <numeric>

namespace aspect
{
  namespace MaterialModel::BulkCohesive::Tip
  {
    /** Fixed material frame, with the coordinate origin of the specimen input
     * as datum. beta is a signed turn from this tangent, in radians. */
    struct ReferenceFrame
    {
      explicit ReferenceFrame(const double degrees=0.)
      {
        const double radians=degrees*numbers::PI/180.;
        tangent[0]=std::cos(radians);
        tangent[1]=std::sin(radians);
        if (std::remainder(degrees,180.)==0.) tangent[1]=0.;
        if (std::remainder(degrees-90.,180.)==0.) tangent[0]=0.;
      }

      template <int dim>
      double xi(const Point<dim> &p) const { return tangent[0]*p[0]+tangent[1]*p[1]; }

      // Components u,v such that tau:S(beta) = u sin(2 beta) + v cos(2 beta).
      std::array<double,2> traction_moments(const Tensor &tau) const
      {
        const double u=.5*(tau[1][1]-tau[0][0]), v=tau[0][1];
        const double c=tangent[0]*tangent[0]-tangent[1]*tangent[1], s=2.*tangent[0]*tangent[1];
        return {{c*u-s*v,s*u+c*v}};
      }

      Tensor shear(const double beta) const
      {
        const double tx=tangent[0]*std::cos(beta)-tangent[1]*std::sin(beta);
        const double ty=tangent[1]*std::cos(beta)+tangent[0]*std::sin(beta);
        Tensor S;
        // Preserve the certified arithmetic on the horizontal reference.
        if (tangent[0]==1. && tangent[1]==0.)
          {
            S[0][0]=-.5*std::sin(2.*beta);
            S[0][1]=.5*std::cos(2.*beta);
          }
        else
          {
            S[0][0]=-tx*ty;
            S[0][1]=.5*(tx*tx-ty*ty);
          }
        S[1][1]=-S[0][0];
        return S;
      }

      dealii::Tensor<1,2> tangent;
    };

    /** Immutable cell rules for a fixed mesh. Each integration point belongs
     * to exactly one physical chord, including on a nonaligned mesh. */
    template <int dim>
    struct SectionGeometry
    {
      ReferenceFrame frame;
      bool cut=false, strips=false;
      const char *archive_suffix() const
      {
        return strips ? "; cell-local material strips v1" : cut ? "; fixed-frame line quadrature v1" : "";
      }
      std::map<CellId,Quadrature<dim>> rules;

      template <class Cells, class Sections>
      void initialize(const Cells &cells, const MPI_Comm comm, const unsigned int order,
                      const double degrees, const bool exact_lines, const double initial_age, Sections &sections,
                      std::map<CellId,std::vector<unsigned int>> &ids,
                      std::map<CellId,std::vector<double>> &normal_weights,
                      const unsigned int strip_subdivisions=1)
      {
        frame=ReferenceFrame(degrees);
        std::set<double> local;
        unsigned int non_cartesian=0, nonaligned=0;
        double altitude=std::numeric_limits<double>::max();
        for (const auto &entry:cells)
          {
            const auto &cell=entry.second;
            const auto a=cell->vertex(1)-cell->vertex(0), b=cell->vertex(2)-cell->vertex(0);
            AssertThrow((cell->vertex(3)-cell->vertex(2)-a).norm()<=
                        128.*std::numeric_limits<double>::epsilon()*cell->diameter(),
                        ExcMessage("Material sections require affine quadrilateral cells."));
            if (a[1]!=0. || b[0]!=0. || a[0]<=0. || b[1]<=0.) non_cartesian=1;
            const auto normal_component=[&](const auto &v){return frame.tangent[0]*v[1]-frame.tangent[1]*v[0];};
            const auto aligned=[&](const auto &v){return std::min(std::abs(normal_component(v)),
                std::abs(frame.tangent[0]*v[0]+frame.tangent[1]*v[1]))<=128.*std::numeric_limits<double>::epsilon()*v.norm();};
            if (!aligned(a) || !aligned(b)) nonaligned=1;
            altitude=std::min(altitude,std::abs(a[0]*b[1]-a[1]*b[0])/std::max(a.norm(),b.norm()));
            for (const auto v:cell->vertex_indices()) local.insert(frame.xi(cell->vertex(v)));
          }
        strips=!exact_lines && Utilities::MPI::max(nonaligned,comm);
        std::vector<double> knots;
        const double lo=Utilities::MPI::min(local.empty() ? std::numeric_limits<double>::max() : *local.begin(),comm);
        const double hi=Utilities::MPI::max(local.empty() ? std::numeric_limits<double>::lowest() : *local.rbegin(),comm);
        const double tolerance=64.*std::numeric_limits<double>::epsilon()*std::max({std::abs(lo),std::abs(hi),hi-lo});
        if (strips)
          {
            const double base_count=std::max(1.,std::ceil((hi-lo)/Utilities::MPI::min(altitude,comm)));
            AssertThrow(strip_subdivisions>0 && base_count*strip_subdivisions<std::numeric_limits<unsigned int>::max(),
                        ExcMessage("Invalid number of material strips."));
            const unsigned int count=static_cast<unsigned int>(base_count)*strip_subdivisions;
            for (unsigned int i=0;i<=count;++i) knots.push_back(lo+(hi-lo)*i/count);
            for (unsigned int i=0;i<count;++i)
              sections.push_back({.5*(knots[i]+knots[i+1]),knots[i+1]-knots[i],initial_age,0.,0.,0.});
          }
        else
          {
            std::set<double> global;
            for (const auto &part:Utilities::MPI::all_gather(comm,std::vector<double>(local.begin(),local.end())))
              global.insert(part.begin(),part.end());
            for (const double x:global)
              if (knots.empty() || x-knots.back()>tolerance) knots.push_back(x);
          }
        const QGauss<1> line(order);
        for (unsigned int i=1;!strips && i<knots.size();++i)
          for (unsigned int j=0;j<order;++j)
            {
              const double width=knots[i]-knots[i-1];
              sections.push_back({knots[i-1]+width*line.point(j)[0],width*line.weight(j),initial_age,0.,0.,0.});
            }
        // Exact tensor-product order and reference weights for the inherited
        // horizontal mesh. Hanging projected vertices use the chord rule.
        for (const auto &entry:cells)
          {
            const auto &cell=entry.second;
            const auto it=std::upper_bound(knots.begin(),knots.end(),cell->vertex(0)[0]);
            if (it==knots.end() || *it!=cell->vertex(1)[0]) non_cartesian=1;
          }
        cut=Utilities::MPI::max(non_cartesian,comm) || degrees!=0.;
        const auto lower_station=[&](const double x)
        {
          return std::lower_bound(sections.begin(),sections.end(),x,
                                  [](const auto &section,double value){return section.coordinate<value;})-sections.begin();
        };
        for (const auto &entry:cells)
          {
            const auto &cell=entry.second;
            const auto a=cell->vertex(1)-cell->vertex(0), b=cell->vertex(2)-cell->vertex(0);
            const double area=std::abs(a[0]*b[1]-a[1]*b[0]);
            auto &indices=ids[entry.first];
            auto &normal=normal_weights[entry.first];
            if (!cut)
              {
                const unsigned int first=lower_station(cell->vertex(0)[0]);
                for (unsigned int j=0;j<order;++j)
                  for (unsigned int i=0;i<order;++i)
                    {
                      indices.push_back(first+i);
                      normal.push_back(b[1]*line.weight(j));
                    }
                continue;
              }
            std::array<double,4> vertex;
            for (unsigned int v=0;v<4;++v) vertex[v]=frame.xi(cell->vertex(v));
            const auto bounds=std::minmax_element(vertex.begin(),vertex.end());
            std::vector<Point<dim>> points;
            std::vector<double> weights;
            const auto add_chord=[&](const unsigned int k,const double xi,const double longitudinal_weight)
              {
                std::vector<Point<dim>> ends;
                for (const auto &edge: {std::pair<unsigned int,unsigned int>{0,1},{1,3},{3,2},{2,0}})
                  {
                    const auto v=edge.first, w=edge.second;
                    if (xi<=std::min(vertex[v],vertex[w]) || xi>=std::max(vertex[v],vertex[w])) continue;
                    const double t=(xi-vertex[v])/(vertex[w]-vertex[v]);
                    const auto p=GeometryInfo<dim>::unit_cell_vertex(v);
                    ends.push_back(p+t*(GeometryInfo<dim>::unit_cell_vertex(w)-p));
                  }
                AssertThrow(ends.size()==2,ExcMessage("Degenerate material-section chord at a projected vertex."));
                const auto delta=ends[1]-ends[0];
                const double length=(delta[0]*a+delta[1]*b).norm();
                for (unsigned int j=0;j<order;++j)
                  {
                    points.push_back(ends[0]+line.point(j)[0]*delta);
                    normal.push_back(longitudinal_weight/sections[k].weight*length*line.weight(j));
                    weights.push_back(sections[k].weight*normal.back()/area);
                    indices.push_back(k);
                  }
              };
            if (strips)
              {
                // Split only this cell's intersected strips at its own corners.
                // Each chord is quadrature for the same strip-average RSF state.
                const unsigned int first=std::max<std::ptrdiff_t>(0,std::upper_bound(knots.begin(),knots.end(),*bounds.first)-knots.begin()-1);
                for (unsigned int k=first;k<sections.size() && knots[k]<*bounds.second;++k)
                  {
                    std::array<double,6> breaks={{std::max(knots[k],*bounds.first),std::min(knots[k+1],*bounds.second)}};
                    for (unsigned int j=0;j<4;++j) breaks[2+j]=std::clamp(vertex[j],breaks[0],breaks[1]);
                    std::sort(breaks.begin(),breaks.end());
                    for (unsigned int j=1;j<breaks.size();++j)
                      {
                        const double width=breaks[j]-breaks[j-1];
                        if (width<=tolerance) continue;
                        for (unsigned int i=0;i<order;++i)
                          add_chord(k,breaks[j-1]+width*line.point(i)[0],width*line.weight(i));
                      }
                  }
              }
            else
              for (unsigned int k=lower_station(*bounds.first);k<lower_station(*bounds.second);++k)
                add_chord(k,sections[k].coordinate,sections[k].weight);
            AssertThrow(std::abs(std::accumulate(weights.begin(),weights.end(),0.)-1.)<
                        1e-12*std::max(1.,double(weights.size())/100.),
                        ExcMessage("Section quadrature did not cover the material cell."));
            rules.emplace(entry.first,Quadrature<dim>(points,weights));
          }
      }
    };

    /** Reuse ordinary FEValues on the horizontal path. Cut cell histories and
     * their observers must use the very same reference points and ordering. */
    template <int dim>
    class SectionValues
    {
      public:
        SectionValues(const Mapping<dim> &mapping, const FiniteElement<dim> &fe,
                      const SectionGeometry<dim> &geometry, const unsigned int order, const UpdateFlags flags)
          : mapping(mapping),fe(fe),geometry(geometry),flags(flags)
        {
          if (!geometry.cut) values=std::make_unique<FEValues<dim>>(mapping,fe,QGauss<dim>(order),flags);
        }

        template <class Cell>
        const FEValues<dim> &reinit(const Cell &cell)
        {
          if (geometry.cut)
            values=std::make_unique<FEValues<dim>>(mapping,fe,geometry.rules.at(cell->id()),flags);
          values->reinit(cell);
          return *values;
        }

      private:
        const Mapping<dim> &mapping;
        const FiniteElement<dim> &fe;
        const SectionGeometry<dim> &geometry;
        UpdateFlags flags;
        std::unique_ptr<FEValues<dim>> values;
    };
  }
}
#endif
