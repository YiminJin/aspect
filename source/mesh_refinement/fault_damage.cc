/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/mesh_refinement/fault_damage.h>
#include <numeric>

namespace aspect::MeshRefinement::FaultDamage
{
  namespace
  {
    using Patch = BernsteinPatch<2>;
    using Box = BoundingBox<2>;
    using Interval = std::pair<long double,long double>;
    using Polynomial = std::array<long double,5>;

    long double value(const Polynomial &p, unsigned int degree, const long double x)
    {
      long double result=p[degree];
      while (degree>0)
        result=result*x+p[--degree];
      return result;
    }

    // The exact degree is at most two on a Cartesian face, four on the
    // inclined seed line. Split at the derivative's roots to bracket every
    // change of sign, including two crossings between consecutive FE nodes.
    std::vector<long double> roots(const Polynomial &p, unsigned int degree)
    {
      while (degree>0 && p[degree]==0.)
        --degree;
      if (degree==0)
        return {};
      if (degree==1)
        {
          const long double x=-p[0]/p[1];
          return (x>0. && x<1.) ? std::vector<long double> {x} :
                 std::vector<long double> {};
        }
      Polynomial derivative {};
      for (unsigned int i=1; i<=degree; ++i)
        derivative[i-1]=i*p[i];
      auto ends=roots(derivative,degree-1);
      ends.push_back(1.);
      long double start=0.;
      std::vector<long double> result;
      for (const auto end : ends)
        {
          const auto left=value(p,degree,start), right=value(p,degree,end);
          if (left!=0. && right!=0. && (left<0.)!=(right<0.))
            {
              long double a=start,b=end;
              for (unsigned int i=0; i<std::numeric_limits<long double>::digits; ++i)
                {
                  const auto mid=(a+b)/2;
                  const auto f=value(p,degree,mid);
                  if (f==0.)
                    {
                      a=b=mid;
                      break;
                    }
                  if ((left<0.)==(f<0.))
                    a=mid;
                  else
                    b=mid;
                }
              result.push_back((a+b)/2);
            }
          if (end<1. && right==0.)
            result.push_back(end);
          start=end;
        }
      return result;
    }

    // Positive-length intervals of phi >= contour. A constant at the contour
    // has full support. An isolated equality cannot make a face connection.
    std::vector<Interval> superlevel(const Polynomial &p, const unsigned int degree)
    {
      auto ends=roots(p,degree);
      ends.push_back(1.);
      std::vector<Interval> result;
      long double a=0.;
      for (const long double b : ends)
        {
          if (a<b && value(p,degree,(a+b)/2)>=0.)
            result.emplace_back(a,b);
          a=b;
        }
      return result;
    }

    using Row = std::array<double,3>;

    Row face_row(const Patch &patch, const unsigned int axis, const unsigned int side)
    {
      const unsigned int start=side*(axis==0 ? 2 : 6), stride=axis==0 ? 3 : 1;
      return {{patch[start],patch[start+stride],patch[start+2*stride]}};
    }

    std::vector<Interval> face_intervals(const Row &row)
    {
      const auto [lo,hi]=std::minmax_element(row.begin(),row.end());
      if (*lo>=damage_contour)
        return {{0.,1.}};
      if (*hi<=damage_contour)
        return {};
      const long double a=row[0], b=row[1], c=row[2];
      return superlevel({{a-damage_contour,2*(b-a),a-2*b+c,0.,0.}},2);
    }

    enum class Excursion { empty, connected, ambiguous };

    Excursion classify_interior(const Patch &patch)
    {
      // If phi is monotone in x, every superlevel point can reach the high
      // x edge on its own horizontal line. One interval on that edge then
      // proves connectivity of the whole excursion (likewise for y).
      for (unsigned int axis=0; axis<2; ++axis)
        {
          bool increasing=true,decreasing=true;
          const unsigned int stride=axis==0 ? 1 : 3;
          for (unsigned int i=0; i<patch.size(); ++i)
            if ((i/stride)%3 != 2)
              {
                increasing &= patch[i+stride]>=patch[i];
                decreasing &= patch[i+stride]<=patch[i];
              }
          if (increasing || decreasing)
            {
              const Row high=face_row(patch,axis,increasing);
              // A monotone patch takes its maximum on this edge. If both
              // endpoints lie below the contour, the quadratic has a
              // positive interval precisely when its middle control value
              // (relative to the contour) exceeds the geometric mean of the
              // two endpoint deficits. Equality can give a ridge of zero
              // area even when the Bernstein hull extends above the contour.
              if (high[0]<=damage_contour && high[2]<=damage_contour)
                {
                  const long double middle=static_cast<long double>(high[1])-damage_contour;
                  const long double left=static_cast<long double>(damage_contour)-high[0];
                  const long double right=static_cast<long double>(damage_contour)-high[2];
                  if (middle<=0. || middle*middle<=left*right)
                    return Excursion::empty;
                }
              // A quadratic has disconnected superlevel intervals only if
              // both ends reach the contour and an interior minimum dips below.
              if (std::min(high[0],high[2])<damage_contour ||
                  high[1]>=std::min(high[0],high[2]) ||
                  face_intervals(high).size()==1)
                return Excursion::connected;
            }
        }
      return Excursion::ambiguous;
    }

    struct Leaf
    {
      Box bounds;
      Patch phase;
      unsigned int cell;
    };

    void split_excursion(const Leaf &leaf, const unsigned int depth, std::vector<Leaf> &leaves)
    {
      const auto [lo,hi]=std::minmax_element(leaf.phase.begin(),leaf.phase.end());
      if (*hi<damage_contour || (*hi==damage_contour && *lo<damage_contour))
        return; // Strictly below in the open cell by positivity of all nine basis functions.
      const Excursion interior=*lo>=damage_contour ? Excursion::connected : classify_interior(leaf.phase);
      if (interior==Excursion::empty)
        return;
      if (interior==Excursion::connected || depth==5)
        {
          leaves.push_back(leaf);
          return;
        }
      const auto children=PhaseField::subdivide_bernstein<2>(leaf.phase);
      const auto &[lower,upper]=leaf.bounds.get_boundary_points();
      const auto center=leaf.bounds.center();
      for (unsigned int d=0; d<2; ++d)
        AssertThrow(center[d]>lower[d] && center[d]<upper[d],
                    dealii::ExcMessage("The physical coordinates cannot resolve the fault damage subdivision."));
      for (unsigned int i=0; i<4; ++i)
        {
          const Point<2> a((i&1) ? center[0] : lower[0], (i&2) ? center[1] : lower[1]);
          const Point<2> b((i&1) ? upper[0] : center[0], (i&2) ? upper[1] : center[1]);
          split_excursion({Box({a,b}),children[i],leaf.cell},depth+1,leaves);
        }
    }

    // Distance in metres along anchor + s*tangent. Clip against the physical
    // cell first; no root-grid coordinates or assumed isotropic cell sizes.
    std::optional<Interval> incoming_line(const Box         &bounds,
                                          const Point<2>    &anchor,
                                          const Tensor<1,2> &tangent,
                                          long double        end)
    {
      long double start=-std::numeric_limits<long double>::infinity();
      const auto &[lo,hi]=bounds.get_boundary_points();
      for (unsigned int d=0; d<2; ++d)
        if (tangent[d]==0.)
          {
            if (anchor[d]<lo[d] || anchor[d]>hi[d]) return {};
          }
        else
          {
            const long double a=(static_cast<long double>(lo[d])-anchor[d])/tangent[d];
            const long double b=(static_cast<long double>(hi[d])-anchor[d])/tangent[d];
            start=std::max(start,std::min(a,b));
            end=std::min(end,std::max(a,b));
          }
      if (start>end) return {};
      return Interval {start,end};
    }

    std::vector<Interval> supported_line(const Leaf         &leaf,
                                         const Point<2>     &anchor,
                                         const Tensor<1,2>  &tangent,
                                         const long double   end)
    {
      const auto line=incoming_line(leaf.bounds,anchor,tangent,end);
      if (!line) return {};
      const auto [a,b]=*line;
      if (*std::min_element(leaf.phase.begin(),leaf.phase.end())>=damage_contour)
        return {*line};
      // Restrict the tensor polynomial to the clipped physical line. Each
      // one-dimensional Bernstein basis factor is quadratic in line position.
      using Basis = std::array<std::array<long double,3>,3>;
      std::array<Basis,2> basis;
      for (unsigned int d=0; d<2; ++d)
        {
          const auto lo=leaf.bounds.get_boundary_points().first[d];
          const auto h=leaf.bounds.side_length(d);
          const long double u=(static_cast<long double>(anchor[d])-lo+a*tangent[d])/h;
          const long double du=(b-a)*tangent[d]/h;
          basis[d]= {{{{(1-u) *(1-u),-2*(1-u) *du,du*du}},
              {{2*u *(1-u),2*(1-2*u) *du,-2*du*du}},
              {{u*u,2*u*du,du*du}}
            }
          };
        }
      Polynomial trace {};
      for (unsigned int j=0; j<3; ++j)
        for (unsigned int i=0; i<3; ++i)
          for (unsigned int m=0; m<3; ++m)
            for (unsigned int n=0; n<3; ++n)
              trace[m+n]+=(static_cast<long double>(leaf.phase[3*j+i])-damage_contour)*basis[0][i][m]*basis[1][j][n];
      auto intervals=superlevel(trace,4);
      for (auto &[lo,hi] : intervals)
        {
          lo=(lo==0.) ? a : a+(b-a)*lo;
          hi=(hi==1.) ? b : a+(b-a)*hi;
        }
      return intervals;
    }

    // Sweep actual one-sided superlevel intervals on each coordinate plane.
    // Equal endpoints have zero overlap: two diagonally touching tiles, or
    // DG traces supported on opposite halves of a face, do not communicate.
    std::vector<unsigned int> face_components(const std::vector<Leaf> &leaves)
    {
      std::vector<unsigned int> parents(leaves.size());
      std::iota(parents.begin(),parents.end(),0u);
      const auto root=[&](unsigned int i)
      {
        while (parents[i]!=i)
          {
            parents[i]=parents[parents[i]];
            i=parents[i];
          }
        return i;
      };
      using Span = std::tuple<long double,long double,unsigned int>;
      for (unsigned int axis=0; axis<2; ++axis)
        {
          const unsigned int transverse=1-axis;
          std::map<double,std::array<std::vector<Span>,2>> faces;
          for (unsigned int i=0; i<leaves.size(); ++i)
            {
              const auto &leaf=leaves[i];
              const auto &[lo,hi]=leaf.bounds.get_boundary_points();
              for (unsigned int side=0; side<2; ++side)
                for (const auto &[a,b] : face_intervals(face_row(leaf.phase,axis,side)))
                  {
                    const long double x=lo[transverse], length=static_cast<long double>(hi[transverse])-x;
                    faces[(side ? hi : lo)[axis]][side].emplace_back(a==0. ? x : x+length*a,
                                                                     b==1. ? hi[transverse] : x+length*b,i);
                  }
            }
          for (auto &[coordinate,sides] : faces)
            {
              (void)coordinate;
              for (auto &side : sides) std::sort(side.begin(),side.end());
              unsigned int i=0,j=0;
              while (i<sides[0].size() && j<sides[1].size())
                {
                  const auto [a,b,k]=sides[0][i];
                  const auto [c,d,l]=sides[1][j];
                  if (std::max(a,c)<std::min(b,d))
                    parents[root(k)]=root(l);
                  if (b<=d) ++i;
                  if (d<=b) ++j;
                }
            }
        }
      for (auto &p : parents) p=root(p);
      return parents;
    }

    Component locate(const std::vector<Cell>     &owned,
                     const Point<2>              &anchor,
                     const Point<2>              &tip,
                     const Tensor<1,2>           &tangent,
                     MPI_Comm                     comm,
                     const bool                   line_only,
                     const std::optional<Source> &pinned)
    {
      const long double end=std::max(0.,(tip-anchor)*tangent);
      std::vector<Cell> active;
      for (const auto &cell : owned)
        if (may_contain_damage(cell.phase) && (!line_only || incoming_line(cell.bounds,anchor,tangent,end)))
          active.push_back(cell);
      std::vector<Cell> cells;
      for (auto &shard : dealii::Utilities::MPI::all_gather(comm,active))
        cells.insert(cells.end(),shard.begin(),shard.end());
      // Geometric source ties at DG faces are independent of MPI ownership.
      std::sort(cells.begin(),cells.end(),[](const Cell &a,const Cell &b)
      {
        return a.id<b.id;
      });
      std::vector<Leaf> leaves;
      for (unsigned int i=0; i<cells.size(); ++i)
        split_excursion({cells[i].bounds,cells[i].phase,i},0,leaves);
      const auto groups=face_components(leaves);
      std::vector<std::tuple<long double,long double,unsigned int>> intervals;
      std::optional<unsigned int> group;
      Component result;
      long double nearest=std::numeric_limits<long double>::infinity();
      for (unsigned int i=0; i<leaves.size(); ++i)
        for (const auto &[a,b] : supported_line(leaves[i],anchor,tangent,end))
          {
            intervals.emplace_back(a,b,groups[i]);
            const auto &id=cells[leaves[i].cell].id;
            if (pinned)
              {
                if (!group && id==pinned->cell && a<=pinned->distance_from_anchor && b>=pinned->distance_from_anchor)
                  {
                    group=groups[i];
                    nearest=pinned->distance_from_anchor;
                    result.source=pinned;
                  }
              }
            else
              {
                const long double first=std::max(0.L,a);
                if ((first<b || (end==0. && b==0.)) && first<nearest)
                  {
                    nearest=first;
                    group=groups[i];
                    // Select an actual point inside the supported interval to
                    // carry into the trial. An earlier detached island may
                    // appear nearer the anchor as the phase increases.
                    result.source=Source {id,(first+b)/2};
                  }
              }
          }
      if (!group) return result;
      result.anchored=true;
      if (!line_only)
        for (unsigned int i=0; i<leaves.size(); ++i)
          if (groups[i]==*group)
            {
              const auto &cell=cells[leaves[i].cell];
              result.cells.emplace(cell.id,cell.bounds);
            }
      std::sort(intervals.rbegin(),intervals.rend());
      long double rear=nearest;
      for (const auto &[a,b,component] : intervals)
        if (component==*group && a<=rear && b>=rear)
          rear=a;
      result.rear=anchor+static_cast<double>(rear)*tangent;
      return result;
    }
  }

  bool may_contain_damage(const BernsteinPatch<2> &phase)
  {
    return *std::max_element(phase.begin(),phase.end())>=damage_contour;
  }

  Component connected_component(const std::vector<Cell> &locally_owned,
                                const Point<2>          &anchor,
                                const Point<2>          &tip,
                                const Tensor<1,2>       &tangent,
                                MPI_Comm                 comm,
                                std::optional<Source>    parent_source)
  {
    return locate(locally_owned,anchor,tip,tangent,comm,false,parent_source);
  }

  Component seed_on_line(const std::vector<Cell> &locally_owned,
                         const Point<2>          &anchor,
                         const Point<2>          &tip,
                         const Tensor<1,2>       &tangent,
                         MPI_Comm                 comm,
                         std::optional<Source>    parent_source)
  {
    return locate(locally_owned,anchor,tip,tangent,comm,true,parent_source);
  }
}
