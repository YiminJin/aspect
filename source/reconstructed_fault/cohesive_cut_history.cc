/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_bulk_history.h>
#include <aspect/reconstructed_fault/cohesive_geometry.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <sstream>
#include <set>
#include <algorithm>
#include <cmath>
#include <limits>

namespace aspect { namespace CohesiveInterface
{
  template<int dim> unsigned int BulkHistory<dim>::cut_region(const Point<dim> &x,const int side) const
  {
    const auto &nodes=cut_geometry->vertices();
    if(x[0]<nodes.front() || x[0]>nodes.back())return 0;
    const double d=x[1]-cut_geometry->sample(x[0]).y;
    if(std::abs(d)>=cut_geometry->width())return 0;
    const auto j=std::min<std::size_t>(nodes.size()-2,
      std::upper_bound(nodes.begin(),nodes.end(),x[0])-nodes.begin()-1);
    // Match Geometry::lifting's existing sided-trace convention exactly.
    return 1+2*j+((std::abs(d)<1e-10 && side) ? side>0 : d>=0.);
  }

  template<int dim> Point<dim> BulkHistory<dim>::trace_point(const Cell &cell,const Point<dim> &unit) const
  {
    auto u=unit;
    if(cut_geometry)
      for(unsigned int d=0;d<dim;++d)
        {
          // Select the interior limit, not a new physical displacement or a
          // localization floor. Polynomial values still use the original unit.
          const double eps=64*std::numeric_limits<double>::epsilon();
          if(u[d]==0.)u[d]=eps;
          if(u[d]==1.)u[d]=1.-eps;
        }
    return this->get_mapping().transform_unit_to_real_cell(cell,u);
  }

  template<int dim> typename BulkHistory<dim>::Tensor BulkHistory<dim>::cut_value(
    const CellId &id,const Point<dim> &u,const Point<dim> &,const int side) const
  {
    const auto key=cut_region(trace_point(cells.at(id),u),side);
    const auto &regions=cut_tensor.at(id);
    const auto it=regions.find(key);
    AssertThrow(it!=regions.end(),ExcMessage("Cut Maxwell history lacks requested material-side trace, cell="+id.to_string()+", region="+std::to_string(key)));
    const auto &r=it->second;
    const double x=2*(u[0]-r.bounds[0])/(r.bounds[2]-r.bounds[0])-1.,
                 y=2*(u[1]-r.bounds[1])/(r.bounds[3]-r.bounds[1])-1.;
    double values[3]={};
    for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<3;++i)for(unsigned int j=0;j<3;++j)
      values[c]+=r.coefficients[9*c+3*i+j]*std::pow(x,i)*std::pow(y,j);
    Tensor result;result[0][0]=values[0];result[1][1]=values[1];result[0][1]=values[2];
    result[2][2]=-values[0]-values[1];return result;
  }

  template<int dim> typename BulkHistory<dim>::Coefficients BulkHistory<dim>::restrict_polynomial(
    const Coefficients &source,const std::array<double,4> &bounds) const
  {
    Coefficients result{};
    const double transform[3][3]={{0,1,0},{-.5,0,.5},{.5,-1,.5}};
    for(unsigned int a=0;a<3;++a)for(unsigned int b=0;b<3;++b)
      {
        Point<dim> u;u[0]=bounds[0]+.5*a*(bounds[2]-bounds[0]);u[1]=bounds[1]+.5*b*(bounds[3]-bounds[1]);
        double value[3]={};
        for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)
          value[c]+=source[9*c+i]*element.shape_value(i,u);
        for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<3;++i)for(unsigned int j=0;j<3;++j)
          result[9*c+3*i+j]+=transform[i][a]*transform[j][b]*value[c];
      }
    return result;
  }

  template<int dim> std::shared_ptr<BulkHistory<dim>> BulkHistory<dim>::cut_rotated_anchor(
    const PhysicalField &physical,const GradientField &gradient,const double dt,
    const unsigned int order,const Geometry &formed_geometry) const
  {
    AssertThrow(dim==2 && dt>0. && std::isfinite(dt) && order>=8,
                ExcMessage("Cut objective Maxwell history requires plane strain, positive dt and Gauss8."));
    check_layout();
    auto result=std::make_shared<BulkHistory>(this->get_simulator(),G,eta);
    result->native=native;result->projection=projection;result->tensor=tensor;
    result->cut_geometry=std::make_shared<Geometry>(formed_geometry);
    unsigned int nonzero_spin=0;
    std::map<CellId,Quadrature<dim>> rules;
    std::map<CellId,std::vector<dealii::Tensor<2,dim>>> gradients;
    for(const auto &entry:cells)
      {
        auto rule=formed_geometry.quadrature(entry.second->vertex(0),entry.second->vertex(3),order);
        auto values=gradient(entry.second,rule);
        AssertDimension(values.size(),rule.size());
        for(const auto &L:values)nonzero_spin|=L[1][0]!=L[0][1];
        rules.emplace(entry.first,std::move(rule));gradients.emplace(entry.first,std::move(values));
      }
    if(Utilities::MPI::max(nonzero_spin,this->get_mpi_communicator())==0)
      {
        result->cut_geometry=cut_geometry;result->cut_tensor=cut_tensor;
        return result;
      }
    for(const auto &entry:cells)
      {
        const auto cell=entry.second;
        const auto lo=cell->vertex(0),hi=cell->vertex(3);
        for(const auto v:cell->vertex_indices())for(unsigned int d=0;d<dim;++d)
          AssertThrow(cell->vertex(v)[d]==((v&(1u<<d)) ? hi[d] : lo[d]),
                      ExcMessage("Cut lifting-history projection currently requires affine Cartesian cells; native remainder is not discarded on other meshes."));
        const auto &rule=rules.at(entry.first);
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_JxW_values|update_quadrature_points);
        fe.reinit(cell);
        const auto tau=physical(cell,rule);
        const auto &L=gradients.at(entry.first);
        AssertDimension(tau.size(),rule.size());AssertDimension(L.size(),rule.size());
        const auto &dg=tensor.at(entry.first);
        struct Sample {Point<dim> unit;double weight;Tensor target;};
        std::map<unsigned int,std::vector<Sample>> samples;
        FullMatrix<double> global_mass(9,9);
        std::array<Vector<double>,3> global_rhs;for(auto &v:global_rhs)v.reinit(9);
        for(unsigned int q=0;q<rule.size();++q)
          {
            Tensor value;
            if(cut_geometry)value=cut_value(entry.first,rule.point(q),fe.quadrature_point(q));
            else
              {
                for(unsigned int i=0;i<9;++i)
                  {
                    const double p=element.shape_value(i,rule.point(q));
                    value[0][0]+=dg[i]*p;value[1][1]+=dg[9+i]*p;value[0][1]+=dg[18+i]*p;
                  }
                value[2][2]=-value[0][0]-value[1][1];
              }
            SymmetricTensor<2,2> in_plane;
            in_plane[0][0]=tau[q][0][0];in_plane[1][1]=tau[q][1][1];in_plane[0][1]=tau[q][0][1];
            if constexpr(dim==2)
              {
                const auto delta=ReconstructedFaultUtilities::rotate_plane_strain_stress_increment(in_plane,L[q],dt);
                value[0][0]+=delta[0][0];value[1][1]+=delta[1][1];value[0][1]+=delta[0][1];
              }
            AssertThrow(std::isfinite(value.norm()),ExcMessage("Nonfinite complete cut-tensor rotation target."));
            samples[result->cut_region(fe.quadrature_point(q))].push_back({rule.point(q),fe.JxW(q),value});
            const double components[3]={value[0][0],value[1][1],value[0][1]};
            for(unsigned int i=0;i<9;++i)
              {
                const double w=fe.JxW(q)*element.shape_value(i,rule.point(q));
                for(unsigned int j=0;j<9;++j)global_mass(i,j)+=w*element.shape_value(j,rule.point(q));
                for(unsigned int c=0;c<3;++c)global_rhs[c][i]+=w*components[c];
              }
          }
        global_mass.gauss_jordan();
        for(unsigned int c=0;c<3;++c)
          {
            Vector<double> values(9);global_mass.vmult(values,global_rhs[c]);
            for(unsigned int i=0;i<9;++i)result->tensor[entry.first][9*c+i]=values[i];
          }
        for(const auto &region:samples)
          {
            const auto &points=region.second;
            AssertThrow(points.size()>=9,ExcMessage("Cut history region has insufficient samples for nine independent modes."));
            CutRegion fitted{};
            fitted.bounds={{points.front().unit[0],points.front().unit[1],points.front().unit[0],points.front().unit[1]}};
            for(const auto &p:points)for(unsigned int d=0;d<2;++d)
              {fitted.bounds[d]=std::min(fitted.bounds[d],p.unit[d]);fitted.bounds[d+2]=std::max(fitted.bounds[d+2],p.unit[d]);}
            LAPACKFullMatrix<double> basis(points.size(),9);
            for(unsigned int q=0;q<points.size();++q)
              {
                const double x=2*(points[q].unit[0]-fitted.bounds[0])/(fitted.bounds[2]-fitted.bounds[0])-1.,
                             y=2*(points[q].unit[1]-fitted.bounds[1])/(fitted.bounds[3]-fitted.bounds[1])-1.;
                for(unsigned int i=0;i<3;++i)for(unsigned int j=0;j<3;++j)
                  basis(q,3*i+j)=std::sqrt(points[q].weight)*std::pow(x,i)*std::pow(y,j);
              }
            basis.compute_svd();
            AssertThrow(basis.singular_value(8)>1e-10*basis.singular_value(0),
                        ExcMessage("Cut history space is rank deficient; no sliver floor or regularized inverse is admitted."));
            basis.compute_inverse_svd(1e-12);
            for(unsigned int c=0;c<3;++c)
              {
                const unsigned int i=c==1 ? 1 : 0,j=c==0 ? 0 : 1;
                Vector<double> rhs(points.size()),values(9);
                for(unsigned int q=0;q<points.size();++q)rhs[q]=std::sqrt(points[q].weight)*points[q].target[i][j];
                basis.vmult(values,rhs);
                for(unsigned int k=0;k<9;++k)fitted.coefficients[9*c+k]=values[k];
              }
            result->cut_tensor[entry.first].emplace(region.first,std::move(fitted));
          }
      }
    result->check_layout();return result;
  }

  template<int dim> void BulkHistory<dim>::pack_cut(CellRecords &global) const
  {
    CellRecords local;
    for(const auto &entry:cut_tensor)
      {
        auto &row=local[entry.first.to_string()];row.push_back(entry.second.size());
        for(const auto &r:entry.second)
          {
            row.push_back(r.first);row.insert(row.end(),r.second.bounds.begin(),r.second.bounds.end());
            row.insert(row.end(),r.second.coefficients.begin(),r.second.coefficients.end());
          }
      }
    std::set<std::string> seen;
    for(const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))
      for(const auto &entry:part)
        {
          AssertThrow(seen.insert(entry.first).second,ExcMessage("Duplicate cut-history CellId owner."));
          auto &row=global.at(entry.first);row.insert(row.end(),entry.second.begin(),entry.second.end());
        }
    AssertThrow(seen.size()==global.size(),ExcMessage("Incomplete cut-history checkpoint coverage."));
    std::ostringstream out;{aspect::oarchive archive(out);archive<<*cut_geometry;}
    std::vector<double> metadata;for(const unsigned char c:out.str())metadata.push_back(c);
    global.emplace("@objective_cut_geometry_v1",std::move(metadata));
  }

  template<int dim> void BulkHistory<dim>::restore_cut(const CellRecords &records)
  {
    std::string data;
    for(const double c:records.at("@objective_cut_geometry_v1"))
      {
        AssertThrow(std::isfinite(c) && c>=0. && c<=255. && std::floor(c)==c,ExcMessage("Corrupt cut-geometry checkpoint metadata."));
        data.push_back(static_cast<unsigned char>(c));
      }
    auto geometry=std::make_shared<Geometry>();std::istringstream in(data);
    {aspect::iarchive archive(in);archive>>*geometry;}cut_geometry=std::move(geometry);
    for(const auto &entry:cells)
      {
        const auto &row=records.at(entry.first.to_string());
        unsigned int at=vertices.at(entry.first).size()*dim+27;
        for(unsigned int i=0;i<this->get_fe().n_dofs_per_cell();++i)
          at+=this->get_fe().system_to_component_index(i).first<dim;
        const double count=row.at(at++);
        AssertThrow(std::isfinite(count) && count>=1. && count<=2*cut_geometry->vertices().size()
                    && std::floor(count)==count,ExcMessage("Invalid cut-history region count."));
        for(unsigned int r=0;r<count;++r)
          {
            const double key=row.at(at++);
            AssertThrow(std::isfinite(key) && key>=0. && key<=2*cut_geometry->vertices().size()
                        && std::floor(key)==key,ExcMessage("Invalid cut-history material-region key."));
            CutRegion region{};for(auto &b:region.bounds)b=row.at(at++);
            for(auto &c:region.coefficients){c=row.at(at++);AssertThrow(std::isfinite(c),ExcMessage("Nonfinite cut-history coefficient."));}
            AssertThrow(std::all_of(region.bounds.begin(),region.bounds.end(),[](double v){return std::isfinite(v);})
                        && region.bounds[2]>region.bounds[0] && region.bounds[3]>region.bounds[1],ExcMessage("Invalid cut-history basis bounds."));
            AssertThrow(cut_tensor[entry.first].emplace(static_cast<unsigned int>(key),std::move(region)).second,
                        ExcMessage("Duplicate cut-history material region."));
          }
        AssertDimension(at,row.size());
      }
  }

  template std::shared_ptr<BulkHistory<2>> BulkHistory<2>::cut_rotated_anchor(const PhysicalField &,const GradientField &,double,unsigned int,const Geometry &) const;
  template Point<2> BulkHistory<2>::trace_point(const Cell &,const Point<2> &) const;
  template unsigned int BulkHistory<2>::cut_region(const Point<2> &,int) const;
  template BulkHistory<2>::Tensor BulkHistory<2>::cut_value(const CellId &,const Point<2> &,const Point<2> &,int) const;
  template BulkHistory<2>::Coefficients BulkHistory<2>::restrict_polynomial(const Coefficients &,const std::array<double,4> &) const;
  template void BulkHistory<2>::pack_cut(CellRecords &) const;
  template void BulkHistory<2>::restore_cut(const CellRecords &);
}}
