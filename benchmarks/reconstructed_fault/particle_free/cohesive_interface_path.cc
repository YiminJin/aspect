/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
// Isolated fixed-background vector mixed reference with nested jump functions.
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <cstdint>
#include <cstring>
#include <zlib.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <map>
#include <set>
#include <sstream>
#include "cohesive_interface_path_parallel.h"
using namespace dealii;
namespace
{
constexpr double G=1e6, size=12., axial=.41954981558863996;
PathColumnParallel column_mpi;
struct Arc {double x,y,end,a,b;unsigned cells;double c1=0,c2=0,c3=0;bool hermite=false;};
std::vector<Arc> arcs;
std::vector<double> nodes;
double envelope, normal_envelope, mesh_transition;
bool fitted_mesh=false;
unsigned affine_modes=1;
struct Geometry {double y,slope,second;Tensor<1,2> t,dt;};
Geometry geometry(double x)
{
  const auto it=std::find_if(arcs.begin(),arcs.end(),[&](const Arc &a){return x<=a.end+1e-12;});
  AssertThrow(it!=arcs.end() && x>=arcs.front().x-1e-12,ExcMessage("Outside graph"));
  const auto &a=*it;const double h=a.end-a.x,z=std::max(0.,std::min(1.,(x-a.x)/h));
  Geometry g;
  if(a.hermite)
    {
      g.y=a.y+h*(a.a*z+.5*a.c1*z*z+a.c2*z*z*z/3+a.c3*z*z*z*z/4);
      g.slope=a.a+a.c1*z+a.c2*z*z+a.c3*z*z*z;
      g.second=(a.c1+2*a.c2*z+3*a.c3*z*z)/h;
    }
  else
    {
      g.y=a.y+a.a*h*z+(a.b-a.a)*h*(z*z*z-.5*z*z*z*z);
      g.slope=a.a+(a.b-a.a)*(3*z*z-2*z*z*z);g.second=(a.b-a.a)*(6*z-6*z*z)/h;
    }
  g.t[0]=1;g.t[1]=g.slope;g.t/=g.t.norm();
  g.dt[0]=-g.slope*g.second;g.dt[1]=g.second;g.dt/=std::pow(1+g.slope*g.slope,1.5);
  return g;
}
struct Mode {unsigned k;Tensor<2,2> grad;Tensor<1,2> value;};
double mesh_graph(double x)
{return geometry(std::max(arcs.front().x,std::min(arcs.back().end,x))).y;}
double mesh_blend(double y)
{return std::abs(y)<=mesh_transition?1.:(6-std::abs(y))/(6-mesh_transition);}
bool normal_active(const Point<2> &p)
{return p[0]>nodes.front() && p[0]<nodes.back() && std::abs(p[1]-geometry(p[0]).y)<normal_envelope;}
Mode normal_bubble(const Point<2> &p)
{
  auto next=std::upper_bound(nodes.begin(),nodes.end(),p[0]);unsigned j=std::min<unsigned>(nodes.size()-2,next-nodes.begin()-1);
  const auto g=geometry(p[0]);const double h=nodes[j+1]-nodes[j],z=(p[0]-nodes[j])/h,d=p[1]-g.y;
  const double rho=.5*(d>=0?1.:-1.)*(1-std::abs(d)/normal_envelope),b=4*z*(1-z),db=4*(1-2*z)/h;
  Tensor<1,2>n,dn;n[0]=-g.t[1];n[1]=g.t[0];dn[0]=-g.dt[1];dn[1]=g.dt[0];Mode m;m.k=j;
  m.value=b*rho*n;
  for(unsigned a=0;a<2;++a){m.grad[a][0]=(db*rho+b*g.slope/(2*normal_envelope))*n[a]+b*rho*dn[a];m.grad[a][1]=-b*n[a]/(2*normal_envelope);}
  return m;
}
std::vector<Mode> modes(const Point<2> &p,int side=0)
{
  if(p[0]<nodes.front() || p[0]>nodes.back())return {};
  const auto g=geometry(p[0]);const double d=p[1]-g.y;
  if(std::abs(d)>=envelope)return {};
  const double sign=std::abs(d)<1e-10 && side ? side : (d>=0?1.:-1.);
  const double rho=.5*sign*(1-std::abs(d)/envelope);
  auto next=std::upper_bound(nodes.begin(),nodes.end(),p[0]);
  unsigned j=std::min<unsigned>(nodes.size()-2,next-nodes.begin()-1);
  const double h=nodes[j+1]-nodes[j],z=(p[0]-nodes[j])/h;
  std::vector<Mode> result;
  for(unsigned v=0;v<2;++v)
    {
      const unsigned index=j+v;if(index==0 || index+1==nodes.size())continue;
      const double b=v?z:1-z,db=(v?1.:-1.)/h;Mode m;m.k=index+affine_modes-1;
      m.value=b*rho*g.t;
      for(unsigned a=0;a<2;++a)
        {
          m.grad[a][0]=(db*rho+b*g.slope/(2*envelope))*g.t[a]+b*rho*g.dt[a];
          m.grad[a][1]=-b*g.t[a]/(2*envelope);
        }
      result.push_back(m);
    }
  return result;
}
SymmetricTensor<2,3> dev(const SymmetricTensor<2,2> &e)
{
  SymmetricTensor<2,3>d;for(unsigned i=0;i<2;++i)for(unsigned j=i;j<2;++j)d[i][j]=e[i][j];
  const double tr=trace(d)/3;for(unsigned i=0;i<3;++i)d[i][i]-=tr;return d;
}
std::ofstream file(const std::string &dir,const std::string &name)
{std::ofstream f(column_mpi.coordinator()?dir+"/"+name:"/dev/null");AssertThrow(f,ExcMessage("Cannot open output"));f<<std::setprecision(17);return f;}
void matrix(const std::string &dir,const std::string &name,const FullMatrix<double>&m)
{auto f=file(dir,name);for(unsigned i=0;i<m.m();++i){for(unsigned j=0;j<m.n();++j)f<<(j?",":"")<<m(i,j);f<<'\n';}}
void add_cut(std::vector<double>&cuts,double x,double l,double r)
{if(x>l+1e-11 && x<r-1e-11)cuts.push_back(x);}
// Split all graph/support crossings. The slope is monotone on each arc;
// splitting at its possible zero gives monotone graph pieces for bisection.
std::vector<double> cuts(double l,double r,double bottom,double top)
{
  std::vector<double> result={l,r};for(double x:nodes)add_cut(result,x,l,r);
  for(const auto &arc:arcs)
    {
      double a=std::max(l,arc.x),b=std::min(r,arc.end);if(b<=a+1e-12)continue;
      std::vector<double> monotone={a,b};
      if(geometry(a).slope*geometry(b).slope<0)
        {
          double u=a,v=b;for(unsigned k=0;k<60;++k){double mid=.5*(u+v);if(geometry(u).slope*geometry(mid).slope<=0)v=mid;else u=mid;}
          monotone.insert(monotone.begin()+1,.5*(u+v));
        }
      std::vector<double>levels;
      for(double boundary:{bottom,top})for(double width:{envelope,normal_envelope})for(int sign:{-1,0,1})
        {
          const double divisor=fitted_mesh?1-mesh_blend(boundary):1.;
          if(divisor>1e-12)levels.push_back((boundary+sign*width)/divisor);
        }
      for(double level:levels)
        for(unsigned i=1;i<monotone.size();++i)
          {
            double u=monotone[i-1],v=monotone[i],fu=geometry(u).y-level,fv=geometry(v).y-level;
            if(fu*fv>=0)continue;
            for(unsigned k=0;k<60;++k){double mid=.5*(u+v),fm=geometry(mid).y-level;if(fu*fm<=0)v=mid;else{u=mid;fu=fm;}}
            add_cut(result,.5*(u+v),l,r);
          }
    }
  std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end(),[](double a,double b){return std::abs(a-b)<1e-10;}),result.end());
  return result;
}
Quadrature<2> cut_rule(const Point<2>&lo,const Point<2>&hi,unsigned order)
{
  QGauss<1> q(order);std::vector<Point<2>> points;std::vector<double> weights;
  const auto xs=cuts(lo[0],hi[0],lo[1],hi[1]);
  for(unsigned k=1;k<xs.size();++k)for(unsigned i=0;i<order;++i)
    {
      const double x=xs[k-1]+(xs[k]-xs[k-1])*q.point(i)[0],wx=(xs[k]-xs[k-1])*q.weight(i);
      const double g=fitted_mesh?mesh_graph(x):0.;
      const double bottom=lo[1]+(fitted_mesh?mesh_blend(lo[1])*g:0.),top=hi[1]+(fitted_mesh?mesh_blend(hi[1])*g:0.);
      std::vector<double> ys={bottom,top};
      if(x>nodes.front() && x<nodes.back())
        {double y=geometry(x).y;for(double level:{y-envelope,y,y+envelope,y-normal_envelope,y+normal_envelope})add_cut(ys,level,bottom,top);}
      std::sort(ys.begin(),ys.end());
      ys.erase(std::unique(ys.begin(),ys.end()),ys.end());
      for(unsigned j=1;j<ys.size();++j)for(unsigned v=0;v<order;++v)
        {
          const double y=ys[j-1]+(ys[j]-ys[j-1])*q.point(v)[0];
          const double middle=.5*(lo[1]+hi[1]);
          const double determinant=fitted_mesh && std::abs(middle)>mesh_transition?1-(middle>0?g:-g)/(6-mesh_transition):1.;
          const double reference_y=fitted_mesh?middle+(y-(middle+mesh_blend(middle)*g))/determinant:y;
          points.emplace_back((x-lo[0])/(hi[0]-lo[0]),(reference_y-lo[1])/(hi[1]-lo[1]));
          weights.push_back(wx*(ys[j]-ys[j-1])*q.weight(v)/((hi[0]-lo[0])*(hi[1]-lo[1])*determinant));
        }
    }
  return Quadrature<2>(points,weights);
}
std::string column_hash(const FullMatrix<double> &m,unsigned j)
{
  std::uint64_t hash=14695981039346656037ULL;
  for(unsigned i=0;i<m.m();++i){double value=m(i,j);unsigned char bytes[sizeof(value)];std::memcpy(bytes,&value,sizeof(value));for(auto byte:bytes){hash^=byte;hash*=1099511628211ULL;}}
  std::ostringstream s;s<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return s.str();
}
}
int main(int argc,char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
      column_mpi.initialize(argc>14 && std::string(argv[14])=="column-mpi");
      try
      {
      AssertThrow(argc>=5,ExcMessage("usage: path nx degree geometry.csv output [order]"));
      const unsigned nx=std::stoul(argv[1]),degree=std::stoul(argv[2]),order=argc>5?std::stoul(argv[5]):6;
      const std::string out=argv[4];const double mesh_h=size/nx;
      envelope=argc>7?std::stod(argv[7]):mesh_h;
      normal_envelope=argc>8?std::stod(argv[8]):envelope;
      fitted_mesh=argc>9 && std::string(argv[9])=="fitted";
      affine_modes=argc>11 && std::string(argv[11])=="two-affine"?2:1;
      mesh_transition=std::max(envelope,normal_envelope);
      AssertThrow(envelope>0 && normal_envelope>0 && mesh_transition<3,ExcMessage("Invalid numerical lifting support"));
      std::ifstream input(argv[3]);std::string line;std::getline(input,line);
      while(std::getline(input,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream s(line);Arc a;s>>a.x>>a.y>>a.end>>a.a>>a.b>>a.cells;AssertThrow(s && a.end>a.x && a.cells>0,ExcMessage("Invalid path arc"));a.hermite=bool(s>>a.c1>>a.c2>>a.c3);arcs.push_back(a);}
      AssertThrow(!arcs.empty(),ExcMessage("Empty path"));
      for(const auto &a:arcs){if(nodes.empty())nodes.push_back(a.x);else AssertThrow(std::abs(nodes.back()-a.x)<1e-11,ExcMessage("Disconnected path"));for(unsigned j=1;j<=a.cells;++j)nodes.push_back(a.x+(a.end-a.x)*j/a.cells);}
      const unsigned nj=nodes.size()-2,nc=nj+affine_modes;const auto start=std::chrono::steady_clock::now();
      std::vector<double>xs,ys;
      if(fitted_mesh)
        {
          xs={0.};
          for(double x=mesh_h;x<nodes.front()-1e-10;x+=mesh_h)xs.push_back(x);
          for(double x:nodes)xs.push_back(x);
          for(double x=nodes.back()+mesh_h;x<12-1e-10;x+=mesh_h)xs.push_back(x);xs.push_back(12.);
          ys={-6.,-mesh_transition,-envelope,0.,envelope,mesh_transition,6.};
          for(unsigned j=1;j<nx;++j){double y=-6+j*mesh_h;bool near=false;for(double v:ys)near|=std::abs(y-v)<.25*mesh_h;if(!near)ys.push_back(y);}
          std::sort(ys.begin(),ys.end());ys.erase(std::unique(ys.begin(),ys.end()),ys.end());
        }
      else for(unsigned j=0;j<=nx;++j){xs.push_back(j*mesh_h);ys.push_back(-6+j*mesh_h);}
      std::vector<std::vector<double>>steps(2);for(unsigned j=1;j<xs.size();++j)steps[0].push_back(xs[j]-xs[j-1]);for(unsigned j=1;j<ys.size();++j)steps[1].push_back(ys[j]-ys[j-1]);
      Triangulation<2> tria;
      if(fitted_mesh)GridGenerator::subdivided_hyper_rectangle(tria,steps,Point<2>(0,-6),Point<2>(12,6));
      else GridGenerator::subdivided_hyper_rectangle(tria,std::vector<unsigned>{nx,nx},Point<2>(0,-6),Point<2>(12,6));
      FESystem<2>geometry_fe(FE_Q<2>(4),2);DoFHandler<2>geometry_dofs(tria);Vector<double>coordinates;std::unique_ptr<Mapping<2>>mapping_ptr;
      if(fitted_mesh)
        {
          geometry_dofs.distribute_dofs(geometry_fe);coordinates.reinit(geometry_dofs.n_dofs());std::vector<types::global_dof_index>gi(geometry_fe.n_dofs_per_cell());MappingQ1<2>reference;
          for(const auto &cell:geometry_dofs.active_cell_iterators())
            {cell->get_dof_indices(gi);for(unsigned j=0;j<gi.size();++j){auto pt=reference.transform_unit_to_real_cell(cell,geometry_fe.get_unit_support_points()[j]);coordinates[gi[j]]=geometry_fe.system_to_component_index(j).first==0?pt[0]:pt[1]+mesh_blend(pt[1])*mesh_graph(pt[0]);}}
          mapping_ptr=std::make_unique<MappingFEField<2,2,Vector<double>>>(geometry_dofs,coordinates);
        }
      else mapping_ptr=std::make_unique<MappingQ1<2>>();
      const auto &mapping=*mapping_ptr;
      FE_Q<2>scalar(degree);FESystem<2> fe(scalar,2,FE_DGP<2>(degree-1),1);DoFHandler<2>dofs(tria);dofs.distribute_dofs(fe);
      const FEValuesExtractors::Vector u(0);const FEValuesExtractors::Scalar p(2);
      std::map<types::global_dof_index,unsigned> master,component;std::map<types::global_dof_index,Point<2>> location;
      std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
      for(const auto &cell:dofs.active_cell_iterators()){cell->get_dof_indices(ids);for(unsigned j=0;j<ids.size();++j){unsigned c=fe.system_to_component_index(j).first;component[ids[j]]=c;if(c<2)location[ids[j]]=mapping.transform_unit_to_real_cell(cell,scalar.get_unit_support_points()[fe.system_to_base_index(j).second]);}}
      types::global_dof_index gauge=numbers::invalid_dof_index;for(auto c:component)if(c.second==2){gauge=c.first;break;}
      for(auto c:component){bool boundary=false;if(c.second<2){auto x=location.at(c.first);boundary=x[0]<1e-11||x[0]>12-1e-11||std::abs(x[1])>6-1e-11;}if(!boundary && c.first!=gauge)master[c.first]=master.size();}
      const unsigned nm=master.size(),normal_cells=nodes.size()-1;DynamicSparsityPattern dsp(nm);
      for(const auto &cell:dofs.active_cell_iterators()){cell->get_dof_indices(ids);for(auto i:ids)for(auto j:ids)if(master.count(i)&&master.count(j))dsp.add(master.at(i),master.at(j));}
      SparsityPattern sp;sp.copy_from(dsp);SparseMatrix<double>A(sp);FullMatrix<double>B(nm,nc),D(nc,nc),Bn(nm,normal_cells),Dn(normal_cells,nc);Vector<double>mean(nm),normal_div(normal_cells);
      SymmetricTensor<2,3> affine;affine[0][0]=-axial;affine[1][1]=axial;affine[0][1]=.5;
      std::vector<SymmetricTensor<2,3>> backgrounds(affine_modes,affine);
      if(affine_modes==2){backgrounds[1]=0.;backgrounds[1][0][0]=-.5;backgrounds[1][1][1]=.5;backgrounds[1][0][1]=-axial;}
      QGauss<2> aq(6);FEValues<2> av(mapping,fe,aq,update_gradients|update_values|update_JxW_values);
      double volume=0;
      const unsigned columns=xs.size()-1;std::vector<DoFHandler<2>::active_cell_iterator> cells(columns*(ys.size()-1));
      for(const auto &cell:dofs.active_cell_iterators())
        {
          const auto center=cell->center();const unsigned ix=std::upper_bound(xs.begin(),xs.end(),center[0])-xs.begin()-1,iy=std::upper_bound(ys.begin(),ys.end(),center[1])-ys.begin()-1;cells[ix+columns*iy]=cell;
          if(!column_mpi.coordinator())continue;
          cell->get_dof_indices(ids);av.reinit(cell);
          for(unsigned q=0;q<aq.size();++q)
            {
              double w=av.JxW(q);volume+=w;
              for(unsigned a=0;a<affine_modes;++a)for(unsigned b=0;b<affine_modes;++b)D(a,b)+=w*2*(backgrounds[a]*backgrounds[b]);
              for(unsigned i=0;i<ids.size();++i)if(master.count(ids[i]))
                {
                  unsigned ii=master.at(ids[i]);auto ei=dev(av[u].symmetric_gradient(i,q));double di=av[u].divergence(i,q),pi=av[p].value(i,q);mean[ii]+=w*pi;
                  for(unsigned a=0;a<affine_modes;++a)B(ii,a)+=w*2*(ei*backgrounds[a]);
                  for(unsigned j=0;j<ids.size();++j)if(master.count(ids[j]))A.add(ii,master.at(ids[j]),w*(2*ei*dev(av[u].symmetric_gradient(j,q))-di*av[p].value(j,q)-pi*av[u].divergence(j,q)));
                }
            }
          auto qr=cut_rule(cell->vertex(0),cell->vertex(3),order);FEValues<2>v(mapping,fe,qr,update_gradients|update_values|update_JxW_values|update_quadrature_points);v.reinit(cell);
          for(unsigned q=0;q<qr.size();++q)
            {
              const auto mm=modes(v.quadrature_point(q));const double w=v.JxW(q);
              if(normal_active(v.quadrature_point(q)))
                {
                  const auto normal=normal_bubble(v.quadrature_point(q));const auto en=dev(symmetrize(normal.grad));const double divn=trace(normal.grad);
                  normal_div[normal.k]+=w*divn;for(unsigned a=0;a<affine_modes;++a)Dn(normal.k,a)+=w*2*(en*backgrounds[a]);
                  for(const auto &m:mm)Dn(normal.k,m.k)+=w*2*(en*dev(symmetrize(m.grad)));
                  for(unsigned i=0;i<ids.size();++i)if(master.count(ids[i]))Bn(master.at(ids[i]),normal.k)+=w*(2*dev(v[u].symmetric_gradient(i,q))*en-v[p].value(i,q)*divn);
                }
              for(const auto &m:mm)
                {
                  const auto em=dev(symmetrize(m.grad));const double dm=trace(m.grad);
                  for(unsigned a=0;a<affine_modes;++a){D(a,m.k)+=w*2*(backgrounds[a]*em);D(m.k,a)+=w*2*(backgrounds[a]*em);}
                  for(const auto &n:mm)D(m.k,n.k)+=w*2*(em*dev(symmetrize(n.grad)));
                  for(unsigned i=0;i<ids.size();++i)if(master.count(ids[i]))B(master.at(ids[i]),m.k)+=w*(2*dev(v[u].symmetric_gradient(i,q))*em-v[p].value(i,q)*dm);
                }
            }
        }
      column_mpi.sparse_from_root(A);column_mpi.from_root(B);column_mpi.from_root(D);column_mpi.from_root(Bn);column_mpi.from_root(Dn);
      column_mpi.broadcast(mean.begin(),mean.size(),0);column_mpi.broadcast(normal_div.begin(),normal_div.size(),0);column_mpi.broadcast(&volume,1,0);
      const auto assembled=std::chrono::steady_clock::now();
      SparseDirectUMFPACK solver;solver.initialize(A);FullMatrix<double>response(nm,nc);Vector<double>rhs(nm),x(nm),residual(nm);std::vector<double>pmean(nc);double maxres=0;
      for(unsigned k=0;k<nc;++k)if(column_mpi.owns(k)){for(unsigned i=0;i<nm;++i)rhs[i]=-B(i,k);solver.vmult(x,rhs);A.vmult(residual,x);residual-=rhs;maxres=std::max(maxres,residual.linfty_norm());pmean[k]=mean*x/volume;for(unsigned i=0;i<nm;++i)response(i,k)=x[i];}
      column_mpi.injected_failure();column_mpi.columns(response);column_mpi.scalars(pmean);maxres=column_mpi.maximum(maxres);
      const auto solved=std::chrono::steady_clock::now();
      FullMatrix<double>K(D);for(unsigned k=0;k<nc;++k)for(unsigned l=0;l<nc;++l)if(column_mpi.owns(l)){for(unsigned i=0;i<nm;++i)K(k,l)+=B(i,k)*response(i,l);K(k,l)*=G;}
      column_mpi.columns(K);
      matrix(out,"stiffness.csv",K);
      auto hashes=file(out,"columns.csv");hashes<<"column,B_FNV1a,response_FNV1a,pressure_mean\n";for(unsigned j=0;j<nc;++j)hashes<<j<<','<<column_hash(B,j)<<','<<column_hash(response,j)<<','<<pmean[j]<<'\n';
      bool cell_field_phase=false;
      struct BulkShape {unsigned index;SymmetricTensor<2,3>strain;double pressure;};
      std::vector<BulkShape>shape_data;shape_data.reserve(ids.size());
      auto stress=[&](const auto &v,unsigned q,const std::vector<Mode>&mm)
      {
        // Same native shape values and addition order. Evaluate each shape
        // once per point, rather than once for every response column.
        shape_data.clear();for(unsigned j=0;j<ids.size();++j)if(master.count(ids[j]))
          shape_data.push_back({master.at(ids[j]),dev(v[u].symmetric_gradient(j,q)),v[p].value(j,q)});
        std::vector<SymmetricTensor<2,3>> tau(nc);std::vector<double> pressure(nc);
        for(unsigned a=0;a<affine_modes;++a)if(cell_field_phase || column_mpi.owns(a))tau[a]=2*G*backgrounds[a];for(const auto &m:mm)if(cell_field_phase || column_mpi.owns(m.k))tau[m.k]+=2*G*dev(symmetrize(m.grad));
        for(unsigned k=0;k<nc;++k)if(cell_field_phase || column_mpi.owns(k))
          {
            pressure[k]=-G*pmean[k];
            for(const auto &shape:shape_data){const double c=response(shape.index,k);tau[k]+=2*G*c*shape.strain;pressure[k]+=G*c*shape.pressure;}
          }
        return std::make_pair(tau,pressure);
      };
      std::vector<Tensor<1,2>> last_displacement(nc);
      std::vector<std::pair<unsigned,Tensor<1,2>>> displacement_shapes;displacement_shapes.reserve(ids.size());
      auto point_stress=[&](Point<2>pt,int side)
      {
        Point<2> loc=pt;if(side){const auto g=geometry(pt[0]);loc[0]-=side*1e-9*envelope*g.t[1];loc[1]+=side*1e-9*envelope*g.t[0];}
        if(fitted_mesh){const double g=mesh_graph(loc[0]),d=loc[1]-g;loc[1]=std::abs(d)<=mesh_transition?d:std::copysign(mesh_transition,d)+(d-std::copysign(mesh_transition,d))/(1-(d>0?g:-g)/(6-mesh_transition));}
        unsigned ix=std::min<unsigned>(xs.size()-2,std::upper_bound(xs.begin(),xs.end(),loc[0])-xs.begin()-1),iy=std::min<unsigned>(ys.size()-2,std::upper_bound(ys.begin(),ys.end(),loc[1])-ys.begin()-1);
        const auto cell=cells[ix+columns*iy];auto unit=mapping.transform_real_to_unit_cell(cell,pt);for(unsigned a=0;a<2;++a)unit[a]=std::max(0.,std::min(1.,unit[a]));
        FEValues<2>fv(mapping,fe,Quadrature<2>(std::vector<Point<2>>{unit}),update_gradients|update_values);fv.reinit(cell);cell->get_dof_indices(ids);const auto mm=modes(pt,side);
        displacement_shapes.clear();for(unsigned j=0;j<ids.size();++j)if(master.count(ids[j]))displacement_shapes.emplace_back(master.at(ids[j]),fv[u].value(j,0));
        std::fill(last_displacement.begin(),last_displacement.end(),Tensor<1,2>());
        for(unsigned k=0;k<affine_modes;++k)if(cell_field_phase || column_mpi.owns(k))for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)last_displacement[k][a]+=backgrounds[k][a][b]*pt[b];
        for(const auto &m:mm)if(cell_field_phase || column_mpi.owns(m.k))last_displacement[m.k]+=m.value;
        for(unsigned k=0;k<nc;++k)if(cell_field_phase || column_mpi.owns(k))for(const auto &shape:displacement_shapes)last_displacement[k]+=response(shape.first,k)*shape.second;
        return stress(fv,0,mm);
      };
      const unsigned nq=3*(nodes.size()-1);FullMatrix<double>Q(nq,nj),Nm(nq,nc),Np(nq,nc),S(nq,nc),Jx(nq,nc),Jy(nq,nc);QGauss<1>sq(3);
      auto surface=file(out,"surface.csv");surface<<"q,cell,x_m,y_m,weight_m,tx,ty\n";
      for(unsigned c=0;c+1<nodes.size();++c)for(unsigned q=0;q<3;++q)
        {
          const unsigned row=3*c+q;double z=sq.point(q)[0],h=nodes[c+1]-nodes[c],xx=nodes[c]+h*z;auto g=geometry(xx);Tensor<1,2>normal;normal[0]=-g.t[1];normal[1]=g.t[0];
          if(c>0)Q(row,c-1)=1-z;if(c+1<nodes.size()-1)Q(row,c)=z;
          surface<<row<<','<<c<<','<<xx<<','<<g.y<<','<<h*sq.weight(q)/g.t[0]<<','<<g.t[0]<<','<<g.t[1]<<'\n';
          for(int side:{-1,1}){auto field=point_stress(Point<2>(xx,g.y),side);for(unsigned k=0;k<nc;++k){auto &tau=field.first[k];double nn=0,tn=0;for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b){nn+=normal[a]*tau[a][b]*normal[b];tn+=g.t[a]*tau[a][b]*normal[b];}(side<0?Nm:Np)(row,k)=field.second[k]-nn;S(row,k)+=.5*tn;Jx(row,k)+=side*last_displacement[k][0];Jy(row,k)+=side*last_displacement[k][1];}}
        }
      column_mpi.columns(Nm);column_mpi.columns(Np);column_mpi.columns(S);column_mpi.columns(Jx);column_mpi.columns(Jy);
      matrix(out,"Q.csv",Q);matrix(out,"normal-minus.csv",Nm);matrix(out,"normal-plus.csv",Np);matrix(out,"shear.csv",S);
      FullMatrix<double>Nwork(nq,nc);auto normal_mass=file(out,"normal-measure.csv");normal_mass<<"cell,bulk_mass_m,surface_mass_m\n";
      for(unsigned j=0;j<normal_cells;++j)
        {
          const double mass=-normal_div[j];AssertThrow(mass>0,ExcMessage("Invalid normal work measure"));double surface_mass=0;
          for(unsigned q=0;q<3;++q){double z=sq.point(q)[0],xx=nodes[j]+(nodes[j+1]-nodes[j])*z;surface_mass+=4*z*(1-z)*(nodes[j+1]-nodes[j])*sq.weight(q)/geometry(xx).t[0];}
          normal_mass<<j<<','<<mass<<','<<surface_mass<<'\n';
          for(unsigned k=0;k<nc;++k)if(column_mpi.owns(k)){double force=Dn(j,k)+normal_div[j]*pmean[k];for(unsigned i=0;i<nm;++i)force+=Bn(i,j)*response(i,k);for(unsigned q=0;q<3;++q)Nwork(3*j+q,k)=G*force/mass;}
        }
      column_mpi.columns(Nwork);
      matrix(out,"normal-work.csv",Nwork);
      matrix(out,"jump-x.csv",Jx);matrix(out,"jump-y.csv",Jy);
      double normal_error=0,tangent_error=0,gradient_error=0;
      for(unsigned q=0;q<nq;++q){double xx=0;unsigned cell=q/3;xx=nodes[cell]+(nodes[cell+1]-nodes[cell])*sq.point(q%3)[0];auto g=geometry(xx);for(unsigned k=0;k<nc;++k){normal_error=std::max(normal_error,std::abs(-g.t[1]*Jx(q,k)+g.t[0]*Jy(q,k)));tangent_error=std::max(tangent_error,std::abs(g.t[0]*Jx(q,k)+g.t[1]*Jy(q,k)-(k>=affine_modes?Q(q,k-affine_modes):0.)));}}
      for(const auto &arc:arcs)
        {
          double xx=arc.x+.37*(arc.end-arc.x);Point<2>pt(xx,geometry(xx).y+.31*envelope);auto mm=modes(pt);if(mm.empty())continue;
          const auto m=mm.front();Tensor<2,2>fd;const double eps=1e-6*std::min(envelope,nodes[1]-nodes[0]);
          auto value=[&](Point<2>p){for(const auto &v:modes(p))if(v.k==m.k)return v.value;return Tensor<1,2>();};
          for(unsigned b=0;b<2;++b){auto plus=pt,minus=pt;plus[b]+=eps;minus[b]-=eps;auto v=(value(plus)-value(minus))/(2*eps);for(unsigned a=0;a<2;++a)fd[a][b]=v[a];}
          gradient_error=std::max(gradient_error,(fd-m.grad).norm()/m.grad.norm());
        }
      auto nd=file(out,"nodes.csv");nd<<"mode,x_m,y_m\n";for(unsigned j=1;j+1<nodes.size();++j)nd<<j-1<<','<<nodes[j]<<','<<geometry(nodes[j]).y<<'\n';
      const auto surface_done=std::chrono::steady_clock::now();
      // Independent higher-order energy probes and physical cell means.
      const unsigned probes=5;FullMatrix<double>C(probes,nc);for(unsigned j=0;j<probes;++j){C(j,0)=.001;for(unsigned k=1;k<nc;++k)C(j,k)=1e-4*std::sin((j+1)*k*.739);}
      std::vector<double>energy(probes);FullMatrix<double>normal_check(normal_cells,nc);Vector<double>normal_check_mass(normal_cells);
      std::vector<double>physical_coefficients(nc);const bool physical_diagnostic=argc>10;
      if(physical_diagnostic){std::ifstream incoming(argv[10]);for(auto &c:physical_coefficients)AssertThrow(bool(incoming>>c),ExcMessage("Incomplete diagnostic coefficients"));}
      cell_field_phase=column_mpi.enabled;
      double diagnostic[3][7]={};
      const bool compress=argc>6 && std::string(argv[6])=="compress";
      std::ofstream bin,centers;gzFile gzip_mean=nullptr,gzip_center=nullptr;
      if(column_mpi.coordinator())
        {
          if(compress){gzip_mean=gzopen((out+"/field-mean.bin.gz").c_str(),"wb6");gzip_center=gzopen((out+"/field-center.bin.gz").c_str(),"wb6");AssertThrow(gzip_mean && gzip_center,ExcMessage("Cannot open compressed fields"));}
          else{bin.open(out+"/field-mean.bin",std::ios::binary);centers.open(out+"/field-center.bin",std::ios::binary);}
        }
      auto write_field=[&](const double *values,unsigned count,bool center)
      {if(!column_mpi.coordinator())return;if(compress){unsigned bytes=count*sizeof(double);AssertThrow(gzwrite(center?gzip_center:gzip_mean,values,bytes)==int(bytes),ExcMessage("Compressed field write failed"));}
       else (center?centers:bin).write(reinterpret_cast<const char*>(values),count*sizeof(double));};
      std::vector<double>field_means,column_centers,cell_geometry;
      std::map<unsigned,std::vector<double>>cell_probes;
      if(column_mpi.enabled){field_means.resize(tria.n_active_cells()*nc*5);column_centers.resize(field_means.size());cell_geometry.resize(3*tria.n_active_cells());}
      auto cf=file(out,"cells.csv");cf<<"cell,x_m,y_m,area_m2\n";
      const auto end=geometry(nodes.back());Tensor<1,2> tip=end.t;const Point<2> probe_center(nodes.back()+.75*tip[0],end.y+.75*tip[1]);FullMatrix<double>probe(5,nc);double probe_weight=0,affine_error=0;
      const bool candidate_probes=argc>12 && std::string(argv[12])=="candidate-probes";
      std::vector<Point<2>> endpoint_centers;
      if(candidate_probes)for(const auto &arc:arcs)
        {const auto g=geometry(arc.end);endpoint_centers.emplace_back(arc.end+.75*g.t[0],g.y+.75*g.t[1]);}
      FullMatrix<double>endpoint_probes(5*endpoint_centers.size(),nc);
      std::vector<double>endpoint_weights(endpoint_centers.size());
      for(const auto &cell:dofs.active_cell_iterators())
        {
          if(column_mpi.enabled && !column_mpi.owns(cell->active_cell_index()))continue;
          cell->get_dof_indices(ids);auto qr=cut_rule(cell->vertex(0),cell->vertex(3),order+2);FEValues<2>v(mapping,fe,qr,update_gradients|update_values|update_JxW_values|update_quadrature_points);v.reinit(cell);
          std::vector<double> values(5*nc),cell_probe(1+5*nc);double area=0;
          for(unsigned q=0;q<qr.size();++q)
            {
              const auto pt=v.quadrature_point(q);const auto mm=modes(pt);auto f=stress(v,q,mm);double w=v.JxW(q);area+=w;
              if(physical_diagnostic)
                {
                  SymmetricTensor<2,3>t;double p=0;for(unsigned k=0;k<nc;++k){t+=physical_coefficients[k]*f.first[k];p+=physical_coefficients[k]*f.second[k];}
                  const double divergence=-1.5*t[2][2]/G;auto perturbation=t;for(unsigned a=0;a<affine_modes;++a)perturbation-=2*G*physical_coefficients[a]*backgrounds[a];
                  for(unsigned region=0;region<3;++region)
                    {
                      const bool included=region==0 || (pt[0]>=3 && pt[0]<=nodes.back() && std::abs(pt[1]-mesh_graph(pt[0]))<.75 && (region==1 || (pt[0]>=5.75 && pt[0]<=7.75)));
                      if(!included)continue;
                      auto &d=diagnostic[region];d[0]+=w;d[1]+=w*(t*t)/(4*G);d[2]+=w*(perturbation*perturbation)/(4*G);d[3]+=w*divergence*divergence;d[4]+=w*p*p;d[5]+=w*p*divergence;d[6]=std::max(d[6],std::abs(divergence));
                    }
                }
              if(normal_active(pt))
                {
                  const auto nv=normal_bubble(pt);const auto en=dev(symmetrize(nv.grad));const double divn=trace(nv.grad);
                  normal_check_mass[nv.k]-=w*divn;
                  for(unsigned k=0;k<nc;++k)normal_check(nv.k,k)+=w*(f.first[k]*en-f.second[k]*divn);
                }
              for(unsigned k=0;k<nc;++k){const auto&t=f.first[k];double a[5]={t[0][0],t[1][1],t[2][2],t[0][1],f.second[k]};for(unsigned j=0;j<5;++j)values[5*k+j]+=w*a[j];}
              for(unsigned a=0;a<affine_modes;++a)affine_error=std::max(affine_error,(f.first[a]-2*G*backgrounds[a]).norm());
              for(unsigned j=0;j<probes;++j){SymmetricTensor<2,3>t;for(unsigned k=0;k<nc;++k)t+=C(j,k)*f.first[k];energy[j]+=w*(t*t)/(4*G);}
              const double rr=pt.distance_square(probe_center)/(.75*.75);
              if(rr<1)
                {
                  double ww=w*(1-rr)*(1-rr);
                  if(column_mpi.enabled){cell_probe[0]+=ww;for(unsigned k=0;k<nc;++k){const auto&t=f.first[k];double a[5]={t[0][0],t[1][1],t[2][2],t[0][1],f.second[k]};for(unsigned j=0;j<5;++j)cell_probe[1+5*k+j]+=ww*a[j];}}
                  else{probe_weight+=ww;for(unsigned k=0;k<nc;++k){const auto&t=f.first[k];probe(0,k)+=ww*t[0][0];probe(1,k)+=ww*t[1][1];probe(2,k)+=ww*t[2][2];probe(3,k)+=ww*t[0][1];probe(4,k)+=ww*f.second[k];}}
                }
            }
          for(auto &a:values)a/=area;
          if(column_mpi.enabled)std::copy(values.begin(),values.end(),field_means.begin()+cell->active_cell_index()*nc*5);
          else write_field(values.data(),values.size(),false);
          const auto center=mapping.transform_unit_to_real_cell(cell,Point<2>(.5,.5));auto f=point_stress(center,0);for(unsigned k=0;k<nc;++k){const auto&t=f.first[k];double a[5]={t[0][0],t[1][1],t[2][2],t[0][1],f.second[k]};if(column_mpi.enabled)std::copy(a,a+5,column_centers.begin()+5*(cell->active_cell_index()*nc+k));else write_field(a,5,true);}
          if(column_mpi.enabled)
            {const unsigned i=cell->active_cell_index();cell_geometry[3*i]=center[0];cell_geometry[3*i+1]=center[1];cell_geometry[3*i+2]=area;if(cell_probe[0]>0)cell_probes[i]=std::move(cell_probe);}
          else cf<<cell->active_cell_index()<<','<<center[0]<<','<<center[1]<<','<<area<<'\n';
        }
      if(column_mpi.enabled)
        {
          column_mpi.cell_fields(field_means,tria.n_active_cells(),5*nc);column_mpi.cell_fields(column_centers,tria.n_active_cells(),5*nc);column_mpi.cell_fields(cell_geometry,tria.n_active_cells(),3);
          for(unsigned cell=0;cell<tria.n_active_cells();++cell)
            {write_field(field_means.data()+cell*nc*5,nc*5,false);for(unsigned k=0;k<nc;++k)write_field(column_centers.data()+5*(cell*nc+k),5,true);cf<<cell<<','<<cell_geometry[3*cell]<<','<<cell_geometry[3*cell+1]<<','<<cell_geometry[3*cell+2]<<'\n';}
          const auto records=column_mpi.cells(cell_probes,1+5*nc);
          if(column_mpi.coordinator())for(const auto &record:records){probe_weight+=record.second[0];for(unsigned k=0;k<nc;++k)for(unsigned j=0;j<5;++j)probe(j,k)+=record.second[1+5*k+j];}
          column_mpi.from_root(probe);column_mpi.broadcast(&probe_weight,1,0);
          column_mpi.sum(normal_check);column_mpi.sum(normal_check_mass.begin(),normal_check_mass.size());column_mpi.sum(energy.data(),energy.size());
          for(auto &d:diagnostic){column_mpi.sum(d,6);d[6]=column_mpi.maximum(d[6]);}
        }
      if(compress && column_mpi.coordinator()){AssertThrow(gzclose(gzip_mean)==Z_OK && gzclose(gzip_center)==Z_OK,ExcMessage("Compressed field close failed"));}
      affine_error=column_mpi.maximum(affine_error);
      const auto fields_done=std::chrono::steady_clock::now();
      probe*=1/probe_weight;matrix(out,"probe.csv",probe);
      if(candidate_probes)
        {
          // The selection measure must not depend on unused candidate cuts.
          // This fixed bulk-cell quadrature affects only the extra diagnostic
          // disks, never mechanics, source, energy or normal-work integration.
          const unsigned probe_order=argc>13?std::stoul(argv[13]):8;
          QGauss<2>pq(probe_order);FEValues<2>pv(mapping,fe,pq,update_gradients|update_values|update_JxW_values|update_quadrature_points);
          std::map<unsigned,std::vector<double>> endpoint_cells;
          const unsigned block=1+5*nc;
          for(const auto &cell:dofs.active_cell_iterators())
            {
              if(column_mpi.enabled && !column_mpi.owns(cell->active_cell_index()))continue;
              std::vector<double>local(endpoint_centers.size()*block);bool contributed=false;
              cell->get_dof_indices(ids);pv.reinit(cell);
              for(unsigned q=0;q<pq.size();++q)
                {
                  const auto pt=pv.quadrature_point(q);bool included=false;
                  for(const auto &center:endpoint_centers)included|=pt.distance_square(center)<.75*.75;
                  if(!included)continue;
                  const auto f=stress(pv,q,modes(pt));
                  for(unsigned a=0;a<endpoint_centers.size();++a)
                    {
                      const double r=pt.distance_square(endpoint_centers[a])/(.75*.75);if(r>=1)continue;
                      const double ww=pv.JxW(q)*(1-r)*(1-r);contributed=true;
                      if(column_mpi.enabled)
                        {local[a*block]+=ww;for(unsigned k=0;k<nc;++k){const auto&t=f.first[k];double v[5]={t[0][0],t[1][1],t[2][2],t[0][1],f.second[k]};for(unsigned j=0;j<5;++j)local[a*block+1+5*k+j]+=ww*v[j];}}
                      else
                        {endpoint_weights[a]+=ww;for(unsigned k=0;k<nc;++k){const auto&t=f.first[k];endpoint_probes(5*a,k)+=ww*t[0][0];endpoint_probes(5*a+1,k)+=ww*t[1][1];endpoint_probes(5*a+2,k)+=ww*t[2][2];endpoint_probes(5*a+3,k)+=ww*t[0][1];endpoint_probes(5*a+4,k)+=ww*f.second[k];}}
                    }
                }
              if(column_mpi.enabled && contributed)endpoint_cells[cell->active_cell_index()]=std::move(local);
            }
          if(column_mpi.enabled)
            {
              const auto records=column_mpi.cells(endpoint_cells,endpoint_centers.size()*block);
              if(column_mpi.coordinator())for(const auto &record:records)for(unsigned a=0;a<endpoint_centers.size();++a)
                {endpoint_weights[a]+=record.second[a*block];for(unsigned k=0;k<nc;++k)for(unsigned j=0;j<5;++j)endpoint_probes(5*a+j,k)+=record.second[a*block+1+5*k+j];}
              column_mpi.from_root(endpoint_probes);column_mpi.broadcast(endpoint_weights.data(),endpoint_weights.size(),0);
            }
          auto mass=file(out,"formed-probe-mass.csv");mass<<"arc,x_m,y_m,weight_m2,quadrature_order\n";
          for(unsigned a=0;a<endpoint_centers.size();++a)
            {
              AssertThrow(endpoint_weights[a]>0,ExcMessage("Empty formed-endpoint sampling disk"));
              mass<<a<<','<<endpoint_centers[a][0]<<','<<endpoint_centers[a][1]<<','<<endpoint_weights[a]<<','<<probe_order<<'\n';
              for(unsigned i=0;i<5;++i)for(unsigned k=0;k<nc;++k)endpoint_probes(5*a+i,k)/=endpoint_weights[a];
            }
          matrix(out,"formed-probes.csv",endpoint_probes);
        }
      FullMatrix<double>Nwork_check(nq,nc);double normal_work_error=0;
      for(unsigned j=0;j<normal_cells;++j)for(unsigned k=0;k<nc;++k)for(unsigned q=0;q<3;++q)
        {const double value=normal_check(j,k)/normal_check_mass[j];Nwork_check(3*j+q,k)=value;normal_work_error=std::max(normal_work_error,std::abs(value-Nwork(3*j+q,k))/std::max(1.,std::abs(value)));}
      matrix(out,"normal-work-check.csv",Nwork_check);
      if(physical_diagnostic){auto d=file(out,"physical-diagnostic.csv");d<<"region,area_m2,bulk_J_m,perturbation_J_m,divergence_squared_m2,pressure_squared_Pa2_m2,pressure_divergence_J_m,maximum_divergence\n";for(unsigned region=0;region<3;++region){d<<region;for(double v:diagnostic[region])d<<','<<v;d<<'\n';}}
      auto en=file(out,"energy-check.csv");en<<"probe,physical_J_m,condensed_J_m,relative\n";double energy_error=0;
      for(unsigned j=0;j<probes;++j){double e=0;for(unsigned k=0;k<nc;++k)for(unsigned l=0;l<nc;++l)e+=.5*C(j,k)*K(k,l)*C(j,l);double error=std::abs(e-energy[j])/std::abs(energy[j]);energy_error=std::max(energy_error,error);en<<j<<','<<energy[j]<<','<<e<<','<<error<<'\n';}
      auto meta=file(out,"native.json");meta<<"{\"affine_modes\":"<<affine_modes<<",\"nx\":"<<nx<<",\"degree\":"<<degree<<",\"order\":"<<order<<",\"cells\":"<<tria.n_active_cells()<<",\"bulk_masters\":"<<nm<<",\"jump_modes\":"<<nj<<",\"mixed_residual\":"<<maxres<<",\"energy_relative\":"<<energy_error<<",\"normal_work_check\":"<<normal_work_error<<",\"affine_unit_strain_error_Pa\":"<<affine_error<<",\"probe_weight_m2\":"<<probe_weight<<",\"normal_operator_error\":"<<normal_error<<",\"tangent_operator_error\":"<<tangent_error<<",\"gradient_FD_relative\":"<<gradient_error<<",\"lifting_width_m\":"<<envelope<<",\"normal_test_width_m\":"<<normal_envelope<<",\"fitted_mesh\":"<<(fitted_mesh?"true":"false")<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
      const auto finished=std::chrono::steady_clock::now();struct rusage usage;getrusage(RUSAGE_SELF,&usage);
      const double timing[]={std::chrono::duration<double>(assembled-start).count(),std::chrono::duration<double>(solved-assembled).count(),
        std::chrono::duration<double>(surface_done-solved).count(),std::chrono::duration<double>(fields_done-surface_done).count(),
        std::chrono::duration<double>(finished-fields_done).count(),double(usage.ru_maxrss),double((nc+column_mpi.ranks-1-column_mpi.rank)/column_mpi.ranks),
        double(usage.ru_utime.tv_sec+usage.ru_stime.tv_sec)+1e-6*(usage.ru_utime.tv_usec+usage.ru_stime.tv_usec)};
      std::vector<double>rank_timing(8*column_mpi.ranks);
      MPI_Gather(timing,8,MPI_DOUBLE,rank_timing.data(),8,MPI_DOUBLE,0,MPI_COMM_WORLD);
      auto times=file(out,"parallel-cost.csv");times<<"rank,assembly_s,factor_solve_broadcast_s,surface_s,fields_s,probes_output_s,peak_rss_KiB,owned_columns,cpu_seconds\n";
      if(column_mpi.coordinator())for(unsigned rank=0;rank<column_mpi.ranks;++rank){times<<rank;for(unsigned j=0;j<8;++j)times<<','<<rank_timing[8*rank+j];times<<'\n';}
      auto owners=file(out,"column-owners.csv");owners<<"column,owner_rank\n";for(unsigned k=0;k<nc;++k)owners<<k<<','<<(column_mpi.enabled?k%column_mpi.ranks:0)<<'\n';
      return 0;
      }
      catch(const std::exception &e)
        {std::cerr<<"Native path rank "<<column_mpi.rank<<": "<<e.what()<<'\n';if(column_mpi.ranks>1)MPI_Abort(MPI_COMM_WORLD,1);throw;}
    }
  catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
