/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
// Isolated fitted exact-jump mixed-FE reference. No production path is selected.
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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <aspect/reconstructed_fault/cohesive_interface.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <map>
#include <set>
using namespace dealii;
namespace {
constexpr double G=1e6;
bool smooth_turn=false;
double turn_slope=0.,turn_width=.25;
double base_graph(double x)
{const double z=(x-.5)/turn_width;return z<=0?0.:z>=1?turn_slope*(x-.5-.5*turn_width):turn_slope*turn_width*(z*z*z-.5*z*z*z*z);}
double graph(double x)
{
 if(!smooth_turn)return .5*std::max(0.,x-.5);
 if(x<=.75)return base_graph(x);
 // Mesh-coordinate completion only: the physical fault ends at x=.75.
 const double z=std::min(1.,(x-.75)/.125);
 return base_graph(.75)+turn_slope*.125*(z-z*z*z+.5*z*z*z*z);
}
std::pair<double,double> graph_derivatives(double x)
{
 if(!smooth_turn)return {x>.5?.5:0.,0.};
 const double z=(x-.5)/turn_width;
 if(z<=0)return {0.,0.};
 if(x<=.75)return z>=1?std::make_pair(turn_slope,0.):std::make_pair(turn_slope*(3*z*z-2*z*z*z),turn_slope*(6*z-6*z*z)/turn_width);
 const double v=(x-.75)/.125;
 if(v>=1)return {0.,0.};
 return {turn_slope*(1-3*v*v+2*v*v*v),turn_slope*(-6*v+6*v*v)/.125};
}
Tensor<1,2> tangent(double x) {Tensor<1,2> t;t[0]=1;t[1]=graph_derivatives(x).first;return t/t.norm();}
Tensor<1,2> tangent_derivative(double x)
{const auto d=graph_derivatives(x);Tensor<1,2> t;t[0]=-d.first*d.second;t[1]=d.second;return t/std::pow(1+d.first*d.first,1.5);}
SymmetricTensor<2,3> dev(const SymmetricTensor<2,2> &e)
{SymmetricTensor<2,3>d;for(unsigned i=0;i<2;++i)for(unsigned j=i;j<2;++j)d[i][j]=e[i][j];double tr=trace(d)/3;for(unsigned i=0;i<3;++i)d[i][i]-=tr;return d;}
std::ofstream file(const std::string &dir,const std::string &name)
{std::ofstream f(dir+"/"+name);AssertThrow(f,ExcMessage("Cannot open output"));f<<std::setprecision(17);return f;}
void matrix(const std::string &dir,const std::string &name,const FullMatrix<double>&m)
{auto f=file(dir,name);for(unsigned i=0;i<m.m();++i){for(unsigned j=0;j<m.n();++j)f<<(j?",":"")<<m(i,j);f<<'\n';}}
}
int main(int argc,char **argv)
{
 try {
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  AssertThrow(argc>=4,ExcMessage("usage: kink nx ny output [quadrature]"));
  const unsigned nx=std::stoul(argv[1]),ny=std::stoul(argv[2]),order=argc>4?std::stoul(argv[4]):3;
  smooth_turn=argc>5 && std::string(argv[5])=="smooth";
  if(smooth_turn){AssertThrow(argc>=10,ExcMessage("smooth requires angle width axial-ratio shear-sign"));turn_slope=std::tan(std::stod(argv[6])*numbers::PI/180.);turn_width=std::stod(argv[7]);
    AssertThrow(turn_width>=.125 && turn_width<=.25 && std::abs(nx*turn_width-std::round(nx*turn_width))<1e-12,ExcMessage("Aligned finite turn required"));}
  AssertThrow(nx%8==0 && ny%2==0,ExcMessage("Aligned material endpoints required"));
  const std::string out=argv[3];const auto start=std::chrono::steady_clock::now();
  Triangulation<2> tria;GridGenerator::subdivided_hyper_rectangle(tria,std::vector<unsigned>{nx,ny},Point<2>(0,-.5),Point<2>(1,.5));
  if(!smooth_turn)GridTools::transform([](const Point<2>&p){return Point<2>(p[0],p[1]+(1-2*std::abs(p[1]))*graph(p[0]));},tria);
  FE_Q<2> scalar(2);FESystem<2> fe(scalar,2,FE_DGP<2>(1),1);
  FESystem<2> geometry_fe(FE_Q<2>(4),2);DoFHandler<2> geometry_dofs(tria);Vector<double> coordinates;
  std::unique_ptr<Mapping<2>> mapping_ptr;
  if(smooth_turn){geometry_dofs.distribute_dofs(geometry_fe);coordinates.reinit(geometry_dofs.n_dofs());std::vector<types::global_dof_index> gi(geometry_fe.n_dofs_per_cell());MappingQ1<2> reference;
   for(const auto &cell:geometry_dofs.active_cell_iterators()){cell->get_dof_indices(gi);for(unsigned j=0;j<gi.size();++j){const auto pt=reference.transform_unit_to_real_cell(cell,geometry_fe.get_unit_support_points()[j]);const unsigned c=geometry_fe.system_to_component_index(j).first;
     coordinates[gi[j]]=c==0?pt[0]:pt[1]+(1-2*std::abs(pt[1]))*graph(pt[0]);}}
   mapping_ptr=std::make_unique<MappingFEField<2,2,Vector<double>>>(geometry_dofs,coordinates);
  }else mapping_ptr=std::make_unique<MappingQ1<2>>();
  const auto &mapping=*mapping_ptr;DoFHandler<2> dofs(tria);dofs.distribute_dofs(fe);
  const FEValuesExtractors::Vector u(0);const FEValuesExtractors::Scalar p(2);
  std::map<types::global_dof_index,unsigned> component,master;
  std::map<types::global_dof_index,Point<2>> location;
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  for(const auto &cell:dofs.active_cell_iterators()){
   cell->get_dof_indices(ids);
   for(unsigned j=0;j<ids.size();++j){unsigned c=fe.system_to_component_index(j).first;component[ids[j]]=c;
    if(c<2)location[ids[j]]=mapping.transform_unit_to_real_cell(cell,scalar.get_unit_support_points()[fe.system_to_base_index(j).second]);}}
  types::global_dof_index gauge=numbers::invalid_dof_index;
  for(auto c:component)if(c.second==2){gauge=c.first;break;}
  std::map<unsigned,unsigned> jump_ids;std::vector<double> jump_x;
  for(unsigned i=nx/2+1;i<3*nx/2;++i)if(smooth_turn || i!=nx){jump_ids[i]=jump_x.size();jump_x.push_back(double(i)/(2*nx));}
  const unsigned nj=jump_x.size();
  for(auto c:component){bool boundary=false;if(c.second<2){const auto x=location.at(c.first);boundary=x[0]<1e-12||x[0]>1-1e-12||std::abs(x[1])>.5-1e-12;}
   if(!boundary && c.first!=gauge)master[c.first]=master.size();}
  const unsigned nm=master.size();DynamicSparsityPattern dsp(nm);
  for(const auto &cell:dofs.active_cell_iterators()){cell->get_dof_indices(ids);for(auto i:ids)for(auto j:ids)if(master.count(i)&&master.count(j))dsp.add(master.at(i),master.at(j));}
  SparsityPattern sp;sp.copy_from(dsp);SparseMatrix<double>A(sp);FullMatrix<double>B(nm,nj+1),D(nj+1,nj+1);
  QGauss<2> rule(order);FEValues<2> v(mapping,fe,rule,update_gradients|update_values|update_JxW_values|update_quadrature_points);
  SymmetricTensor<2,3> affine;
  const bool axial=argc>5 && std::string(argv[5])=="axial";
  if(axial){affine[0][0]=1.;affine[1][1]=-1.;}else affine[0][1]=.5;
  if(smooth_turn){affine[0][0]=-std::stod(argv[8]);affine[1][1]=std::stod(argv[8]);affine[0][1]=.5*std::stod(argv[9]);}
  auto trace_modes=[&](const auto &cell){std::vector<std::tuple<unsigned,unsigned,double>> r;
   if(smooth_turn || cell->center()[1]>graph(cell->center()[0]))for(unsigned j=0;j<ids.size();++j)if(smooth_turn?component.at(ids[j])==0:component.at(ids[j])<2){auto x=location.at(ids[j]);
    if(std::abs(x[1]-graph(x[0]))<1e-12){unsigned k=std::lround(2*nx*x[0]);if(jump_ids.count(k))r.emplace_back(j,jump_ids.at(k)+1,smooth_turn?(cell->center()[1]>0?.5:-.5):tangent(x[0])[component.at(ids[j])]);}}
   return r;};
  auto lift_gradient=[&](const auto &values,unsigned q,unsigned j,double factor){
   if(!smooth_turn)return factor*values[u].gradient(j,q);
   const double xx=values.quadrature_point(q)[0];const auto t=tangent(xx),dt=tangent_derivative(xx);const auto grad=values.shape_grad(j,q);const double phi=values.shape_value(j,q);Tensor<2,2> F;
   for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)F[a][b]=factor*(t[a]*grad[b]+(b==0?dt[a]*phi:0.));
   return F;};
  Vector<double> mean(nm);double volume=0;
  for(const auto &cell:dofs.active_cell_iterators()){
   cell->get_dof_indices(ids);v.reinit(cell);auto jm=trace_modes(cell);
   for(unsigned q=0;q<rule.size();++q){const double w=v.JxW(q);volume+=w;
    std::vector<SymmetricTensor<2,3>> E(nj+1);E[0]=affine;std::vector<double> div(nj+1,0.);
    for(auto a:jm){unsigned j,k;double t;std::tie(j,k,t)=a;if(smooth_turn){const auto F=lift_gradient(v,q,j,t);E[k]+=dev(symmetrize(F));div[k]+=trace(F);}else{E[k]+=t*dev(v[u].symmetric_gradient(j,q));div[k]+=t*v[u].divergence(j,q);}}
    for(unsigned i=0;i<ids.size();++i)if(master.count(ids[i])){unsigned ii=master.at(ids[i]);auto ei=dev(v[u].symmetric_gradient(i,q));double di=v[u].divergence(i,q),pi=v[p].value(i,q);mean[ii]+=w*pi;
     for(unsigned k=0;k<=nj;++k)B(ii,k)+=w*(2*ei*E[k]-pi*div[k]);
     for(unsigned j=0;j<ids.size();++j)if(master.count(ids[j]))A.add(ii,master.at(ids[j]),w*(2*ei*dev(v[u].symmetric_gradient(j,q))-di*v[p].value(j,q)-pi*v[u].divergence(j,q)));}
    for(unsigned k=0;k<=nj;++k)for(unsigned l=0;l<=nj;++l)D(k,l)+=w*2*(E[k]*E[l]);}}
  SparseDirectUMFPACK solver;solver.initialize(A);FullMatrix<double> response(nm,nj+1);Vector<double> rhs(nm),x(nm),residual(nm);double maxres=0.;
  std::vector<double> pmean(nj+1);
  for(unsigned k=0;k<=nj;++k){for(unsigned i=0;i<nm;++i)rhs[i]=-B(i,k);solver.vmult(x,rhs);A.vmult(residual,x);residual-=rhs;maxres=std::max(maxres,residual.linfty_norm());pmean[k]=mean*x/volume;for(unsigned i=0;i<nm;++i)response(i,k)=x[i];}
  FullMatrix<double>K(D);for(unsigned k=0;k<=nj;++k)for(unsigned l=0;l<=nj;++l){for(unsigned i=0;i<nm;++i)K(k,l)+=B(i,k)*response(i,l);K(k,l)*=G;}
  matrix(out,"stiffness.csv",K);matrix(out,"response.csv",response);
  auto nodes=file(out,"nodes.csv");nodes<<"mode,x_m,y_m,tx,ty,branch\n";for(unsigned j=0;j<nj;++j){auto t=tangent(jump_x[j]);nodes<<j<<','<<jump_x[j]<<','<<graph(jump_x[j])<<','<<t[0]<<','<<t[1]<<','<<(jump_x[j]>.5?1:0)<<'\n';}
  // Native complete physical stress responses, pressure gauge is mean zero.
  auto stress=[&](const auto &values,unsigned q,unsigned k,const auto &jm){SymmetricTensor<2,3> e;double pressure=-pmean[k];if(k==0)e=affine;
   for(auto a:jm){unsigned j,l;double t;std::tie(j,l,t)=a;if(k==l)e+=smooth_turn?dev(symmetrize(lift_gradient(values,q,j,t))):t*dev(values[u].symmetric_gradient(j,q));}
   for(unsigned j=0;j<ids.size();++j)if(master.count(ids[j])){double c=response(master.at(ids[j]),k);e+=c*dev(values[u].symmetric_gradient(j,q));pressure+=c*values[p].value(j,q);}
   return std::make_pair(2*G*e,G*pressure);};
  QGauss<1> srule(3);FullMatrix<double>Q(3*nx/2,nj);FullMatrix<double>normal_minus(Q.m(),nj+1),normal_plus(Q.m(),nj+1),shear(Q.m(),nj+1);
  FullMatrix<double>jump_trace_x(Q.m(),nj+1),jump_trace_y(Q.m(),nj+1);
  auto surface=file(out,"surface.csv");surface<<"q,x_m,y_m,weight_m,tx,ty,branch\n";
  for(unsigned seg=nx/4;seg<3*nx/4;++seg)for(unsigned q=0;q<3;++q){unsigned row=3*(seg-nx/4)+q;double xi=srule.point(q)[0],xx=(seg+xi)/nx;auto t=tangent(xx);Tensor<1,2>n;n[0]=-t[1];n[1]=t[0];
   double shapes[3]={(1-xi)*(1-2*xi),4*xi*(1-xi),xi*(2*xi-1)};
   for(unsigned j=0;j<3;++j){auto it=jump_ids.find(2*seg+j);if(it!=jump_ids.end())Q(row,it->second)=shapes[j];}
   surface<<row<<','<<xx<<','<<graph(xx)<<','<<srule.weight(q)/(nx*t[0])<<','<<t[0]<<','<<t[1]<<','<<(xx>.5?1:0)<<'\n';
   for(const auto &cell:dofs.active_cell_iterators())if(std::abs(cell->center()[0]-(seg+.5)/nx)<1e-12){
    for(unsigned face=0;face<4;++face)if(std::abs(cell->face(face)->center()[1]-(smooth_turn?0.:graph(cell->face(face)->center()[0])))<1e-12){
     Point<2> phys(xx,graph(xx));auto unit=mapping.transform_real_to_unit_cell(cell,phys);FEValues<2>fv(mapping,fe,Quadrature<2>(std::vector<Point<2>>{unit}),update_values|update_gradients|update_quadrature_points);fv.reinit(cell);cell->get_dof_indices(ids);auto jm=trace_modes(cell);bool upper=cell->center()[1]>(smooth_turn?0.:graph(cell->center()[0]));
     for(unsigned k=0;k<=nj;++k){auto val=stress(fv,0,k,jm);auto tau=val.first;double nn=0,tn=0;for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b){nn+=n[a]*tau[a][b]*n[b];tn+=t[a]*tau[a][b]*n[b];}
      (upper?normal_plus:normal_minus)(row,k)=val.second-nn;shear(row,k)+=.5*tn;
      if(smooth_turn){Tensor<1,2> displacement;for(auto a:jm){unsigned j,l;double factor;std::tie(j,l,factor)=a;if(k==l)displacement+=factor*fv.shape_value(j,0)*t;}
       for(unsigned j=0;j<ids.size();++j)if(master.count(ids[j]))displacement+=response(master.at(ids[j]),k)*fv[u].value(j,0);
       jump_trace_x(row,k)+=(upper?1.:-1.)*displacement[0];jump_trace_y(row,k)+=(upper?1.:-1.)*displacement[1];}
      }}}}
  matrix(out,"Q.csv",Q);matrix(out,"normal-minus.csv",normal_minus);matrix(out,"normal-plus.csv",normal_plus);matrix(out,"shear.csv",shear);
  double normal_gap=0.,trace_error=0.;
  if(smooth_turn){matrix(out,"jump-x.csv",jump_trace_x);matrix(out,"jump-y.csv",jump_trace_y);
   for(unsigned seg=nx/4;seg<3*nx/4;++seg)for(unsigned q=0;q<3;++q){const unsigned row=3*(seg-nx/4)+q;const auto t=tangent((seg+srule.point(q)[0])/nx);for(unsigned k=0;k<=nj;++k){normal_gap=std::max(normal_gap,std::abs(-t[1]*jump_trace_x(row,k)+t[0]*jump_trace_y(row,k)));trace_error=std::max(trace_error,std::abs(t[0]*jump_trace_x(row,k)+t[1]*jump_trace_y(row,k)-(k?Q(row,k-1):0.)));}}}
  QGauss<2> check_rule(smooth_turn?8:6);FEValues<2> cv(mapping,fe,check_rule,update_values|update_gradients|update_quadrature_points|update_JxW_values);
  std::ofstream fields,field_binary,cell_table,curves;
  if(smooth_turn){field_binary.open(out+"/field-cell-response.bin",std::ios::binary);cell_table=file(out,"field-cells.csv");cell_table<<"cell,x_m,y_m,area_m2\n";curves=file(out,"mesh-curves.csv");curves<<"cell,face,point,x_m,y_m\n";}
  else{fields=file(out,"field-response.csv");fields<<"cell,q,x_m,y_m,weight_m2,mode,tau_xx_Pa,tau_yy_Pa,tau_zz_Pa,tau_xy_Pa,pprime_Pa\n";}
  FullMatrix<double>energy(nj+1,nj+1);double gradient_fd=0.,affine_error=0.;bool gradient_checked=false;
  auto mesh=file(out,"mesh.csv");mesh<<"cell,vertex,x_m,y_m\n";
  for(const auto &cell:dofs.active_cell_iterators()){cell->get_dof_indices(ids);cv.reinit(cell);auto jm=trace_modes(cell);
   std::vector<double> cellvalues(5*(nj+1));double area=0.;Point<2> centroid;
   for(unsigned j=0;j<4;++j){const auto pt=smooth_turn?mapping.transform_unit_to_real_cell(cell,GeometryInfo<2>::unit_cell_vertex(j)):cell->vertex(j);mesh<<cell->active_cell_index()<<','<<j<<','<<pt[0]<<','<<pt[1]<<'\n';}
   if(smooth_turn)for(unsigned face=0;face<4;++face)for(unsigned k=0;k<=4;++k){const auto pt=mapping.transform_unit_to_real_cell(cell,face<2?Point<2>(face,double(k)/4):Point<2>(double(k)/4,face-2));curves<<cell->active_cell_index()<<','<<face<<','<<k<<','<<pt[0]<<','<<pt[1]<<'\n';}
   if(smooth_turn && !gradient_checked && !jm.empty() && cell->center()[0]>.5 && cell->center()[0]<.5+turn_width){
    unsigned j,k;double factor;std::tie(j,k,factor)=jm.front();const auto pt=cv.quadrature_point(0);const auto exact=lift_gradient(cv,0,j,factor);Tensor<2,2> fd;const double eps=1e-4/nx;
    for(unsigned b=0;b<2;++b){auto plus=pt,minus=pt;plus[b]+=eps;minus[b]-=eps;const auto up=mapping.transform_real_to_unit_cell(cell,plus),um=mapping.transform_real_to_unit_cell(cell,minus);const auto vp=factor*fe.shape_value(j,up)*tangent(plus[0]),vm=factor*fe.shape_value(j,um)*tangent(minus[0]);for(unsigned a=0;a<2;++a)fd[a][b]=(vp[a]-vm[a])/(2*eps);}
    gradient_fd=(fd-exact).norm()/std::max(exact.norm(),1e-30);gradient_checked=true;
   }
   for(unsigned q=0;q<check_rule.size();++q){std::vector<SymmetricTensor<2,3>>tau(nj+1);for(unsigned k=0;k<=nj;++k){auto val=stress(cv,q,k,jm);tau[k]=val.first;auto pt=cv.quadrature_point(q);
     if(!smooth_turn)fields<<cell->active_cell_index()<<','<<q<<','<<pt[0]<<','<<pt[1]<<','<<cv.JxW(q)<<','<<k<<','<<tau[k][0][0]<<','<<tau[k][1][1]<<','<<tau[k][2][2]<<','<<tau[k][0][1]<<','<<val.second<<'\n';
     else{const double data[5]={tau[k][0][0],tau[k][1][1],tau[k][2][2],tau[k][0][1],val.second};for(unsigned a=0;a<5;++a)cellvalues[5*k+a]+=cv.JxW(q)*data[a];if(k==0)affine_error=std::max(affine_error,std::max((tau[k]-2*G*affine).norm(),std::abs(val.second)));}}
    if(smooth_turn){area+=cv.JxW(q);centroid+=cv.JxW(q)*cv.quadrature_point(q);}
    for(unsigned k=0;k<=nj;++k)for(unsigned l=0;l<=nj;++l)energy(k,l)+=cv.JxW(q)*(tau[k]*tau[l])/(2*G);}
   if(smooth_turn){centroid/=area;cell_table<<cell->active_cell_index()<<','<<centroid[0]<<','<<centroid[1]<<','<<area<<'\n';for(auto &value:cellvalues)value/=area;field_binary.write(reinterpret_cast<const char*>(cellvalues.data()),cellvalues.size()*sizeof(double));}
  }
  matrix(out,"energy-gauss6.csv",energy);
  // Independent samples of the unchanged production point evaluator for port checks.
  auto returns=file(out,"point-reference.csv");returns<<"damage,jump,reference,theta,increment,pressure,dt,resistance,derivative,pressure_derivative,new_damage,new_theta,storage,friction,radiation\n";
  aspect::CohesiveInterface::Parameters law;
  for(double a:{.0,.25,.7,1.})for(double inc:{-1e-5,0.,2e-6,1e-4})for(double pn:{8000.,11000.}){
   aspect::CohesiveInterface::PointState old;old.damage=a;old.jump=a*.0002;old.reference=a*.0001;old.theta=1.2;auto r=aspect::CohesiveInterface::evaluate(law,old,inc,pn,.002);
   returns<<a<<','<<old.jump<<','<<old.reference<<','<<old.theta<<','<<inc<<','<<pn<<",.002,"<<r.resistance<<','<<r.derivative<<','<<r.pressure_derivative<<','<<r.state.damage<<','<<r.state.theta<<','<<r.storage<<','<<r.friction<<','<<r.radiation<<'\n';}
  auto meta=file(out,"native.json");meta<<"{\"nx\":"<<nx<<",\"ny\":"<<ny<<",\"quadrature\":"<<order<<",\"cells\":"<<tria.n_active_cells()<<",\"bulk_masters\":"<<nm<<",\"jump_modes\":"<<nj<<",\"mixed_residual\":"<<maxres<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  if(smooth_turn)meta<<",\"smooth_turn\":true,\"turn_width_m\":"<<turn_width<<",\"angle_deg\":"<<std::stod(argv[6])<<",\"axial_ratio\":"<<std::stod(argv[8])<<",\"shear_sign\":"<<std::stod(argv[9])<<",\"gradient_FD_relative\":"<<gradient_fd<<",\"affine_stress_error_Pa\":"<<affine_error<<",\"normal_jump_operator_error\":"<<normal_gap<<",\"tangent_jump_operator_error\":"<<trace_error<<",\"independent_energy_order\":8";
  meta<<"}\n";
  return 0;
 }catch(std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
