#include "coulomb_point.h"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <deal.II/base/exceptions.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <cmath>
#include <cstring>
namespace BC=aspect::MaterialModel::BulkCohesive;namespace TR=BC::TensorReturn;namespace CR=BC::CoulombReturn;
using BC::Tensor;
std::map<std::string,double> defects;std::map<std::string,unsigned int> counts;std::string context;
void check(std::string key,double error,double bound=1e-12)
{defects[key]=std::max(defects[key],std::abs(error));AssertThrow(std::isfinite(error)&&std::abs(error)<=bound,dealii::ExcMessage(key+" error="+std::to_string(error)+" "+context));}
void require(bool pass,std::string text){AssertThrow(pass,dealii::ExcMessage(text+" "+context));}
Tensor shear(double xx,double xy){Tensor a;a[0][0]=xx;a[1][1]=-xx;a[0][1]=xy;return a;}
Tensor rotate(Tensor a,double angle)
{const double c=cos(angle),s=sin(angle);Tensor b;b[0][0]=c*c*a[0][0]+s*s*a[1][1]-2*c*s*a[0][1];b[1][1]=s*s*a[0][0]+c*c*a[1][1]+2*c*s*a[0][1];b[0][1]=c*s*(a[0][0]-a[1][1])+(c*c-s*s)*a[0][1];b[2][2]=a[2][2];return b;}
TR::State rotate(TR::State a,double angle){a.T=rotate(a.T,angle);a.A=rotate(a.A,angle);a.Q=rotate(a.Q,angle);a.K=rotate(a.K,angle);return a;}
std::string bytes(const TR::State &x){std::ostringstream s;{boost::archive::binary_oarchive ar(s);ar<<x;}return s.str();}
std::vector<double> pack(const TR::Response &r)
{std::vector<double> a;for(const auto *t:{&r.state.T,&r.state.A,&r.state.K,&r.state.Q,&r.stress,&r.flow_rate})for(auto ij:{std::make_pair(0,0),std::make_pair(1,1),std::make_pair(2,2),std::make_pair(0,1)})a.push_back((*t)[ij.first][ij.second]);for(double x:{r.state.L,r.speed,r.driving,r.bulk_energy,r.cohesive_energy,r.bulk_dissipation,r.cohesive_dissipation,r.friction_dissipation})a.push_back(x);return a;}
std::vector<double> pack(const TR::Variation &d)
{TR::Response r;r.state.T=d.T;r.state.A=d.A;r.state.K=d.K;r.state.Q=d.Q;r.stress=d.stress;r.flow_rate=d.flow_rate;r.state.L=d.L;r.speed=d.speed;r.driving=d.driving;r.bulk_energy=d.bulk_energy;r.cohesive_energy=d.cohesive_energy;r.bulk_dissipation=d.bulk_dissipation;r.cohesive_dissipation=d.cohesive_dissipation;r.friction_dissipation=d.friction_dissipation;return pack(r);}
double scale(unsigned int k){if(k<24){unsigned int group=k/4;return group==0||group==3||group==4?1000.:.001;}const double a[]={.001,.001,8.,1.,1.,.01,.01,.01};return a[k-24];}
void compare(const TR::Response &a,const TR::Response &b,std::string key)
{auto x=pack(a),y=pack(b);for(unsigned int i=0;i<x.size();++i)check(key,(x[i]-y[i])/std::max(scale(i),std::abs(y[i])));}
int main(int argc,char **argv)
{
 try{
  require(argc==2,"Output directory argument required");const std::string out=argv[1];
  const BC::Parameters material{1e6,1e8,4.,1000.};const double pi=std::acos(-1.);
  std::ofstream angles(out+"/angles.csv"),cases(out+"/cases.csv");angles<<std::setprecision(17);cases<<std::setprecision(17);
  angles<<"mu,rotation_rad,shear_sign,branch,plane_rad,compression_axis_rad,analytical_alpha_rad,normal_Pa,return_Q_norm_Pa,flow_Wxx,flow_Wxy\n";
  for(double mu:{.1,.6})for(double angle:{0.,.37,1.13})for(double sign:{-1.,1.})
   {
    const CR::Parameters p{material,mu,0.};const double pressure=10000.,radius=mu*pressure/std::sqrt(1+mu*mu),dt=.1,beta=1/(1+dt*material.G/material.eta),phi=.2,h=BC::compliance(phi,4.).h;
    const Tensor T=rotate(shear(0.,sign*radius),angle);const double compression=.5*atan2(-2*T[0][1],-(T[0][0]-T[1][1])),alpha=pi/4-.5*atan(mu);
    auto candidates=CR::planes(T,Tensor(),pressure,mu,0.);
    for(unsigned int i=0;i<2;++i)
     {
      double error=1.;for(double expected:{compression-alpha,compression+alpha})error=std::min(error,std::abs(sin(candidates[i].theta-expected)));check("analytical_Coulomb_angle",error);
      const int k=i==0?1:-1;const double theta=candidates[i].theta;
      Tensor g=shear(-k*sin(2*theta),k*cos(2*theta));TR::State old;old.phi=phi;old.Q=-300./beta*g;old.A=h/(2*material.G)*old.Q;old.K=old.A;old.L=std::sqrt(2.)*old.K.norm();old.T=2*material.G*old.A+old.Q;const Tensor inc=(T/beta-old.T)/(2*material.G);
      const auto r=CR::update(p,old,inc,phi,dt,pressure);check("known_return_endpoint_Q",r.point.state.Q.norm()/1000.);
      check("known_active_plane",std::abs(sin(r.planes[i].theta-theta)));
      check("known_tensor_increment",(r.point.flow_rate-300*h/(2*material.G*beta*dt)*g).norm()/.001);
      angles<<mu<<','<<angle<<','<<sign<<','<<k<<','<<r.planes[i].theta<<','<<compression<<','<<alpha<<','<<r.planes[i].normal<<','<<r.point.state.Q.norm()<<','<<r.point.flow_rate[0][0]<<','<<r.point.flow_rate[0][1]<<'\n';
     }
   }
  cases<<"case,mu,old_phi,old_Txx,old_Tyy,old_Tzz,old_Txy,old_Axx,old_Axy,old_Qxx,old_Qxy,Exx,Exy,phi,dt,pressure,active,Qxx,Qxy,gamma_plus,gamma_minus,theta_plus,theta_minus\n";
  unsigned int id=0;double asymmetry=0.;
  for(double mu:{.1,.6})for(double ax:{-.0002,0.,.0002})for(double ay:{-.0001,.0001})for(auto e:{shear(0.,0.),shear(.001,.001),shear(.0024,-.001),shear(-.0015,.0024),shear(.0024,.0024),shear(-.0024,-.0024),shear(.0001,-.0002)})
   {
    context="case="+std::to_string(id);CR::Parameters p{material,mu,0.};TR::State old;old.phi=.1;old.A=shear(ax,ay);old.Q=2*material.G/BC::compliance(old.phi,4.).h*old.A;old.K=old.A;old.L=std::sqrt(2.)*old.K.norm();old.T=2*material.G*old.A+old.Q+shear(100.,200.);old.T[0][0]+=100.;old.T[1][1]+=100.;old.T[2][2]=-200.;
    const std::string initial=bytes(old);const double phi=.2,dt=.1,pressure=10000.;const auto r=CR::update(p,old,e,phi,dt,pressure);counts["active_"+std::to_string(r.active)]++;
    require(bytes(old)==initial,"Trial changed accepted histories");const auto retry=CR::update(p,old,e,phi,dt,pressure);require(bytes(retry.point.state)==bytes(r.point.state),"Retry not deterministic");
    auto ledger=TR::account(material,old,r.point,e,dt);check("energy_ledger",ledger.error/std::max({1.,std::abs(ledger.internal_work),std::abs(ledger.storage_change)}));
    require(r.point.friction_dissipation>=0. && r.system_speed+1e-14>=r.point.speed,"Dissipation or net/system slip inequality");
    double diss=0.;for(unsigned int i=0;i<2;++i)diss+=(p.residual_cohesion+mu*r.planes[i].normal)*BC::compliance(phi,4.).h*r.gamma[i]/(material.G*r.point.beta*dt);
    check("independent_plane_dissipation",(diss-r.point.friction_dissipation)/std::max(.01,std::abs(diss)));
    cases<<id++<<','<<mu<<','<<old.phi<<','<<old.T[0][0]<<','<<old.T[1][1]<<','<<old.T[2][2]<<','<<old.T[0][1]<<','<<ax<<','<<ay<<','<<old.Q[0][0]<<','<<old.Q[0][1]<<','<<e[0][0]<<','<<e[0][1]<<','<<phi<<','<<dt<<','<<pressure<<','<<r.active<<','<<r.point.state.Q[0][0]<<','<<r.point.state.Q[0][1]<<','<<r.gamma[0]<<','<<r.gamma[1]<<','<<r.planes[0].theta<<','<<r.planes[1].theta<<'\n';
    const auto rotated=CR::update(p,rotate(old,.43),rotate(e,.43),phi,dt,pressure);auto back=rotated.point;back.state=rotate(back.state,-.43);back.stress=rotate(back.stress,-.43);back.flow_rate=rotate(back.flow_rate,-.43);compare(back,r.point,"basis_covariance");
    auto reflect=[](Tensor x){x[0][1]*=-1;return x;};auto mirrored=old;mirrored.T=reflect(old.T);mirrored.A=reflect(old.A);mirrored.K=reflect(old.K);mirrored.Q=reflect(old.Q);
    auto reversed=CR::update(p,mirrored,reflect(e),phi,dt,pressure).point;reversed.state.T=reflect(reversed.state.T);reversed.state.A=reflect(reversed.state.A);reversed.state.K=reflect(reversed.state.K);reversed.state.Q=reflect(reversed.state.Q);reversed.stress=reflect(reversed.stress);reversed.flow_rate=reflect(reversed.flow_rate);compare(reversed,r.point,"reversed_shear_covariance");
    const double eps=1e-5;
    for(unsigned int probe=0;probe<4;++probe)
     {
      const auto du=probe==0||probe==1?shear(.0003,-.0004):Tensor();
      const double dphi=probe==0||probe==2?.02:0.,dp=probe==0||probe==3?300.:0.;
      const auto action=CR::derivative(p,old,r,du,dphi,dt,dp);const auto d=pack(action.point);
      const auto rp=CR::update(p,old,e+eps*du,phi+eps*dphi,dt,pressure+eps*dp),rm=CR::update(p,old,e-eps*du,phi-eps*dphi,dt,pressure-eps*dp);const auto plus=pack(rp.point),minus=pack(rm.point);
      for(unsigned int i=0;i<d.size();++i)check("complete_FD",((plus[i]-minus[i])/(2*eps)-d[i])/std::max(scale(i),std::abs(d[i])),1e-6);
      check("system_speed_FD",((rp.system_speed-rm.system_speed)/(2*eps)-action.system_speed)/.001,1e-6);
      for(unsigned int i=0;i<2;++i)if(r.planes[i].defined)
       {
        const double delta=rp.planes[i].theta-rm.planes[i].theta;
        check("plane_direction_FD",.5*atan2(sin(2*delta),cos(2*delta))/(2*eps)-action.theta[i],1e-6);
        check("normal_traction_FD",((rp.planes[i].normal-rm.planes[i].normal)/(2*eps)-action.normal[i])/1000.,1e-6);
        check("effective_shear_FD",((rp.planes[i].shear-rm.planes[i].shear)/(2*eps)-action.shear[i])/1000.,1e-6);
        check("slip_multiplier_FD",((rp.gamma[i]-rm.gamma[i])/(2*eps)-action.gamma[i])/1000.,1e-6);
        check("yield_margin_FD",((rp.planes[i].margin-rm.planes[i].margin)/(2*eps)-action.margin[i])/1000.,1e-6);
       }
     }
    const Tensor b1=shear(.5,0.),b2=shear(0.,.5);auto d1=CR::derivative(p,old,r,b1,0.,dt,0.).point,d2=CR::derivative(p,old,r,b2,0.,dt,0.).point;
    asymmetry=std::max(asymmetry,std::abs(d1.stress*b2-d2.stress*b1)/material.G);
    auto z=p;z.mu=0.;z.residual_cohesion=material.resistance;const auto zr=CR::update(z,old,e,phi,dt,pressure);compare(zr.point,TR::update(material,old,e,phi,dt),"constant_radial_identity");
    const auto za=CR::derivative(z,old,zr,shear(.0003,-.0004),.02,dt,300.);const auto zp=CR::update(z,old,e+eps*shear(.0003,-.0004),phi+eps*.02,dt,pressure+eps*300.),zm=CR::update(z,old,e-eps*shear(.0003,-.0004),phi-eps*.02,dt,pressure-eps*300.);
    for(unsigned int i=0;i<2;++i)if(zr.planes[i].defined){const double d=zp.planes[i].theta-zm.planes[i].theta;check("mu_zero_plane_FD",.5*atan2(sin(2*d),cos(2*d))/(2*eps)-za.theta[i],1e-6);check("mu_zero_normal_FD",((zp.planes[i].normal-zm.planes[i].normal)/(2*eps)-za.normal[i])/1000.,1e-6);}

   }
  {
   CR::Parameters p{material,.1,0.};TR::State virgin;virgin.phi=.2;
   const auto r=CR::update(p,virgin,shear(0.,.0015),.2,.1,10000.);
   require(r.active==3,"Symmetric virgin loading must exercise both conjugate planes");
   check("virgin_net_flow_remains_radial",r.point.flow_rate[0][0]/.001);
   check("virgin_equal_conjugate_slip",(r.gamma[0]-r.gamma[1])/1000.);
  }
  require(counts["active_0"] && counts["active_1"] && counts["active_2"] && counts["active_3"],"Not all active sets exercised");require(asymmetry>1e-6,"Nonsymmetric tangent negative control not exercised");
  for(double mu:{.1,.6})for(double phi:{0.,1e-20,1e-12})
   {
    CR::Parameters p{material,mu,0.};TR::State old;const double pressure=10000.,dt=.1;const Tensor e=shear(.001,.003);
    const auto r=CR::update(p,old,e,phi,dt,pressure);const double eps=1e-5;const auto d=pack(CR::derivative(p,old,r,Tensor(),.02,dt,0.).point),a=pack(r.point),b=pack(CR::update(p,old,e,phi+eps*.02,dt,pressure).point),c=pack(CR::update(p,old,e,phi+2*eps*.02,dt,pressure).point);
    for(unsigned int i=0;i<d.size();++i)check("intact_right_FD",((-3*a[i]+4*b[i]-c[i])/(2*eps)-d[i])/std::max(scale(i),std::abs(d[i])),1e-6);
   }
  for(double mu:{.1,.6})
   {
    CR::Parameters p{material,mu,0.};TR::State old;const double R=mu*10000.,qcrit=2000.;const double peak=(qcrit+std::sqrt((1+mu*mu)*R*R-mu*mu*qcrit*qcrit))/(1+mu*mu),beta=1/(1+.1*material.G/material.eta);
    for(double factor:{.999,1.001}){auto r=CR::update(p,old,shear(0.,peak*factor/(2*material.G*beta)),0.,.1,10000.);require((r.point.driving>8.)==(factor>1.),"Intact strength calibration");}
   }
  TR::State old,shadow;bool replaying=false;old.phi=.2;unsigned int publications=0;std::ofstream path(out+"/path.csv");path<<std::setprecision(17)<<"step,Txx,Txy,Axx,Axy,Kxx,Kxy,L,tauxx,tauxy,Y,Df,active,Wxx,Wxy,net_speed,system_speed,theta_plus,theta_minus\n";
  for(unsigned int step=1;step<=60;++step)
   {
    CR::Parameters p{material,.1,0.};Tensor e=step<=30?shear(0.,.0001):shear(.0001,0.);auto before=bytes(old);auto r=CR::update(p,old,e,.2,.1,10000.);auto reject=CR::update(p,old,.5*e,.2,.1,10000.);(void)reject;require(bytes(old)==before,"Rejected point trial mutated owner");auto retry=CR::update(p,old,e,.2,.1,10000.);require(bytes(retry.point.state)==bytes(r.point.state),"Path retry");if(replaying){auto uninterrupted=CR::update(p,shadow,e,.2,.1,10000.);require(bytes(uninterrupted.point.state)==bytes(r.point.state),"Restart versus uninterrupted state");compare(uninterrupted.point,r.point,"restart_responses");shadow=uninterrupted.point.state;}old=r.point.state;++publications;
    if(step==30){shadow=old;replaying=true;{std::ofstream file(out+"/restart-point.bin",std::ios::binary);boost::archive::binary_oarchive ar(file);ar<<old;}TR::State restored;{std::ifstream file(out+"/restart-point.bin",std::ios::binary);boost::archive::binary_iarchive ar(file);ar>>restored;}require(bytes(old)==bytes(restored),"Tensor restart serialization");old=restored;}
    path<<step<<','<<old.T[0][0]<<','<<old.T[0][1]<<','<<old.A[0][0]<<','<<old.A[0][1]<<','<<old.K[0][0]<<','<<old.K[0][1]<<','<<old.L<<','<<r.point.stress[0][0]<<','<<r.point.stress[0][1]<<','<<r.point.driving<<','<<r.point.friction_dissipation<<','<<r.active<<','<<r.point.flow_rate[0][0]<<','<<r.point.flow_rate[0][1]<<','<<r.point.speed<<','<<r.system_speed<<','<<r.planes[0].theta<<','<<r.planes[1].theta<<'\n';
   }
  require(publications==60,"Point publication count");bool refused=false;try{CR::update({material,.1,0.},TR::State(),shear(0.,.001),.2,.1,0.);}catch(const std::exception&){refused=true;}require(refused,"Tensile/unconfined state was not rejected");
  {
    // Closed trial stress does not suffice when stored backstress is released.
    // This is a pressure reduction from an admissible old fixed-resistance state.
    const double dt=.1,beta=1./(1.+dt*material.G/material.eta),phi=.2,h=BC::compliance(phi,material.m).h;
    TR::State incoming;incoming.phi=phi;incoming.Q=shear(0.,1000.);incoming.A=h/(2*material.G)*incoming.Q;
    incoming.T=2*material.G*incoming.A+shear(0.,900.);
    const Tensor increment=(h*incoming.Q-incoming.T)/(2*material.G);
    auto scalar=material;scalar.resistance=100.;
    const auto unrestricted=TR::update(scalar,incoming,increment,phi,dt);
    require((beta*(incoming.T+2*material.G*increment)-2*material.G*beta*incoming.A).norm()<1e-12,"Return-admission control trial is not zero");
    require(std::abs(unrestricted.stress[0][1])>400.,"Return-admission control does not open a plane");
    bool rejected=false;
    try{CR::update({material,0.,100.},incoming,increment,phi,dt,400.);}
    catch(const std::exception&){rejected=true;}
    require(rejected,"Zero-friction control admitted a noncompressive returned state");
  }
  std::ofstream summary(out+"/checks.json");summary<<std::setprecision(17)<<"{\n\"defects\":{";bool first=true;for(auto x:defects){if(!first)summary<<',';first=false;summary<<'"'<<x.first<<"\":"<<x.second;}summary<<"},\n\"counts\":{";first=true;for(auto x:counts){if(!first)summary<<',';first=false;summary<<'"'<<x.first<<"\":"<<x.second;}summary<<"},\n\"tangent_major_asymmetry_over_G\":"<<asymmetry<<",\n\"point_publications\":60,\n\"closed_contact_rejection_passed\":true,\n\"returned_contact_rejection_passed\":true\n}\n";
  std::cout<<"Coulomb point checks passed; maximum tangent asymmetry/G="<<asymmetry<<std::endl;
 }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
