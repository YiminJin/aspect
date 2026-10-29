/* P5a: standalone point-return and small accepted-history fixture only. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive_tensor.h>
#include <deal.II/base/exceptions.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <map>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <cstring>

namespace BC=aspect::MaterialModel::BulkCohesive;
namespace TR=BC::TensorReturn;
using BC::Tensor;
const BC::Parameters parameters{1e6,1e8,4.,1000.};
const double Kf=8.;
std::map<std::string,double> defects;
std::map<std::string,unsigned int> counts;
std::string context;

void measure(const std::string &name,double error,const double bound=1e-12)
{
  error=std::abs(error);defects[name]=std::max(defects[name],error);
  std::ostringstream message;message<<std::setprecision(17)<<name<<" error="<<error<<" bound="<<bound<<" "<<context;
  AssertThrow(std::isfinite(error) && error<=bound,dealii::ExcMessage(message.str()));
}
void require(const bool pass,const std::string &message)
{AssertThrow(pass,dealii::ExcMessage(message));}
Tensor shear(const double xx,const double xy)
{Tensor t;t[0][0]=xx;t[1][1]=-xx;t[0][1]=xy;return t;}
Tensor original_direction(){return shear(0.,.5);}
Tensor rotate(const Tensor &t,const double angle)
{
  const double c=std::cos(angle),s=std::sin(angle);
  Tensor y;y[0][0]=c*c*t[0][0]+s*s*t[1][1]-2*c*s*t[0][1];
  y[1][1]=s*s*t[0][0]+c*c*t[1][1]+2*c*s*t[0][1];
  y[0][1]=c*s*(t[0][0]-t[1][1])+(c*c-s*s)*t[0][1];y[2][2]=t[2][2];return y;
}
TR::State rotate(TR::State x,const double angle)
{x.T=rotate(x.T,angle);x.A=rotate(x.A,angle);x.K=rotate(x.K,angle);x.Q=rotate(x.Q,angle);return x;}
TR::State embed(const BC::State &s)
{
  TR::State x;x.T=s.T;x.phi=s.phi;x.A=s.a*original_direction();
  x.K=s.signed_shear*original_direction();x.Q=2*s.q*original_direction();x.L=s.absolute_shear;return x;
}
std::string bytes(const TR::State &s)
{std::ostringstream out;{boost::archive::binary_oarchive archive(out);archive<<s;}return out.str();}
bool same_state(const TR::State &a,const TR::State &b)
{
  // Check the actual fields, independently of the serialization list.
  const auto same=[](const double x,const double y){return std::memcmp(&x,&y,sizeof(double))==0;};
  for(const auto &p:{std::make_pair(a.T,b.T),std::make_pair(a.A,b.A),std::make_pair(a.K,b.K),std::make_pair(a.Q,b.Q)})
    for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)if(!same(p.first[i][j],p.second[i][j]))return false;
  return same(a.phi,b.phi) && same(a.L,b.L);
}

void compare(const TR::Response &r,const BC::Response &b,const std::string &key)
{
  const auto mapped=embed(b.state);
  const auto tensor=[&](const Tensor &actual,const Tensor &reference,const double floor)
    {measure(key,(actual-reference).norm()/std::max(floor,reference.norm()));};
  tensor(r.state.T,mapped.T,1000.);tensor(r.state.Q,mapped.Q,1000.);
  tensor(r.state.A,mapped.A,.001);tensor(r.state.K,mapped.K,.001);
  tensor(r.stress,b.stress,1000.);tensor(r.flow_rate,b.w*original_direction(),.001);
  const auto scalar=[&](const double actual,const double reference,const double floor)
    {measure(key,(actual-reference)/std::max(floor,std::abs(reference)));};
  scalar(r.driving,b.driving,8.);scalar(r.bulk_energy,b.bulk_energy,1.);
  scalar(r.cohesive_energy,b.cohesive_energy,1.);scalar(r.state.L,b.state.absolute_shear,.001);
  scalar(r.bulk_dissipation,b.bulk_dissipation,.01);scalar(r.cohesive_dissipation,b.cohesive_dissipation,.01);
  scalar(r.friction_dissipation,b.friction_dissipation,.01);
}

void fixed_identity()
{
  const auto S=original_direction();
  for(const double phi:{0.,1e-20,1e-12,.005,.2,.6})
    {
      BC::State old;old.phi=phi;old.a=BC::compliance(phi,parameters.m).h*250./parameters.G;
      old.q=phi>0.?250.:0.;old.signed_shear=old.a;old.absolute_shear=std::abs(old.a);
      old.T[0][0]=750.;old.T[1][1]=-250.;old.T[2][2]=-500.;old.T[0][1]=parameters.G*old.a+600.;
      for(const double drive:{-8000.,0.,8000.})
        for(const double step:{.00125,.1,1.})
          {
            context="fixed identity phi="+std::to_string(phi)+" drive="+std::to_string(drive)+" dt="+std::to_string(step);
            auto strain=shear(.0001,drive/(2*parameters.G));
            const auto t=TR::update(parameters,embed(old),strain,phi,step,&S);
            const auto s=BC::update(parameters,old,strain,phi,step);compare(t,s,"fixed_identity");
            for(const auto &perturb:{std::make_pair(shear(.001,.0003),0.),std::make_pair(Tensor(),.02)})
              {
                const auto d=TR::derivative(parameters,embed(old),t,perturb.first,perturb.second,step,&S);
                const double dt0=2*parameters.G*t.beta*perturb.first[0][1];
                const double da=s.da_dt*dt0+s.da_dphi*perturb.second;
                const double dq=s.dq_dt*dt0+s.dq_dphi*perturb.second;
                measure("fixed_tangent_A",(d.A-da*S).norm()/std::max(.001,(da*S).norm()));
                measure("fixed_tangent_Q",(d.Q-2*dq*S).norm()/std::max(1000.,(2*dq*S).norm()));
                const auto tau=2*parameters.G*t.beta*perturb.first-2*parameters.G*da*S;
                measure("fixed_tangent_stress",(d.stress-tau).norm()/std::max(1000.,tau.norm()));
                const double dY=s.driving_dt*dt0+s.driving_dphi*perturb.second;
                measure("fixed_tangent_driving",(d.driving-dY)/std::max(8.,std::abs(dY)));
              }
            const auto negative=-S;
            const auto opposite=TR::update(parameters,embed(old),strain,phi,step,&negative);
            measure("fixed_direction_sign",(opposite.state.A-t.state.A).norm()/.001);
            measure("fixed_direction_sign",(opposite.flow_rate-t.flow_rate).norm()/.001);
            measure("fixed_direction_sign",(opposite.state.K-t.state.K).norm()/.001);
            measure("fixed_direction_sign",(2*(opposite.flow_rate*negative)+2*(t.flow_rate*S))/.001);
          }
    }
}

std::vector<double> pack(const TR::Response &r)
{
  std::vector<double> v;
  for(const auto *t:{&r.state.T,&r.state.A,&r.state.K,&r.state.Q,&r.stress,&r.flow_rate})
    for(const auto &ij:{std::make_pair(0,0),std::make_pair(1,1),std::make_pair(2,2),std::make_pair(0,1)})
      v.push_back((*t)[ij.first][ij.second]);
  for(const double x:{r.state.L,r.speed,r.driving,r.bulk_energy,r.cohesive_energy,
                      r.bulk_dissipation,r.cohesive_dissipation,r.friction_dissipation})v.push_back(x);
  return v;
}
std::vector<double> pack(const TR::Variation &d)
{
  TR::Response r;r.state.T=d.T;r.state.A=d.A;r.state.K=d.K;r.state.Q=d.Q;r.stress=d.stress;r.flow_rate=d.flow_rate;
  r.state.L=d.L;r.speed=d.speed;r.driving=d.driving;r.bulk_energy=d.bulk_energy;r.cohesive_energy=d.cohesive_energy;
  r.bulk_dissipation=d.bulk_dissipation;r.cohesive_dissipation=d.cohesive_dissipation;r.friction_dissipation=d.friction_dissipation;
  return pack(r);
}
double scale(const unsigned int k)
{if(k<24){const auto group=k/4;return group==0 || group==3 || group==4 ? 1000. : .001;}
 const double s[]={.001,.001,8.,1.,1.,.01,.01,.01};return s[k-24];}

void derivative_checks(const std::string &out)
{
  std::ofstream cases(out+"/derivative-inputs.csv");cases<<std::setprecision(17);
  cases<<"case,restricted,old_phi,old_Txx,old_Tyy,old_Txy,old_Axx,old_Axy,old_Qxx,old_Qxy,phi,dt,Exx,Eyy,Exy,sliding,Axx,Axy,Qxx,Qxy,tauxx,tauyy,tauzz,tauxy,Y,Eb,Ec,Db,Dc,Df,Wxx,Wxy,L,Kxx,Kxy\n";
  for(unsigned int test=0;test<5;++test)
    for(bool restricted:{false,true})
      {
        context="derivative case="+std::to_string(test)+" restricted="+std::to_string(restricted);
        TR::State old;old.phi=test<3?.1:0.;old.A=restricted?shear(0.,.0001):shear(.00006,.0001);
        if(test>=3)old.A=0.;
        const double h0=BC::compliance(old.phi,parameters.m).h;
        old.Q=h0>0.?2*parameters.G/h0*old.A:Tensor();old.K=old.A;old.L=std::sqrt(2.)*old.K.norm();
        old.T=2*parameters.G*old.A+old.Q+(restricted?shear(0.,350.):shear(150.,350.));
        // An orthogonal full plane-strain mode must survive unchanged.
        old.T[0][0]+=100.;old.T[1][1]+=100.;old.T[2][2]=-200.;
        Tensor increment;
        if(test==1)increment=shear(.004,.005);
        if(test==2)increment=shear(-.004,-.005);
        if(test==4)increment=shear(.002,.002);
        const double phi=test<3?.2:0.,dt=.1;const auto S=original_direction();const Tensor *rule=restricted?&S:nullptr;
        const auto r=TR::update(parameters,old,increment,phi,dt,rule);
        cases<<test<<','<<restricted<<','<<old.phi<<','<<old.T[0][0]<<','<<old.T[1][1]<<','<<old.T[0][1]<<','
             <<old.A[0][0]<<','<<old.A[0][1]<<','<<old.Q[0][0]<<','<<old.Q[0][1]<<','<<phi<<','<<dt<<','
             <<increment[0][0]<<','<<increment[1][1]<<','<<increment[0][1]<<','<<r.sliding;
        for(const double x:{r.state.A[0][0],r.state.A[0][1],r.state.Q[0][0],r.state.Q[0][1],r.stress[0][0],r.stress[1][1],r.stress[2][2],r.stress[0][1],r.driving,
                            r.bulk_energy,r.cohesive_energy,r.bulk_dissipation,r.cohesive_dissipation,r.friction_dissipation,r.flow_rate[0][0],r.flow_rate[0][1],r.state.L,r.state.K[0][0],r.state.K[0][1]})cases<<','<<x;
        cases<<'\n';
        std::vector<std::pair<Tensor,double>> probes{{shear(.001,0.),0.},{shear(0.,.001),0.},{Tensor(),.02},{shear(.0007,-.0004),.02}};
        Tensor normal;normal[0][0]=.001;normal[1][1]=.0004;normal[2][2]=-.0014;probes.emplace_back(normal,0.);
        for(const auto &probe:probes)
          {
            const auto d=pack(TR::derivative(parameters,old,r,probe.first,probe.second,dt,rule));
            const auto v=pack(r);
            for(const double epsilon:{1e-4,5e-5})
              {
                const auto a=pack(TR::update(parameters,old,increment+epsilon*probe.first,phi+epsilon*probe.second,dt,rule));
                const bool one_sided=phi==old.phi && probe.second>0.;
                const auto b=pack(TR::update(parameters,old,increment+(one_sided?2.:-1.)*epsilon*probe.first,
                                            phi+(one_sided?2.:-1.)*epsilon*probe.second,dt,rule));
                for(unsigned int k=0;k<v.size();++k)
                  {
                    const double fd=one_sided?(-3*v[k]+4*a[k]-b[k])/(2*epsilon):(a[k]-b[k])/(2*epsilon);
                    measure("complete_return_FD",(fd-d[k])/std::max(scale(k),std::abs(d[k])),1e-6);
                  }
              }
          }
        for(const double angle:{.37,1.11})
          {
            const auto rotatedS=rotate(S,angle),e=rotate(increment,angle);
            const auto rr=TR::update(parameters,rotate(old,angle),e,phi,dt,restricted?&rotatedS:nullptr);
            const auto expected=rotate(r.state,angle);
            for(const auto &pair:{std::make_pair(rr.state.T-expected.T,1000.),std::make_pair(rr.state.Q-expected.Q,1000.),
                                 std::make_pair(rr.state.A-expected.A,.001),std::make_pair(rr.state.K-expected.K,.001),
                                 std::make_pair(rr.stress-rotate(r.stress,angle),1000.),std::make_pair(rr.flow_rate-rotate(r.flow_rate,angle),.001)})
              measure("passive_basis_covariance",pair.first.norm()/pair.second);
            measure("passive_basis_covariance",(rr.driving-r.driving)/8.);
            const auto d=TR::derivative(parameters,old,r,shear(.001,.0003),.02,dt,rule);
            const auto rd=TR::derivative(parameters,rotate(old,angle),rr,rotate(shear(.001,.0003),angle),.02,dt,restricted?&rotatedS:nullptr);
            measure("tangent_basis_covariance",(rd.A-rotate(d.A,angle)).norm()/.001);
            measure("tangent_basis_covariance",(rd.Q-rotate(d.Q,angle)).norm()/1000.);
            measure("tangent_basis_covariance",(rd.stress-rotate(d.stress,angle)).norm()/1000.);
            measure("tangent_basis_covariance",(rd.driving-d.driving)/8.);
          }
        const auto ledger=TR::account(parameters,old,r,increment,dt);
        measure("tensor_BE_ledger",ledger.error/std::max(1.,r.bulk_energy+r.cohesive_energy),1e-10);
        // Local continuous power, with the actual flow and arbitrary phase rate.
        const Tensor D=shear(.001,-.0005);const double phidot=.02;
        const auto Adot=r.flow_rate-parameters.G/parameters.eta*r.state.A;
        const auto taudot=2*parameters.G*D-parameters.G/parameters.eta*r.state.T-2*parameters.G*Adot;
        const double stored_rate=(r.stress*taudot)/(2*parameters.G)+(r.state.Q*Adot)-r.driving*phidot;
        if(!r.intact)measure("continuous_power",(stored_rate-(r.stress*D)+r.bulk_dissipation+r.cohesive_dissipation+r.friction_dissipation+r.driving*phidot)/std::max(1.,std::abs(stored_rate)));
      }
}

void intact_and_ties(const std::string &out)
{
  context="tiny positive phase, zero load";
  TR::State zero;
  const auto tiny_zero=TR::update(parameters,zero,Tensor(),1e-303,.1);
  require(tiny_zero.driving==0. && tiny_zero.state.Q.norm()==0. && tiny_zero.speed==0.,"Tiny positive-h zero state must stay zero.");
  for(const double x:pack(TR::derivative(parameters,zero,tiny_zero,Tensor(),.02,.1)))
    require(std::isfinite(x) && x==0.,"Tiny zero-state phase derivative must vanish.");
  context="tiny positive phase, representable nonzero flow";
  zero.T=shear(0.,4000.);
  const auto tiny_flow=TR::update(parameters,zero,Tensor(),1e-200,.1);
  require(tiny_flow.flow_rate[0][1]!=0. && tiny_flow.direction_defined(),"Representable tiny flow was lost in a squared norm.");
  measure("tiny_flow_magnitude",tiny_flow.speed/(2*std::abs(tiny_flow.flow_rate[0][1]))-1.);
  const auto d_tiny=TR::derivative(parameters,zero,tiny_flow,shear(0.,.001),0.,.1);
  require(d_tiny.speed>0.,"Tiny flow derivative lost its magnitude.");
  measure("tiny_flow_tangent",d_tiny.speed/(2*d_tiny.flow_rate[0][1])-1.);
  auto forbidden=zero;forbidden.A=shear(0.,1e-200);bool rejected=false;
  try{TR::update(parameters,forbidden,Tensor(),0.,.1);}catch(const std::exception &){rejected=true;}
  require(rejected,"A nonzero intact tensor was erased because its squared norm underflowed.");
  std::ofstream f(out+"/intact-threshold.csv");f<<"Txy_Pa,Y_Pa,A_norm,flow_norm\n"<<std::setprecision(17);
  TR::State old;
  for(const double stress:{-3001.,-3000.,-2999.,-1000.,0.,1000.,2999.,3000.,3001.})
    {
      old.T=shear(0.,stress);const auto r=TR::update(parameters,old,Tensor(),0.,.00125);
      const double tp=r.state.T[0][1],q=std::copysign(std::max(0.,std::abs(tp)-1000.),tp);
      measure("intact_limit",(r.driving-4*q*q/(2*parameters.G))/8.);
      require(r.state.A.norm()==0. && r.flow_rate.norm()==0. && !r.direction_defined(),"Intact flow/direction must vanish.");
      // Exact threshold in terms of ACTUAL post-Maxwell trial stress.
      old.T=shear(0.,stress/r.beta);const auto exact=TR::update(parameters,old,Tensor(),0.,.00125);
      f<<exact.state.T[0][1]<<','<<exact.driving<<','<<exact.state.A.norm()<<','<<exact.flow_rate.norm()<<'\n';
      if(std::abs(stress)==3000.)measure("pure_shear_threshold",(exact.driving-Kf)/Kf);
      if(std::abs(stress)==2999.)require(exact.driving<Kf,"Below threshold");
      if(std::abs(stress)==3001.)require(exact.driving>Kf,"Above threshold");
      for(const double phi:{1e-20,1e-12,1e-7})
        {const auto a=TR::update(parameters,old,Tensor(),phi,.00125);
         measure("intact_limit",(a.state.Q-exact.state.Q/(1+BC::compliance(phi,4.).h)).norm()/1000.);}
    }
  for(const double phi:{0.,.2})
    {
      old=TR::State();old.phi=phi;const double dt=.1,beta=1/(1+dt*.01);
      old.T=shear(0.,1000./beta);
      const auto tie=TR::update(parameters,old,Tensor(),phi,dt);
      require(!tie.sliding && !tie.direction_defined(),"Yield equality selects sticking.");
      const auto d=TR::derivative(parameters,old,tie,shear(0.,-.001),0.,dt);
      const double epsilon=1e-5;
      const auto inside=TR::update(parameters,old,shear(0.,-.001*epsilon),phi,dt);
      const auto outside=TR::update(parameters,old,shear(0.,.001*epsilon),phi,dt);
      measure("yield_inward_derivative",((inside.state.A-tie.state.A)/epsilon-d.A).norm()/.001,1e-6);
      require(!inside.sliding && outside.sliding,"Both sides of the stick/slip switch must be exercised.");
      const auto dp=TR::derivative(parameters,old,outside,shear(0.,.001),0.,dt);
      const auto further=TR::update(parameters,old,shear(0.,.002*epsilon),phi,dt);
      measure("yield_outward_derivative",((further.state.A-outside.state.A)/epsilon-dp.A).norm()/.001,1e-6);
    }
}

// Fixture-only transaction: the production native owner/format is untouched.
struct History
{
  TR::State state;
  unsigned int publications=0;
  double time=0.,work=0.,Db=0.,Dc=0.,Df=0.,numerical=0.;
  struct Trial{TR::Response response;BC::StepAccounting ledger;unsigned int parent;double dt;bool consumed=false;};
  Trial trial(const Tensor &e,const double phi,const double dt,const Tensor *S=nullptr)const
    {const auto r=TR::update(parameters,state,e,phi,dt,S);return {r,TR::account(parameters,state,r,e,dt),publications,dt,false};}
  void finish(Trial &t,const bool accept)
    {
      require(!t.consumed && t.parent==publications,"Duplicate/stale point publication.");t.consumed=true;
      if(!accept)return;
      state=t.response.state;++publications;time+=t.dt;work+=t.ledger.internal_work;
      Db+=t.dt*t.response.bulk_dissipation;Dc+=t.dt*t.response.cohesive_dissipation;Df+=t.dt*t.response.friction_dissipation;
      numerical+=t.ledger.stress_loss+t.ledger.traction_loss+t.ledger.compliance_work;
    }
  template<class Archive>void serialize(Archive &a,const unsigned int)
    {a & state & publications & time & work & Db & Dc & Df & numerical;}
  std::string save()const
    {std::ostringstream s;{boost::archive::binary_oarchive a(s);const std::string tag="P5a tensor point history v1";a<<tag<<*this;}return s.str();}
  static History restore(const std::string &record)
    {History h;std::istringstream s(record);boost::archive::binary_iarchive a(s);std::string tag;a>>tag;require(tag=="P5a tensor point history v1","Wrong point history format.");a>>h;return h;}
};

void populated_history()
{
  History h;h.state.phi=.2;h.state.A=shear(.0002,.0003);h.state.K=shear(-.0001,.0007);h.state.L=.003;
  h.state.Q=2*parameters.G/BC::compliance(.2,4.).h*h.state.A;
  h.state.T=shear(0.,2500.);h.state.T[0][0]=310.;h.state.T[1][1]=-110.;h.state.T[2][2]=-200.;
  h.publications=7;h.time=2.5;h.work=.12;h.Db=.01;h.Dc=.02;h.Df=.03;h.numerical=.004;
  const auto restored=History::restore(h.save());
  require(same_state(restored.state,h.state) && restored.save()==h.save(),"Populated restart lost a tensor component or metadata.");
  ++counts["fully_populated_records_checked"];
  const auto old=h.state;auto first=h.trial(shear(.0004,.0002),.21,.1);auto stale=h.trial(shear(.0004,.0002),.21,.1);
  require(same_state(h.state,old),"Stateless return modified incoming tensor history.");
  h.finish(first,true);const auto accepted=h.save();bool threw=false;
  try{h.finish(stale,true);}catch(const std::exception&){threw=true;}
  require(threw && h.save()==accepted,"Stale trial published over a new accepted state.");
  ++counts["stale_publication_rejected"];
  auto inconsistent=old;inconsistent.Q[0][1]+=1.;threw=false;
  try{TR::update(parameters,inconsistent,Tensor(),.2,.1);}catch(const std::exception&){threw=true;}
  require(threw,"Inconsistent accepted physical A/Q was silently accepted.");
  threw=false;try{TR::update(parameters,old,Tensor(),.19,.1);}catch(const std::exception&){threw=true;}
  require(threw,"Physical phase obstacle was ignored.");
}

void path(const std::string &out,const bool restricted)
{
  const auto S=original_direction();const Tensor *rule=restricted?&S:nullptr;
  const std::string name=restricted?"fixed":"radial";
  std::ofstream f(out+"/"+name+"-history.csv");f<<std::setprecision(17);
  f<<"step,time_s,dt_s,phi,engineering_xy,engineering_normal,Txx_Pa,Tyy_Pa,Tzz_Pa,Txy_Pa,tauxx_Pa,tauyy_Pa,tauzz_Pa,tauxy_Pa,Axx,Axy,Kxx,Kxy,L,Qxx_Pa,Qxy_Pa,Wxx_s_inv,Wxy_s_inv,speed_s_inv,direction_defined,Y_Pa,Eb_J_m3,Ec_J_m3,Db_W_m3,Dc_W_m3,Df_W_m3,work_J_m3,cumulative_Db_J_m3,cumulative_Dc_J_m3,cumulative_Df_J_m3,numerical_J_m3,ledger_J_m3,old_Axx,old_Axy,relaxed_Axx,relaxed_Axy,new_Axx,new_Axy\n";
  History h;h.state.phi=.2;BC::State scalar;scalar.phi=.2;
  History resumed;bool restarted=false;double gamma=0.,normal=0.;const double dt=.25;
  for(unsigned int step=1;step<=800;++step)
    {
      context=name+" step="+std::to_string(step);
      const double time=step*dt,rate_xy=time<=50.?1e-4:0.,rate_n=time>90. && time<=150.?1e-4:0.;
      const auto increment=shear(.5*dt*rate_n,.5*dt*rate_xy);
      gamma+=dt*rate_xy;normal+=dt*rate_n;
      const auto before=h.save();const auto old=h.state;
      auto t=h.trial(increment,.2,dt,rule);
      if(step==500)
        {
          auto repeated=h.trial(increment,.2,dt,rule);
          require(same_state(t.response.state,repeated.response.state),"Repeated point trials differ.");
          h.finish(t,false);require(h.save()==before && same_state(h.state,old),"Rejected point trial changed history.");
          t=h.trial(increment,.2,dt,rule);require(same_state(t.response.state,repeated.response.state),"Point retry differs.");
          h.finish(repeated,false);require(h.save()==before,"Second rejected trial changed history.");
          counts["discarded_trials"]+=2;
        }
      measure("path_BE_ledger",t.ledger.error/std::max(1.,t.response.bulk_energy+t.response.cohesive_energy),1e-10);
      require(t.response.bulk_dissipation>=0. && t.response.cohesive_dissipation>=0. && t.response.friction_dissipation>=0.,"Negative physical dissipation.");
      const auto &r=t.response;
      measure("memory_recurrence",(r.state.A-r.beta*old.A-r.beta*dt*r.flow_rate).norm()/.001);
      const auto effective=TR::in_plane(r.stress-r.state.Q);
      const double yielding=restricted?std::abs(effective*S):TR::magnitude(effective);
      if(!r.intact)measure("yield_admissibility",std::max(0.,yielding-parameters.resistance)/1000.);
      measure("friction_complementarity",((r.stress-r.state.Q)*r.flow_rate-r.friction_dissipation)/std::max(1.,r.friction_dissipation));
      h.finish(t,true);require(h.publications==step,"Point owner published more than once.");
      if(step==500)
        {const auto accepted=h.save();bool threw=false;try{h.finish(t,true);}catch(const std::exception&){threw=true;}
         require(threw && h.save()==accepted,"Duplicate publication not rejected atomically.");++counts["duplicate_publication_rejected"];}
      if(restricted){const auto sr=BC::update(parameters,scalar,increment,.2,dt);compare(r,sr,"fixed_path_identity");scalar=sr.state;}
      if(restarted)
        {auto z=resumed.trial(increment,.2,dt,rule);resumed.finish(z,true);require(resumed.save()==h.save() && same_state(resumed.state,h.state),"Restarted tensor owner differs.");++counts[name+"_restart_states_compared"];}
      if(step==480)
        {const auto b=h.save();const std::string filename=out+"/"+name+"-point.restart";
         {std::ofstream binary(filename,std::ios::binary);binary.write(b.data(),b.size());}
         std::ifstream binary(filename,std::ios::binary);const std::string data((std::istreambuf_iterator<char>(binary)),std::istreambuf_iterator<char>());
         resumed=History::restore(data);require(resumed.save()==b && same_state(resumed.state,h.state),"Point file restart changed a history.");restarted=true;}
      if((time>50. && time<=90.) || time>150.)
        {
          require(!r.direction_defined(),"Hold unexpectedly active in the selected point path.");
          measure("hold_Maxwell_memory",(r.state.A-r.beta*old.A).norm()/.001);
          require(r.state.K==old.K && r.state.L==old.L,"No-flow hold changed accumulated history.");
        }
      f<<step<<','<<time<<','<<dt<<",0.2,"<<gamma<<','<<normal;
      for(const auto *v:{&r.state.T,&r.stress})f<<','<<(*v)[0][0]<<','<<(*v)[1][1]<<','<<(*v)[2][2]<<','<<(*v)[0][1];
      for(double v:{r.state.A[0][0],r.state.A[0][1],r.state.K[0][0],r.state.K[0][1],r.state.L,r.state.Q[0][0],r.state.Q[0][1],
                    r.flow_rate[0][0],r.flow_rate[0][1],r.speed,static_cast<double>(r.direction_defined()),r.driving,r.bulk_energy,r.cohesive_energy,
                    r.bulk_dissipation,r.cohesive_dissipation,r.friction_dissipation,h.work,h.Db,h.Dc,h.Df,h.numerical,t.ledger.error,
                    old.A[0][0],old.A[0][1],r.beta*old.A[0][0],r.beta*old.A[0][1],r.beta*dt*r.flow_rate[0][0],r.beta*dt*r.flow_rate[0][1]})f<<','<<v;
      f<<'\n';
    }
  counts[name+"_publications"]=h.publications;
  require(restarted && h.publications==800,"Incomplete point fixture.");
}

int main()
{
  dealii::deal_II_exceptions::disable_abort_on_exception();
  const char *out=std::getenv("ASPECT_P5A_OUTPUT");if(!out){std::cerr<<"Set ASPECT_P5A_OUTPUT to an existing directory.\n";return 2;}
  const auto start=std::chrono::steady_clock::now();
  try
    {
      fixed_identity();derivative_checks(out);intact_and_ties(out);populated_history();path(out,false);path(out,true);
      std::ofstream f(std::string(out)+"/native-checks.json");f<<std::setprecision(17)<<"{\"passed\":true,\"defects\":{";
      bool first=true;for(const auto &entry:defects){if(!first)f<<',';first=false;f<<'\"'<<entry.first<<"\":"<<entry.second;}
      f<<"},\"counts\":{";first=true;
      for(const auto &entry:counts){if(!first)f<<',';first=false;f<<'\"'<<entry.first<<"\":"<<entry.second;}
      f<<"},\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
      std::cout<<"P5a point checks and both history paths passed.\n";
    }
  catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
