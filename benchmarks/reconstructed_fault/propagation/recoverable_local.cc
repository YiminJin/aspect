// Native state-history tests and independent evolving-phase energy derivatives.
#include "../../../tests/phase_field_fault_test_access.h"
#include <aspect/phase_field.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <fstream>
#include <array>
#include <iomanip>

namespace aspect { namespace Postprocess {
template<int dim> class RecoverableLocal : public Interface<dim>, public SimulatorAccess<dim> {
public:
 std::pair<std::string,std::string> execute(TableHandler &) override {
  using F=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
  AssertThrow(this->get_timestep_number()==0,ExcMessage("Local energy experiment only runs at initialization."));
  auto &phase=this->get_phase_field_handler();
  const auto g=[&](double phi){return phase.energetic_degradation({.5,.5},phi);};
  const double G=1e6,phi=.6,gg=g(phi),h=1/gg-1,q0=600;
  const double H0=F::recoverable_phase_driving_force(G,gg,q0);
  std::ofstream out(this->get_output_directory()+"recoverable-local.csv");
  out << std::setprecision(17) << "eta,dt,path,step,time,q,Y,H,H0\n";
  unsigned int count=0;
  for(double eta:{1e8,1e22}) for(double dt:{.125,.0625,.03125})
   for(const std::string path:{"loading","relaxation","cycle","hold"}) {
    double H=H0;
    const unsigned int n=std::lround((path=="cycle"?2.:1.)/dt);
    auto traction=[&](double t) {
     if(path=="loading") return 600+200*t;
     if(path=="relaxation") return 600*std::exp(-G*t/eta);
     if(path=="cycle") return 600-200*std::pow(std::sin(numbers::PI*t),2);
     return 600.;
    };
    const auto c=F::maxwell_coefficients(eta,G,dt);
    for(unsigned int k=1;k<=n;++k) {
     const double t=k*dt,q=traction(t),Y=F::recoverable_phase_driving_force(G,gg,q);
     H=std::max(H,Y);
     const double expected=path=="loading"?q*q/(2*G*gg*gg):H0;
     AssertThrow(std::abs(H-expected)<=1e-13*H0,ExcMessage("Recoverable peak path failed."));
     if(path=="relaxation") {
      const auto r=F::compute_cohesive_response(c.first,c.second,100,100,traction(t-dt),0,h,h);
      AssertThrow(std::abs(r.cohesive_traction-q)<1e-12*q0,ExcMessage("Native relaxation failed."));
     }
     AssertThrow(F::recoverable_phase_driving_force(G,1.,q)==q*q/(2*G),ExcMessage("Intact energy limit failed."));
     out<<eta<<','<<dt<<','<<path<<','<<k<<','<<t<<','<<q<<','<<Y<<','<<H<<','<<H0<<'\n';++count;
    }
   }
  // Three positive transverse weights, nonuniform evolving phi, fixed material.
  // Derivative is independently taken from stored energy, holding Z/G fixed.
  std::ofstream energy(this->get_output_directory()+"recoverable-local-energy.csv");
  energy << std::setprecision(17) << "eta,t,I,Z,q,stored,dissipation,power,phase_power,dE_dt,identity_error,phase_derivative_error\n";
  const double w[3]={.2,.5,.3},eps=1e-6;
  for(double eta:{1e8,1e22}) for(unsigned int k=0;k<=20;++k) {
   const double t=k/20.;
   auto phases=[](double time){return std::array<double,3>{{.1+.2*time,.6+.05*std::sin(numbers::PI*time),.3+.04*time}};};
   auto integral=[&](const std::array<double,3>&p){double I=0;for(unsigned int j=0;j<3;++j)I+=w[j]*(1/g(p[j])-1);return I;};
   const auto p=phases(t);const double I=integral(p),Z=G*(.01+.002*t),q=Z/I;
   const double stored=Z*Z/(2*G*I),D=I*q*q/eta,V=.002+Z/eta;
   double derivative_error=0;
   for(unsigned int j=0;j<3;++j) {
    auto plus=p,minus=p;plus[j]+=eps;minus[j]-=eps;
    const double fd=(Z*Z/(2*G*integral(plus))-Z*Z/(2*G*integral(minus)))/(2*eps);
    const double gp=(g(p[j]+eps)-g(p[j]-eps))/(2*eps);
    const double analytic=w[j]*gp*F::recoverable_phase_driving_force(G,g(p[j]),q);
    derivative_error=std::max(derivative_error,std::abs(fd-analytic)/std::max(1.,std::abs(analytic)));
   }
   const double Idot=(integral(phases(t+eps))-integral(phases(t-eps)))/(2*eps);
   const double phase_power=-q*q/(2*G)*Idot;
   const auto E=[&](double time){double z=G*(.01+.002*time);return z*z/(2*G*integral(phases(time)));};
   const double dE=(E(t+eps)-E(t-eps))/(2*eps);
   const double error=dE-(q*V-D+phase_power);
   AssertThrow(derivative_error<1e-7 && std::abs(error)<1e-8*std::max(1.,std::abs(q*V)),ExcMessage("Evolving-phase recoverable energy identity failed."));
   energy<<eta<<','<<t<<','<<I<<','<<Z<<','<<q<<','<<stored<<','<<D<<','<<q*V<<','<<phase_power<<','<<dE<<','<<error<<','<<derivative_error<<'\n';
  }
  this->get_pcout()<<"Recoverable native paths: "<<count<<" checks; 42 evolving-phase energy states passed."<<std::endl;
  return {"Recoverable local model:","verified"};
 }
};
ASPECT_REGISTER_POSTPROCESSOR(RecoverableLocal,"P2 recoverable local","Native recoverable-state and independent energy checks.")
}}
