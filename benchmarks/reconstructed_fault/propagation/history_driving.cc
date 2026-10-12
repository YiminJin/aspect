// Isolated native-function experiment. No simulator histories are written.
#include "../../../tests/phase_field_fault_test_access.h"
#include <aspect/phase_field.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <fstream>
#include <iomanip>

namespace aspect {
namespace Postprocess {
template <int dim>
class HistoryDriving : public Interface<dim>, public SimulatorAccess<dim> {
public:
  std::pair<std::string,std::string> execute(TableHandler &) override {
    using F = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
    AssertThrow(this->get_timestep_number()==0, ExcMessage("Isolated history test requires end time zero."));
    auto &phase = this->get_phase_field_handler();
    const std::vector<double> fractions{.5,.5};
    const double phi=.6, G=1e6, g=phase.energetic_degradation(fractions,phi);
    const double h=1/g-1, q0=600, H0=q0*q0/(2*G*g*g);
    std::ofstream out(this->get_output_directory()+"history-native.csv");
    out << std::setprecision(17)
        << "eta,dt,path,step,time,phi,g,h,beta,kappa,q_old,q,candidate,H_max,H0,state_energy,signed_sum,positive_sum,intact_candidate,near_intact_candidate,potential_derivative\n";
    unsigned int checks=0;
    // eta=1e22 realizes the elastic limit to double precision in beta.
    for (const double eta : {1e8,1e22})
      for (const double dt : {.125,.0625,.03125})
        for (const std::string path : {"loading","relaxation","cycle","hold"}) {
          const auto c=F::maxwell_coefficients(eta,G,dt);
          const double beta=c.first, kappa=c.second;
          const auto traction=[&](const double t) {
            if (path=="loading") return q0+200*t;
            if (path=="relaxation") return q0*std::exp(-G*t/eta);
            if (path=="cycle") return q0-200*std::pow(std::sin(numbers::PI*t),2);
            return q0;
          };
          double H=H0, sum=0, positive_sum=0;
          const unsigned int n=std::lround((path=="cycle" ? 2. : 1.)/dt);
          for (unsigned int k=1;k<=n;++k) {
            const double t=k*dt, oldq=traction(t-dt), q=traction(t);
            const double C=F::compute_crack_driving_force_candidate(dt,beta,kappa,g,h,q,oldq);
            H=std::max(H,C); sum+=C; positive_sum+=std::max(0.,C);
            const double energy=q*q/(2*G*g*g);
            // Independent derivative of the documented one-step potential,
            // holding upsilon and previous h*q fixed while varying current phi.
            const double upsilon=h*(q-beta*oldq)/kappa;
            const double bracket=.5*kappa*upsilon*upsilon+beta*h*oldq*upsilon;
            const double eps=1e-6;
            const double gp=phase.energetic_degradation(fractions,phi+eps);
            const double gm=phase.energetic_degradation(fractions,phi-eps);
            const double derivative=dt*bracket*(1/(1/gp-1)-1/(1/gm-1))/(gp-gm);
            AssertThrow(std::abs(derivative-C)<2e-7*std::max(1.,std::abs(C)),
                        ExcMessage("Native candidate differs from incremental potential derivative."));
            if (path=="relaxation") {
              const auto response=F::compute_cohesive_response(beta,kappa,100.,100.,oldq,0.,h,h);
              AssertThrow(std::abs(response.cohesive_traction-q)<1e-12*q0
                          && std::abs(C)<1e-9*H0,
                          ExcMessage("Zero-input native Maxwell relaxation mismatch."));
            }
            if (eta==1e22)
              AssertThrow(std::abs(C-(q*q-oldq*oldq)/(2*G*g*g))<1e-12*H0,
                          ExcMessage("Native elastic candidate is not the energy increment."));
            const double intact=F::compute_crack_driving_force_candidate(dt,beta,kappa,1.,0.,q,oldq);
            const double near_g=phase.energetic_degradation(fractions,1e-10);
            const double near=F::compute_crack_driving_force_candidate(dt,beta,kappa,near_g,1/near_g-1,q,oldq);
            out << eta << ',' << dt << ',' << path << ',' << k << ',' << t << ',' << phi << ','
                << g << ',' << h << ',' << beta << ',' << kappa << ',' << oldq << ',' << q << ','
                << C << ',' << H << ',' << H0 << ',' << energy << ',' << sum << ',' << positive_sum
                << ',' << intact << ',' << near << ',' << derivative << '\n';
            ++checks;
          }
        }
    this->get_pcout() << "Native history path checks: " << checks << " passed; no physical histories advanced." << std::endl;
    return {"Native history semantics:","incremental potential verified; feedback qualification unresolved"};
  }
};
ASPECT_REGISTER_POSTPROCESSOR(HistoryDriving, "P2 history driving",
                             "Read-only native prescribed-traction history experiment.")
}}
