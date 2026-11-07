/* Benchmark-only equilibrated boundary reference, GNU GPL v2 or later. */
#include "background.h"
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/simulator_signals.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    template<int dim> class LayeredBackground : public Interface<dim>, public SimulatorAccess<dim>
    {
        LayeredReference::Background reference;
        LayeredReference::Field accepted, candidate;
        std::vector<double> primitive;
        double translation=0., accepted_time=0., candidate_time=0., candidate_dt=.1, traction=0.;
        unsigned accepted_step=0, candidate_step=0, retries=0;
        bool initialized=false, pending=false;
        std::string identity;
        unsigned long long calls=0;

        void prepare()
        {
          const unsigned step=this->get_timestep_number();
          const double dt=step ? this->get_timestep() : .1;
          AssertThrow(!step || (initialized && step==accepted_step+1 &&
                                std::abs(this->get_time()-dt-accepted_time)<1e-11),
                      ExcMessage("Layered boundary parent does not match the attempted native step."));
          if (pending && candidate_step==step)++retries;
          candidate=reference.advance(accepted,dt,traction);
          primitive=reference.primitive(candidate.gamma);
          candidate_step=step;
          candidate_time=this->get_time();
          candidate_dt=dt;
          pending=true;
          AssertThrow(std::abs(reference.integral(candidate.gamma)-2.)<1e-12,
                      ExcMessage("Layered reference violates prescribed velocity difference."));
        }
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Layered background");
          prm.declare_entry("Translation","0",Patterns::Double());
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          AssertThrow(dim==2 && prm.get("Pressure normalization")=="no" && prm.get_double("Surface pressure")==0.,
                      ExcMessage("Layered benchmark requires 2D, natural pressure reference zero."));
        prm.enter_subsection("Postprocess");
        AssertThrow(!prm.get_bool("Run postprocessors on nonlinear iterations"),
                      ExcMessage("Background publication requires accepted-step postprocessing."));
        prm.leave_subsection();
          prm.enter_subsection("Geometry model");
          AssertThrow(prm.get("Model name")=="box",ExcMessage("Layered reference requires its Cartesian box."));
          prm.enter_subsection("Box");
          AssertThrow(prm.get_double("Y extent")==4. && prm.get_double("Box origin Y coordinate")==-2.,
                      ExcMessage("Layered reference requires depth interval [-2,2]."));
          prm.leave_subsection();
          prm.leave_subsection();
          prm.enter_subsection("Material model");
          prm.enter_subsection("Phase field fault");
          AssertThrow(prm.get_double("Reference viscosities")==10. && prm.get_double("Elastic shear moduli")==10.
                      && prm.get_double("Initial time step")==.1,ExcMessage("Layered benchmark coefficient/initialization mismatch."));
          prm.enter_subsection("Material viscosity multiplier");
          AssertThrow(prm.get("Function expression")==".55+.45*tanh(y/.35)",ExcMessage("Layered benchmark requires its declared eta(y)."));
          prm.leave_subsection();
          prm.leave_subsection();
          prm.leave_subsection();
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Layered background");
          translation=prm.get_double("Translation");
          prm.leave_subsection();
          prm.leave_subsection();
          std::ostringstream id;
          id<<std::setprecision(17)<<"layered-Maxwell-v1:G=10:eta=1..10:width=.35:du=2:init=.1:pref=0:panels=64:U="<<translation;
          identity=id.str();
        }
        void initialize() override
        {
          accepted=LayeredReference::Field(reference.y.size());
          // Initial mechanics uses .1 s, but its stress is never published.
          candidate=reference.advance(accepted,.1,traction);
          primitive=reference.primitive(candidate.gamma);
          this->get_signals().start_timestep.connect([this](const auto &)
          {
            prepare();
          });
          auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(
                        Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()));
          model.set_material_exterior_stress(identity,[this](const Point<dim> &x,double parent,double stage,double dt)
          {
            AssertThrow(initialized && pending && std::abs(parent-accepted_time)<1e-11
                        && std::abs(dt-candidate_dt)<1e-12 && stage>=parent-1e-11 && stage<=parent+dt+1e-11,
                        ExcMessage("Layered exterior callback is not using the accepted transport parent."));
            ++calls;
            SymmetricTensor<2,dim> tau;
            tau[0][0]=reference.value(accepted.a,x[1]);
            tau[1][1]=-tau[0][0];
            tau[0][1]=reference.value(accepted.xy,x[1]);
            return tau;
          });
        }
        double horizontal_velocity(double y) const
        {
          return reference.velocity(candidate,primitive,y,translation);
        }
        double shear_traction() const
        {
          return traction;
        }
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(pending && candidate_step==this->get_timestep_number(),ExcMessage("No matching layered boundary candidate to publish."));
          const auto total_calls=Utilities::MPI::sum(calls,this->get_mpi_communicator());
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
            {
              const auto path=this->get_output_directory()+"layered-background.csv";
              std::ifstream old(path);
              const bool header=old.peek()==std::ifstream::traits_type::eof();
              std::ofstream out(path,std::ios::app);
              if (header)out<<"step,time,dt,parent_time,q,velocity_difference,retries,inflow_calls\n";
              out<<std::setprecision(17)<<candidate_step<<','<<candidate_time<<','<<candidate_dt<<','<<accepted_time<<','<<traction<<','<<reference.integral(candidate.gamma)<<','<<retries<<','<<total_calls<<'\n';
              std::ofstream field(this->get_output_directory()+"background-"+std::to_string(candidate_step)+".csv");
              field<<"y,eta,parent_xx,parent_xy,xx,xy,gamma,vx,pressure\n"<<std::setprecision(17);
              for (unsigned i=0; i<reference.y.size(); ++i)
                field<<reference.y[i]<<','<<reference.eta[i]<<','<<accepted.a[i]<<','<<accepted.xy[i]<<','<<candidate.a[i]<<','<<candidate.xy[i]<<','<<candidate.gamma[i]<<','<<horizontal_velocity(reference.y[i])<<','<<-candidate.a[i]<<'\n';
            }
          // execute is called only after the ordinary controller accepts and native
          // histories publish. Initialization retains zero stress but accepted gamma.
          if (candidate_step)accepted=candidate;
          else accepted.gamma=candidate.gamma;
          accepted_step=candidate_step;
          accepted_time=candidate_time;
          initialized=true;
          pending=false;
          calls=0;
          return {"Layered reference","accepted parent published"};
        }
        void save(std::map<std::string,std::string> &status) const override
        {
          AssertThrow(initialized && !pending,ExcMessage("Cannot checkpoint a pending layered boundary trial."));
          std::ostringstream out;
          out<<identity<<'\n'<<std::setprecision(17)<<accepted_step<<' '<<accepted_time<<' '<<retries<<'\n';
          for (unsigned i=0; i<accepted.a.size(); ++i)out<<accepted.a[i]<<' '<<accepted.xy[i]<<' '<<accepted.gamma[i]<<'\n';
          status["Layered benchmark background v1"]=out.str();
        }
        void load(const std::map<std::string,std::string> &status) override
        {
          auto p=status.find("Layered benchmark background v1");
          AssertThrow(p!=status.end(),ExcMessage("Missing layered background checkpoint; conversion is not supported."));
          std::istringstream in(p->second);
          std::string stored;
          std::getline(in,stored);
          AssertThrow(stored==identity,ExcMessage("Layered background checkpoint identity mismatch."));
          in>>accepted_step>>accepted_time>>retries;
          for (unsigned i=0; i<accepted.a.size(); ++i)in>>accepted.a[i]>>accepted.xy[i]>>accepted.gamma[i];
          AssertThrow(bool(in),ExcMessage("Incomplete layered background checkpoint."));
          initialized=true;
          pending=false;
          calls=0;
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(LayeredBackground,"layered background","Accepted-time benchmark exterior, equilibrium velocity and top traction.")
  }
  namespace BoundaryVelocity
  {
    template<int dim> class LayeredVelocity : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x) const override
        {
          Tensor<1,dim> v;
          v[0]=this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::LayeredBackground<dim>>().horizontal_velocity(x[1]);
          return v;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(LayeredVelocity,"layered background","Equilibrated split-background boundary velocity; interior mechanics stays solved.")
  }
  namespace BoundaryTraction
  {
    template<int dim> class LayeredTraction : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_traction(const types::boundary_id,const Point<dim> &,const Tensor<1,dim> &) const override
        {
          Tensor<1,dim> t;
          t[0]=this->get_postprocess_manager().template get_matching_active_plugin<Postprocess::LayeredBackground<dim>>().shear_traction();
          return t;
        }
    };
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(LayeredTraction,"layered background","Constant equilibrium shear traction and zero normal total traction, p=tau_yy.")
  }
}
