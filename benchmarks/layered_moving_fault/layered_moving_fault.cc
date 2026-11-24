/* Benchmark-only equilibrated boundary reference, GNU GPL v2 or later. */
#include "background.h"
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/simulator_signals.h>
#include <aspect/simulator/assemblers/fault_stress.h>
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
        mutable unsigned long long calls=0;

        void prepare()
        {
          const unsigned step=this->get_timestep_number();
          const double dt=step ? this->get_timestep() : .1;
          AssertThrow(!step || (initialized && step==accepted_step+1 &&
                                std::abs(this->get_time()-dt-accepted_time)<1e-11),
                      ExcMessage("Layered boundary parent does not match the attempted native step."));
          if (pending && candidate_step==step)++retries;
          calls=0;
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
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Layered background");
          translation=prm.get_double("Translation");
          prm.leave_subsection();
          prm.leave_subsection();
          prm.enter_subsection("Material model");
          prm.enter_subsection("Phase field fault");
          AssertThrow(prm.get_double("Shear modulus")==10. && prm.get_double("Initialization interval")==.1,
                      ExcMessage("The benchmark column requires G=10 Pa and a 0.1 s initialization."));
          prm.leave_subsection();
          prm.enter_subsection("Layered mature fault");
          reference=LayeredReference::Background(64,prm.get_bool("Homogeneous"));
          prm.leave_subsection();
          prm.leave_subsection();
          std::ostringstream id;
          id<<std::setprecision(17)<<"layered-Maxwell-v1:G=10:eta=1..10:width=.35:du=2:init=.1:pref=0:panels=64:U="<<translation<<":constant="<<reference.homogeneous;
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
        }
        double parent_component(double y,double stage,unsigned int component) const
        {
          AssertThrow(initialized && pending && stage>=accepted_time-1.e-11 && stage<=candidate_time+1.e-11,
                      ExcMessage("Layered exterior requires the accepted column at the current RK stage."));
          ++calls;
          return component==2 ? reference.value(accepted.xy,y) :
                 (component==0 ? 1. : -1.)*reference.value(accepted.a,y);
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
  namespace MaterialModel
  {
    // Isolate one ordinary assembled RHS before the velocity constraints act.
    template <int dim> class ColumnLoad : public Assemblers::FaultStressLoad<dim>
    {
      public:
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                     internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override
        {
          const auto &scratch=dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&>(scratch_base);
          const auto lo=scratch.cell->vertex(0);
          if (lo[0]!=0. || lo[1]!=0.)
            return Assemblers::FaultStressLoad<dim>::execute(scratch_base,data_base);
          auto &data=dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(data_base);
          internal::Assembly::CopyData::StokesSystem<dim> isolated(data.local_rhs.size(),false);
          Assemblers::FaultStressLoad<dim>::execute(scratch_base,isolated);
          data.local_rhs+=isolated.local_rhs;
          const auto &intro=this->introspection();
          const auto &fe=this->get_fe();
          double work=0.;
          for (unsigned int i=0; i<data.local_rhs.size(); ++i)
            {
              const auto local=intro.stokes_dof_info[i].local_dof_index;
              const auto component=fe.system_to_component_index(local).first;
              if (component!=intro.component_indices.velocities[0] && component!=intro.component_indices.velocities[1]) continue;
              const auto base=fe.system_to_base_index(local);
              const auto x=this->get_mapping().transform_unit_to_real_cell(scratch.cell,fe.base_element(base.first.first).unit_support_point(base.second));
              work+=isolated.local_rhs(i)*(component==intro.component_indices.velocities[0] ? x[1]*(1.+x[0]) : .4*x[0]);
            }
          std::ofstream out(this->get_output_directory()+"load-"+std::to_string(this->get_timestep_number())+".txt");
          out<<std::setprecision(17)<<scratch.cell->id()<<' '<<work<<'\n';
        }
    };

    template <int dim>
    class LayeredFault : public PhaseFieldFault<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Material model");
          prm.enter_subsection("Layered mature fault");
          prm.declare_entry("Homogeneous","false",Patterns::Bool(),"Use the 10 Pa s constant limit of the benchmark.");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void initialize() override
        {
          PhaseFieldFault<dim>::initialize();
          this->get_signals().set_assemblers.connect([](const SimulatorAccess<dim> &,Assemblers::Manager<dim> &assemblers)
          {
            for (auto &assembler:assemblers.stokes_system)
              if (dynamic_cast<Assemblers::FaultStressLoad<dim>*>(assembler.get()))
                {
                  assembler=std::make_unique<ColumnLoad<dim>>();
                  return;
                }
            AssertThrow(false,ExcMessage("Missing mature fault Stokes load."));
          });
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          PhaseFieldFault<dim>::parse_parameters(prm);
          prm.enter_subsection("Material model");
          prm.enter_subsection("Layered mature fault");
          homogeneous=prm.get_bool("Homogeneous");
          const auto averaging=this->get_parameters().material_averaging;
          AssertThrow(homogeneous || averaging==MaterialAveraging::none || averaging==MaterialAveraging::default_averaging,
                      ExcMessage("Layered fault mechanics requires pointwise Stokes viscosity without material averaging."));
          prm.leave_subsection();
          prm.leave_subsection();
        }
        double bulk_viscosity(const Point<dim> &x) const override
        {
          return homogeneous ? 10. : 5.5+4.5*std::tanh(x[1]/.35);
        }
      private:
        bool homogeneous=false;
        class Exterior : public Function<dim>
        {
          public:
            explicit Exterior(const LayeredFault &material): Function<dim>(3), material(material) {}
            double value(const Point<dim> &x, const unsigned int component) const override
            {
              return material.get_postprocess_manager().template get_matching_active_plugin<Postprocess::LayeredBackground<dim>>()
                     .parent_component(x[1],this->get_time(),component);
            }
          private:
            const LayeredFault &material;
        };
        Exterior exterior {*this};
        Function<dim> &exterior_stress() override
        {
          return exterior;
        }
    };
    ASPECT_REGISTER_MATERIAL_MODEL(LayeredFault,"layered mature fault","Benchmark Maxwell viscosity sampled at the Eulerian constitutive point.")
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


namespace aspect
{
  namespace Postprocess
  {
    template <int dim> class MovingFaultSamples : public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &material=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
          const auto layered=dynamic_cast<const MaterialModel::LayeredFault<dim>*>(&material);
          const auto viscosity=[&](const Point<dim> &x)
          {
            return layered ? layered->bulk_viscosity(x) : 10.;
          };
          const auto stem=this->get_output_directory()+"material-"+std::to_string(this->get_timestep_number());
          const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          std::ofstream surface(stem+"-surface-"+std::to_string(rank)+".csv");
          surface<<std::setprecision(17)<<"i,x,y,Theta_s,Z_Pa_m,I_m,V_m_s,width_m,eta,slip\n";
          const auto &labels=material.labels();
          for (unsigned int i=0; i<labels.size(); ++i)
            {
              const auto &p=labels[i];
              surface<<i<<','<<p.position[0]<<','<<p.position[1]<<','<<p.age<<','<<p.memory<<','
                     <<material.phase_profile().normal_integrals()[i]<<','<<p.rate<<','<<p.half_width<<','
                     <<viscosity(p.position)<<','<<p.slip<<'\n';
            }
          std::ofstream fields(stem+"-rank"+std::to_string(rank)+".csv");
          fields<<std::setprecision(17)<<"cell,x,y,q,phi,vx,vy,tauxx,tauyy,tauxy,Lxx,Lxy,Lyx,Lyy,eta,pressure,chi,station,offset_xx,offset_yy,offset_xy\n";
          const QGauss<dim> rule(3);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_gradients|update_quadrature_points);
          std::vector<Tensor<1,dim>> velocity(rule.size());
          std::vector<Tensor<2,dim>> grad(rule.size());
          std::vector<double> pressure(rule.size());
          std::array<std::vector<double>,4> composition;
          const auto &ex=this->introspection().extractors;
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                const auto &sol=this->get_solution();
                fe[ex.velocities].get_function_values(sol,velocity);
                fe[ex.velocities].get_function_gradients(sol,grad);
                fe[ex.pressure].get_function_values(sol,pressure);
                for (unsigned int c=0; c<4; ++c)
                  {
                    composition[c].resize(rule.size());
                    fe[ex.compositional_fields[c]].get_function_values(sol,composition[c]);
                  }
                for (unsigned int q=0; q<rule.size(); ++q)
                  {
                    const auto &x=fe.quadrature_point(q);
                    const auto transfer=material.phase_profile().sample(x,composition[3][q]);
                    const auto &seg=material.phase_profile().geometry.segment(transfer.foot.left);
                    const double eta=viscosity(x);
                    const double dt=this->get_timestep_number() ? this->get_timestep() : .1;
                    const double k=-eta*std::expm1(-10.*dt/eta);
                    const auto offset=material.endpoint_stress(cell,x,composition[3][q],symmetrize(grad[q]))-
                                      2.*k*Utilities::Tensors::consistent_deviator(symmetrize(grad[q]));
                    fields<<cell->id()<<','<<x[0]<<','<<x[1]<<','<<q<<','<<composition[3][q]<<','<<velocity[q][0]<<','<<velocity[q][1];
                    for (unsigned int c=0; c<3; ++c) fields<<','<<composition[c][q];
                    fields<<','<<grad[q][0][0]<<','<<grad[q][0][1]<<','<<grad[q][1][0]<<','<<grad[q][1][1]
                          <<','<<eta<<','<<pressure[q]<<','<<transfer.chi<<','<<seg.station+transfer.foot.xi *seg.length
                          <<','<<offset[0][0]<<','<<offset[1][1]<<','<<offset[0][1]<<'\n';
                  }
              }
          return {"Material samples","accepted endpoint"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MovingFaultSamples,"moving fault samples","Physical Q3 observations and persistent labels at joint publication.")
  }
}
