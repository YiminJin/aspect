#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/generator/reference_cell.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <fstream>
#include <iomanip>
#include <cstring>
#include "../../bp3/plugin/runtime.h"
namespace aspect
{
  namespace Particle { namespace Generator {
    template<int dim> class CrossingReference : public ReferenceCell<dim>
    {
      double gap=1e-8;
      bool staggered=false;
      public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Generator");prm.enter_subsection("Crossing reference cell");
        prm.declare_entry("Staggered","false",Patterns::Bool(),"Cross two rows at step one and the third at step two.");
        prm.declare_entry("Gap","1e-8",Patterns::Double(0),"Initial distance below the internal face, in metres.");
        prm.leave_subsection();prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        ReferenceCell<dim>::parse_parameters(prm);
        prm.enter_subsection("Generator");prm.enter_subsection("Crossing reference cell");
        gap=prm.get_double("Gap");staggered=prm.get_bool("Staggered");prm.leave_subsection();prm.leave_subsection();
      }
      void generate_particles(Particles::ParticleHandler<dim> &ph) override
      {
        ReferenceCell<dim>::generate_particles(ph);
        for(auto &p:ph)
          if(p.get_location()[1]<1500.)
          {
            auto x=p.get_location();x[0]+=100.*(std::floor(x[1]/500.)-1.);x[1]=2000.-(staggered && x[1]<1000. ? 1e-8 : gap);p.set_location(x);
            auto cell=p.get_surrounding_cell();
            p.set_reference_location(this->get_mapping().transform_real_to_unit_cell(cell,x));
          }
        // Histories have not been initialized yet: native initialization uses
        // these actual positions. Only this separately labeled lifecycle case moves them.
      }
    };
    ASPECT_REGISTER_PARTICLE_GENERATOR(CrossingReference,"crossing reference cell","Test-only pre-initialization near-boundary arrangement.")
  }}
  namespace Postprocess
  {
    template<int dim> class CoupledReplenishment : public Interface<dim>,public SimulatorAccess<dim>
    {
      unsigned int reject_step=0;
      bool rejected=false, migrate=false;
      double resume_dt=0.;
      std::string corrupt_audit;
      types::particle_index previous_next=0;
      std::set<types::particle_index> previous;
      unsigned int observations=0;
      static void hash_double(uint64_t &h,const double x)
      {uint64_t bits;std::memcpy(&bits,&x,8);h^=bits;h*=1099511628211ull;}
      void capture(const std::string &stage)
      {
        auto &pm=this->get_particle_manager(0);auto &ph=pm.get_particle_handler();
        const auto &info=pm.get_property_manager().get_data_info();unsigned int s=info.get_position_by_field_name("maxwell stress");
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        std::string path=this->get_output_directory()+"lifecycle_rank"+std::to_string(rank)+".csv";
        bool header=!std::ifstream(path).good();std::ofstream out(path,std::ios::app);out<<std::setprecision(17);
        if(header)out<<"stage,step,time,dt,n,next_id,born_or_received,lost_or_sent,position_hash,property_hash,integrator_hash,stress_max,stress_mean,pressure_min,pressure_max,mapped_stress_max,H_min,H_negative,rng_hash,membership_hash,audit_size,audit_hash,partition_hash,new_ids,received_ids\n";
        std::map<types::particle_index,std::vector<double>> data;
        std::set<types::particle_index> ids;
        for(const auto &p:ph){ids.insert(p.get_id());auto &a=data[p.get_id()];for(unsigned int d=0;d<dim;++d)a.push_back(p.get_location()[d]);for(double x:p.get_properties())a.push_back(x);}
        unsigned int fresh=0;for(auto id:ids)fresh+=!previous.count(id) && id>=previous_next;
        unsigned int born=0,lost=0;for(auto id:ids)born+=!previous.count(id);for(auto id:previous)lost+=!ids.count(id);
        uint64_t hp=1469598103934665603ull,hq=hp,hi=hp;double maxstress=0,sum=0;
        const auto internal=info.get_position_by_field_name("internal: integrator properties");
        for(auto &p:data){hp^=p.first;hq^=p.first;hi^=p.first;
          for(unsigned int d=0;d<dim;++d)hash_double(hp,p.second[d]);
          for(unsigned int d=0;d<info.n_components();++d)hash_double(d>=internal?hi:hq,p.second[dim+d]);
          for(unsigned int d=0;d<3;++d)maxstress=std::max(maxstress,std::abs(p.second[dim+s+d]));
          sum+=p.second[dim+s];}
        double Hmin=std::numeric_limits<double>::max();unsigned int Hnegative=0;const auto Hindex=info.get_position_by_field_name("crack_driving_force");
        for(const auto &p:ph){Hmin=std::min(Hmin,p.get_properties()[Hindex]);Hnegative+=p.get_properties()[Hindex]<0;}
        double pmin=std::numeric_limits<double>::max(),pmax=-pmin,mapped=0;
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),QGauss<dim>(3),update_values);
        std::vector<Vector<double>> values(fe.n_quadrature_points,Vector<double>(this->get_fe().n_components()));
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
        {fe.reinit(cell);fe.get_function_values(this->get_solution(),values);for(auto &v:values){auto p=v[this->introspection().component_indices.pressure];pmin=std::min(pmin,p);pmax=std::max(pmax,p);for(const auto &m:this->get_parameters().mapped_particle_properties)if(m.second.first=="maxwell stress")mapped=std::max(mapped,std::abs(v[this->introspection().component_indices.compositional_fields[m.first]]));}}
        uint64_t rng=1469598103934665603ull, membership=rng, audit=rng, partition=rng;
        for(const char c:pm.get_random_number_state()){rng^=static_cast<unsigned char>(c);rng*=1099511628211ull;}
        for(auto id:ids){membership^=id;membership*=1099511628211ull;}
        for(const auto &entry:BP3Benchmark::work_initial_H){audit^=entry.first;hash_double(audit,entry.second);}
        for(const auto &cell:this->get_triangulation().active_cell_iterators())if(cell->is_locally_owned())
          for(const char c:cell->id().to_string()){partition^=static_cast<unsigned char>(c);partition*=1099511628211ull;}
        std::ofstream state(this->get_output_directory()+"rng-"+stage+"-"+std::to_string(this->get_timestep_number())+"-rank"+std::to_string(rank)+".txt");
        state<<pm.get_random_number_state();
        out<<stage<<','<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<ids.size()<<','<<ph.get_next_free_particle_index()<<','<<(previous.empty()?0:born)<<','<<lost<<','<<hp<<','<<hq<<','<<hi<<','<<maxstress<<','<<sum/std::max(size_t(1),ids.size())<<','<<pmin<<','<<pmax<<','<<mapped<<','<<Hmin<<','<<Hnegative<<','<<rng<<','<<membership<<','<<BP3Benchmark::work_initial_H.size()<<','<<audit<<','<<partition<<','<<fresh<<','<<(born-fresh)<<'\n';previous=ids;previous_next=ph.get_next_free_particle_index();

      }
      public:
      static void declare_parameters(ParameterHandler &prm)
      {prm.enter_subsection("Postprocess");prm.enter_subsection("Coupled replenishment");
       prm.declare_entry("Migrate retained particles","false",Patterns::Bool(),"Test-only native transfer of one retained particle to a ghost neighbor per rank.");
       prm.declare_entry("Corrupt audit","none",Patterns::Selection("none|value|missing"),"Negative test on a retained particle, after native restore.");
       prm.declare_entry("Reject step","0",Patterns::Integer(0),"Controlled precommit failure; zero disables.");
       prm.declare_entry("Resume interval","0",Patterns::Double(0),"Test-only requested first restart interval.");prm.leave_subsection();prm.leave_subsection();}
      void parse_parameters(ParameterHandler &prm) override
      {prm.enter_subsection("Postprocess");prm.enter_subsection("Coupled replenishment");migrate=prm.get_bool("Migrate retained particles");corrupt_audit=prm.get("Corrupt audit");reject_step=prm.get_integer("Reject step");resume_dt=prm.get_double("Resume interval");prm.leave_subsection();prm.leave_subsection();}
      void initialize() override
      {
        this->get_signals().post_simulator_initialization.connect([this](const SimulatorAccess<dim>&)
        {
          this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&){capture("generated");});
          auto &surface=this->get_reconstructed_fault_surface_system();
          surface.set_normal_traction_diagnostic([](const Point<dim>&){return false;});
          surface.normal_diagnostic_observer=[this](const ReconstructedFaultSurfaceResidual &r,const auto &)
          {
            capture("incoming");
            if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
            {
              const std::string path=this->get_output_directory()+"traction.csv";bool head=!std::ifstream(path).good();std::ofstream out(path,std::ios::app);out<<std::setprecision(17);
              if(head)out<<"step,time,observation,fault,node,raw_normal,filter_coefficient,V,Theta\n";
              const auto &m=this->get_reconstructed_fault_manager();auto state=m.get_property_information()[m.get_property_index("phase field fault state")].position;
              for(unsigned int f=0;f<r.raw_normal_traction.size();++f)for(unsigned int j=0;j<r.raw_normal_traction[f].size();++j)
                out<<this->get_timestep_number()<<','<<this->get_time()<<','<<observations<<','<<f<<','<<j<<','<<r.raw_normal_traction[f][j]<<','<<r.normal_filter_coefficients[f][j]<<','<<m.get_slip_rate(f)[j]<<','<<m.get_fault(f).get_properties(j)[state]<<'\n';
            }++observations;
          };
        });
        this->get_signals().post_resume_time_step.connect([this](const auto &,double &dt){if(resume_dt>0.)dt=std::min(dt,resume_dt);});
        this->get_signals().post_particle_backup.connect([this](auto &){capture("backup");});
        this->get_signals().post_particle_restore.connect([this](auto &){capture("restored");});
        this->get_signals().post_particle_management.connect([this](auto &){capture("managed");});
        this->get_signals().pre_checkpoint_store_user_data.connect([this](auto &){capture("checkpoint");});
        this->get_signals().post_resume_load_user_data.connect([this](auto &){capture("resume");});
        this->get_signals().post_restore_particles.connect([this](Particle::Manager<dim>&pm){capture("restored_before_advection");
          if(this->get_timestep_number()==1 && migrate)
          {
            // Separate transport diagnostic, not a physical BP3 trajectory:
            // move one retained particle into a ghost neighbor and let native
            // sorting transfer its unmodified properties and auxiliary audit.
            auto &ph=pm.get_particle_handler();bool moved=false;
            for(const auto &cell:this->get_triangulation().active_cell_iterators())
              if(cell->is_locally_owned() && !moved && ph.n_particles_in_cell(cell)>0)
                for(unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell && !moved;++f)
                  if(!cell->at_boundary(f) && cell->neighbor(f)->is_ghost())
                  {
                    auto p=ph.particles_in_cell(cell).begin();
                    p->set_location(cell->neighbor(f)->center());moved=true;
                  }
            AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(moved),this->get_mpi_communicator()),ExcMessage("Migration fixture needs a ghost neighbor on every rank."));
            ph.sort_particles_into_subdomains_and_cells();
            ph.exchange_ghost_particles();
            capture("migrated");
          }
          if(this->get_timestep_number()==1 && corrupt_audit!="none")
          {
            auto p=pm.get_particle_handler().begin();
            if(p!=pm.get_particle_handler().end())
              if(corrupt_audit=="missing") BP3Benchmark::work_initial_H.erase(p->get_id());
              else p->get_properties()[pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force")]+=1.;
          }
        });
        this->get_signals().post_reconstructed_fault_linear_solver.connect([this](const auto &,const auto &,const auto &,const auto &,const auto &,const auto &,double,unsigned int,double)
        {if(reject_step && !rejected && this->get_timestep_number()==reject_step){rejected=true;capture("rejected");throw ExcNonlinearSolverNoConvergence();}});
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {capture("accepted");return {"Coupled replenishment:","sampled"};}
    };
    ASPECT_REGISTER_POSTPROCESSOR(CoupledReplenishment,"coupled replenishment observer","Compact test-only birth/transfer/lifecycle observation.")
  }
}
