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
      public:
      static void declare_parameters(ParameterHandler &) {}
      void generate_particles(Particles::ParticleHandler<dim> &ph) override
      {
        ReferenceCell<dim>::generate_particles(ph);
        for(auto &p:ph)
          if(p.get_location()[1]<1500.)
          {
            auto x=p.get_location();x[0]+=100.*(std::floor(x[1]/500.)-1.);x[1]=2000.-1e-8;p.set_location(x);
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
      bool reject=false,rejected=false,track_births=false;
      std::set<types::particle_index> previous,attempt_births;
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
        if(header)out<<"stage,step,time,dt,n,next_id,born_or_received,lost_or_sent,position_hash,property_hash,integrator_hash,stress_max,stress_mean,pressure_min,pressure_max,mapped_stress_max,H_min,H_negative\n";
        std::map<types::particle_index,std::vector<double>> data;
        std::set<types::particle_index> ids;
        for(const auto &p:ph){ids.insert(p.get_id());auto &a=data[p.get_id()];for(unsigned int d=0;d<dim;++d)a.push_back(p.get_location()[d]);for(double x:p.get_properties())a.push_back(x);}
        unsigned int born=0,lost=0;for(auto id:ids)born+=!previous.count(id);for(auto id:previous)lost+=!ids.count(id);
        uint64_t hp=1469598103934665603ull,hq=hp,hi=hp;double maxstress=0,sum=0;
        const auto internal=info.get_position_by_field_name("internal: integrator properties");
        for(auto &p:data){hp^=p.first;hq^=p.first;hi^=p.first;
          for(unsigned int d=0;d<dim;++d)hash_double(hp,p.second[d]);
          for(unsigned int d=0;d<info.n_components();++d)hash_double(d>=internal?hi:hq,p.second[dim+d]);
          for(unsigned int d=0;d<3;++d)maxstress=std::max(maxstress,std::abs(p.second[dim+s+d]));sum+=p.second[dim+s];}
        double Hmin=std::numeric_limits<double>::max();unsigned int Hnegative=0;const auto Hindex=info.get_position_by_field_name("crack_driving_force");
        for(const auto &p:ph){Hmin=std::min(Hmin,p.get_properties()[Hindex]);Hnegative+=p.get_properties()[Hindex]<0;}
        double pmin=std::numeric_limits<double>::max(),pmax=-pmin,mapped=0;
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),QGauss<dim>(3),update_values);
        std::vector<Vector<double>> values(fe.n_quadrature_points,Vector<double>(this->get_fe().n_components()));
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
        {fe.reinit(cell);fe.get_function_values(this->get_solution(),values);for(auto &v:values){auto p=v[this->introspection().component_indices.pressure];pmin=std::min(pmin,p);pmax=std::max(pmax,p);for(const auto &m:this->get_parameters().mapped_particle_properties)if(m.second.first=="maxwell stress")mapped=std::max(mapped,std::abs(v[this->introspection().component_indices.compositional_fields[m.first]]));}}
        out<<stage<<','<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<ids.size()<<','<<ph.get_next_free_particle_index()<<','<<(previous.empty()?0:born)<<','<<lost<<','<<hp<<','<<hq<<','<<hi<<','<<maxstress<<','<<sum/std::max(size_t(1),ids.size())<<','<<pmin<<','<<pmax<<','<<mapped<<','<<Hmin<<','<<Hnegative<<'\n';previous=ids;
        if(track_births && stage=="incoming" && this->get_timestep_number()>0)
        {
          // Test-only audit adapter: baseline each genuinely new ID before any
          // mechanical history update. Existing IDs retain the original H check.
          const unsigned int H=info.get_position_by_field_name("crack_driving_force");
          std::vector<std::pair<types::particle_index,double>> fresh;
          for(const auto &p:ph)if(!BP3Benchmark::work_initial_H.count(p.get_id()))fresh.emplace_back(p.get_id(),p.get_properties()[H]);
          for(auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),fresh))for(auto &entry:part){BP3Benchmark::work_initial_H.emplace(entry);attempt_births.insert(entry.first);}
        }
      }
      public:
      static void declare_parameters(ParameterHandler &prm)
      {prm.enter_subsection("Postprocess");prm.enter_subsection("Coupled replenishment");
       prm.declare_entry("Reject first step","false",Patterns::Bool(),"Controlled precommit failure at the first nonzero step.");
       prm.declare_entry("Track birth audit","false",Patterns::Bool(),"Test-only H audit baseline for newly initialized IDs.");prm.leave_subsection();prm.leave_subsection();}
      void parse_parameters(ParameterHandler &prm) override
      {prm.enter_subsection("Postprocess");prm.enter_subsection("Coupled replenishment");reject=prm.get_bool("Reject first step");track_births=prm.get_bool("Track birth audit");prm.leave_subsection();prm.leave_subsection();}
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
        this->get_signals().post_restore_particles.connect([this](Particle::Manager<dim>&){capture("restored_before_advection");});
        this->get_signals().post_reconstructed_fault_linear_solver.connect([this](const auto &,const auto &,const auto &,const auto &,const auto &,const auto &,double,unsigned int,double)
        {if(reject && !rejected && this->get_timestep_number()==1){rejected=true;capture("rejected");for(auto id:attempt_births)BP3Benchmark::work_initial_H.erase(id);attempt_births.clear();throw ExcNonlinearSolverNoConvergence();}});
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {capture("accepted");attempt_births.clear();return {"Coupled replenishment:","sampled"};}
    };
    ASPECT_REGISTER_POSTPROCESSOR(CoupledReplenishment,"coupled replenishment observer","Compact test-only birth/transfer/lifecycle observation.")
  }
}
