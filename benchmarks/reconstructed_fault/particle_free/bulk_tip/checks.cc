/* Bounded bulk-tip diagnostics; reuse the existing benchmark friendship only. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/endpoint_phase_constraints.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

namespace aspect
{
  namespace Postprocess
  {
    template<int dim> class MomentCycle:public Interface<dim>,public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      bool rejected=false;
      Model &model()const{return const_cast<Model&>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));}
      Simulator<dim> &sim()const{return const_cast<Simulator<dim>&>(this->get_simulator());}
      void gate(const std::string &name,const bool local)const
      {
        const bool pass=Utilities::MPI::min(static_cast<unsigned int>(local),this->get_mpi_communicator());
        if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
          {std::ofstream f(this->get_output_directory()+"tip-checks.csv",std::ios::app);f<<name<<','<<pass<<'\n';}
        AssertThrow(pass,ExcMessage("Bulk tip check failed: "+name));
      }
      std::vector<double> persistent()const
      {
        const auto &s=*model().bulk_cohesive_state;
        std::vector<double> a{s.time,double(s.step),double(s.publications)};
        a.insert(a.end(),s.phase.begin(),s.phase.end());
        for(const auto &v:s.sections)for(double x:{v.coordinate,v.weight,v.age,v.slip,v.rate,v.force})a.push_back(x);
        for(const auto &cell:s.history)for(const auto &p:cell.second)
          {for(const auto *t:{&p.stress,&p.A})for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)a.push_back((*t)[i][j]);
           for(double x:{p.rate,p.q,p.signed_shear,p.absolute_shear})a.push_back(x);}
        return a;
      }
      void recovery()
      {
        const auto before=persistent();const auto owner=model().bulk_cohesive_state;
        const auto original=sim().solution;
        sim().compute_current_constraints();sim().solve_timestep();
        gate("trial keeps complete accepted parent",before==persistent() && owner==model().bulk_cohesive_state && bool(sim().pending_dg_endpoint_publication));
        const auto trial=sim().solution;
        auto reject=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};reject(false);
        gate("rollback restores fields and all histories",before==persistent() && original==sim().solution);
        sim().solve_timestep();gate("retry is deterministic",before==persistent() && trial==sim().solution);
        auto discard=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};discard(false);
        gate("second rollback unchanged",before==persistent() && original==sim().solution);rejected=true;
      }
    public:
      void initialize()override
      {
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&)
          {if(std::getenv("ASPECT_TIP_RECOVERY") && !rejected && this->get_timestep_number()==2)recovery();});
      }
      std::pair<std::string,std::string> execute(TableHandler&)override
      {
        const auto &s=*model().bulk_cohesive_state;
        gate("single publication",s.step==this->get_timestep_number() && s.publications==s.step);
        if(s.step>0)return {"Bulk tip","accepted histories checked"};
        std::vector<double> fractions(model().elastic_shear_moduli.size(),0.);fractions[0]=1.;
        const auto &law=model().fault_friction;
        for(double v:{0.,1e-20,-1e-20,.001,-.001,.01,-.01})
          {
            const auto r=law.signed_endpoint(fractions,v,10.,.005,1000.,1000.);
            const double eps=v==0.?1e-24:std::abs(v)*1e-5;
            const auto a=law.signed_endpoint(fractions,v+eps,10.,.005,1000.,1000.);
            const auto b=law.signed_endpoint(fractions,v-eps,10.,.005,1000.,1000.);
            gate("signed RSF derivative",std::abs((a.traction-b.traction)/(2*eps)-r.derivative)<2e-6*std::max(1.,std::abs(r.derivative)));
            gate("positive BE age and signed dissipation",r.age>0. && r.traction*v>=0.);
          }
        for(double free:{0.,.001,-.001,.1,-.1})for(double mobility:{0.,1e-7,1e-4})
          {
            const auto r=law.section_endpoint(fractions,free,mobility,10.,.005,1000.,1000.);
            gate("implicit section balance",std::abs(r.rate+mobility*r.law.traction-free)<1e-14);
            if(free!=0.)
              {
                const double eps=1e-6*std::abs(free);
                const auto a=law.section_endpoint(fractions,free+eps,mobility,10.,.005,1000.,1000.);
                const auto b=law.section_endpoint(fractions,free-eps,mobility,10.,.005,1000.,1000.);
                gate("implicit section derivative",std::abs((a.law.traction-b.law.traction)/(2*eps)-r.force_derivative)<2e-6*std::max(1.,std::abs(r.force_derivative)));
              }
          }
        const auto &handler=this->get_phase_field_handler();
        const auto b=this->introspection().phase_field().block_index;
        const auto comm=this->get_mpi_communicator();
        LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm),field(this->get_solution());
        owned=field;const auto n=owned.block(b).size();std::vector<double> direction(n);
        for(unsigned int i=0;i<n;++i)direction[i]=std::sin(1.7*i)+.3*std::cos(.37*i);
        for(const auto i:owned.block(b).locally_owned_elements())owned.block(b)[i]=.2+.01*std::sin(.7*i);
        owned.compress(VectorOperation::insert);field=owned;
        std::map<std::pair<unsigned int,unsigned int>,double> H;
        const auto energy=handler.evaluate_cell_fracture(field,true,&H);
        double action=0.;for(unsigned int i=0;i<n;++i)action+=energy.gradient[i]*direction[i];
        const double eps=1e-6;
        for(const auto i:owned.block(b).locally_owned_elements())owned.block(b)[i]+=eps*direction[i];
        owned.compress(VectorOperation::insert);field=owned;
        const auto plus=handler.evaluate_cell_fracture(field,false);
        for(const auto i:owned.block(b).locally_owned_elements())owned.block(b)[i]-=2*eps*direction[i];
        owned.compress(VectorOperation::insert);field=owned;
        const auto minus=handler.evaluate_cell_fracture(field,false);
        gate("native SIPG energy derivative",std::abs((plus.energy-minus.energy)/(2*eps)-action)<1e-6*std::max(1.,std::abs(action)));
        double error=0.,norm=0.;std::vector<double> hd(n,0.);
        for(const auto &entry:H)hd[entry.first.first]+=entry.second*direction[entry.first.second];
        for(unsigned int i=0;i<n;++i){error=std::max(error,std::abs((plus.gradient[i]-minus.gradient[i])/(2*eps)-hd[i]));norm=std::max(norm,std::abs(hd[i]));}
        gate("native SIPG Hessian derivative",error<1e-6*std::max(1.,norm));
        return {"Bulk tip","RSF and native fracture derivatives checked"};
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"bulk tip checks","Bounded bulk section-RSF diagnostics and opt-in rejection exercise.")
  }
}
