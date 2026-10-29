/* P5b bounded native owner/tangent/lifecycle fixture; no model ownership. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/time_stepping/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/particle/manager.h>
#include <aspect/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>

namespace aspect
{
  namespace GeometryModel
  {
    template<int dim> class BulkTensorBox : public Box<dim>
    {
      public:
        void create_coarse_mesh(parallel::distributed::Triangulation<dim> &grid) const override
        {
          Box<dim>::create_coarse_mesh(grid);
          if(!std::getenv("ASPECT_P5B_UNIFORM"))
            {
              for(const auto &cell:grid.active_cell_iterators())
                if(std::abs(cell->center()[0]-.125)<.032 && std::abs(cell->center()[1]-.125)<.032)
                  cell->set_refine_flag();
              grid.execute_coarsening_and_refinement();
            }
        }
    };
    ASPECT_REGISTER_GEOMETRY_MODEL(BulkTensorBox,"bulk tensor box","Small fixed hanging-node material block; no runtime AMR.")
  }
  namespace BoundaryVelocity
  {
    template<int dim> class BulkTensorLoading : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x) const override
        {
          const double t=this->get_time();
          const double xy=t<=.3+1e-12 ? .015 : 0.;
          const double normal=t>.3+1e-12 && t<=.6+1e-12 ? .015 : 0.;
          Tensor<1,dim> v;v[0]=xy*(x[1]-.125)+.5*normal*(x[0]-.125);
          v[1]=-.5*normal*(x[1]-.125);return v;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(BulkTensorLoading,"bulk cohesive loading","Trace-free affine xy, normal-difference load and hold.")
  }
  namespace TimeStepping
  {
    template<int dim> class BulkTensorSteps : public Interface<dim>, public SimulatorAccess<dim>
    {public: double execute() override {return this->get_parameters().maximum_time_step;}};
    ASPECT_REGISTER_TIME_STEPPING_MODEL(BulkTensorSteps,"bulk cohesive steps","Bounded P5b interval; no adaptive-law changes.")
  }
  namespace Postprocess
  {
    // Reuse the established restricted fixture-access friend, rather than
    // adding another production access API. Load only this fixture library.
    template<int dim> class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      using State=typename Model::BulkCohesiveState;
      using Tensor3=MaterialModel::BulkCohesive::Tensor;
      std::uint64_t publications=0;
      bool recovered=false;
      Model &model() const {return const_cast<Model&>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));}
      Simulator<dim> &sim() const {return const_cast<Simulator<dim>&>(this->get_simulator());}
      void check(const std::string &label,const bool pass) const
      {
        std::ofstream out(this->get_output_directory()+"gates.csv",std::ios::app);
        if(out.tellp()==0)out<<"gate,passed\n";
        out<<label<<','<<pass<<std::endl;AssertThrow(pass,ExcMessage("P5b native fixture: "+label));
      }
      static Tensor3 original_direction() {Tensor3 S;S[0][1]=.5;return S;}
      static std::array<Tensor3,4> histories(const typename State::LocalHistory &h,const bool tensor)
      {
        if(tensor)return {{h.A,h.K,h.Q,h.W}};
        const auto S=original_direction();return {{h.a*S,h.signed_shear*S,2*h.q*S,h.rate*S}};
      }
      std::vector<double> persistent() const
      {
        const auto &s=*model().bulk_cohesive_state;std::vector<double> v;
        for(const auto *owner:{s.tensor.get(),s.native_projection.get()})
          for(const auto &field:owner->trial_stress)for(unsigned int i=0;i<field.size();++i)v.push_back(field[i]);
        const auto &native=s.native_history.block(this->introspection().block_indices.velocities);
        v.insert(v.end(),native.begin(),native.end());
        for(const auto &entry:s.history)for(const auto &h:entry.second)
          {
            for(const auto &t:histories(h,s.tensor_flow()))
              for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)v.push_back(t[i][j]);
            v.push_back(s.tensor_flow()?h.L:h.absolute_shear);
          }
        v.insert(v.end(),s.phase.begin(),s.phase.end());v.push_back(s.step);v.push_back(s.time);v.push_back(s.publications);
        return v;
      }
      static bool same(const std::vector<double> &a,const std::vector<double> &b)
      {return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;}
      void initialize_phase()
      {
        const auto pc=this->introspection().variable("phase_field").first_component_index;
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())for(const auto vertex:cell->vertex_indices())
          sim().solution[cell->vertex_dof_index(vertex,pc)]=.2;
        this->get_current_constraints().distribute(sim().solution);
      }
      void recovery()
      {
        const auto before=persistent();const auto original=sim().solution;
        sim().compute_current_constraints();sim().solve_timestep();
        check("converged trial leaves all accepted histories unchanged",same(before,persistent()) && bool(sim().pending_dg_endpoint_publication));
        const auto candidate=sim().solution;
        auto reject=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};reject(false);
        check("outer rejection restores solution and physical owners",same(before,persistent()) && sim().solution==original);
        sim().solve_timestep();
        check("retry reproduces solution without publication",same(before,persistent()) && sim().solution==candidate);
        auto discard=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};discard(false);
        check("second rejection leaves accepted state unchanged",same(before,persistent()) && sim().solution==original);
        recovered=true;
      }
      void export_state() const
      {
        namespace BC=MaterialModel::BulkCohesive;
        const auto &s=*model().bulk_cohesive_state;const auto &p=s.parameters;
        const auto rule=model().mechanical_quadrature();
        const auto &element=this->get_fe();
        FEValues<dim> fe(this->get_mapping(),element,rule,update_values|update_gradients|update_JxW_values|update_quadrature_points);
        const auto pc=this->introspection().variable("phase_field").first_component_index;
        const auto label=Utilities::int_to_string(s.step,4);
        std::ofstream points(this->get_output_directory()+"history-"+label+".csv");
        points<<std::setprecision(17)<<"cell,q,x_m,y_m,JxW_m2,phi,Txx_Pa,Tyy_Pa,Tzz_Pa,Txy_Pa,tauxx_Pa,tauyy_Pa,tauzz_Pa,tauxy_Pa";
        for(const auto name:{"A","K","Q","W"})for(const auto suffix:{"xx","xy"})points<<','<<name<<suffix;
        points<<",L,Y_Pa,Eb_J_m3,Ec_J_m3\n";
        Vector<double> stress_load(sim().solution.size()),total_load(sim().solution.size());
        std::vector<double> phi(rule.size()),pressure(rule.size());
        std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
          {
            fe.reinit(cell);cell->get_dof_indices(ids);
            fe[FEValuesExtractors::Scalar(pc)].get_function_values(sim().solution,phi);
            fe[this->introspection().extractors.pressure].get_function_values(sim().solution,pressure);
            const auto source=s.evaluate_tensor(model(),cell->id(),rule);
            for(unsigned int q=0;q<rule.size();++q)
              {
                const auto &local=s.history.at(cell->id())[q];const auto values=histories(local,s.tensor_flow());
                Tensor3 T;T[0][0]=source[q][0][0];T[1][1]=source[q][1][1];T[0][1]=source[q][0][1];T[2][2]=-T[0][0]-T[1][1];
                const auto tau=T-2*p.G*values[0];const auto h=BC::compliance(phi[q],p.m);
                const auto point=fe.quadrature_point(q);
                points<<cell->id()<<','<<q<<','<<point[0]<<','<<point[1]<<','<<fe.JxW(q)<<','<<phi[q];
                for(const auto &t:{T,tau})points<<','<<t[0][0]<<','<<t[1][1]<<','<<t[2][2]<<','<<t[0][1];
                for(const auto &t:values)points<<','<<t[0][0]<<','<<t[0][1];
                points<<','<<(s.tensor_flow()?local.L:local.absolute_shear)<<','<<h.first*(values[2]*values[2])/(4*p.G)
                      <<','<<(tau*tau)/(4*p.G)<<','<<h.h*(values[2]*values[2])/(4*p.G)<<'\n';
                for(unsigned int i=0;i<ids.size();++i)
                  {
                    const auto raw=fe[this->introspection().extractors.velocities].symmetric_gradient(i,q);
                    const auto dev=Utilities::Tensors::consistent_deviator(raw);
                    Tensor3 e;e[0][0]=dev[0][0];e[1][1]=dev[1][1];e[2][2]=-e[0][0]-e[1][1];e[0][1]=dev[0][1];
                    const double force=fe.JxW(q)*(tau*e);
                    stress_load[ids[i]]+=force;
                    total_load[ids[i]]+=force-fe.JxW(q)*pressure[q]*fe[this->introspection().extractors.velocities].divergence(i,q);
                  }
              }
          }
        std::ofstream loads(this->get_output_directory()+"weak-"+label+".csv");loads<<std::setprecision(17)<<"dof,stress_N_m,total_N_m\n";
        for(unsigned int i=0;i<stress_load.size();++i)loads<<i<<','<<stress_load[i]<<','<<total_load[i]<<'\n';
        const auto record=persistent();std::ofstream binary(this->get_output_directory()+"record-"+label+".bin",std::ios::binary);
        binary.write(reinterpret_cast<const char*>(record.data()),record.size()*sizeof(double));
        std::ofstream solution(this->get_output_directory()+"solution-"+label+".bin",std::ios::binary);
        for(unsigned int i=0;i<sim().solution.size();++i){const double v=sim().solution[i];solution.write(reinterpret_cast<const char*>(&v),sizeof(v));}
      }
      public:
        void initialize() override
        {
          this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&){initialize_phase();});
          this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&)
            {if(std::getenv("ASPECT_P5B_RECOVERY") && !recovered && this->get_time()>.39)recovery();});
        }
        std::pair<std::string,std::string> execute(TableHandler&) override
        {
          const auto &s=*model().bulk_cohesive_state;
          check("exactly one accepted publication",s.publications==this->get_timestep_number() && (!s.publications || s.publications==publications+1));
          publications=s.publications;export_state();
          struct rusage usage;getrusage(RUSAGE_SELF,&usage);
          std::ofstream out(this->get_output_directory()+"fixture.csv",std::ios::app);
          if(out.tellp()==0)out<<"step,time,publications,cells,DoFs,peak_KiB\n";
          out<<std::setprecision(17)<<s.step<<','<<s.time<<','<<s.publications<<','<<this->get_triangulation().n_global_active_cells()<<','<<this->get_dof_handler().n_dofs()<<','<<usage.ru_maxrss<<'\n';
          return {"P5b native fixture","accepted"};
        }
        void save(std::map<std::string,std::string> &status) const override
        {status["P5b publications"]=std::to_string(publications);status["P5b recovered"]=std::to_string(recovered);}
        void load(const std::map<std::string,std::string> &status) override
        {publications=std::stoull(status.at("P5b publications"));recovered=std::stoul(status.at("P5b recovered"));}
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"bulk cohesive checks","P5b native accepted owner/restart/weak-load observer.")
  }
}
