/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Fixture owns loading/initial prescription/diagnostics, never physical history.
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/time_stepping/interface.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#ifdef ASPECT_BULK_ENERGY_FIXTURE
#include <aspect/utilities.h>
#endif

namespace aspect
{
  namespace BoundaryVelocity
  {
    template<int dim> class BulkCohesiveLoading:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x) const override
        {
          // Prescribed simple shear, followed by a short hold. Endpoint values
          // make the rate right-continuous with respect to accepted intervals.
          const double t=this->get_time();
#ifdef ASPECT_BULK_CONTINUED_FIXTURE
          const double loading_end=.175;
#else
          const double loading_end=.15;
#endif
          const double rate=t>.05+1e-12 && t<=loading_end+1e-12 ? .018 : 0.;
          Tensor<1,dim> v;v[0]=rate*(x[1]-.5);return v;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(BulkCohesiveLoading,"bulk cohesive loading",
      "Bounded bulk cohesive fixture: unloaded hold, affine shear ramp and hold.")
  }
  namespace TimeStepping
  {
    template<int dim> class BulkCohesiveSteps:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        double execute() override {return this->get_parameters().maximum_time_step;}
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(BulkCohesiveSteps,"bulk cohesive steps","Bounded prescribed bulk-model timestep.")
  }
  namespace Postprocess
  {
    template<int dim> class MomentCycle:public Interface<dim>,public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      using BCState=MaterialModel::BulkCohesive::State;
      unsigned int last_publications=0;
      bool recovery_done=false;
      bool source_failure_checked=false;
      Model &model()const{return const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));}
      Simulator<dim> &sim()const{return const_cast<Simulator<dim>&>(this->get_simulator());}
      std::ofstream output(const std::string &name)const
        {std::ofstream f(this->get_output_directory()+name);f<<std::setprecision(17);return f;}
      void check(const std::string &name,const bool value)const
      {
        std::ofstream out(this->get_output_directory()+"gates.csv",std::ios::app);
        if(out.tellp()==0)out<<"gate,passed\n";
        out<<name<<','<<value<<std::endl;AssertThrow(value,ExcMessage("Bulk fixture gate: "+name));
      }
      std::vector<double> dg_local_phase_record()const
      {
        const auto &s=*model().bulk_cohesive_state;std::vector<double> v;
        for(const auto &field:s.tensor->trial_stress)for(unsigned int i=0;i<field.size();++i)v.push_back(field[i]);
        for(const auto &entry:s.history)for(const auto &h:entry.second)
          {v.push_back(h.a);v.push_back(h.signed_shear);v.push_back(h.absolute_shear);v.push_back(h.q);v.push_back(h.rate);}
        v.insert(v.end(),s.phase.begin(),s.phase.end());
        v.push_back(s.step);v.push_back(s.time);v.push_back(s.publications);
        return v;
      }
      std::vector<double> persistent()const
      {
        auto v=dg_local_phase_record();const auto &s=*model().bulk_cohesive_state;
        const auto &memory=s.native_history.block(this->introspection().block_indices.velocities);
        for(unsigned int i=0;i<memory.size();++i)v.push_back(memory[i]);
        for(const auto &field:s.native_projection->trial_stress)
          for(unsigned int i=0;i<field.size();++i)v.push_back(field[i]);
        return v;
      }
      static bool same(const std::vector<double> &a,const std::vector<double> &b)
        {return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;}
      void seed()
      {
        static const PhaseField::GeometricFunction a(.15625,1.,8./3.);
        static const PhaseField::DegradationFunction g(1.,128.);
        static const PhaseField::PhaseFieldProfile profile(a,g,.6);
        const auto component=this->introspection().variable("phase_field").first_component_index;
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())for(auto v:cell->vertex_indices())
          {const auto x=cell->vertex(v);sim().solution[cell->vertex_dof_index(v,component)]=profile.value(std::hypot(std::max(0.,x[0]-.5),x[1]-.5));}
        this->get_current_constraints().distribute(sim().solution);
      }
      void recovery()
      {
        const auto old=persistent();const auto original=sim().solution;
        sim().compute_current_constraints();
        sim().solve_timestep();
        check("converged trial does not publish",same(old,persistent()) && bool(sim().pending_dg_endpoint_publication));
        const auto trial=sim().solution;
        auto reject=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};reject(false);
        check("outer rejection restores native solution and all bulk histories",same(old,persistent()) && sim().solution==original);
        sim().solve_timestep();
        check("identical retry native solution",sim().solution==trial && same(old,persistent()));
        auto discard=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};discard(false);
        check("repeated rejection leaves accepted histories unchanged",same(old,persistent()));
        recovery_done=true;
      }
      void source_checks()
      {
        namespace BC=MaterialModel::BulkCohesive;
        const auto &m=model();const auto &s=*m.bulk_cohesive_state;
        const std::string label=Utilities::int_to_string(s.step,4);
        const FE_Q<dim> scalar(2);
        const Quadrature<dim> nodes(scalar.get_unit_support_points());
        std::vector<Point<dim>> samples;
        for(const auto &p:std::vector<std::array<double,2>>{{{.13,.27}},{{.34,.77}},{{.5,.5}},{{.92,.08}},{{.03,.97}},{{0.,0.}},{{1.,.31}}})
          {Point<dim> x;x[0]=p[0];x[1]=p[1];samples.push_back(x);}
        const Quadrature<dim> points(samples);
        auto coefficients=output("cell-history-"+label+".csv");
        coefficients<<"cell,u,v,Ax,Ay,DGxx,DGyy,DGxy,Pxx,Pyy,Pxy\n";
        auto values=output("point-history-"+label+".csv");values<<"cell,u,v,x,y,Txx,Tyy,Txy\n";
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),nodes,update_values);
        std::vector<Tensor<1,dim>> A(nodes.size());
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          {
            fe.reinit(cell);fe[this->introspection().extractors.velocities].get_function_values(s.native_history,A);
            const auto dg=m.evaluate_dg_trial_tensor(*s.tensor,cell->id(),nodes);
            const auto projection=m.evaluate_dg_trial_tensor(*s.native_projection,cell->id(),nodes);
            for(unsigned int q=0;q<nodes.size();++q)
              coefficients<<cell->id()<<','<<nodes.point(q)[0]<<','<<nodes.point(q)[1]<<','<<A[q][0]<<','<<A[q][1]<<','
                          <<dg[q][0][0]<<','<<dg[q][1][1]<<','<<dg[q][0][1]<<','<<projection[q][0][0]<<','<<projection[q][1][1]<<','<<projection[q][0][1]<<'\n';
            const auto T=s.evaluate_tensor(m,cell->id(),points);
            for(unsigned int q=0;q<points.size();++q)
              {
                const auto x=this->get_mapping().transform_unit_to_real_cell(cell,points.point(q));
                values<<cell->id()<<','<<points.point(q)[0]<<','<<points.point(q)[1]<<','<<x[0]<<','<<x[1]<<','<<T[q][0][0]<<','<<T[q][1][1]<<','<<T[q][0][1]<<'\n';
              }
          }
        if(!source_failure_checked && s.step>0)
          {
            const auto before=persistent();auto bad=s;
            bad.tensor=std::make_shared<typename Model::DGHistory>(*s.tensor);
            const auto id=s.native_cells.begin()->first;
            std::vector<types::global_dof_index> ids(9);s.tensor->space->cells.at(id)->get_dof_indices(ids);
            bad.tensor->trial_stress[0][ids[0]]+=.01;
            bad.tensor->trial_stress[0].compress(VectorOperation::insert);
            std::vector<BC::Tensor> expected;
            for(const auto &T:s.evaluate_tensor(m,id,points))
              {BC::Tensor t;t[0][0]=T[0][0];t[1][1]=T[1][1];t[2][2]=-T[0][0]-T[1][1];t[0][1]=T[0][1];expected.push_back(t);}
            bool failed=false;
            try{bad.check_tensor(m,id,points,expected);}
            catch(const std::exception &e){failed=std::string(e.what()).find("independent QP check")!=std::string::npos;}
            check("inconsistent source history fails unchanged tolerance",failed && same(before,persistent()));
            const auto directory=this->get_output_directory();
            AssertThrow(std::rename((directory+"bulk-source-consistency-failure.csv").c_str(),
                                    (directory+"deliberate-source-consistency-failure.csv").c_str())==0,ExcInternalError());
            source_failure_checked=true;
          }
      }
#ifdef ASPECT_BULK_ENERGY_FIXTURE
      // Diagnostic only: assemble the actual unconstrained boundary reactions
      // from accepted stress and physical pressure, also during a fixed hold.
      void boundary_work()
      {
        const auto &m=model();const auto &s=*m.bulk_cohesive_state;
        struct Reaction{Point<dim> point;unsigned int component=0;double force=0.;};
        std::map<types::global_dof_index,Reaction> reactions;
        const auto nv=sim().solution.block(this->introspection().block_indices.velocities).size();
        for(const auto &line:this->get_current_constraints().get_lines())
          if(line.index<nv && line.entries.empty())reactions.emplace(line.index,Reaction());
        const auto quadrature=m.mechanical_quadrature();
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),quadrature,update_values|update_gradients|update_JxW_values);
        std::vector<double> pressure(quadrature.size());
        std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell());
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          {
            if(!cell->at_boundary())continue;
            cell->get_dof_indices(ids);fe.reinit(cell);
            fe[this->introspection().extractors.pressure].get_function_values(sim().solution,pressure);
            const auto T=s.evaluate_tensor(m,cell->id(),quadrature);
            for(unsigned int i=0;i<ids.size();++i)
              if(reactions.count(ids[i]))
                {
                  auto &r=reactions.at(ids[i]);r.component=this->get_fe().system_to_component_index(i).first;
                  r.point=this->get_mapping().transform_unit_to_real_cell(cell,this->get_fe().get_unit_support_points()[i]);
                  for(unsigned int q=0;q<quadrature.size();++q)
                    {
                      const auto raw=fe[this->introspection().extractors.velocities].symmetric_gradient(i,q);
                      const auto E=Utilities::Tensors::consistent_deviator(raw);
                      const double xy=T[q][0][1]-s.parameters.G*s.history.at(cell->id())[q].a;
                      const double contraction=T[q][0][0]*E[0][0]+T[q][1][1]*E[1][1]
                        +(T[q][0][0]+T[q][1][1])*(E[0][0]+E[1][1])+2*xy*E[0][1];
                      r.force+=fe.JxW(q)*(contraction-pressure[q]*trace(raw));
                    }
                }
          }
        const auto label=Utilities::int_to_string(s.step,4);
        auto out=output("boundary-reactions-"+label+".csv");out<<"dof,component,x,y,force,increment\n";
        for(const auto &entry:reactions)
          {
            const auto &r=entry.second;
            AssertThrow(r.point[0]==0. || r.point[0]==1.5 || r.point[1]==0. || r.point[1]==1.,
                        ExcMessage("The energy fixture reaction must be on the prescribed physical boundary."));
            out<<entry.first<<','<<r.component<<','<<r.point[0]<<','<<r.point[1]<<','<<r.force<<','
               <<(s.step ? this->get_timestep()*sim().solution[entry.first] : 0.)<<'\n';
          }
      }
#endif
      void export_state()
      {
        namespace BC=MaterialModel::BulkCohesive;
        const auto &m=model();const auto &s=*m.bulk_cohesive_state;
        const std::string label=Utilities::int_to_string(this->get_timestep_number(),4);
        const auto pb=this->introspection().variable("phase_field").block_index;
        const auto pc=this->introspection().variable("phase_field").first_component_index;
        const auto po=sim().solution.get_block_indices().block_start(pb);
        auto nodes=output("phase-"+label+".csv");nodes<<"dof,x,y,phi\n";
        std::map<unsigned int,Point<dim>> positions;
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())for(auto v:cell->vertex_indices())
          positions[cell->vertex_dof_index(v,pc)-po]=cell->vertex(v);
        for(const auto &node:positions)
          {nodes<<node.first<<','<<node.second[0]<<','<<node.second[1]<<','<<s.phase[node.first]<<'\n';}
        const auto quad=m.mechanical_quadrature();
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points|update_JxW_values);
        std::vector<double> phi(quad.size());
#ifdef ASPECT_BULK_ENERGY_FIXTURE
#ifdef ASPECT_BULK_CONTINUED_FIXTURE
        const bool dense=s.step==0 || std::abs(s.time-.15)<1e-12
                         || std::abs(s.time-.175)<1e-12 || std::abs(s.time-.185)<1e-12;
        // Original, unfiltered local constitutive rate at its native Gauss8
        // history points. Earlier frames can be recovered from q and phi via
        // the same accepted a recurrence; dense histories independently check it.
        std::ofstream qp_rate;
        if(s.time>=.12-1e-12)qp_rate=output("quadrature-rate-"+label+".bin");
#else
        const bool dense=s.step==0 || std::abs(s.time-.15)<1e-12 || std::abs(s.time-.16)<1e-12;
#endif
        auto qp_history=output("quadrature-q-"+label+".bin");
#else
        const bool dense=this->get_timestep_number()==0 || this->get_timestep_number()%4==0
                         || this->get_time()>=.149999 || (std::getenv("ASPECT_BULK_SOURCE_EXPORT") && this->get_time()>=.13-1e-12);
#endif
        std::ofstream fields;if(dense){fields=output("bulk-"+label+".csv");fields<<"cell,x,y,JxW,phi,Txx,Tyy,Txy,tauxx,tauyy,tauxy,a,q,k,L,driving,bulk,cohesive,DGxx,DGyy,DGxy,Rxx,Ryy,Rxy\n";}
        // Normal columns use the actual tensor Gauss weights from all normal
        // cells; no surface association is read to select constitutive points.
#ifdef ASPECT_BULK_ORIENTATION_FIXTURE
        // The isolated rotated-mesh fixture has no tensor-rule normal columns.
        // Retain native QP histories and positive physical measures instead of
        // interpolating them to fictitious columns. This area moment identity
        // is diagnostic only, never a surface normalization or constitutive input.
        std::vector<double> area_weights;
        std::vector<BC::Response> area_samples;
#else
        std::map<unsigned int,std::pair<std::vector<double>,std::vector<BC::Response>>> columns;
        std::map<unsigned int,double> column_x;
#endif
        double maxdrive=0.;
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          {
            fe.reinit(cell);fe[FEValuesExtractors::Scalar(pc)].get_function_values(sim().solution,phi);
            const auto T=s.evaluate_tensor(m,cell->id(),quad);
            const auto polynomial=m.evaluate_dg_trial_tensor(*s.tensor,cell->id(),quad);
            for(unsigned int q=0;q<quad.size();++q)
              {
                const auto &h=s.history.at(cell->id())[q];BC::Response r;
#ifdef ASPECT_BULK_ENERGY_FIXTURE
                qp_history.write(reinterpret_cast<const char*>(&h.q),sizeof(double));
#ifdef ASPECT_BULK_CONTINUED_FIXTURE
                if(qp_rate.is_open())qp_rate.write(reinterpret_cast<const char*>(&h.rate),sizeof(double));
#endif
#endif
                r.state.a=h.a;r.state.q=h.q;r.state.phi=phi[q];r.w=h.rate;
                const auto c=BC::compliance(phi[q],s.parameters.m);
                r.cohesive_energy=c.h*h.q*h.q/(2*s.parameters.G);
                r.cohesive_dissipation=c.h*h.q*h.q/s.parameters.eta;
                const auto x=fe.quadrature_point(q);
                const double drive=c.first*h.q*h.q/(2*s.parameters.G);
                maxdrive=std::max(maxdrive,drive);
#ifdef ASPECT_BULK_ORIENTATION_FIXTURE
                area_weights.push_back(fe.JxW(q));area_samples.push_back(r);
#else
                const QGauss<1> normal_rule(s.quadrature);
                const unsigned int normal=q/s.quadrature;
                const double dx=cell->vertex(1)[0]-cell->vertex(0)[0];
                const unsigned int column_index=std::lround(cell->vertex(0)[0]/dx)*s.quadrature+q%s.quadrature;
                auto &column=columns[column_index];column_x[column_index]=x[0];
                column.first.push_back((cell->vertex(2)[1]-cell->vertex(0)[1])*normal_rule.weight(normal));
                column.second.push_back(r);
#endif
                if(dense)fields<<cell->id().to_string()<<','<<x[0]<<','<<x[1]<<','<<fe.JxW(q)<<','<<phi[q]<<','<<T[q][0][0]<<','<<T[q][1][1]<<','<<T[q][0][1]<<','
                     <<T[q][0][0]<<','<<T[q][1][1]<<','<<T[q][0][1]-s.parameters.G*h.a<<','<<h.a<<','<<h.q<<','<<h.signed_shear<<','<<h.absolute_shear<<','<<drive<<','
                     <<((T[q]*T[q])+std::pow(trace(T[q]),2)-4*s.parameters.G*h.a*T[q][0][1]+2*s.parameters.G*s.parameters.G*h.a*h.a)/(4*s.parameters.G)<<','<<r.cohesive_energy<<','
                     <<polynomial[q][0][0]<<','<<polynomial[q][1][1]<<','<<polynomial[q][0][1]<<','<<T[q][0][0]-polynomial[q][0][0]<<','<<T[q][1][1]-polynomial[q][1][1]<<','<<T[q][0][1]-polynomial[q][0][1]<<'\n';
              }
          }
        const auto old=persistent();
#ifdef ASPECT_BULK_ORIENTATION_FIXTURE
        const auto area_result=BC::observe_profile(s.parameters,area_weights,area_samples);
        // Extended accumulation avoids O(number of QPs) double-sum drift on
        // the refined fixture; the physical coverage tolerance is unchanged.
        long double area=0.;for(const auto weight:area_weights)area+=weight;
        check("rotated fixture positive quadrature covers physical area",std::abs(area-1.5L)<1e-12L);
        check("rotated fixture area moment decomposition",std::abs(area_result.bulk_cohesive
              -area_result.surface_cohesive-area_result.residual_cohesive)<1e-12);
        auto observer=output("area-observer-"+label+".csv");
        observer<<"area,integrated_h,integrated_a,G_integrated_a,cohesive,mean_mode,residual,moment_defect\n";
        observer<<area<<','<<area_result.I<<','<<area_result.elastic_slip<<','<<area_result.Z<<','
                <<area_result.bulk_cohesive<<','<<area_result.surface_cohesive<<','
                <<area_result.residual_cohesive<<','<<area_result.residual_moment<<'\n';
#else
        auto observer=output("observer-"+label+".csv");observer<<"x,I,se,Z,residual_energy,moment_defect,original_surface,rebuilt_surface\n";
        for(const auto &entry:columns)
          {
            const auto result=BC::observe_profile(s.parameters,entry.second.first,entry.second.second);
            double covered=0.;for(double weight:entry.second.first)covered+=weight;
            check("observer normal quadrature covers full domain",std::abs(covered-1.)<1e-12);
            const double x=column_x.at(entry.first);
            if(result.normalized)observer<<x<<','<<result.I<<','<<result.elastic_slip<<','<<result.Z<<','<<result.residual_cohesive<<','<<result.residual_moment<<','<<(x<=.5)<<','<<(x<=.75)<<'\n';
          }
#endif
        check("zero-time observer rebuild preserves every physical bulk history",same(old,persistent()));
        // Preserve the old DG/local/phase record layout. The new persistent FE
        // owner is a separate export and is included in all lifecycle checks.
#ifdef ASPECT_BULK_ENERGY_FIXTURE
#ifdef ASPECT_BULK_CONTINUED_FIXTURE
        // Preserve complete DG/native-FE/local/phase owners around any abrupt
        // response for a later temporal study; no trial is published here.
        if(dense || s.time>=.15-1e-12)
#else
        if(dense)
#endif
          {
#endif
        const auto record=dg_local_phase_record();auto bytes=output("state-"+label+".bin");
        bytes.write(reinterpret_cast<const char*>(record.data()),record.size()*sizeof(double));
        auto memory=output("native-history-"+label+".bin");
        const auto &velocity_memory=s.native_history.block(this->introspection().block_indices.velocities);
        for(unsigned int i=0;i<velocity_memory.size();++i){const double value=velocity_memory[i];memory.write(reinterpret_cast<const char*>(&value),sizeof(double));}
        auto native=output("solution-"+label+".bin");
        for(unsigned int i=0;i<sim().solution.size();++i){const double value=sim().solution[i];native.write(reinterpret_cast<const char*>(&value),sizeof(double));}
#ifdef ASPECT_BULK_ENERGY_FIXTURE
          }
#endif
        auto metrics=std::ofstream(this->get_output_directory()+"fixture.csv",std::ios::app);
        if(metrics.tellp()==0)metrics<<"step,time,dt,publications,max_phase,max_driving,points,cells,DG_DoFs\n";
        metrics<<std::setprecision(17)<<s.step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<s.publications<<','<<*std::max_element(s.phase.begin(),s.phase.end())<<','<<maxdrive<<','<<quad.size()*s.history.size()<<','<<s.history.size()<<','<<3*s.tensor->space->dofs.n_dofs()<<'\n';
      }
      public:
        void initialize()override
        {
          AssertThrow(model().uses_bulk_cohesive_phase(),ExcMessage("Bulk fixture requires the default-off bulk model."));
          this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&){seed();});
          this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&)
            {if(!recovery_done && std::getenv("ASPECT_BULK_RECOVERY") && this->get_time()>.119)recovery();});
        }
        std::pair<std::string,std::string> execute(TableHandler&)override
        {
          const auto &s=*model().bulk_cohesive_state;
          check("single accepted physical publication",s.publications==this->get_timestep_number()
                && (s.publications==0 || s.publications==last_publications+1));
          last_publications=s.publications;
#ifdef ASPECT_BULK_ENERGY_FIXTURE
          boundary_work();
#endif
          export_state();
          if(std::getenv("ASPECT_BULK_VERIFY_SOURCE_HISTORY"))source_checks();
          return {"Bulk cohesive native fixture","accepted"};
        }
        void save(std::map<std::string,std::string> &status)const override
          {status["bulk fixture publications"]=std::to_string(last_publications);status["bulk recovery done"]=std::to_string(recovery_done);status["bulk source failure checked"]=std::to_string(source_failure_checked);}
        void load(const std::map<std::string,std::string> &status)override
          {last_publications=std::stoul(status.at("bulk fixture publications"));recovery_done=std::stoul(status.at("bulk recovery done"));
           source_failure_checked=status.count("bulk source failure checked") && std::stoul(status.at("bulk source failure checked"));}
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"bulk cohesive checks","Observer of native bulk cohesive state and lifecycle.")
  }
}
