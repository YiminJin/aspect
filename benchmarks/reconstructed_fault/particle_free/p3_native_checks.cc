/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Frozen geometry transaction, using native material/manager/checkpoint owners.
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/phase_field.h>
#include <aspect/particle/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

namespace aspect { namespace Postprocess
{
  // Existing narrow test friend; this DSO is never loaded with other fixtures
  // defining MomentCycle. It contains no alternate persistent history owner.
  template<int dim> class MomentCycle:public Interface<dim>,public SimulatorAccess<dim>
  {
    using Model=MaterialModel::PhaseFieldFault<dim>;
    using BV=LinearAlgebra::BlockVector;
    unsigned int slip_property=0;
    bool sensitive=false;
    Model &model()const{return const_cast<Model &>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));}
    Simulator<dim> &sim()const{return const_cast<Simulator<dim>&>(this->get_simulator());}
    std::ofstream output(const std::string &name)const
    {std::ofstream f(this->get_output_directory()+name);f<<std::setprecision(17);return f;}
    void gate(const std::string &name,bool pass)const
    {
      std::ofstream f(this->get_output_directory()+"gates.csv",std::ios::app);
      if(f.tellp()==0)f<<"gate,passed\n";
      f<<name<<','<<pass<<std::endl;AssertThrow(pass,ExcMessage("P3 native gate: "+name));
    }
    template<class F> bool rejects(F f,const std::string &required="")const
    {
      dealii::deal_II_exceptions::disable_abort_on_exception();
      bool result=false;
      try{f();}catch(const std::exception &error)
        {
          result=required.empty() || std::string(error.what()).find(required)!=std::string::npos;
          std::ofstream out(this->get_output_directory()+"expected-rejections.txt",std::ios::app);
          out<<error.what()<<'\n';
        }
      dealii::deal_II_exceptions::enable_abort_on_exception();
      return result;
    }
    static double phase(const Point<dim> &x)
    {
      static const PhaseField::GeometricFunction a(.15625,1.,8./3.);
      static const PhaseField::DegradationFunction g(1.,128.);
      static const PhaseField::PhaseFieldProfile profile(a,g,.6);
      return profile.value(std::hypot(std::max(0.,x[0]-.5),x[1]-.5));
    }
    static std::array<double,3> tensor(const Point<dim> &p)
    {
      const double x=p[0],y=p[1];
      return {{400+71*x+33*y+19*x*y+11*x*x+13*y*y+7*x*x*y+5*x*y*y+3*x*x*y*y,
        -130+17*x-21*y+9*x*y+5*x*x-7*y*y-4*x*x*y+3*x*y*y+2*x*x*y*y,
        800-29*x+41*y-17*x*y+23*x*x+31*y*y+11*x*x*y-13*x*y*y+7*x*x*y*y}};
    }
    static bool same(const std::vector<double>&a,const std::vector<double>&b)
    {return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;}
    std::vector<double> persistent()const
    {
      const auto &m=model();const auto &s=*m.fe_strain_state;
      auto v=m.dg_history_values();v.insert(v.end(),s.Z.begin(),s.Z.end());v.insert(v.end(),s.I.begin(),s.I.end());
      for(unsigned int i=0;i<s.displacement_history.size();++i)v.push_back(s.displacement_history[i]);
      for(const auto *field:{&sim().solution,&sim().old_solution,&sim().old_old_solution})
        for(unsigned int i=0;i<field->size();++i)v.push_back((*field)[i]);
      v.push_back(s.accepted_publications);v.push_back(s.geometry_publications);v.push_back(s.committed_step);
      const auto &manager=this->get_reconstructed_fault_manager();
      for(unsigned int j=0;j<s.Z.size();++j)
        {
          for(double x:manager.get_fault(0).get_properties(j))v.push_back(x);
          for(unsigned int d=0;d<dim;++d)v.push_back(manager.get_fault(0).vertex(j)[d]);
        }
      for(double x:manager.get_slip_rate(0))v.push_back(x);
      for(double x:manager.get_timestep_committed_slip_rate(0))v.push_back(x);
      for(const auto &p:this->get_particle_manager(0).get_particle_handler())
        {v.push_back(p.get_id());for(unsigned int d=0;d<dim;++d)v.push_back(p.get_location()[d]);for(double x:p.get_properties())v.push_back(x);}
      return v;
    }
    void export_state(const std::string &name)
    {
      auto &m=model();const auto &s=*m.fe_strain_state;auto &manager=this->get_reconstructed_fault_manager();
      auto constraint_rows=output(name+"-constraints.txt");
      this->get_current_constraints().print(constraint_rows);
      auto out=output(name+"-surface.csv");out<<"vertex,x,y,Z,cumulative_slip,rate,I,q\n";
      const auto cp=manager.get_property_information()[m.fault_property_indices.cohesive_traction].position;
      const auto sp=manager.get_property_information()[slip_property].position;
      for(unsigned int j=0;j<s.Z.size();++j)
        out<<j<<','<<s.vertices[j][0]<<','<<s.vertices[j][1]<<','<<s.Z[j]<<','<<manager.get_fault(0).get_properties(j)[sp]<<','
           <<manager.get_timestep_committed_slip_rate(0)[j]<<','<<s.I[j]<<','<<manager.get_fault(0).get_properties(j)[cp]<<'\n';
      auto cells=output(name+"-cells-0.csv");cells<<"cell,x0,y0,hx,hy";
      for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)cells<<",c"<<c<<'_'<<i;
      cells<<'\n';
      const auto &space=*s.dg_history->space;std::vector<types::global_dof_index> ids(9);
      std::map<std::pair<double,double>,std::pair<double,double>> nodes;
      const auto pc=this->introspection().variable("phase_field").first_component_index;
      for(const auto &entry:s.cells)
        {
          const auto cell=entry.second;space.cells.at(entry.first)->get_dof_indices(ids);
          cells<<entry.first.to_string()<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]<<','
            <<cell->vertex(1)[0]-cell->vertex(0)[0]<<','<<cell->vertex(2)[1]-cell->vertex(0)[1];
          for(const auto &component:s.dg_history->trial_stress)for(auto i:ids)cells<<','<<component[i];
          cells<<'\n';
          for(auto v:cell->vertex_indices())
            {auto i=cell->vertex_dof_index(v,pc);auto p=cell->vertex(v);nodes[{p[1],p[0]}]={sim().solution[i],s.displacement_history[i]};}
        }
      auto phases=output(name+"-phase.csv");phases<<"x,y,phi,obstacle\n";
      for(const auto &row:nodes)phases<<row.first.second<<','<<row.first.first<<','<<row.second.first<<','<<row.second.second<<'\n';
      auto basis=output("basis-0.csv");basis<<"i,u,v\n";
      for(unsigned int i=0;i<9;++i)basis<<i<<','<<space.element.get_unit_support_points()[i][0]<<','<<space.element.get_unit_support_points()[i][1]<<'\n';
      // Reconstruct independently through the actual material point sampler.
      auto stress=output(name+"-stress.csv");stress<<"cell,x,y,Txx,Tyy,Txy,tauxx,tauyy,tauxy\n";
      const QGauss<dim> samples(3);
      for(const auto &entry:s.cells)
        {
          const auto T=m.evaluate_dg_trial_tensor(*s.dg_history,entry.first,samples);
          for(unsigned int q=0;q<samples.size();++q)
            {
              const auto x=this->get_mapping().transform_unit_to_real_cell(entry.second,samples.point(q));
              const auto tau=m.evaluate_dg_history_stress(*s.dg_history,s.Z,s.I,entry.first,x);
              stress<<entry.first.to_string()<<','<<x[0]<<','<<x[1]<<','<<T[q][0][0]<<','<<T[q][1][1]<<','<<T[q][0][1]<<','
                <<tau[0][0]<<','<<tau[1][1]<<','<<tau[0][1]<<'\n';
            }
        }
      // Bind the canonical simulator-owned surface object, not a fixture weak
      // functional. Commit must discard it before its geometry can be reused.
      sim().reconstructed_fault_surface_system=std::make_unique<ReconstructedFaultSurfaceSystem<dim>>(sim());
      auto &surface=*sim().reconstructed_fault_surface_system;
      // Birth records retain exactly V=0. Native friction evaluation requires
      // V>=Vmin; this fixture has identically mu_s=mu_d=.1, damping=0 and dt=0,
      // so every admissible observation rate gives exactly the same loads.
      // This temporary argument is never assigned to any persistent rate.
      auto observation_rates=manager.get_slip_rate(0);
      for(auto &rate:observation_rates)rate=std::max(rate,m.fault_friction.get_minimum_slip_rate());
      const auto loads=surface.evaluate_accepted_cohesive_loads(sim().solution,{observation_rates});
      auto weak=output(name+"-native-loads.csv");weak<<"vertex,shear,cohesive,friction,residual\n";
      for(unsigned int j=0;j<s.Z.size();++j)weak<<j<<','<<loads.shear_traction[0][j]<<','<<loads.cohesive_traction[0][j]<<','<<loads.friction_traction[0][j]<<','<<loads.values[0][j]<<'\n';
      const auto energy=m.evaluate_fe_phase_energy(sim().solution,false,false);
      const auto fracture=this->get_phase_field_handler().evaluate_cell_fracture(sim().solution,false);
      auto ledger=output(name+"-ledger.csv");ledger<<"bulk,cohesive,fracture,geometry_publications,physical_publications,step,time,dt,particles,bulk_cells,constraints\n";
      ledger<<energy.bulk<<','<<energy.cohesive<<','<<fracture.energy<<','<<s.geometry_publications<<','<<s.accepted_publications<<','
        <<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<s.n_particles_global<<','<<s.n_cells_global<<','<<this->get_current_constraints().n_constraints()<<'\n';
      const auto state_values=persistent();
      auto all=output(name+"-persistent.csv");all<<"value\n";for(double value:state_values)all<<value<<'\n';
      std::ofstream bytes(this->get_output_directory()+name+"-persistent.bin",std::ios::binary);
      bytes.write(reinterpret_cast<const char *>(state_values.data()),state_values.size()*sizeof(double));
    }
    void initialize_loaded()
    {
      auto &m=model();auto &v=sim();m.initialize_fe_shared_phase(v.solution);
      v.old_solution=v.solution;v.old_old_solution=v.solution;v.current_linearization_point=v.solution;
      auto &manager=this->get_reconstructed_fault_manager();manager.reconstruct_initial_faults();m.initialize_prescribed_dg_geometry();
      auto &s=*m.fe_strain_state;auto &space=*s.dg_history->space;
      std::array<LinearAlgebra::Vector,3> owned;for(auto &x:owned)x.reinit(space.dofs.locally_owned_dofs(),this->get_mpi_communicator());
      std::vector<types::global_dof_index> ids(9);
      for(const auto &entry:s.cells)
        {space.cells.at(entry.first)->get_dof_indices(ids);for(unsigned int i=0;i<9;++i)
            {const auto x=this->get_mapping().transform_unit_to_real_cell(entry.second,space.element.get_unit_support_points()[i]);auto T=tensor(x);for(unsigned int c=0;c<3;++c)owned[c][ids[i]]=T[c];}}
      for(unsigned int c=0;c<3;++c){owned[c].compress(VectorOperation::insert);s.dg_history->trial_stress[c]=owned[c];}
      const auto cp=manager.get_property_information()[m.fault_property_indices.cohesive_traction].position;
      const auto sp=manager.get_property_information()[slip_property].position;
      std::vector<double> rates(s.Z.size());
      for(unsigned int j=0;j<s.Z.size();++j)
        {
          const double x=s.vertices[j][0];double z=sensitive?1000*std::min({1.,x/(2./64),(.5-x)/(2./64)}):1000*std::pow(std::sin(2*numbers::PI*x),2);
          if(j==0 || j+1==s.Z.size())z=0.;
          s.Z[j]=z;
          manager.get_fault(0).get_properties(j)[cp]=z/s.I[j];
          manager.get_fault(0).get_properties(j)[sp]=2*z/1e6;rates[j]=z*1e-8;
        }
      manager.initialize_slip_rate(0,rates);
      m.validate_fe_strain_restrictions();
    }
    void append()
    {
      auto &m=model();auto &manager=this->get_reconstructed_fault_manager();
      const auto parent=persistent();const auto incoming=m.fe_strain_state;
      const auto identity=m.prepare_dg_geometry_append({},{},{},{});
      sim().commit_reconstructed_fault_geometry(identity);
      gate("no-extension native identity",same(parent,persistent()));
      export_state("old");sim().create_snapshot();
      std::vector<Point<dim>> nodes;for(unsigned int j=33;j<=48;++j){Point<dim>x;x[0]=j/64.;x[1]=.5;nodes.push_back(x);}
      const unsigned int width=manager.get_fault(0).get_properties(0).size();
      gate("old tip has zero cohesive slip and accumulated slip",
           incoming->Z.back()==0. && manager.get_fault(0).get_properties(incoming->Z.size()-1)
           [manager.get_property_information()[slip_property].position]==0.);
      // Deliberate inadmissible zero-phase tip: native normalization must fail
      // after candidate geometry is bound, and restore every accepted owner.
      auto unsupported=nodes;
      for(unsigned int j=49;j<=56;++j){Point<dim>x;x[0]=j/64.;x[1]=.5;unsupported.push_back(x);}
      gate("zero-support append preparation rejects",rejects([&](){m.prepare_dg_geometry_append(
        unsupported,std::vector<double>(unsupported.size(),.5),
        std::vector<std::vector<double>>(unsupported.size(),std::vector<double>(width)),std::vector<double>(unsupported.size()));}));
      gate("preparation exception rolls back native owners",same(parent,persistent()));
      const auto make=[&](){return m.prepare_dg_geometry_append(nodes,std::vector<double>(nodes.size(),.5),std::vector<std::vector<double>>(nodes.size(),std::vector<double>(width)),std::vector<double>(nodes.size()));};
      auto rejected=make();gate("prepare leaves all accepted owners unchanged",same(parent,persistent()));
      const auto rejected_values=[&](){std::vector<double> a;for(const auto &field:rejected->next->dg_history->trial_stress)for(unsigned int i=0;i<field.size();++i)a.push_back(field[i]);return a;}();
      // Inject an invalid acceptance precondition after complete preparation.
      // M5 must reject it before touching any actual owner or canonical cache.
      rejected->next->accepted_publications=1;
      gate("native rejected commit",rejects([&](){sim().commit_reconstructed_fault_geometry(rejected);}));
      gate("rejection preserves all native owners",same(parent,persistent()));rejected.reset();
      auto retry=make();std::vector<double> retry_values;for(const auto &field:retry->next->dg_history->trial_stress)for(unsigned int i=0;i<field.size();++i)retry_values.push_back(field[i]);
      gate("deterministic retry candidate",same(rejected_values,retry_values));
      gate("candidate phase obstacle exact",retry->next->displacement_history==incoming->displacement_history);
      sim().commit_reconstructed_fault_geometry(retry);
      gate("geometry caches expired before reuse",!sim().reconstructed_fault_surface_system && !sim().reconstructed_fault_stokes_coupling && !sim().endpoint_velocity_inverse && !sim().endpoint_bulk_direct);
      gate("exactly one geometry publication and no physical publication",m.fe_strain_state->geometry_publications==1 && m.fe_strain_state->accepted_publications==0 && m.fe_strain_state->committed_step==0);
      gate("old Z preserved",std::equal(incoming->Z.begin(),incoming->Z.end(),m.fe_strain_state->Z.begin()));
      gate("duplicate commit rejected",rejects([&](){sim().commit_reconstructed_fault_geometry(retry);}));
      gate("physical finite-tip evolution rejected",rejects([&](){sim().solve_reconstructed_fault_endpoint_phase();},"Finite-tip"));
      gate("physical mechanical solve rejected",rejects([&](){m.prepare_reconstructed_fault_mechanical_solve();},"Finite-tip"));
      export_state("new");sim().create_snapshot();
    }
  public:
    void initialize()override
    {
      AssertThrow(dim==2,ExcNotImplemented());
      if(const auto *name=std::getenv("ASPECT_P3_CASE"))sensitive=std::string(name)=="sensitive";
      model().enable_prescribed_dg_geometry(phase);
      slip_property=this->get_reconstructed_fault_manager().register_property("P3 accumulated slip",1);
    }
    std::pair<std::string,std::string> execute(TableHandler &)override
    {
      if(!this->get_parameters().resume_computation)initialize_loaded();
      else
        {
          gate("native restart finite-tip evolution rejected",rejects([&](){sim().solve_reconstructed_fault_endpoint_phase();},"Finite-tip"));
          export_state("restored");
        }
      if(model().fe_strain_state->geometry_publications==0)append();
      else gate("restored single geometry publication",model().fe_strain_state->geometry_publications==1 && model().fe_strain_state->accepted_publications==0);
      return {"P3 native frozen geometry","complete"};
    }
    void save(std::map<std::string,std::string> &status)const override{status["P3 prescribed geometry profile"]=sensitive?"sensitive":"original";}
    void load(const std::map<std::string,std::string> &status)override{sensitive=status.at("P3 prescribed geometry profile")=="sensitive";}
  };
  ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"P3 native geometry","Frozen finite-tip append in native DG, cohesive, physical obstacle and simulator restart owners; no evolution.")
}}
