// Fixed-mesh benchmark only. Native history is deliberately cell-local.
#include "clean_stress_cycle.cc"
#include <aspect/simulator.h>
#include <deal.II/base/mpi.h>
#include <cstdint>
#include <cstring>

namespace aspect::Postprocess
{
  template <int dim>
  class MomentCycle : public CleanStressCycle<dim>
  {
    using Tensor = SymmetricTensor<2,dim>;
    using Field = std::map<CellId,std::vector<Tensor>>;
    struct CellGeometry { Point<dim> lo,hi; };
    struct Loads
    {
      LinearAlgebra::BlockVector free;
      double absolute=0.,top=0.,bottom=0.;
    };
    std::string mode;
    double eta=0.,G=0.,initial_dt=0.;
    Field native;
    std::map<CellId,CellGeometry> geometry;
    unsigned int last_commit=0,newton=0,krylov=0;
    double alpha=1.,relative=0.;
    std::vector<double> initial_invariants;

    auto &model() const
    {
      return const_cast<MaterialModel::PhaseFieldFault<dim>&>(
        Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()));
    }
    static double lagrange(const double x,const unsigned int i)
    {
      const QGauss<1> q(3);double v=1.;
      for (unsigned int j=0;j<3;++j) if (j!=i) v*=(x-q.point(j)[0])/(q.point(i)[0]-q.point(j)[0]);
      return v;
    }
    Tensor sample(const Field &field,const CellId &id,const Point<dim> &p) const
    {
      const auto &g=geometry.at(id);const auto &v=field.at(id);Tensor answer;
      const double x=(p[0]-g.lo[0])/(g.hi[0]-g.lo[0]), y=(p[1]-g.lo[1])/(g.hi[1]-g.lo[1]);
      for (unsigned int j=0;j<3;++j) for (unsigned int i=0;i<3;++i)
        answer+=lagrange(x,i)*lagrange(y,j)*v[3*j+i];
      return answer;
    }
    Field read_history(const LinearAlgebra::BlockVector &state) const
    {
      const auto &intro=this->introspection();const QGauss<dim> q(3);
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),q,update_values);
      Field result;std::array<std::vector<double>,3> values;
      for (auto &v:values) v.resize(q.size());
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
        {
          fe.reinit(cell);auto &r=result[cell->id()];r.resize(q.size());
          for (unsigned int c=0;c<3;++c)
            fe[intro.extractors.compositional_fields[c]].get_function_values(state,values[c]);
          for (unsigned int k=0;k<q.size();++k)
            {r[k][0][0]=values[0][k];r[k][1][1]=values[1][k];r[k][0][1]=values[2][k];}
        }
      return result;
    }
    Field read_native() const
    {
      Field result;
      const QGauss<dim> q(3);
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),q,update_quadrature_points);
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
        {fe.reinit(cell);for (unsigned int k=0;k<q.size();++k)
          result[cell->id()].push_back(model().benchmark_retained_stress(cell->id(),fe.quadrature_point(k)));}
      return result;
    }
    Loads loads(const Field &field,const bool reverse=false) const
    {
      const auto &intro=this->introspection();const auto &element=this->get_fe();
      FEValues<dim> fe(this->get_mapping(),element,QGauss<dim>(3),update_gradients|update_JxW_values);
      Loads r;r.free.reinit(intro.index_sets.system_partitioning,this->get_mpi_communicator());
      Vector<double> local(element.n_dofs_per_cell());std::vector<types::global_dof_index> indices(local.size());
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
        {
          fe.reinit(cell);cell->get_dof_indices(indices);local=0.;const auto &T=field.at(cell->id());
          for (unsigned int i=0;i<local.size();++i)
            if (intro.component_masks.velocities[element.system_to_component_index(i).first])
              {
                for (unsigned int index=0;index<T.size();++index)
                  {
                    const unsigned int q=reverse?T.size()-1-index:index;
                    const double value=fe.JxW(q)*(T[q]*fe[intro.extractors.velocities].symmetric_gradient(i,q));
                    local[i]+=value;r.absolute+=std::abs(value);
                  }
                if (element.system_to_component_index(i).first==intro.component_indices.velocities[0])
                  {
                    const auto p=this->get_mapping().transform_unit_to_real_cell(cell,element.get_unit_support_points()[i]);
                    if (std::abs(p[1]-.5)<1e-13) r.top+=local[i];
                    if (std::abs(p[1]+.5)<1e-13) r.bottom+=local[i];
                  }
              }
          // Vector-only distribution implements C_v^T (no inhomogeneous lift).
          this->get_current_constraints().distribute_local_to_global(local,indices,r.free);
        }
      r.free.compress(VectorOperation::add);
      for (double *v:{&r.absolute,&r.top,&r.bottom}) *v=Utilities::MPI::sum(*v,this->get_mpi_communicator());
      return r;
    }
    Field subtract(const Field &a,const Field &b) const
    {
      Field r=a;for (auto &[id,v]:r) for (unsigned int q=0;q<v.size();++q) v[q]-=b.at(id)[q];return r;
    }
    // Only the diagnostic copy is transferred. Restore live published data on
    // both success and failure; no aging or mechanics is called by this probe.
    Field production_transfer()
    {
      auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
      const LinearAlgebra::BlockVector saved(sim.solution);
      try
        {
          this->get_particle_manager(0).get_particle_handler().exchange_ghost_particles();
          sim.interpolate_particle_properties({AdvectionField::composition(0),AdvectionField::composition(1),AdvectionField::composition(2)});
          LinearAlgebra::BlockVector owned;
          owned.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          owned=sim.solution;this->get_current_constraints().distribute(owned);
          LinearAlgebra::BlockVector working(sim.solution);working=owned;
          Field r=read_history(working);sim.solution=saved;return r;
        }
      catch (...) {sim.solution=saved;throw;}
    }
  public:
    static void declare_parameters(ParameterHandler &prm)
    {
      // CleanStressCycle is also registered in this library and declares its
      // own entries. This class adds only the explicitly selected history mode.
      prm.enter_subsection("Postprocess");prm.enter_subsection("Moment cycle");
      prm.declare_entry("History mode","production",Patterns::Selection("production|native_history_reference|horizontal_moment_update"));
      prm.leave_subsection();prm.leave_subsection();
    }
    void parse_parameters(ParameterHandler &prm) override
    {
      CleanStressCycle<dim>::parse_parameters(prm);
      prm.enter_subsection("Postprocess");prm.enter_subsection("Moment cycle");mode=prm.get("History mode");
      prm.leave_subsection();prm.leave_subsection();
      prm.enter_subsection("Material model");prm.enter_subsection("Phase field fault");
      eta=prm.get_double("Reference viscosities");G=prm.get_double("Elastic shear moduli");initial_dt=prm.get_double("Initial time step");
      prm.leave_subsection();prm.leave_subsection();
    }
    void initialize() override
    {
      CleanStressCycle<dim>::initialize();
      AssertThrow(this->compact,ExcMessage("Moment cycle requires compact output."));
      AssertThrow(!this->advect_particles || mode=="production",
                  ExcMessage("The advecting comparison must use production particle history."));
      AssertThrow(this->fault_angle==0. || mode!="horizontal_moment_update",
                  ExcMessage("The horizontal moment update is not an inclined-fault method."));
      this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&)
      {
        for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
          {
            const auto ends=cell->bounding_box().get_boundary_points();
            geometry[cell->id()]={ends.first,ends.second};native[cell->id()]=std::vector<Tensor>(9);
          }
        if (mode=="native_history_reference")
          model().benchmark_retained_stress=[this](const CellId &id,const Point<dim> &p){return sample(native,id,p);};
      });
      this->get_signals().post_reconstructed_fault_solver.connect([this](unsigned int n,unsigned int k,double a,const auto &)
      {newton=n;krylov=k;alpha=a;});
      this->get_signals().post_nonlinear_solver.connect([this](const SolverControl &c){relative=c.last_value();});
    }

    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      AssertThrow(this->converged,ExcMessage("Moment cycle did not converge."));
      const auto step=this->get_timestep_number();
      AssertThrow(step==0 || step==last_commit+1,ExcMessage("Repeated/skipped moment publication."));
      this->write_particles("after");
      // Verify realized normal flux, not just the prescribed function text.
      // Periodic side faces remain geometrical boundary faces in this mesh.
      std::array<double,4> boundary_flux{};
      FEFaceValues<dim> face(this->get_mapping(),this->get_fe(),QGauss<dim-1>(3),
                            update_values|update_normal_vectors|update_JxW_values);
      std::vector<dealii::Tensor<1,dim>> face_velocity(face.n_quadrature_points);
      for (const auto &cell:this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int f:cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                face.reinit(cell,f);
                face[this->introspection().extractors.velocities].get_function_values(this->get_solution(),face_velocity);
                for (unsigned int q=0;q<face.n_quadrature_points;++q)
                  boundary_flux.at(cell->face(f)->boundary_id())+=face.JxW(q)*(face_velocity[q]*face.normal_vector(q));
              }
      for (double &v:boundary_flux) v=Utilities::MPI::sum(v,this->get_mpi_communicator());
      if (this->get_pcout().is_active())
        {
          std::ofstream out(this->get_output_directory()+"boundary_flux.csv",step?std::ios::app:std::ios::out);
          out<<std::setprecision(17);
          if (!step) out<<"step,left,right,bottom,top,total\n";
          out<<step;double total=0.;
          for (const double v:boundary_flux) {out<<','<<v;total+=v;}
          out<<','<<total<<'\n';
        }
      // Particles do not yet exist at post_set_initial_state. Hash the retained
      // initial data after timestep zero, which does not evolve stress/state.
      if (step==0)
        {
          std::uint64_t hash=0,count=0;
          for (const auto &p:this->get_particle_manager(0).get_particle_handler())
            {
              std::uint64_t h=1469598103934665603ULL;
              auto add=[&](double v){std::uint64_t bits;std::memcpy(&bits,&v,8);h^=bits;h*=1099511628211ULL;};
              for (unsigned int d=0;d<dim;++d) add(p.get_location()[d]);
              for (const auto v:p.get_properties()) add(v);
              hash+=h;++count;
            }
          hash=Utilities::MPI::sum(hash,this->get_mpi_communicator());
          count=Utilities::MPI::sum(count,this->get_mpi_communicator());
          AssertThrow(count>0,ExcMessage("Initial particle hash has no particles."));
          if (this->get_pcout().is_active())
            {std::ofstream out(this->get_output_directory()+"initial_hash.txt");out<<count<<' '<<hash<<'\n';}
        }
      const auto &intro=this->introspection();const auto &manager=this->get_reconstructed_fault_manager();
      if (step==0 && this->get_pcout().is_active())
        {
          std::ofstream out(this->get_output_directory()+"finite_elements.txt");
          out<<this->get_fe().get_name()<<'\n';
          for (unsigned int c=0;c<intro.n_compositional_fields;++c)
            out<<intro.name_for_compositional_index(c)<<' '
               <<this->get_fe().base_element(intro.base_elements.compositional_fields[c]).get_name()<<'\n';
        }
      const auto &q=intro.quadratures.velocities;AssertThrow(q.size()==9,ExcMessage("Moment cycle needs native 3x3 quadrature."));
      const double dt=step?this->get_timestep():initial_dt;
      AssertThrow(dt==.1,ExcMessage("Moment-cycle physical clock changed."));
      const auto coefficients=model().compute_maxwell_coefficients(eta,G,dt);
      Field incoming=mode=="native_history_reference"?read_native():read_history(this->get_current_linearization_point());
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),q,update_values|update_gradients|update_quadrature_points|update_JxW_values);
      std::vector<Tensor> strain(q.size());
      std::vector<dealii::Tensor<2,dim>> gradients(q.size());std::vector<dealii::Tensor<1,dim>> velocities(q.size());
      std::vector<double> phi(q.size()),pressure(q.size());Field current,projected;
      std::map<CellId,std::pair<Tensor,Tensor>> fits;
      double mean=0.,volume=0.,rough=0.,removed=0.,moment0=0.,moment1=0.,homogeneity=0.;
      std::vector<double> row_min(64*3*3,std::numeric_limits<double>::max()),row_max(64*3*3,-std::numeric_limits<double>::max());
      std::vector<double> invariants;
      std::ofstream coefficient_output;
      if (this->advect_particles)
        coefficient_output.open(this->get_output_directory()+"coefficients_"+std::to_string(step)+"_rank"+this->rank()+".bin",std::ios::binary);
      std::ofstream fields(this->get_output_directory()+"fields_"+std::to_string(step)+"_rank"+this->rank()+".bin",std::ios::binary);
      std::ofstream probe(this->get_output_directory()+"probe_"+std::to_string(step)+"_rank"+this->rank()+".csv");
      probe<<std::setprecision(17)<<"cell,q,x,y,Txx,Tyy,Txy,fitxx,fityy,fitxy\n";
      std::ofstream traction;
      if (this->fault_angle!=0.)
        {
          traction.open(this->get_output_directory()+"traction_"+std::to_string(step)+"_rank"+this->rank()+".csv");
          traction<<std::setprecision(17)<<"cell,q,x,y,w,chi,segment,xi,nx,ny,p,d,sigma,old_xx,old_yy,old_xy\n";
          if (step==0 && this->get_pcout().is_active())
            {
              std::ofstream out(this->get_output_directory()+"geometry.csv");out<<std::setprecision(17)<<"vertex,x,y\n";
              const auto &fault=manager.get_fault(0);
              for (unsigned int i=0;i<fault.n_vertices();++i)
                out<<i<<','<<fault.vertex(i)[0]<<','<<fault.vertex(i)[1]<<'\n';
            }
        }
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
        {
          fe.reinit(cell);fe[intro.extractors.velocities].get_function_symmetric_gradients(this->get_solution(),strain);
          fe[intro.extractors.velocities].get_function_gradients(this->get_solution(),gradients);
          fe[intro.extractors.velocities].get_function_values(this->get_solution(),velocities);
          fe[intro.extractors.pressure].get_function_values(this->get_solution(),pressure);
          fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(this->get_solution(),phi);
          const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),q,fe.get_quadrature_points());
          auto &T=current[cell->id()];T.resize(q.size());Tensor b0,b1;double m0=0,m1=0,m2=0.;
          for (unsigned int k=0;k<q.size();++k)
            {
              Tensor crack;double chi=0.,V=0.;
              if (associations[k].active)
                {
                  const auto &a=associations[k];typename MaterialModel::PhaseFieldFault<dim>::ReconstructedFaultBulkPointInputs in;
                  in.fault_index=a.fault_index;in.segment_index=a.segment_index;in.xi=a.xi;
                  in.phase_field=in.previous_phase_field=phi[k];in.temperature=293.;in.bulk_material_fractions={1.};
                  const auto response=model().evaluate_reconstructed_fault_bulk_point(in);
                  AssertThrow(response.eta_ve==coefficients.eta_ve && response.history_correction==0.,ExcMessage("Coefficient/source mismatch."));
                  chi=response.localization_factor;V=manager.interpolate_slip_rate(a.fault_index,a.segment_index,a.xi);
                  crack=chi*V*symmetrize(outer_product(a.tangent,a.normal));
                }
              T[k]=model().compute_maxwell_stress(coefficients,strain[k]-crack,incoming.at(cell->id())[k]);
              if (this->fault_angle!=0. && associations[k].active)
                {
                  const auto &a=associations[k];const auto p=fe.quadrature_point(k);
                  const double d=-(a.normal*(T[k]*a.normal));
                  traction<<cell->id()<<','<<k<<','<<p[0]<<','<<p[1]<<','<<fe.JxW(k)<<','<<chi
                          <<','<<a.segment_index<<','<<a.xi<<','<<a.normal[0]<<','<<a.normal[1]
                          <<','<<pressure[k]<<','<<d<<','<<pressure[k]+d;
                  const auto &old=incoming.at(cell->id())[k];
                  traction<<','<<old[0][0]<<','<<old[1][1]<<','<<old[0][1]<<'\n';
                }
              const unsigned int row=static_cast<unsigned int>(std::lround((cell->center()[1]+.5)*64-.5))*3+k/3;
              for (unsigned int c=0;c<3;++c)
                {const double v=T[k][Tensor::unrolled_to_component_indices(c)];
                 row_min[3*row+c]=std::min(row_min[3*row+c],v);row_max[3*row+c]=std::max(row_max[3*row+c],v);}
              const double w=fe.JxW(k),xi=q.point(k)[1]-.5;
              m0+=w;m1+=w*xi;m2+=w*xi*xi;b0+=w*T[k];b1+=w*xi*T[k];
              const auto p=fe.quadrature_point(k);
              for (const double v:{p[0],p[1],phi[k],chi,V}) invariants.push_back(v);
              if (this->advect_particles)
                {
                  const double data[]={p[0],p[1],phi[k],chi,V,coefficients.beta,coefficients.eta_ve};
                  coefficient_output.write(reinterpret_cast<const char*>(data),sizeof(data));
                }
              // Compact matched-point fields: x,y,w,u_x,u_y, four gradients,
              // current tensor and incoming tensor. All doubles, 15 columns.
              const double data[]={p[0],p[1],w,velocities[k][0],velocities[k][1],gradients[k][0][0],gradients[k][0][1],gradients[k][1][0],gradients[k][1][1],T[k][0][0],T[k][1][1],T[k][0][1],incoming.at(cell->id())[k][0][0],incoming.at(cell->id())[k][1][1],incoming.at(cell->id())[k][0][1]};
              fields.write(reinterpret_cast<const char*>(data),sizeof(data));
            }
          const double determinant=m0*m2-m1*m1;
          const Tensor A=(m2*b0-m1*b1)/determinant,B=(m0*b1-m1*b0)/determinant;
          fits[cell->id()]={A,B};auto &fit=projected[cell->id()];fit.resize(q.size());Tensor e0,e1;
          for (unsigned int k=0;k<q.size();++k)
            {
              const double w=fe.JxW(k),xi=q.point(k)[1]-.5;fit[k]=A+xi*B;
              e0+=w*(fit[k]-T[k]);e1+=w*xi*(fit[k]-T[k]);
              rough+=w*((T[k]-b0/m0)*(T[k]-b0/m0));removed+=w*((T[k]-fit[k])*(T[k]-fit[k]));
              mean+=w*T[k][0][1];volume+=w;
              // Tangential symmetry within each cell (also checked globally offline).
              homogeneity=std::max(homogeneity,(T[k]-T[3*(k/3)]).norm());
              if (std::abs(cell->center()[0]-.125)<.016 && std::abs(cell->center()[1]-.0546875)<.008)
                {const auto p=fe.quadrature_point(k);probe<<cell->id()<<','<<k<<','<<p[0]<<','<<p[1];
                 for (const auto &t:{T[k],fit[k]}) probe<<','<<t[0][0]<<','<<t[1][1]<<','<<t[0][1];
                 probe<<'\n';}
            }
          moment0=std::max(moment0,(e0/m0).norm());moment1=std::max(moment1,(e1/m0).norm());
        }
      if (step==0) initial_invariants=invariants;
      AssertThrow(initial_invariants==invariants,ExcMessage("Phase/source/geometry changed."));
      homogeneity=Utilities::MPI::max(homogeneity,this->get_mpi_communicator());
      Utilities::MPI::min(row_min,this->get_mpi_communicator(),row_min);
      Utilities::MPI::max(row_max,this->get_mpi_communicator(),row_max);
      for (unsigned int i=0;i<row_min.size();++i) homogeneity=std::max(homogeneity,row_max[i]-row_min[i]);
      if (step && mode=="horizontal_moment_update") AssertThrow(homogeneity<1e-5,ExcMessage("Horizontal fit lacks tangential symmetry."));
      const auto current_load=loads(current),fit_load=loads(projected);
      Field next=incoming;
      double particle_variance=0.;
      if (step)
        {
          auto &pm=this->get_particle_manager(0);const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
          if (mode=="horizontal_moment_update")
            for (auto &p:pm.get_particle_handler())
              {
                const auto &fit=fits.at(p.get_surrounding_cell()->id());const Tensor value=fit.first+(p.get_reference_location()[1]-.5)*fit.second;
                for (unsigned int c=0;c<3;++c) p.get_properties()[sp+c]=value[Tensor::unrolled_to_component_indices(c)];
              }
          if (mode=="native_history_reference") {native=current;next=read_native();}
          else next=production_transfer();
          // Parent roughness is meaningful only for A/C; B's particles are shadow data.
          if (mode!="native_history_reference")
            {
              std::map<CellId,std::vector<double>> by_cell;double count=0.;
              for (const auto &p:pm.get_particle_handler()) by_cell[p.get_surrounding_cell()->id()].push_back(p.get_properties()[sp+2]);
              for (const auto &[id,v]:by_cell) {double a=0;for (double x:v)a+=x; a/=v.size();for (double x:v)particle_variance+=(x-a)*(x-a);count+=v.size();}
              particle_variance=Utilities::MPI::sum(particle_variance,this->get_mpi_communicator())/Utilities::MPI::sum(count,this->get_mpi_communicator());
            }
          last_commit=step;
        }
      const auto next_load=loads(next),projection_jump=loads(subtract(projected,current)),mapping_jump=loads(subtract(next,projected)),total_jump=loads(subtract(next,current));
      auto reversed=loads(current,true);reversed.free-=current_load.free;
      const double assembly_floor=reversed.free.l2_norm();
      std::ofstream histories(this->get_output_directory()+"histories_"+std::to_string(step)+"_rank"+this->rank()+".bin",std::ios::binary);
      double next_mean=0.,next_m0=0.,next_m1=0.;
      for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
        {fe.reinit(cell);Tensor d0,d1;double w0=0;
         for (unsigned int k=0;k<q.size();++k) {double w=fe.JxW(k);next_mean+=w*next.at(cell->id())[k][0][1];w0+=w;
           const auto &h=next.at(cell->id())[k],&f=projected.at(cell->id())[k];const auto p=fe.quadrature_point(k);
           const double data[]={p[0],p[1],w,h[0][0],h[1][1],h[0][1],f[0][0],f[1][1],f[0][1]};
           histories.write(reinterpret_cast<const char*>(data),sizeof(data));
           d0+=w*(next.at(cell->id())[k]-current.at(cell->id())[k]);d1+=w*(q.point(k)[1]-.5)*(next.at(cell->id())[k]-current.at(cell->id())[k]);}
         next_m0=std::max(next_m0,(d0/w0).norm());next_m1=std::max(next_m1,(d1/w0).norm());}
      for (double *v:{&mean,&volume,&rough,&removed,&next_mean}) *v=Utilities::MPI::sum(*v,this->get_mpi_communicator());
      for (double *v:{&moment0,&moment1,&next_m0,&next_m1}) *v=Utilities::MPI::max(*v,this->get_mpi_communicator());
      // Every rank participates in distributed-vector reductions, even though
      // only root writes the compact summary.
      const double jp=projection_jump.free.l2_norm(),jm=mapping_jump.free.l2_norm(),jt=total_jump.free.l2_norm();
      const double fc=current_load.free.l2_norm(),fn=next_load.free.l2_norm(),jmax=total_jump.free.linfty_norm();
      const double dot=projection_jump.free*mapping_jump.free;
      if (this->get_pcout().is_active())
        {std::ofstream out(this->get_output_directory()+"summary.csv",step?std::ios::app:std::ios::out);out<<std::setprecision(17);
         if (!step) out<<"mode,step,time,dt,beta,kappa,relative,newton,krylov,alpha,mean,next_mean,top,bottom,next_top,next_bottom,native_rough,removed,parent_rough,homogeneity,Jproj,Jmap,Jtotal,Jbeta,Fabs,Fcurrent,Fnext,Jmax,proj_map_dot,fit_m0,fit_m1,next_m0,next_m1,assembly_floor\n";
         out<<mode<<','<<step<<','<<this->get_time()<<','<<dt<<','<<coefficients.beta<<','<<coefficients.eta_ve<<','<<relative<<','<<newton<<','<<krylov<<','<<alpha<<','
            <<mean/volume<<','<<next_mean/volume<<','<<current_load.top<<','<<current_load.bottom<<','<<next_load.top<<','<<next_load.bottom<<','
            <<std::sqrt(rough/volume)<<','<<std::sqrt(removed/volume)<<','<<(mode=="native_history_reference"?-1.:std::sqrt(particle_variance))<<','<<homogeneity<<','
            <<jp<<','<<jm<<','<<jt<<','<<coefficients.beta*jt<<','
            <<current_load.absolute<<','<<fc<<','<<fn<<','<<jmax<<','
            <<dot<<','<<moment0<<','<<moment1<<','<<next_m0<<','<<next_m1<<','<<assembly_floor<<'\n';}
      return {"Moment cycle",mode+" accepted publication measured"};
    }
  };
  ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"moment cycle","Fixed-mesh benchmark-only history moment comparison.")
}
