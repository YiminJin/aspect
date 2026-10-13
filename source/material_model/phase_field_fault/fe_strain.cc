/* Restricted native P2 total-energy model. Existing particle history is unchanged. */
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <aspect/geometry_model/box.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/plugins.h>
#include <aspect/utilities.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/gravity_model/interface.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> bool PhaseFieldFault<dim>::initialize_fe_shared_phase(LinearAlgebra::BlockVector &trial) const
    {
      if(fe_initial_phase_profile.empty())return false;
      AssertThrow(consistent_endpoint_phase && this->get_timestep_number()==0
                  && !this->get_parameters().resume_computation && !fe_strain_state,
                  ExcMessage("Shared profile is fresh pre-existing damage; history conversion is prohibited."));
      std::ifstream input(fe_initial_phase_profile);
      AssertThrow(bool(input),ExcMessage("Cannot open FE initial phase profile."));
      std::vector<std::pair<double,double>> data;double y,value;
      while(input>>y>>value)
        {
          AssertThrow(std::isfinite(y) && value>=0. && value<1.
                      && (data.empty() || y>data.back().first),ExcMessage("Invalid shared phase data."));
          data.emplace_back(y,value);
        }
      AssertThrow(input.eof() && data.size()>=2,ExcMessage("Malformed shared phase data."));
      LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
      owned=trial;
      const auto component=this->introspection().variable("phase_field").first_component_index;
      std::vector<types::global_dof_index> indices(this->get_fe().n_dofs_per_cell());
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        if(!cell->is_artificial())
        {
        cell->get_dof_indices(indices);
        for(unsigned int v=0;v<GeometryInfo<dim>::vertices_per_cell;++v)
          {
            const auto i=indices[this->get_fe().component_to_system_index(component,v)];
            if(!owned.locally_owned_elements().is_element(i))continue;
            const double yy=cell->vertex(v)[1];
            AssertThrow(yy>=data.front().first && yy<=data.back().first,
                        ExcMessage("Shared phase profile does not cover the domain."));
            auto right=std::upper_bound(data.begin(),data.end(),yy,[](double x,const auto &p){return x<p.first;});
            if(right==data.end())owned[i]=data.back().second;
            else if(right==data.begin())owned[i]=right->second;
            else {const auto &left=*(right-1);const double t=(yy-left.first)/(right->first-left.first);
                  owned[i]=(1-t)*left.second+t*right->second;}
          }
        }
      owned.compress(VectorOperation::insert);trial=owned;
      this->get_pcout()<<"   Accepted supplied continuum phase seed; native particle H retained, unloaded FE histories constructed afterward."<<std::endl;
      return true;
    }
    template<int dim> void PhaseFieldFault<dim>::set_fe_diagnostic_reference(
      const LinearAlgebra::BlockVector &reference,const std::vector<double> &Z,const std::vector<double> &I)
    {
      AssertThrow(consistent_endpoint_phase && fe_strain_state && this->get_timestep_number()>0
                  && Z.size()==fe_strain_state->Z.size()
                  && I.size()==Z.size(),ExcMessage("Invalid restricted fixed-entry reference."));
      for(unsigned int i=0;i<I.size();++i)
        AssertThrow(std::isfinite(Z[i]) && std::isfinite(I[i]) && I[i]>0.,ExcMessage("Undefined diagnostic surface history."));
      auto trial=std::make_shared<FEStrainState>(*fe_strain_state);
      trial->accepted=reference;trial->Z=Z;trial->I=I;trial->stress_samples.clear();
      trial->committed_step=this->get_timestep_number()-1;
      fe_strain_state=std::move(trial);
    }

    template<int dim> Quadrature<dim> PhaseFieldFault<dim>::mechanical_quadrature() const
    {
      if(!fe_strain_history)return this->introspection().quadratures.velocities;
      AssertThrow(dim==2,ExcNotImplemented());
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const auto extent=box.get_extents();
      auto cell=this->get_dof_handler().begin_active();
      while(cell!=this->get_dof_handler().end() && !cell->is_locally_owned())++cell;
      AssertThrow(cell!=this->get_dof_handler().end(),ExcMessage("Endpoint rank owns no bulk cells."));
      const double dx=cell->vertex(1)[0]-cell->vertex(0)[0];
      const double dy=cell->vertex(2)[1]-cell->vertex(0)[1];
      const unsigned int n=std::lround(extent[0]/dx), ny=std::lround(extent[1]/dy);
      const auto &faults=this->get_reconstructed_fault_manager().get_faults();
      // Before initial geometry construction no mechanical assembly is allowed.
      AssertThrow(n*ny==this->get_triangulation().n_global_active_cells() && faults.size()==1,
                  ExcMessage("FE strain quadrature requires a uniform Cartesian grid and one fault."));
      const unsigned int ns=faults[0].n_cells();
      const unsigned int panels=std::lcm(n,ns)/n;
      // One rule supplies mechanics, the slip map and its adjoint, and the
      // bulk part of the phase functional. Normalization/CPDI stay separate.
      // Resolve tangential polynomial products as well as sharp Q1 panels.
      const unsigned int tangential_points = std::max(3u,this->get_fe().degree+1);
      QIterated<1>x(QGauss<1>(tangential_points),panels);
      QGauss<1>y(fe_bulk_normal_quadrature_points);
      std::vector<Point<dim>>points;std::vector<double>weights;
      for(unsigned int i=0;i<x.size();++i)for(unsigned int j=0;j<y.size();++j)
        {Point<dim>p;p[0]=x.point(i)[0];p[1]=y.point(j)[0];points.push_back(p);weights.push_back(x.weight(i)*y.weight(j));}
      return Quadrature<dim>(points,weights);
    }

    template<int dim> const LinearAlgebra::BlockVector &
    PhaseFieldFault<dim>::accepted_fe_strain_state() const
    {
      AssertThrow(fe_strain_state,ExcMessage("FE strain history has not been initialized."));
      return fe_strain_state->accepted;
    }

    template<int dim> void PhaseFieldFault<dim>::validate_fe_strain_restrictions() const
    {
      AssertThrow(!benchmark_retained_stress && fe_strain_state,
                  ExcMessage("FE strain cannot consume an external stress override or missing accepted state."));
      const auto&s=*fe_strain_state;const auto &faults=this->get_reconstructed_fault_manager().get_faults();
      AssertThrow(this->get_dof_handler().n_dofs()==s.n_dofs && faults.size()==1
                  && faults[0].geometry_version()==s.geometry_version && faults[0].n_vertices()==s.vertices.size()
                  && this->get_triangulation().n_global_active_cells()==s.n_cells_global,
                  ExcMessage("Accepted FE strain rejects mesh/geometry changes."));
      for(unsigned int i=0;i<s.vertices.size();++i)
        AssertThrow(faults[0].vertex(i)==s.vertices[i],ExcMessage("Accepted FE strain geometry changed."));
      AssertThrow(this->get_triangulation().get_vertices()==s.mesh_vertices,
                  ExcMessage("Accepted FE strain rejects mesh motion."));
      const auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
      AssertThrow(particles.n_global_particles()==s.n_particles_global,ExcMessage("Accepted FE strain rejects particle birth/deletion."));
      unsigned int fixed_particles=1;
      for(const auto&p:particles)
        if(!s.particles.count(p.get_id()) || s.particles.at(p.get_id())!=p.get_location())fixed_particles=0;
      AssertThrow(Utilities::MPI::min(fixed_particles,this->get_mpi_communicator())==1,
                  ExcMessage("Accepted FE strain rejects history transport."));
    }

    template<int dim> void PhaseFieldFault<dim>::prepare_fe_strain_history()
    {
      if(!fe_strain_history)return;
      if(fe_strain_state) {validate_fe_strain_restrictions();return;}
      AssertThrow(this->get_timestep_number()==0 && !this->get_parameters().resume_computation
                  && Plugins::plugin_type_matches<GeometryModel::Box<dim>>(this->get_geometry_model())
                  && !benchmark_retained_stress && normalization_surface_subdivisions==1,
                  ExcMessage("FE strain initializes only on a fresh fixed Cartesian Box without stress overrides."));
      auto state=std::make_shared<FEStrainState>();
      state->accepted.reinit(this->get_solution());state->accepted=0.;
      const auto &intro=this->introspection();const unsigned int b=intro.variable("phase_field").block_index;
      state->accepted.block(b)=this->get_solution().block(b);
      const auto phase_offset=state->accepted.get_block_indices().block_start(b);
      for(auto i:state->accepted.block(b).locally_owned_elements())
        AssertThrow(!this->get_current_constraints().is_constrained(phase_offset+i),
                    ExcMessage("Accepted FE strain currently requires unconstrained Q1 phase degrees of freedom."));
      state->mesh_vertices=this->get_triangulation().get_vertices();
      state->G=elastic_shear_moduli[0];state->eta=reference_viscosities[0];
      state->n_dofs=this->get_dof_handler().n_dofs();
      state->n_cells_global=this->get_triangulation().n_global_active_cells();
      const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
      state->geometry_version=fault.geometry_version();
      const unsigned int ns=fault.n_cells();
      for(unsigned int i=0;i<fault.n_vertices();++i)
        {
          AssertThrow(std::abs(fault.vertex(i)[0]-double(i)/ns)<1.e-13
                      && fault.vertex(i)[1]==.5,ExcMessage("FE strain requires a unit horizontal through-going fault."));
          state->vertices.push_back(fault.vertex(i));
        }
      const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
      const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
      state->n_particles_global=pm.get_particle_handler().n_global_particles();
      for(const auto&p:pm.get_particle_handler())
        {
          for(unsigned int c=0;c<SymmetricTensor<2,dim>::n_independent_components;++c)
            AssertThrow(p.get_properties()[sp+c]==0.,ExcMessage("FE strain requires unloaded initial particle stress; no conversion is performed."));
          state->particles[p.get_id()]=p.get_location();
        }
      const auto fractions=std::vector<double>(elastic_shear_moduli.size(),1./elastic_shear_moduli.size());
      state->m=(1./this->get_phase_field_handler().energetic_degradation(fractions,.5)-1.)/3.;
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const auto extent=box.get_extents(),origin=box.get_origin();
      AssertThrow(extent[0]==1. && origin[0]==0. && extent[1]>0.
                  && std::abs(origin[1]+extent[1]/2-.5)<1.e-13,
                  ExcMessage("FE strain requires unit tangential width and a normal Box centered on y=0.5."));
      prepare_fe_normalization_geometry(*state);
      const auto quad=mechanical_quadrature();
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points|update_JxW_values);
      std::vector<types::global_dof_index>ids(this->get_fe().dofs_per_cell);
      const auto pc=intro.variable("phase_field").first_component_index;
      auto first=this->get_dof_handler().begin_active();
      while(first!=this->get_dof_handler().end() && !first->is_locally_owned())++first;
      AssertThrow(first!=this->get_dof_handler().end(),ExcMessage("Endpoint rank owns no bulk cells."));
      std::array<double,dim> spacing{};
      spacing[0]=first->vertex(1)[0]-first->vertex(0)[0];
      spacing[1]=first->vertex(2)[1]-first->vertex(0)[1];
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {
          if(cell->is_artificial())continue;
          state->evaluation_cells.emplace(cell->id(),cell);
          if(!cell->is_locally_owned())continue;
          fe.reinit(cell);cell->get_dof_indices(ids);state->cells.emplace(cell->id(),cell);
          for(const auto v:cell->vertex_indices())for(unsigned int d=0;d<dim;++d)
            {
              const double coordinate=(cell->vertex(v)[d]-origin[d])/spacing[d];
              AssertThrow(std::abs(coordinate-std::round(coordinate))<1.e-12
                          && cell->vertex(v)[d]>=origin[d]
                          && cell->vertex(v)[d]<=origin[d]+extent[d],
                          ExcMessage("FE strain requires a fixed uniform Cartesian Box mesh."));
            }
          AssertThrow(std::abs(cell->vertex(1)[0]-cell->vertex(0)[0]-spacing[0])<1.e-13
                      && std::abs(cell->vertex(2)[1]-cell->vertex(0)[1]-spacing[1])<1.e-13,
                      ExcMessage("FE strain cell sizes must be uniform."));
          for(unsigned int q=0;q<quad.size();++q)
            {
              typename FEStrainState::BulkPoint p;
              p.cell=cell->id();p.x=fe.quadrature_point(q);p.weight=fe.JxW(q);
              if(consistent_endpoint_phase)
                AssertThrow(this->get_gravity_model().gravity_vector(p.x).norm()==0.,
                            ExcMessage("Endpoint reduced potential requires the unloaded verification fixture without body forces."));
              p.segment=std::min(ns-1,static_cast<unsigned int>(p.x[0]*ns));p.xi=p.x[0]*ns-p.segment;
              for(unsigned int v=0;v<p.dofs.size();++v)
                {const auto j=this->get_fe().component_to_system_index(pc,v);p.dofs[v]=ids[j]-phase_offset;
                 p.shape[v]=fe.shape_value_component(j,q,pc);}
              state->bulk.push_back(p);
            }
        }
      std::stable_sort(state->bulk.begin(),state->bulk.end(),
                       [](const auto&a,const auto&b){return a.cell<b.cell;});
      state->Z.assign(ns+1,0.);state->I=current_normalization_integrals.at(0);
      fe_strain_state=state;
      const auto initial=evaluate_fe_phase_energy(this->get_solution(),false);
      AssertThrow(initial.admissible,ExcMessage("Initial FE strain normalization is undefined."));
      current_normalization_integrals[0]=initial.normalization;state->I=initial.normalization;
      validate_cohesive_state_commit({state->Z});commit_cohesive_state({state->Z});
      this->get_pcout()<<std::setprecision(17)<<"   Accepted FE strain: G="<<state->G
                       <<" eta="<<state->eta<<" lambda="<<state->eta/state->G<<" m="<<state->m
                       <<" bulk QPs="<<state->bulk.size()<<" phase DoFs="<<state->accepted.block(b).size()<<'\n';
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_endpoint_stress(
      const FEEndpointTrial &endpoint,const std::vector<double> &I,const CellId &cell,const Point<dim> &x) const
    {
      validate_fe_strain_restrictions();
      return evaluate_fe_strain_stress(endpoint.strain,endpoint.Z,I,cell,x);
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::retained_history_stress(
      const CellId &id,const Point<dim> &x,const SymmetricTensor<2,dim> &legacy) const
    {
      if(!fe_strain_history)return benchmark_retained_stress?benchmark_retained_stress(id,x):legacy;
      AssertThrow(fe_strain_state && !benchmark_retained_stress,ExcMessage("FE strain needs its single accepted history."));
      const auto&s=*fe_strain_state;
      std::array<double,dim> coordinates;
      for(unsigned int d=0;d<dim;++d)coordinates[d]=x[d];
      const auto key=std::make_pair(id,coordinates);
      const auto cached=s.stress_samples.find(key);
      if(cached!=s.stress_samples.end())return cached->second;
      const auto stress=evaluate_fe_strain_stress(s.accepted,s.Z,s.I,id,x);
      s.stress_samples.emplace(key,stress);
      return stress;
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_strain_stress(
      const LinearAlgebra::BlockVector &state,const std::vector<double>&Znodes,
      const std::vector<double>&Inodes,const CellId&id,const Point<dim>&x,double *evaluation_scale) const
    {
      const auto&s=*fe_strain_state;const auto cell=s.evaluation_cells.at(id);
      const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,x);
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_values|update_gradients);
      fe.reinit(cell);std::vector<SymmetricTensor<2,dim>>R(1);std::vector<double>phi(1);
      fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(state,R);
      fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(state,phi);
      const auto ns=Znodes.size()-1;const auto j=std::min<unsigned int>(ns-1,x[0]*ns);const double t=x[0]*ns-j;
      const double Z=(1-t)*Znodes[j]+t*Znodes[j+1],I=(1-t)*Inodes[j]+t*Inodes[j+1];
      const double h=s.m*phi[0]*(1+phi[0])/std::pow(1-phi[0],2);
      SymmetricTensor<2,dim>S;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
      if(evaluation_scale)
        {
          // Bound differentiation before cancellation of nearby FE values.
          // A constant d_R can have a tiny gradient with O(1) coefficients.
          Vector<double> values(this->get_fe().dofs_per_cell);
          cell->get_dof_values(state,values);double strain_scale=0.;
          for(unsigned int i=0;i<values.size();++i)
            if(this->get_fe().system_to_component_index(i).first<dim)
              strain_scale+=std::abs(values[i])
                            *fe[this->introspection().extractors.velocities].symmetric_gradient(i,0).norm();
          *evaluation_scale=2*s.G*strain_scale+std::abs(2*Z*h/I)*S.norm();
        }
      return 2*s.G*Utilities::Tensors::consistent_deviator(R[0])-2*Z*h/I*S;
    }

    template<int dim> std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
    PhaseFieldFault<dim>::prepare_fe_strain_commit(const LinearAlgebra::BlockVector &bulk,
                                                  const std::vector<std::vector<double>>&tractions) const
    {
      validate_fe_strain_restrictions();const auto&s=*fe_strain_state;
      AssertThrow(s.committed_step!=this->get_timestep_number(),ExcMessage("FE strain cannot commit twice in a physical step."));
      const auto c=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
      auto update=std::make_shared<FEStrainUpdate>();
      LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());owned=s.accepted;
      const auto vb=this->introspection().block_indices.velocities;
      LinearAlgebra::BlockVector owned_bulk(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
      owned_bulk=bulk;
      owned.block(vb)*=c.beta;
      owned.block(vb).add(c.eta_ve/s.G,owned_bulk.block(vb));
      owned.block(this->introspection().variable("phase_field").block_index)=
        bulk.block(this->introspection().variable("phase_field").block_index);
      update->accepted.reinit(s.accepted);update->accepted=owned;
      update->I=current_normalization_integrals.at(0);update->Z.resize(s.Z.size());
      const auto&V=this->get_reconstructed_fault_manager().get_slip_rate(0);
      for(unsigned int j=0;j<s.Z.size();++j)
        {
          update->Z[j]=tractions.at(0).at(j)*update->I[j];
          const double expected=c.beta*s.Z[j]+c.eta_ve*V[j];
          AssertThrow(std::isfinite(update->Z[j]) && std::abs(update->Z[j]-expected)
                      <=64*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(expected),std::abs(update->Z[j])}),
                      ExcMessage("FE strain surface history and accepted Maxwell state disagree."));
        }
      return update;
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::candidate_fe_strain_stress(
      const std::shared_ptr<FEStrainUpdate>&update,const CellId&id,const Point<dim>&x,double *evaluation_scale) const
    {
      double current_scale=0.;
      const auto stress=evaluate_fe_strain_stress(update->accepted,update->Z,update->I,id,x,&current_scale);
      if(evaluation_scale)
        {
          double old_scale=0.;const auto&s=*fe_strain_state;
          evaluate_fe_strain_stress(s.accepted,s.Z,s.I,id,x,&old_scale);
          const auto c=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
          // The triangle inequality also bounds kappa*epsilon by the old
          // and new R terms. Count both endpoint evaluations and forcing.
          *evaluation_scale=2*(current_scale+c.beta*old_scale);
        }
      return stress;
    }

    template<int dim> void PhaseFieldFault<dim>::publish_fe_strain_commit(
      const std::shared_ptr<FEStrainUpdate>&update) noexcept
    {
      fe_strain_state->accepted.swap(update->accepted);
      fe_strain_state->Z.swap(update->Z);fe_strain_state->I.swap(update->I);
      fe_strain_state->stress_samples.clear();
      fe_strain_state->committed_step=this->get_timestep_number();
    }
  }

  template<int dim> Quadrature<dim> fault_mechanical_quadrature(const SimulatorAccess<dim>&sim)
  {
    return fault_mechanical_quadrature(sim.get_material_model(),sim.introspection().quadratures.velocities);
  }
  template<int dim> Quadrature<dim> fault_mechanical_quadrature(
    const MaterialModel::Interface<dim>&material,const Quadrature<dim>&native)
  {
    if(const auto*m=dynamic_cast<const MaterialModel::PhaseFieldFault<dim>*>(&material))
      return m->mechanical_quadrature();
    return native;
  }

  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
    template bool PhaseFieldFault<dim>::initialize_fe_shared_phase(LinearAlgebra::BlockVector &) const; \
    template Quadrature<dim> PhaseFieldFault<dim>::mechanical_quadrature() const; \
    template void PhaseFieldFault<dim>::prepare_fe_strain_history(); \
    template void PhaseFieldFault<dim>::validate_fe_strain_restrictions() const; \
    template const LinearAlgebra::BlockVector &PhaseFieldFault<dim>::accepted_fe_strain_state() const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_endpoint_stress(const FEEndpointTrial &,const std::vector<double>&,const CellId&,const Point<dim>&) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::retained_history_stress(const CellId &,const Point<dim>&,const SymmetricTensor<2,dim>&) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_strain_stress(const LinearAlgebra::BlockVector &,const std::vector<double>&,const std::vector<double>&,const CellId&,const Point<dim>&,double*) const; \
    template std::shared_ptr<PhaseFieldFault<dim>::FEStrainUpdate> PhaseFieldFault<dim>::prepare_fe_strain_commit(const LinearAlgebra::BlockVector &,const std::vector<std::vector<double>>&) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::candidate_fe_strain_stress(const std::shared_ptr<FEStrainUpdate>&,const CellId&,const Point<dim>&,double*) const; \
    template void PhaseFieldFault<dim>::publish_fe_strain_commit(const std::shared_ptr<FEStrainUpdate>&) noexcept;
    INSTANTIATE(2)
    INSTANTIATE(3)
#undef INSTANTIATE
  }
  template Quadrature<2> fault_mechanical_quadrature(const SimulatorAccess<2>&);
  template Quadrature<3> fault_mechanical_quadrature(const SimulatorAccess<3>&);
  template Quadrature<2> fault_mechanical_quadrature(const MaterialModel::Interface<2>&,const Quadrature<2>&);
  template Quadrature<3> fault_mechanical_quadrature(const MaterialModel::Interface<3>&,const Quadrature<3>&);
}
