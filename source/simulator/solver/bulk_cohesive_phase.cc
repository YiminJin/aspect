/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Fixed-mesh bulk cohesive experiment. This is a distinct residual,
// not the surface endpoint potential or an unconverged phase-first pass.
#include <aspect/simulator.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_mpi.h>
#include <aspect/material_model/phase_field_fault/bulk_stability.h>
#include <chrono>
#include <aspect/reconstructed_fault/endpoint_phase_constraints.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace aspect
{
  template<int dim> void Simulator<dim>::solve_bulk_cohesive_phase()
  {
    namespace BC=MaterialModel::BulkCohesive;
    namespace TR=BC::TensorReturn;
    namespace CR=BC::CoulombReturn;
    namespace SG=BC::SlipGradient;
    using Model=MaterialModel::PhaseFieldFault<dim>;
    using BulkState=typename Model::BulkCohesiveState;
    auto &model=Plugins::get_plugin_as_type<Model>(*material_model);
    AssertThrow(model.uses_bulk_cohesive_phase() && dim==2
                && geometry_model->get_periodic_boundary_pairs().empty(),
                ExcMessage("Bulk cohesive native experiment requires nonperiodic 2-D fixed geometry."));
    TimerOutput::Scope timer(computing_timer,"Bulk cohesive: coupled solve");
    const unsigned int rank=Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int ranks=Utilities::MPI::n_mpi_processes(mpi_communicator);
    const bool fixed_coulomb=model.bulk_flow_rule==Model::BulkFlowRule::coulomb_fixed_phase;
    AssertThrow(!fixed_coulomb || !model.evolve_phase_field,
                ExcMessage("Native Coulomb phase evolution is not qualified; phase must remain fixed."));
    using Clock=std::chrono::steady_clock;
    const auto started=Clock::now();
    double assembly_seconds=0.,communication_seconds=0.,linear_seconds=0.,history_seconds=0.,stability_seconds=0.;
    unsigned int evaluations=0,linear_solves=0;
    const auto seconds=[](const auto a){return std::chrono::duration<double>(Clock::now()-a).count();};
    const auto native_constraints=BC::global_constraints(current_constraints,mpi_communicator);
    const auto global_solution=BC::global_values(solution,mpi_communicator);
    const auto pb=introspection.variable("phase_field").block_index;
    const auto pc=introspection.variable("phase_field").first_component_index;
    const auto po=solution.get_block_indices().block_start(pb);
    const auto pressure_begin=solution.get_block_indices().block_start(introspection.block_indices.pressure);
    const auto pressure_end=pressure_begin+solution.block(introspection.block_indices.pressure).size();
    const auto nv=solution.block(introspection.block_indices.velocities).size();
    if(timestep_number==0)
      {
        AssertThrow(!model.bulk_cohesive_state,ExcMessage("Bulk initial history cannot publish twice."));
        model.initialize_bulk_cohesive_history(solution);
        old_solution=solution;old_old_solution=solution;current_linearization_point=solution;
        return; // Genuine unloaded initialization, no artificial interval.
      }
    const auto parent=model.bulk_cohesive_state;
    AssertThrow(parent && parent->step+1==timestep_number
                && parent->mesh==triangulation.get_vertices() && time_step>0.,
                ExcMessage("Bulk cohesive history/mesh/clock changed outside accepted advancement."));
    const auto p=parent->parameters;
    const double dt=time_step,G=p.G,beta=1./(1.+dt*G/p.eta);
    // Bounded continuation diagnostics, inactive in ordinary runs. These
    // select a Newton guess, never alter the incoming history or obstacle.
    const char *diagnostic_time=std::getenv("ASPECT_BULK_CONTINUATION_TIME");
    const bool continuation_diagnostic=diagnostic_time
      && std::abs(time-std::stod(diagnostic_time))<1e-10*dt;
    bool relaxation_guess=false;
    const double ell=phase_field_handler->get_length_scale();
    const double Kf=model.critical_energy_release_rates[0]/((8./3.)*ell);
    const unsigned int native_n=dof_handler.n_dofs(),np=parent->phase.size();
    const bool gradient=parent->slip_length>0.;
    if(gradient)
      {
        AssertThrow(fixed_coulomb && ranks==1 && parent->quadrature==6,
                    ExcMessage("Slip-gradient native prototype requires serial fixed-phase closed Coulomb and Gauss6."));
        if(!parent->slip_space)
          {
            std::vector<SG::Space::Sample> samples;
            FEValues<dim> layout(*mapping,finite_element,model.mechanical_quadrature(),update_quadrature_points|update_JxW_values);
            Point<2> minimum,maximum;for(unsigned int d=0;d<2;++d){minimum[d]=std::numeric_limits<double>::infinity();maximum[d]=-minimum[d];}
            for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
              {
                layout.reinit(cell);
                for(const auto v:cell->vertex_indices())for(unsigned int d=0;d<2;++d)
                  {minimum[d]=std::min(minimum[d],cell->vertex(v)[d]);maximum[d]=std::max(maximum[d],cell->vertex(v)[d]);}
                for(unsigned int q=0;q<layout.n_quadrature_points;++q)
                  samples.push_back({cell->id(),q,Point<2>(layout.quadrature_point(q)[0],layout.quadrature_point(q)[1]),layout.JxW(q)});
              }
            parent->slip_space=std::make_shared<const SG::Space>(samples,minimum,maximum);
          }
      }
    const unsigned int micro_points=gradient?parent->slip_space->points.size():0;
    const unsigned int n=native_n+2*micro_points;
    std::vector<BC::Tensor> oldK(micro_points);
    if(gradient)for(const auto &cell:parent->slip_space->ids)
      for(unsigned int q=0;q<cell.second.size();++q)oldK[cell.second[q]]=parent->history.at(cell.first)[q].K;
    const auto &settings=phase_field_handler->get_solver_parameters();
    AssertThrow(settings.use_bound_constraints && settings.enforce_irreversibility,
                ExcMessage("Bulk cohesive phase requires native physical bounds and irreversibility."));
    const double mechanical_tolerance=parameters.nonlinear_tolerance;
    const double phase_tolerance=settings.nonlinear_solver_tolerance;
    const internal::EndpointPhaseConstraints phase_coordinates(native_constraints,po,np);
    AssertThrow(ranks==1 || phase_coordinates.identity(),ExcMessage("Distributed bulk hanging-node phase evolution is outside this qualification."));
    const Vector<double> lower=phase_coordinates.select(parent->phase);
    AssertThrow(phase_coordinates.expand(lower)==parent->phase,
                ExcMessage("Bulk accepted phase obstacle violates native constraints."));

    // Scratch unknowns: displacement increment, p/G, physical phase. The native
    // accepted velocity and pressure retain their normal physical units.
    Vector<double> x(n);
    for(unsigned int i=0;i<nv;++i)x[i]=dt*global_solution[i];
    for(unsigned int i=pressure_begin;i<pressure_end;++i)
      x[i]=global_solution[i]/G;
    for(unsigned int i=0;i<np;++i)x[po+i]=parent->phase[i];
    if(gradient)for(unsigned int i=0;i<micro_points;++i)
      {x[native_n+2*i]=oldK[i][0][0];x[native_n+2*i+1]=oldK[i][0][1];}
    AffineConstraints<double> base(native_constraints);
    for(const auto &line:native_constraints.get_lines())
      if(line.index<nv)base.set_inhomogeneity(line.index,dt*line.inhomogeneity);
      else if(line.index>=pressure_begin && line.index<pressure_end)
        base.set_inhomogeneity(line.index,line.inhomogeneity/G);
    auto selected=[&](const unsigned int i)
      {return i<nv || (i>=pressure_begin && i<pressure_end) || (!fixed_coulomb && i>=po && i<po+np) || i>=native_n;};
    for(unsigned int i=0;i<n;++i)
      if(!selected(i) && !base.is_constrained(i))
        {base.add_line(i);if(fixed_coulomb && i>=po && i<po+np)base.set_inhomogeneity(i,parent->phase[i-po]);}
    // Closed-velocity boundary experiment: fix one pressure gauge, never
    // impose the frictional p_N as this Lagrange multiplier.
    const bool pinned=std::any_of(base.get_lines().begin(),base.get_lines().end(),
      [&](const auto &line){return line.index>=pressure_begin && line.index<pressure_end && line.entries.empty();});
    // A constant pressure is a null mode only if every velocity test has zero
    // normal trace. Inspect actual boundary-component constraints, including
    // the mixed lateral condition, rather than boundary names or friction p_N.
    unsigned int open_normal=0;
    for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
      for(const auto f:cell->face_indices())if(cell->face(f)->at_boundary())
        for(const auto v:cell->face(f)->vertex_indices())
          {
            const auto normal=cell->face(f)->center()-cell->center();
            for(unsigned int d=0;d<dim;++d)if(std::abs(normal[d])>1e-10*normal.norm())
              if(!native_constraints.is_constrained(cell->face(f)->vertex_dof_index(v,introspection.component_indices.velocities[d])))open_normal=1;
          }
    open_normal=Utilities::MPI::max(open_normal,mpi_communicator);
    if(fixed_coulomb)
      {
        AssertThrow(!open_normal && !pinned && phase_coordinates.identity(),
                    ExcMessage("Closed Coulomb block requires fixed normal boundary velocities, uniform phase space and its own mean perturbation-pressure reference."));
        // Prescribed mean p'=0 fixes physical confinement, not a gauge of a
        // pressure-independent material. Condense the exact Q1 integral once.
        Vector<double> pressure_mass(pressure_end-pressure_begin);
        FEValues<dim> pressure_fe(*mapping,finite_element,model.mechanical_quadrature(),update_values|update_JxW_values);
        std::vector<types::global_dof_index> pressure_ids(finite_element.n_dofs_per_cell());
        for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
          {
            pressure_fe.reinit(cell);cell->get_dof_indices(pressure_ids);
            for(unsigned int i=0;i<pressure_ids.size();++i)if(pressure_ids[i]>=pressure_begin && pressure_ids[i]<pressure_end)
              for(unsigned int q=0;q<pressure_fe.n_quadrature_points;++q)
                pressure_mass[pressure_ids[i]-pressure_begin]+=pressure_fe.JxW(q)*pressure_fe[introspection.extractors.pressure].value(i,q);
          }
        BC::sum(pressure_mass,mpi_communicator);
        const unsigned int reference=pressure_begin;
        AssertThrow(pressure_mass[0]>0.,ExcInternalError());base.add_line(reference);
        for(unsigned int i=1;i<pressure_mass.size();++i)if(pressure_mass[i]!=0.)
          base.add_entry(reference,pressure_begin+i,-pressure_mass[i]/pressure_mass[0]);
      }
    else if(!pinned && !open_normal)
      {
        unsigned int gauge=pressure_begin;
        while(gauge<pressure_end && base.is_constrained(gauge))++gauge;
        AssertThrow(gauge<pressure_end,ExcMessage("No bulk pressure gauge is available."));
        base.add_line(gauge);
      }
    base.close();base.distribute(x);
    // Supported loading-aware continuation for this bulk law. At a change of
    // prescribed velocity (including a hold), the old velocity extrapolation
    // is not a continuation of the new loading. Start at accepted phase with
    // zero interior increment and Maxwell-scaled pressure, then impose the
    // CURRENT boundary increments and solve all coupled variables freely.
    // Under unchanged nonzero loading retain the established predictor.
    double imposed=0.,previous_imposed=0.,loading_change=0.;
    for(const auto &line:base.get_lines())if(line.index<nv && line.entries.empty())
      {
        imposed+=line.inhomogeneity*line.inhomogeneity;
        previous_imposed+=std::pow(dt*global_solution[line.index],2);
        loading_change+=std::pow(line.inhomogeneity-dt*global_solution[line.index],2);
      }
    const bool loading_aware=imposed==0. || std::sqrt(loading_change)>
      128*std::numeric_limits<double>::epsilon()*std::sqrt(std::max(imposed,previous_imposed));
    // Test-only reproduction of the former saddle; it CANNOT disable the
    // publication safeguard. The old diagnostic guess remains reproducible.
    const bool test_predictor=std::getenv("ASPECT_BULK_TEST_PREDICTOR");
    const bool diagnostic_relaxation=continuation_diagnostic && std::getenv("ASPECT_BULK_RELAXATION_GUESS");
    const auto initialize_continuation=[&]()
      {
        for(unsigned int i=0;i<nv;++i)x[i]=0.;
        for(unsigned int i=pressure_begin;i<pressure_end;++i)x[i]=beta*global_solution[i]/G;
        for(unsigned int i=0;i<np;++i)x[po+i]=parent->phase[i];
        base.distribute(x);relaxation_guess=true;
      };
    if((loading_aware && !test_predictor) || diagnostic_relaxation)initialize_continuation();
    AffineConstraints<double> homogeneous(base);
    for(const auto &line:base.get_lines())homogeneous.set_inhomogeneity(line.index,0.);
    DynamicSparsityPattern dynamic(n,n);
    if(ranks==1)
      {
        DynamicSparsityPattern native_dynamic(native_n,native_n);
        DoFTools::make_sparsity_pattern(dof_handler,native_dynamic,base,true);
        for(unsigned int i=0;i<native_n;++i)for(auto it=native_dynamic.begin(i);it!=native_dynamic.end(i);++it)dynamic.add(i,it->column());
        if(gradient)
          {
            for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
              {
                std::vector<types::global_dof_index> local(finite_element.n_dofs_per_cell());cell->get_dof_indices(local);
                for(const auto point:parent->slip_space->ids.at(cell->id()))for(unsigned int c=0;c<2;++c)
                  for(const auto j:local)if(selected(j))
                    {dynamic.add(native_n+2*point+c,j);dynamic.add(j,native_n+2*point+c);}
              }
            for(unsigned int i=0;i<micro_points;++i)
              for(auto it=parent->slip_space->stiffness.begin(i);it!=parent->slip_space->stiffness.end(i);++it)
                for(unsigned int a=0;a<2;++a)for(unsigned int b=0;b<2;++b)dynamic.add(native_n+2*i+a,native_n+2*it->column()+b);
            // Mean-pressure condensation couples the reference pressure to all
            // pressure masters in the new microforce rows, exactly once.
            for(const auto &line:base.get_lines())for(unsigned int i=native_n;i<n;++i)
              if(dynamic.exists(i,line.index))for(const auto &entry:line.entries)
                {dynamic.add(i,entry.first);dynamic.add(entry.first,i);}
          }
      }
    else
      {
        std::vector<std::vector<types::global_dof_index>> local_cells;
        for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
          {
            std::vector<types::global_dof_index> all(finite_element.n_dofs_per_cell()),chosen;
            cell->get_dof_indices(all);
            for(const auto i:all)if(selected(i))chosen.push_back(i);
            local_cells.push_back(std::move(chosen));
          }
        for(const auto &part:Utilities::MPI::all_gather(mpi_communicator,local_cells))
          for(const auto &cell:part)base.add_entries_local_to_global(cell,dynamic,true);
        for(unsigned int i=0;i<n;++i)dynamic.add(i,i);
      }
    SparsityPattern sparsity;sparsity.copy_from(dynamic);
    SparseMatrix<double> jacobian(sparsity);
    const auto quadrature=model.mechanical_quadrature();
    FEValues<dim> fe(*mapping,finite_element,quadrature,
                    update_values|update_gradients|update_JxW_values|update_quadrature_points);
    const auto dpc=finite_element.n_dofs_per_cell();
    std::vector<types::global_dof_index> ids(dpc);
    const auto embed=[](const SymmetricTensor<2,dim> &raw)
      {
        const auto d=Utilities::Tensors::consistent_deviator(raw);
        BC::Tensor t;t[0][0]=d[0][0];t[1][1]=d[1][1];t[2][2]=-d[0][0]-d[1][1];t[0][1]=d[0][1];return t;
      };
    BC::Tensor S;S[0][1]=.5;
    // Native ordinary FE fracture, once. AT1 has an invariant Hessian on this
    // fixed mesh. These raw physical-node entries have not been condensed.
    LinearAlgebra::BlockVector owned_anchor;
    owned_anchor.reinit(introspection.index_sets.system_partitioning,mpi_communicator);owned_anchor=solution;
    internal::endpoint_assign(owned_anchor.block(pb),parent->phase);
    auto anchor=solution;anchor=owned_anchor;
    std::map<std::pair<unsigned int,unsigned int>,double> raw_fracture_hessian;
    const auto fracture=phase_field_handler->evaluate_cell_fracture(anchor,true,&raw_fracture_hessian);
    struct FractureEntry{unsigned int i,j;double value;};
    std::vector<FractureEntry> fracture_entries;
    for(const auto &entry:raw_fracture_hessian)
      if(entry.second!=0.)fracture_entries.push_back({entry.first.first,entry.first.second,entry.second});
    Vector<double> physical_mass(np);
    for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
      for(const auto v:cell->vertex_indices())physical_mass[cell->vertex_dof_index(v,pc)-po]+=cell->measure()/GeometryInfo<dim>::vertices_per_cell;
    BC::sum(physical_mass,mpi_communicator);
    const auto mass=phase_coordinates.transpose(physical_mass);
    std::map<CellId,std::vector<SymmetricTensor<2,dim>>> oldT;
    for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())oldT[cell->id()]=parent->evaluate_tensor(model,cell->id(),quadrature);

    struct Evaluation
    {
      Vector<double> raw,residual,phase_gradient,delta_phase,fracture_gradient;
      std::map<CellId,std::vector<typename BulkState::PointResponse>> points;
      std::map<CellId,std::vector<SG::JointResponse>> joint;
      double norm=0.,mechanical=0.,kkt=0.,bulk=0.,cohesive=0.,fracture=0.;
      double physical=0.,stress_loss=0.,traction_loss=0.,phase_remainder=0.;
      double internal_work=0.,pressure_work=0.,external_work=0.,fracture_loss=0.;
      double Db=0.,Dc=0.,Df=0.,projection_error=0.;
      double storage_change=0.;
      double micro=0.,gradient_energy=0.,gradient_loss=0.;
      bool admissible=true;std::string contact_error;
    };
    auto evaluate=[&](const Vector<double> &v,const bool tangent)
      {
        const auto assembly_start=Clock::now();++evaluations;
        Evaluation e;e.raw.reinit(n);e.delta_phase.reinit(np);
        std::vector<BC::Tensor> K(micro_points),X;
        if(gradient)
          {
            for(unsigned int i=0;i<micro_points;++i){K[i][0][0]=v[native_n+2*i];K[i][1][1]=-K[i][0][0];K[i][0][1]=v[native_n+2*i+1];}
            X=parent->slip_space->apply(K,G,parent->slip_length);
            e.gradient_energy=parent->slip_space->energy(K,G,parent->slip_length);
            e.storage_change=e.gradient_energy-parent->slip_space->energy(oldK,G,parent->slip_length);
            auto delta=K;for(unsigned int i=0;i<micro_points;++i)delta[i]-=oldK[i];
            e.gradient_loss=parent->slip_space->energy(delta,G,parent->slip_length);
          }
        if(tangent)jacobian=0.;
        for(unsigned int i=0;i<np;++i)e.delta_phase[i]=v[po+i]-parent->phase[i];
        Vector<double> fg(np);
        for(unsigned int i=0;i<np;++i)fg[i]=fracture.gradient[i];
        for(const auto &entry:fracture_entries)
          fg[entry.i]+=entry.value*e.delta_phase[entry.j];
        e.fracture=fracture.energy;
        for(unsigned int i=0;i<np;++i)e.fracture+=fracture.gradient[i]*e.delta_phase[i];
        for(const auto &entry:fracture_entries)e.fracture_loss+=.5*entry.value*e.delta_phase[entry.i]*e.delta_phase[entry.j];
        e.fracture+=e.fracture_loss;

        e.fracture_gradient=fg;
        std::vector<SymmetricTensor<2,dim>> gradients(quadrature.size());
        std::vector<double> pressures(quadrature.size()),phis(quadrature.size()),old_phis(quadrature.size());
        Vector<double> native_fields(native_n);for(unsigned int i=0;i<native_n;++i)native_fields[i]=v[i];
        Vector<double> incoming_fields(native_fields);
        for(unsigned int i=0;i<np;++i)incoming_fields[po+i]=parent->phase[i];
        for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
          {
            fe.reinit(cell);cell->get_dof_indices(ids);
            fe[introspection.extractors.velocities].get_function_symmetric_gradients(native_fields,gradients);
            fe[introspection.extractors.pressure].get_function_values(native_fields,pressures);
            fe[FEValuesExtractors::Scalar(pc)].get_function_values(native_fields,phis);
            fe[FEValuesExtractors::Scalar(pc)].get_function_values(incoming_fields,old_phis);
            auto &responses=e.points[cell->id()];responses.resize(quadrature.size());
            if(gradient)e.joint[cell->id()].resize(quadrature.size());
            for(unsigned int q=0;q<quadrature.size();++q)
              {
                try
                {
                const double old_phi=old_phis[q];
                const auto &old=parent->history.at(cell->id())[q];
                BC::State incoming;incoming.T[0][0]=oldT.at(cell->id())[q][0][0];incoming.T[1][1]=oldT.at(cell->id())[q][1][1];
                incoming.T[2][2]=-incoming.T[0][0]-incoming.T[1][1];incoming.T[0][1]=oldT.at(cell->id())[q][0][1];
                incoming.a=old.a;incoming.signed_shear=old.signed_shear;incoming.absolute_shear=old.absolute_shear;incoming.q=old.q;incoming.phi=old_phi;
                const auto increment=embed(gradients[q]);
                TR::State tensor_incoming;
                tensor_incoming.T=incoming.T;tensor_incoming.phi=old_phi;
                tensor_incoming.A=old.A;tensor_incoming.K=old.K;
                tensor_incoming.Q=old.Q;tensor_incoming.L=old.L;
                const BC::Tensor *fixed=parent->flow_rule==Model::BulkFlowRule::fixed_tensor ? &S : nullptr;
                auto &r=responses[q];
                if(parent->tensor_flow())
                  {
                    if(fixed_coulomb)
                      {
                        if(gradient)
                          {
                            const auto point=parent->slip_space->ids.at(cell->id())[q];
                            auto &joint=e.joint.at(cell->id())[q];
                            joint=SG::joint_point({p,parent->coulomb_mu,0.},tensor_incoming,increment,K[point],X[point],dt,parent->background_pressure+G*pressures[q]);
                            r.coulomb_point=std::make_shared<const CR::Response>(joint.local);
                            e.micro=std::max(e.micro,joint.residual.norm()/joint.force_scale);
                            e.raw[native_n+2*point]=joint.residual[0][0]/G;
                            e.raw[native_n+2*point+1]=joint.residual[0][1]/G;
                          }
                        else r.coulomb_point=std::make_shared<const CR::Response>(CR::update({p,parent->coulomb_mu,0.},tensor_incoming,increment,phis[q],dt,parent->background_pressure+G*pressures[q]));
                        r.tensor_point=r.coulomb_point->point;
                      }
                    else r.tensor_point=TR::update(p,tensor_incoming,increment,phis[q],dt,fixed);
                    const auto &t=r.tensor_point;
                    r.state.T=t.state.T;r.state.phi=t.state.phi;
                    // Signed xy projections support existing trace diagnostics;
                    // no tensor history or constitutive action reads these.
                    r.state.a=2*t.state.A[0][1];r.state.q=t.state.Q[0][1];
                    r.state.signed_shear=2*t.state.K[0][1];r.state.absolute_shear=t.state.L;
                    r.w=2*t.flow_rate[0][1];r.beta=t.beta;r.stress=t.stress;r.driving=t.driving;
                    r.bulk_energy=t.bulk_energy;r.cohesive_energy=t.cohesive_energy;
                    r.bulk_dissipation=t.bulk_dissipation;r.cohesive_dissipation=t.cohesive_dissipation;
                    r.friction_dissipation=t.friction_dissipation;r.sliding=t.sliding;r.intact=t.intact;
                  }
                else static_cast<BC::Response &>(r)=BC::update(p,incoming,increment,phis[q],dt);
                const double weight=fe.JxW(q);
                e.bulk+=weight*r.bulk_energy;e.cohesive+=weight*r.cohesive_energy;
                e.Db+=weight*dt*r.bulk_dissipation;e.Dc+=weight*dt*r.cohesive_dissipation;e.Df+=weight*dt*r.friction_dissipation;
                const auto ledger=parent->tensor_flow() ? TR::account(p,tensor_incoming,r.tensor_point,increment,dt)
                                                       : BC::account(p,incoming,r,increment,dt);
                e.storage_change+=weight*ledger.storage_change;
                e.internal_work+=weight*ledger.internal_work;
                const double total_pressure=G*pressures[q]+(fixed_coulomb?parent->background_pressure:0.);
                e.pressure_work+=fixed_coulomb ? weight*total_pressure*trace(gradients[q])
                                             : weight*G*pressures[q]*trace(gradients[q]);
                e.stress_loss+=weight*ledger.stress_loss;e.traction_loss+=weight*ledger.traction_loss;
                const auto h0=BC::compliance(old_phi,p.m),h1=BC::compliance(phis[q],p.m);
                const double traction_energy=parent->tensor_flow() ? (r.tensor_point.state.Q*r.tensor_point.state.Q)/(4*G)
                                                                 : r.state.q*r.state.q/(2*G);
                e.phase_remainder+=weight*traction_energy*(h1.h-h0.h-h1.first*(phis[q]-old_phi));
                std::vector<BC::Tensor> shapeE(dpc);
                std::vector<double> div(dpc),P(dpc),N(dpc);
                for(unsigned int i=0;i<dpc;++i)
                  {
                    shapeE[i]=embed(fe[introspection.extractors.velocities].symmetric_gradient(i,q));
                    div[i]=fe[introspection.extractors.velocities].divergence(i,q);
                    P[i]=fe[introspection.extractors.pressure].value(i,q);
                    N[i]=fe.shape_value_component(i,q,pc);
                    e.raw[ids[i]]+=weight*((r.stress*shapeE[i])/G-(fixed_coulomb?total_pressure/G:pressures[q])*div[i]-P[i]*trace(gradients[q])-N[i]*r.driving/Kf);
                  }
                if(tangent)
                  {
                    std::vector<TR::Variation> actions;
                    if(parent->tensor_flow())
                      {
                        actions.resize(dpc);
                        for(unsigned int j=0;j<dpc;++j)if(selected(ids[j]))
                          actions[j]=fixed_coulomb && !gradient
                            ? CR::derivative({p,parent->coulomb_mu,0.},tensor_incoming,*r.coulomb_point,shapeE[j],N[j],dt,G*P[j]).point
                            : (gradient ? TR::Variation{} : TR::derivative(p,tensor_incoming,r.tensor_point,shapeE[j],N[j],dt,fixed));
                        if(gradient)for(unsigned int j=0;j<dpc;++j)actions[j].stress=2*G*beta*shapeE[j];
                      }
                    for(unsigned int i=0;i<dpc;++i)if(selected(ids[i]))
                      for(unsigned int j=0;j<dpc;++j)if(selected(ids[j]))
                        {
                          double value;
                          if(parent->tensor_flow())
                            value=(actions[j].stress*shapeE[i])/G-div[i]*P[j]-P[i]*div[j]
                              -N[i]*actions[j].driving/Kf;
                          else
                            {
                              const double dt0=2*G*beta*shapeE[j][0][1];
                              const double da=r.da_dt*dt0+r.da_dphi*N[j];
                              value=2*beta*(shapeE[i]*shapeE[j])-2*da*(shapeE[i]*S)
                                -div[i]*P[j]-P[i]*div[j]
                                -N[i]*(r.driving_dt*dt0+r.driving_dphi*N[j])/Kf;
                            }
                          if(value!=0.)jacobian.add(ids[i],ids[j],weight*value);
                        }
                    if(gradient)
                      {
                        const auto point=parent->slip_space->ids.at(cell->id())[q];const auto &joint=e.joint.at(cell->id())[q];
                        BC::Tensor basis[2];basis[0][0][0]=1.;basis[0][1][1]=-1.;basis[1][0][1]=1.;
                        for(unsigned int j=0;j<dpc;++j)if(selected(ids[j]))
                          {
                            const auto a=SG::joint_action({p,parent->coulomb_mu,0.},joint,shapeE[j],{}, {},dt,G*P[j]);
                            jacobian.add(native_n+2*point,ids[j],a.residual[0][0]/G);
                            jacobian.add(native_n+2*point+1,ids[j],a.residual[0][1]/G);
                          }
                        for(unsigned int c=0;c<2;++c)
                          for(unsigned int i=0;i<dpc;++i)if(selected(ids[i]))
                            jacobian.add(ids[i],native_n+2*point+c,-2*weight*beta*(basis[c]*shapeE[i]));
                        for(auto it=parent->slip_space->stiffness.begin(point);it!=parent->slip_space->stiffness.end(point);++it)
                          for(unsigned int c=0;c<2;++c)
                            {
                              const auto dk=it->column()==point?basis[c]:BC::Tensor();
                              const auto dx=2*G*parent->slip_length*parent->slip_length*it->value()/parent->slip_space->mass[point]*basis[c];
                              const auto a=SG::joint_action({p,parent->coulomb_mu,0.},joint,{},dk,dx,dt,0.);
                              jacobian.add(native_n+2*point,native_n+2*it->column()+c,a.residual[0][0]/G);
                              jacobian.add(native_n+2*point+1,native_n+2*it->column()+c,a.residual[0][1]/G);
                            }
                      }
                  }
                }
                catch(const std::exception &error)
                  {
                    if(!fixed_coulomb)throw;
                    if(e.contact_error.empty())
                      {
                        std::ostringstream detail;detail<<std::setprecision(17)<<"cell="<<cell->id()<<", q="<<q
                          <<", reference="<<quadrature.point(q)<<", position="<<fe.quadrature_point(q)
                          <<", total_pressure="<<parent->background_pressure+G*pressures[q]<<": "<<error.what();
                        e.contact_error=detail.str();
                      }
                  }
              }
          }
        if(fixed_coulomb)
          {
            for(const auto &error:Utilities::MPI::all_gather(mpi_communicator,e.contact_error))
              if(!error.empty()){e.admissible=false;e.contact_error=error;break;}
            if(!e.admissible){e.norm=std::numeric_limits<double>::infinity();return e;}
          }
        assembly_seconds+=seconds(assembly_start);
        const auto communication_start=Clock::now();
        BC::sum(e.raw,mpi_communicator);
        if(tangent && ranks>1)
          {
            std::vector<double> entries;entries.reserve(jacobian.n_nonzero_elements());
            for(auto it=jacobian.begin();it!=jacobian.end();++it)entries.push_back(it->value());
            internal::endpoint_sum(entries,mpi_communicator);
            unsigned int k=0;for(auto it=jacobian.begin();it!=jacobian.end();++it)it->value()=entries[k++];
          }
        if(ranks>1)
          for(double *value:{&e.bulk,&e.cohesive,&e.Db,&e.Dc,&e.Df,&e.storage_change,&e.internal_work,
                            &e.pressure_work,&e.stress_loss,&e.traction_loss,&e.phase_remainder})
            *value=Utilities::MPI::sum(*value,mpi_communicator);
        communication_seconds+=seconds(communication_start);
        for(unsigned int i=0;i<np;++i)e.raw[po+i]+=fg[i]/Kf;
        if(tangent)for(const auto &entry:fracture_entries)jacobian.add(po+entry.i,po+entry.j,entry.value/Kf);
        // Test-only export of the actual raw/condensed actions. Independent
        // reconstruction of C checks hanging-node reduction exactly once;
        // these files never select a constitutive or acceptance rule.
        const bool export_constraints=continuation_diagnostic && tangent && rank==0
          && std::getenv("ASPECT_BULK_CHECK_CONSTRAINT_ACTION");
        if(export_constraints)
          {
            std::ofstream raw(parameters.output_directory+"constraint-raw.bin",std::ios::binary);
            const std::uint64_t size=n,count=jacobian.n_nonzero_elements();
            raw.write(reinterpret_cast<const char*>(&size),sizeof(size));
            raw.write(reinterpret_cast<const char*>(&count),sizeof(count));
            for(unsigned int i=0;i<n;++i)for(auto it=jacobian.begin(i);it!=jacobian.end(i);++it)
              {
                const std::uint32_t row=i,column=it->column();const double value=it->value();
                raw.write(reinterpret_cast<const char*>(&row),sizeof(row));
                raw.write(reinterpret_cast<const char*>(&column),sizeof(column));
                raw.write(reinterpret_cast<const char*>(&value),sizeof(value));
              }
            std::ofstream rules(parameters.output_directory+"constraint-rules.csv");
            rules<<std::setprecision(17)<<"slave,master,weight\n";
            for(const auto &line:homogeneous.get_lines())
              if(line.entries.empty())rules<<line.index<<",-1,0\n";
              else for(const auto &entry:line.entries)rules<<line.index<<','<<entry.first<<','<<entry.second<<'\n';
          }
        e.residual=e.raw;
        if(tangent)homogeneous.condense(jacobian,e.residual);
        else homogeneous.condense(e.residual);
        if(export_constraints)
          {
            std::ofstream raw(parameters.output_directory+"constraint-residual.csv");
            raw<<std::setprecision(17)<<"dof,raw,condensed\n";
            for(unsigned int i=0;i<n;++i)raw<<i<<','<<e.raw[i]<<','<<e.residual[i]<<'\n';
          }
        Vector<double> raw_phase(np);
        for(unsigned int i=0;i<np;++i)raw_phase[i]=e.raw[po+i];
        e.phase_gradient=phase_coordinates.transpose(raw_phase);
        const auto phase_values=[&](){Vector<double> a(np);for(unsigned int i=0;i<np;++i)a[i]=v[po+i];return phase_coordinates.select(a);}();
        for(unsigned int i=0;i<native_n;++i)if(selected(i) && !base.is_constrained(i) && !(i>=po && i<po+np))
          e.mechanical+=e.residual[i]*e.residual[i];
        e.mechanical=std::sqrt(e.mechanical);
        double phase_norm=0.;
        for(unsigned int i=0;!fixed_coulomb && i<phase_coordinates.size();++i)
          {
            const double a=std::min(phase_values[i]-lower[i],e.phase_gradient[i]/mass[i]);
            e.kkt=std::max(e.kkt,std::abs(a));phase_norm+=a*a;
          }
        e.norm=std::sqrt(std::pow(e.mechanical/mechanical_tolerance,2)+phase_norm/(phase_tolerance*phase_tolerance)+std::pow(e.micro/1e-12,2));
        for(const auto &line:base.get_lines())
          if(line.index<nv && line.entries.empty())e.external_work+=G*v[line.index]*e.raw[line.index];
        return e;
      };

    // Read-only audit of the actual mixed tangent and its local constitutive
    // actions. No frozen-pressure surrogate or phase-stability certificate.
    const auto export_localization=[&](const Evaluation &evaluation,const std::string &label)
      {
        if(!fixed_coulomb || !std::getenv("ASPECT_COULOMB_LOCALIZATION"))return;
        const std::string stem="localization-"+Utilities::int_to_string(timestep_number,4)+"-"+label;
        std::ofstream points(parameters.output_directory+stem+"-rank-"+std::to_string(rank)+".csv");
        points<<std::setprecision(17)<<"cell,q,x,y,JxW,phi,h,active,dt,beta,P,Y,Wxx,Wxy,gamma_plus,gamma_minus,theta_plus,theta_minus,tauxx,tauyy,tauzz,tauxy,Qxx,Qxy";
        if(gradient)points<<",Xxx,Xxy,force_error,force_bound";
        for(unsigned int j=0;!gradient && j<5;++j)for(const char *component:{"xx","yy","zz","xy","Y"})points<<",d"<<j<<'_'<<component;
        points<<'\n';
        for(const auto &entry:evaluation.points)
          {
            fe.reinit(parent->native_cells.at(entry.first));
            for(unsigned int q=0;q<entry.second.size();++q)
              {
                const auto &response=*entry.second[q].coulomb_point;const auto &r=response.point;
                const auto &old=parent->history.at(entry.first)[q];TR::State incoming;
                incoming.T[0][0]=oldT.at(entry.first)[q][0][0];incoming.T[1][1]=oldT.at(entry.first)[q][1][1];
                incoming.T[2][2]=-incoming.T[0][0]-incoming.T[1][1];incoming.T[0][1]=oldT.at(entry.first)[q][0][1];
                incoming.A=old.A;incoming.K=old.K;incoming.Q=old.Q;incoming.L=old.L;incoming.phi=r.state.phi;
                const auto point=fe.quadrature_point(q);
                points<<entry.first<<','<<q<<','<<point[0]<<','<<point[1]<<','<<fe.JxW(q)<<','<<r.state.phi<<','
                      <<BC::compliance(r.state.phi,p.m).h<<','<<response.active<<','<<dt<<','<<r.beta<<','<<response.pressure<<','<<r.driving<<','
                      <<r.flow_rate[0][0]<<','<<r.flow_rate[0][1]<<','<<response.gamma[0]<<','<<response.gamma[1]<<','
                      <<response.planes[0].theta<<','<<response.planes[1].theta<<','<<r.stress[0][0]<<','<<r.stress[1][1]<<','
                      <<r.stress[2][2]<<','<<r.stress[0][1]<<','<<r.state.Q[0][0]<<','<<r.state.Q[0][1];
                if(gradient)
                  {
                    const auto &joint=evaluation.joint.at(entry.first)[q];
                    const auto X=TR::in_plane(r.stress)-r.state.Q-joint.projection.input+joint.b*dt*r.flow_rate;
                    points<<','<<X[0][0]<<','<<X[0][1]<<','<<joint.residual.norm()<<','<<1e-12*joint.force_scale;
                  }
                for(unsigned int j=0;!gradient && j<5;++j)
                  {
                    SymmetricTensor<2,dim> raw;if(j==0)raw[0][0]=1.;if(j==1)raw[1][1]=1.;if(j==2)raw[0][1]=.5;
                    const auto d=CR::derivative({p,parent->coulomb_mu,0.},incoming,response,embed(raw),j==4?1.:0.,dt,j==3?G:0.).point;
                    for(double value:{d.stress[0][0]/G,d.stress[1][1]/G,d.stress[2][2]/G,d.stress[0][1]/G,d.driving/Kf})points<<','<<value;
                  }
                points<<'\n';
              }
          }
        if(rank!=0)return;
        std::vector<unsigned int> free,index(n,numbers::invalid_unsigned_int);
        for(unsigned int i=0;i<n;++i)if(selected(i) && !base.is_constrained(i)){index[i]=free.size();free.push_back(i);}
        std::uint64_t count=0;
        for(const auto i:free)for(auto it=jacobian.begin(i);it!=jacobian.end(i);++it)
          if(index[it->column()]!=numbers::invalid_unsigned_int && it->value()!=0.)++count;
        std::ofstream matrix(parameters.output_directory+stem+".bin",std::ios::binary);
        const std::uint64_t size=free.size();matrix.write(reinterpret_cast<const char*>(&size),sizeof(size));matrix.write(reinterpret_cast<const char*>(&count),sizeof(count));
        std::ofstream coordinates(parameters.output_directory+stem+"-coordinates.csv");coordinates<<"index,dof,kind\n";
        for(const auto i:free)
          {
            coordinates<<index[i]<<','<<i<<','<<(i>=native_n?"micro":(i<nv?"velocity":"pressure"))<<'\n';
            for(auto it=jacobian.begin(i);it!=jacobian.end(i);++it)
              if(index[it->column()]!=numbers::invalid_unsigned_int && it->value()!=0.)
                {
                  const std::uint32_t row=index[i],column=index[it->column()];const double value=it->value();
                  matrix.write(reinterpret_cast<const char*>(&row),sizeof(row));matrix.write(reinterpret_cast<const char*>(&column),sizeof(column));matrix.write(reinterpret_cast<const char*>(&value),sizeof(value));
                }
          }
        std::ofstream metadata(parameters.output_directory+stem+".json");
        metadata<<std::setprecision(17)<<"{\"time\":"<<time<<",\"dt\":"<<dt<<",\"beta\":"<<beta
                <<",\"parent_publications\":"<<parent->publications<<",\"mechanical_residual\":"<<evaluation.mechanical
                <<",\"label\":\""<<label<<"\"}\n";
      };
    const bool critical_tangent=std::getenv("ASPECT_BULK_CHECK_CRITICAL_TANGENT");
    Evaluation accepted;
    bool converged=false;
    unsigned int stability_retries=0;
    // One initial solve and at most one immutable-parent continuation retry.
    for(unsigned int attempt=0;attempt<2;++attempt)
    {
    converged=false;
    for(unsigned int iteration=0;iteration<parameters.max_nonlinear_iterations;++iteration)
      {
        auto current=evaluate(x,true);
        AssertThrow(current.admissible,ExcMessage("Closed Coulomb current iterate is inadmissible: "+current.contact_error));
        if((fixed_coulomb && std::getenv("ASPECT_BULK_CHECK_TANGENT")
            && current.mechanical<=mechanical_tolerance && (timestep_number==8 || timestep_number==16 || timestep_number==24))
           || (std::getenv("ASPECT_BULK_CHECK_TANGENT") && iteration==0
            && time>=.11-1e-12 && time<.11+.5*dt)
           || (continuation_diagnostic && iteration==0
               && std::getenv("ASPECT_BULK_CHECK_TANGENT")
               && std::getenv("ASPECT_BULK_CHECK_PREDICTOR_TANGENT"))
           || (continuation_diagnostic && (std::getenv("ASPECT_BULK_CHECK_TANGENT") || critical_tangent)
               && current.mechanical<=mechanical_tolerance && current.kkt<=phase_tolerance))
          {
            for(unsigned int probe=0;probe<(fixed_coulomb?3u:1u);++probe)
            {
            Vector<double> d(n);
            std::vector<bool> critical_phase(np,false);
            unsigned int critical_count=0;
            if(critical_tangent)for(unsigned int k=0;k<phase_coordinates.size();++k)
              {
                const auto physical=phase_coordinates.physical_index(k);
                critical_phase[physical]=x[po+physical]-lower[k]>1e-9
                  || current.phase_gradient[k]/mass[k]<=10*phase_tolerance;
                critical_count+=critical_phase[physical];
              }
            const auto restrict_action=[&](Vector<double> &v)
              {
                for(unsigned int i=0;i<n;++i)
                  if(!selected(i) || base.is_constrained(i)
                     || (critical_tangent && i>=po && i<po+np && !critical_phase[i-po]))v[i]=0.;
              };
            for(const auto &cell:dof_handler.active_cell_iterators())if(!cell->is_artificial())
              {
                cell->get_dof_indices(ids);
                for(unsigned int i=0;i<dpc;++i)if(selected(ids[i]))
                  {
                    const auto point=mapping->transform_unit_to_real_cell(cell,finite_element.get_unit_support_points()[i]);
                    const double sx=std::sin(numbers::PI*point[0]),sy=std::sin(numbers::PI*point[1]);
                    const unsigned int c=finite_element.system_to_component_index(i).first;
                    if(c==pc)
                      {
                        // A constant absolute phase probe need not stay in a
                        // smooth return branch near h=0. For the separate
                        // stability check, vary only interior phase masters,
                        // relative to their physical distance from the obstacle.
                        const double gap=x[ids[i]]-parent->phase[ids[i]-po];
                        d[ids[i]]=critical_tangent
                          ? (critical_phase[ids[i]-po]?(gap>1e-9?.01*gap:(x[ids[i]]>0.?.01*x[ids[i]]:1e-5)):0.)
                          :(.01+.005*sx*sx);
                      }
                    else if(ids[i]<nv)d[ids[i]]=fixed_coulomb
                      ? .001*std::sin(numbers::PI*point[0]/.25)*std::sin(numbers::PI*point[1]/.25)*(point[c]-.125)
                      : .001*sx*sy*(point[c]-.5);
                    else d[ids[i]]=fixed_coulomb
                      ? .001*std::cos(2*numbers::PI*point[0]/.25)*std::cos(2*numbers::PI*point[1]/.25)
                      : .001*sx*sy;
                  }
              }
            if(gradient)for(unsigned int k=0;k<micro_points;++k)
              for(unsigned int c=0;c<2;++c)d[native_n+2*k+c]=.0001*std::cos(2*numbers::PI*parent->slip_space->points[k][c]/.25);
            if(fixed_coulomb)
              for(unsigned int i=0;i<n;++i)
                if((probe==1 && i<nv) || (probe==2 && i>=pressure_begin && i<pressure_end))d[i]=0.;
            // Shared coefficients are identical functions of physical position.
            // Sum only uniquely owned DoFs before imposing global constraints.
            if(ranks>1)
              {for(unsigned int i=0;i<n;++i)if(!dof_handler.locally_owned_dofs().is_element(i))d[i]=0.;BC::sum(d,mpi_communicator);}
            homogeneous.distribute(d);
            Vector<double> full_action(n);jacobian.vmult(full_action,d);
            // condense() supplies artificial slave-row diagonals for solving.
            // The derivative of C^T residual lives only on independent rows;
            // a distributed hanging-node direction is nonzero at slaves.
            for(unsigned int i=0;i<n;++i)
              if(!selected(i) || base.is_constrained(i))full_action[i]=0.;
            Vector<double> action(full_action);restrict_action(action);
            double maximum_error=0.,maximum_relative_error=0.,maximum_full_error=0.;
            for(double requested_epsilon:(fixed_coulomb?std::array<double,2>{{1e-4,5e-5}}:std::array<double,2>{{1e-7,5e-8}}))
              {
                double epsilon=requested_epsilon;unsigned int reductions=0;
                Vector<double> a(x),b(x);Evaluation first,second;
                for(;;)
                  {
                    a=x;b=x;a.add(epsilon,d);b.add(2*epsilon,d);
                    first=evaluate(a,false);second=evaluate(b,false);
                    if(!fixed_coulomb)break;
                    unsigned int same=first.admissible && second.admissible;
                    if(same)for(const auto &entry:current.points)for(unsigned int q=0;q<entry.second.size();++q)
                      same= same && entry.second[q].coulomb_point->active==first.points.at(entry.first)[q].coulomb_point->active
                        && entry.second[q].coulomb_point->active==second.points.at(entry.first)[q].coulomb_point->active;
                    same=Utilities::MPI::min(same,mpi_communicator);
                    if(same)break;
                    AssertThrow(reductions<12,ExcMessage("No smooth closed-contact Coulomb FD probe after 12 halvings; preserve the generalized-derivative boundary separately."));
                    epsilon*=.5;++reductions;
                  }
                if(fixed_coulomb && rank==0)
                  {
                    std::ofstream probes(parameters.output_directory+"coulomb-fd-probes.csv",std::ios::app);
                    if(probes.tellp()==0)probes<<"time,probe,requested_epsilon,used_epsilon,branch_reductions\n";
                    probes<<std::setprecision(17)<<time<<','<<probe<<','<<requested_epsilon<<','<<epsilon<<','<<reductions<<'\n';
                  }
                AssertThrow(first.admissible && second.admissible,ExcMessage("Coulomb tangent probe leaves closed contact: "+first.contact_error+second.contact_error));
                Vector<double> fd(first.residual);fd*=4.;fd.add(-1.,second.residual);fd.add(-3.,current.residual);fd/=(2*epsilon);fd-=full_action;
                maximum_full_error=std::max(maximum_full_error,fd.l2_norm()/std::max(1.,full_action.l2_norm()));
                restrict_action(fd);
                const double error=fd.l2_norm()/std::max(1.,action.l2_norm());
                maximum_error=std::max(maximum_error,error);
                maximum_relative_error=std::max(maximum_relative_error,fd.l2_norm()/std::max(1e-100,action.l2_norm()));
                if(error>1e-6 || (critical_tangent && fd.l2_norm()/std::max(1e-100,action.l2_norm())>1e-6))
                  {
                    // A Coulomb switch admits a generalized Newton derivative,
                    // not every classical directional derivative. Preserve the
                    // failed original check and diagnose it on the probed side;
                    // never increase its tolerance or suppress its exception.
                    const auto at_first=evaluate(a,true);
                    Vector<double> right_action(n);jacobian.vmult(right_action,d);
                    Vector<double> c(x);c.add(3*epsilon,d);
                    const auto third=evaluate(c,false);
                    Vector<double> side(first.residual);side*=-3.;side.add(4.,second.residual);
                    side.add(-1.,third.residual);side/=(2*epsilon);side-=right_action;
                    restrict_action(side);restrict_action(right_action);
                    const double side_error=side.l2_norm()/std::max(1.,right_action.l2_norm());
                    unsigned int switches=0,side_switches=0,written=0;
                    std::ofstream points(parameters.output_directory+"bulk-tangent-switches-rank-"+std::to_string(rank)+".csv",std::ios::app);
                    if(points.tellp()==0)points<<"epsilon,cell,qp,x,y,sliding0,sliding1,sliding2,w0,w1,w2,phi0,q0\n";
                    points<<std::setprecision(17);
                    for(const auto &entry:current.points)for(unsigned int q=0;q<entry.second.size();++q)
                      {
                        const auto &r=entry.second[q],&s=first.points.at(entry.first)[q],&t=second.points.at(entry.first)[q];
                        const bool changed=r.sliding!=s.sliding;
                        switches+=changed;side_switches+=s.sliding!=t.sliding;
                        if(changed && written++<32)
                          {
                            const auto point=mapping->transform_unit_to_real_cell(parent->native_cells.at(entry.first),quadrature.point(q));
                            points<<epsilon<<','<<entry.first.to_string()<<','<<q<<','<<point[0]<<','<<point[1]<<','
                                  <<r.sliding<<','<<s.sliding<<','<<t.sliding<<','<<r.w<<','<<s.w<<','<<t.w<<','<<r.state.phi<<','<<r.state.q<<'\n';
                          }
                      }
                    switches=Utilities::MPI::sum(switches,mpi_communicator);
                    side_switches=Utilities::MPI::sum(side_switches,mpi_communicator);
                    if(rank==0)
                      {
                        std::ofstream detail(parameters.output_directory+"bulk-tangent-failure.csv",std::ios::app);
                        if(detail.tellp()==0)detail<<"time,epsilon,error,bound,ratio,probed_side_error,switches,side_switches\n";
                        detail<<std::setprecision(17)<<time<<','<<epsilon<<','<<error<<",1e-6,"<<error/1e-6<<','<<side_error<<','<<switches<<','<<side_switches<<'\n';
                        std::vector<std::pair<double,unsigned int>> order;
                        for(unsigned int i=0;i<n;++i)if(selected(i) && !base.is_constrained(i))order.emplace_back(std::abs(fd[i]),i);
                        std::sort(order.begin(),order.end(),std::greater<>());
                        std::ofstream rows(parameters.output_directory+"bulk-tangent-failed-rows.csv",std::ios::app);
                        if(rows.tellp()==0)rows<<"epsilon,dof,action,error\n";
                        rows<<std::setprecision(17);
                        for(unsigned int k=0;k<std::min<std::size_t>(32,order.size());++k)
                          {const auto i=order[k].second;rows<<epsilon<<','<<i<<','<<action[i]<<','<<fd[i]<<'\n';}
                      }
                    // The diagnostic evaluated another tangent. Restore the
                    // original matrix before any Newton action or export.
                    evaluate(x,true);
                  }
              }
            std::ofstream out;if(rank==0)out.open(parameters.output_directory+(fixed_coulomb?"coulomb-tangent.csv":(critical_tangent?"bulk-critical-tangent.csv":"bulk-tangent.csv")),fixed_coulomb?std::ios::app:std::ios::out);
            if(!fixed_coulomb || out.tellp()==0)
              {out<<"time,action_norm,maximum_FD_error,relative_action_error,full_space_error,critical_phase_dofs";if(fixed_coulomb)out<<",probe";out<<'\n';}
            out<<std::setprecision(17)<<time<<','<<action.l2_norm()<<','<<maximum_error<<','<<maximum_relative_error<<','<<maximum_full_error<<','<<critical_count;if(fixed_coulomb)out<<','<<probe;out<<'\n';
            AssertThrow(maximum_error<=1e-6 && (!(critical_tangent || fixed_coulomb) || maximum_relative_error<=1e-6),
                        ExcMessage("Independent native bulk tangent action failed."));
            }
          }
        pcout<<"   Bulk cohesive Newton "<<iteration<<": mechanics="<<current.mechanical<<", phase KKT="<<current.kkt;
        if(gradient)pcout<<", microforce="<<current.micro;
        pcout<<std::endl;
        if(gradient && current.mechanical<=mechanical_tolerance && current.micro<=1e-12)
          {
            // Sticking equations are exactly DeltaK=0. Distribute those
            // active equations, then reevaluate ALL forces before accepting.
            bool changed=false;
            for(const auto &cell:current.joint)for(unsigned int q=0;q<cell.second.size();++q)
              if(cell.second[q].projection.active==0)
                {
                  const auto k=parent->slip_space->ids.at(cell.first)[q];
                  for(unsigned int c=0;c<2;++c)
                    {const double value=c==0?oldK[k][0][0]:oldK[k][0][1];changed=changed || x[native_n+2*k+c]!=value;x[native_n+2*k+c]=value;}
                }
            if(changed)continue;
          }
        if(current.mechanical<=mechanical_tolerance && current.kkt<=phase_tolerance && current.micro<=1e-12)
          {export_localization(current,"converged");accepted=std::move(current);converged=true;break;}
        const auto linear_start=Clock::now();++linear_solves;
        std::vector<bool> active(n,false);
        Vector<double> direction(n);
        if(gradient)for(const auto &cell:current.joint)for(unsigned int q=0;q<cell.second.size();++q)
          if(cell.second[q].projection.active==0 || (parent->step==0 && iteration==0))
            {
              // Fresh-block initial lifting only: zero interior displacement
              // with imposed boundary values creates a mesh-dependent trial
              // boundary layer. One fixed-K elastic solve supplies a continuous
              // mechanical guess. All slip variables are released immediately
              // afterwards and the full force/residual checks select acceptance.
              const auto k=parent->slip_space->ids.at(cell.first)[q];
              for(unsigned int c=0;c<2;++c)
                {const auto i=native_n+2*k+c;active[i]=true;direction[i]=(c==0?oldK[k][0][0]:oldK[k][0][1])-x[i];}
            }
        for(unsigned int i=0;!fixed_coulomb && i<phase_coordinates.size();++i)
          {
            const auto global=po+phase_coordinates.physical_index(i);
            if(x[global]-lower[i]<=current.phase_gradient[i]/mass[i])
              {active[global]=true;direction[global]=lower[i]-x[global];}
          }
        std::vector<unsigned int> free;
        std::vector<unsigned int> index(n,numbers::invalid_unsigned_int);
        for(unsigned int i=0;i<n;++i)if(selected(i) && !base.is_constrained(i) && !active[i])
          {index[i]=free.size();free.push_back(i);}
        DynamicSparsityPattern ds(free.size(),free.size());
        Vector<double> rhs(free.size()),delta(free.size());
        for(unsigned int k=0;k<free.size();++k)
          for(auto entry=jacobian.begin(free[k]);entry!=jacobian.end(free[k]);++entry)
            if(index[entry->column()]!=numbers::invalid_unsigned_int)ds.add(k,index[entry->column()]);
        SparsityPattern sp;sp.copy_from(ds);SparseMatrix<double> matrix(sp);
        for(unsigned int k=0;k<free.size();++k)
          {
            rhs[k]=-current.residual[free[k]];
            for(auto entry=jacobian.begin(free[k]);entry!=jacobian.end(free[k]);++entry)
              if(index[entry->column()]!=numbers::invalid_unsigned_int)matrix.set(k,index[entry->column()],entry->value());
              else rhs[k]-=entry->value()*direction[entry->column()];
          }
        std::string linear_error;
        if(rank==0)
          try{SparseDirectUMFPACK inverse;inverse.initialize(matrix);inverse.vmult(delta,rhs);}
          catch(const std::exception &error){linear_error=error.what();}
        linear_error=Utilities::MPI::broadcast(mpi_communicator,linear_error,0);
        if(!linear_error.empty())export_localization(current,"failed-linear");
        AssertThrow(linear_error.empty(),ExcMessage(linear_error));
        if(ranks>1){std::vector<double> values(delta.begin(),delta.end());values=Utilities::MPI::broadcast(mpi_communicator,values,0);std::copy(values.begin(),values.end(),delta.begin());}
        linear_seconds+=seconds(linear_start);
        for(unsigned int k=0;k<free.size();++k)direction[free[k]]=delta[k];
        homogeneous.distribute(direction);
        bool found=false;
        for(unsigned int line=0;line<settings.max_newton_line_search_iterations;++line)
          {
            const double alpha=std::ldexp(1.,-static_cast<int>(line));
            Vector<double> trial(x);trial.add(alpha,direction);
            // Feasible projected Newton search in independent master space.
            // A mechanically unequilibrated iterate can predict an inactive
            // phase row while its coupled Newton step points below the lower
            // obstacle. Rejecting every positive step then cannot globalize.
            // Project the trial onto the physical obstacle; never cap phi at
            // an initialization value or regularize its upper endpoint.
            for(unsigned int i=0;i<phase_coordinates.size();++i)
              {
                const auto global=po+phase_coordinates.physical_index(i);
                trial[global]=std::max(lower[i],trial[global]);
              }
            base.distribute(trial);
            bool admissible=true;
            for(unsigned int i=0;i<np;++i)admissible=admissible && trial[po+i]>=parent->phase[i] && trial[po+i]<1.;
            if(!admissible)continue;
            const auto candidate=evaluate(trial,false);
            if(candidate.admissible && candidate.norm<(1.-1e-4*alpha)*current.norm)
              {x=std::move(trial);found=true;break;}
          }
        if(!found)
          {
            export_localization(current,"failed-linesearch");
            std::ofstream failure;if(rank==0)failure.open(parameters.output_directory+"bulk-failed-direction.csv");
            failure<<std::setprecision(17)<<"time,iteration,dof,phi,obstacle,direction,gradient_density,active\n";
            for(unsigned int i=0;i<phase_coordinates.size();++i)
              {
                const auto global=po+phase_coordinates.physical_index(i);
                failure<<time<<','<<iteration<<','<<phase_coordinates.physical_index(i)<<','<<x[global]<<','<<lower[i]<<','<<direction[global]<<','<<current.phase_gradient[i]/mass[i]<<','<<active[global]<<'\n';
              }
            throw ExcNonlinearSolverNoConvergence();
          }
        if(iteration+1==parameters.max_nonlinear_iterations)export_localization(current,"failed-limit");
      }
    if(!converged)throw ExcNonlinearSolverNoConvergence();

    if(fixed_coulomb)
      {
        // There are no admissible phase variations: this is not a Coulomb
        // phase-stability certificate. Its phase-active mode is rejected above.
        for(unsigned int i=0;i<np;++i)AssertThrow(x[po+i]==parent->phase[i],ExcInternalError());
        if(gradient)for(const auto &cell:accepted.joint)for(const auto &joint:cell.second)
          {
            AssertThrow(joint.local.point.friction_dissipation>=0.,ExcMessage("Negative physical slip dissipation at an accepted native quadrature point."));
            for(unsigned int system=0;system<2;++system)
              AssertThrow(joint.local.planes[system].margin<=1e-12*joint.force_scale
                          && joint.local.gamma[system]>=-1e-12,
                          ExcMessage("Accepted gradient Coulomb force/system-rate admissibility failed."));
          }
        break;
      }
    // This potential exists for the retained homogeneous matched-coefficient
    // bulk law only (eligibility is checked by the material model). Critical
    // masters include every strictly interior node and all obstacle nodes
    // whose normalized multiplier is unresolved at the EXISTING KKT tolerance.
    // Positivity on this linear superset suffices for the admissible cone;
    // negativity is called an instability only with a physical feasible mode.
    const auto stability_start=Clock::now();
    std::vector<unsigned int> mechanics,critical;
    Vector<double> metric(phase_coordinates.size());
    for(unsigned int i=0;i<n;++i)
      if(selected(i) && !base.is_constrained(i) && !(i>=po && i<po+np))mechanics.push_back(i);
    for(unsigned int k=0;k<phase_coordinates.size();++k)
      {
        const auto global=po+phase_coordinates.physical_index(k);
        if(x[global]>lower[k] || accepted.phase_gradient[k]/mass[k]<=phase_tolerance)
          {metric[critical.size()]=Kf*mass[k];critical.push_back(global);}
      }
    metric.grow_or_shrink(critical.size());
    bool stable=critical.empty();
    std::string stability_error;
    if(rank==0)
      {
        BC::RelaxedCurvature curvature;
        double feasible_plus=0.,feasible_minus=0.,maximum_growth=0.;
        for(unsigned int i=0;i<np;++i)maximum_growth=std::max(maximum_growth,x[po+i]-parent->phase[i]);
        try
          {
            if(!critical.empty())
              {
                curvature=BC::relaxed_curvature(jacobian,mechanics,critical,metric,beta*G,Kf,mechanical_tolerance);
                stable=curvature.minimum>curvature.uncertainty;
                Vector<double> master_mode(phase_coordinates.size());
                for(unsigned int k=0;k<phase_coordinates.size();++k)
                  {
                    const auto found=std::lower_bound(critical.begin(),critical.end(),po+phase_coordinates.physical_index(k));
                    if(found!=critical.end() && *found==po+phase_coordinates.physical_index(k))
                      master_mode[k]=curvature.mode[found-critical.begin()];
                  }
                auto mode=phase_coordinates.expand(master_mode);mode/=mode.linfty_norm();
                feasible_plus=feasible_minus=std::numeric_limits<double>::max();
                for(unsigned int i=0;i<np;++i)if(mode[i]!=0.)
                  {
                    const double gap=x[po+i]-parent->phase[i],upper=1.-x[po+i];
                    feasible_plus=std::min(feasible_plus,(mode[i]>0.?upper:gap)/std::abs(mode[i]));
                    feasible_minus=std::min(feasible_minus,(mode[i]>0.?gap:upper)/std::abs(mode[i]));
                  }
                if(!stable)
                  {
                    stability_error=curvature.minimum < -curvature.uncertainty && std::max(feasible_plus,feasible_minus)>0.
                      ? "Admissible mechanically relaxed negative curvature"
                      : "Inconclusive critical-cone stability (no admissible negative-mode certificate or unresolved curvature)";
                    std::ofstream mode_out(parameters.output_directory+"bulk-rejected-mode-"+std::to_string(timestep_number)+"-"+std::to_string(attempt)+".csv");
                    mode_out<<std::setprecision(17)<<"phase_dof,phi,obstacle,mode\n";
                    for(unsigned int i=0;i<np;++i)if(mode[i]!=0.)mode_out<<i<<','<<x[po+i]<<','<<parent->phase[i]<<','<<mode[i]<<'\n';
                  }
              }
          }
        catch(const std::exception &error){stable=false;stability_error=error.what();}
        std::ofstream out(parameters.output_directory+"bulk-stability.csv",std::ios::app);
        if(out.tellp()==0)out<<"step,time,attempt,accepted_parent_publications,continuation_guess,critical_masters,minimum_curvature,uncertainty,eigen_residual,linear_residual,asymmetry,cross_symmetry_relative,feasible_plus,feasible_minus,max_phase_increment,stable,seconds\n";
        out<<std::setprecision(17)<<timestep_number<<','<<time<<','<<attempt<<','<<parent->publications<<','<<relaxation_guess<<','<<critical.size()<<','
           <<curvature.minimum<<','<<curvature.uncertainty<<','<<curvature.eigen_residual<<','<<curvature.linear_residual<<','<<curvature.asymmetry<<','<<curvature.cross_symmetry_relative<<','
           <<feasible_plus<<','<<feasible_minus<<','<<maximum_growth<<','<<stable<<','<<seconds(stability_start)<<std::endl;
      }
    stable=Utilities::MPI::broadcast(mpi_communicator,stable,0);
    stability_error=Utilities::MPI::broadcast(mpi_communicator,stability_error,0);
    stability_seconds+=seconds(stability_start);
    AssertThrow(model.bulk_cohesive_state==parent && parent->publications+1==timestep_number
                && !pending_dg_endpoint_publication,ExcMessage("Bulk stability trial changed accepted owners before publication."));
    if(stable)break;
    pcout<<"   Bulk stability rejection before publication: "<<stability_error<<std::endl;
    AssertThrow(attempt==0,ExcMessage("Bulk stable continuation failed after one retry: "+stability_error+". Accepted histories remain unchanged; no publication."));
    ++stability_retries;
    initialize_continuation();
    }

    if(continuation_diagnostic && rank==0)
      {
        // Export the actual condensed tangent on independent, selected DoFs.
        // No artificial constrained-row diagonal enters a stability test.
        // Physical row factors beta*G (mechanics/pressure) and Kf (phase)
        // make the smooth-return tangent symmetric. Eliminating the mixed
        // mechanics block therefore includes constitutive relaxation in the
        // phase second variation. Do not condense these entries a second time.
        std::vector<unsigned int> free,index(n,numbers::invalid_unsigned_int);
        std::vector<unsigned int> phase_master(np,numbers::invalid_unsigned_int);
        for(unsigned int k=0;k<phase_coordinates.size();++k)
          phase_master[phase_coordinates.physical_index(k)]=k;
        for(unsigned int i=0;i<n;++i)if(selected(i) && !base.is_constrained(i))
          {index[i]=free.size();free.push_back(i);}
        std::uint64_t count=0;
        for(const auto i:free)for(auto it=jacobian.begin(i);it!=jacobian.end(i);++it)
          if(index[it->column()]!=numbers::invalid_unsigned_int && it->value()!=0.)++count;
        std::ofstream matrix(parameters.output_directory+"continuation-tangent.bin",std::ios::binary);
        const std::uint64_t size=free.size();
        matrix.write(reinterpret_cast<const char*>(&size),sizeof(size));
        matrix.write(reinterpret_cast<const char*>(&count),sizeof(count));
        std::ofstream coordinates(parameters.output_directory+"continuation-coordinates.csv");
        coordinates<<std::setprecision(17)<<"index,dof,phase_dof,mass,phi,obstacle,physical_gradient\n";
        for(const auto i:free)
          {
            const bool phase=i>=po && i<po+np;
            coordinates<<index[i]<<','<<i<<','<<(phase?static_cast<int>(i-po):-1)<<','
                       <<(phase?mass[phase_master[i-po]]:0.)<<','<<(phase?x[i]:0.)<<','
                       <<(phase?parent->phase[i-po]:0.)<<','<<(phase?Kf*accepted.phase_gradient[phase_master[i-po]]:0.)<<'\n';
            const double scale=phase?Kf:beta*G;
            for(auto it=jacobian.begin(i);it!=jacobian.end(i);++it)
              if(index[it->column()]!=numbers::invalid_unsigned_int && it->value()!=0.)
                {
                  const std::uint32_t row=index[i],column=index[it->column()];
                  const double value=scale*it->value();
                  matrix.write(reinterpret_cast<const char*>(&row),sizeof(row));
                  matrix.write(reinterpret_cast<const char*>(&column),sizeof(column));
                  matrix.write(reinterpret_cast<const char*>(&value),sizeof(value));
                }
          }
        std::ofstream meta(parameters.output_directory+"continuation-state.json");
        meta<<std::setprecision(17)<<"{\"time\":"<<time<<",\"dt\":"<<dt
            <<",\"parent_time\":"<<parent->time<<",\"parent_publications\":"<<parent->publications
            <<",\"beta\":"<<beta<<",\"Kf\":"<<Kf
            <<",\"relaxation_guess\":"<<(relaxation_guess?"true":"false")<<"}\n";
      }

    const auto history_start=Clock::now();
    auto next=std::make_shared<BulkState>(*parent);
    next->tensor=std::make_shared<typename Model::DGHistory>();next->tensor->space=parent->tensor->space;
    for(auto &field:next->tensor->trial_stress)
      field.reinit(next->tensor->space->dofs.locally_owned_dofs(),mpi_communicator);
    // Independent DGQ2 T: actual Gauss3 mass/source projection, retaining all
    // nine coefficients. No projection of the nonlinear cohesive correction.
    const QGauss<dim> source_quadrature(3);
    FEValues<dim> source(*mapping,finite_element,source_quadrature,update_gradients);
    FEValues<dim> dg(*mapping,next->tensor->space->element,source_quadrature,update_values|update_JxW_values);
    std::vector<types::global_dof_index> dids(9);
    std::vector<SymmetricTensor<2,dim>> increment(source_quadrature.size());
    for(const auto &cell:dof_handler.active_cell_iterators())if(cell->is_locally_owned())
      {
        source.reinit(cell);source[introspection.extractors.velocities].get_function_symmetric_gradients(x,increment);
        const auto dc=next->tensor->space->cells.at(cell->id());dg.reinit(dc);dc->get_dof_indices(dids);
        const auto incoming=model.evaluate_dg_trial_tensor(*parent->tensor,cell->id(),source_quadrature);
        FullMatrix<double> mass_matrix(9,9);std::array<Vector<double>,3> rhs;for(auto &v:rhs)v.reinit(9);
        for(unsigned int q=0;q<source_quadrature.size();++q)
          {
            const auto T=beta*(incoming[q]+2*G*Utilities::Tensors::consistent_deviator(increment[q]));
            for(unsigned int i=0;i<9;++i)
              {
                const double weight=dg.JxW(q)*dg.shape_value(i,q);
                for(unsigned int j=0;j<9;++j)mass_matrix(i,j)+=weight*dg.shape_value(j,q);
                rhs[0][i]+=weight*T[0][0];rhs[1][i]+=weight*T[1][1];rhs[2][i]+=weight*T[0][1];
              }
          }
        mass_matrix.gauss_jordan();
        for(unsigned int c=0;c<3;++c)
          {Vector<double> value(9);mass_matrix.vmult(value,rhs[c]);for(unsigned int i=0;i<9;++i)next->tensor->trial_stress[c][dids[i]]=value[i];}
      }
    for(auto &field:next->tensor->trial_stress)field.compress(VectorOperation::insert);
    // Native FE memory retains precisely the part of the physical gradient
    // outside DGQ2 on a non-affine cell. Never feed a projected source into
    // the constitutive residual, tangent or energy. Both owners advance once.
    LinearAlgebra::BlockVector owned_history;
    owned_history.reinit(introspection.index_sets.system_partitioning,mpi_communicator);
    owned_history=parent->native_history;
    for(const auto i:owned_history.block(introspection.block_indices.velocities).locally_owned_elements())
      owned_history[i]=beta*(parent->native_history[i]+x[i]);
    owned_history.compress(VectorOperation::insert);
    next->native_history=owned_history;
    next->rebuild_native_projection(model);
    next->step=timestep_number;next->time=time;next->publications=parent->publications+1;
    for(const auto &entry:accepted.points)
      {
        std::vector<BC::Tensor> expected;
        for(unsigned int q=0;q<entry.second.size();++q)
          {
            const auto &r=entry.second[q];auto &h=next->history[entry.first][q];
            if(parent->tensor_flow())
              {
                h.A=r.tensor_point.state.A;h.K=r.tensor_point.state.K;
                h.Q=r.tensor_point.state.Q;h.W=r.tensor_point.flow_rate;h.L=r.tensor_point.state.L;
              }
            else h={r.state.a,r.state.signed_shear,r.state.absolute_shear,r.state.q,r.w};
            expected.push_back(r.state.T);
          }
        try
          {accepted.projection_error=std::max(accepted.projection_error,next->check_tensor(model,entry.first,quadrature,expected));}
        catch (const std::exception &)
          {
            // Preserve both arithmetic paths before any accepted publication.
            // These local coefficients allow an independent higher-precision
            // reconstruction of a failed check, without rerunning mechanics.
            const auto cell=parent->native_cells.at(entry.first);cell->get_dof_indices(ids);
            std::ofstream coefficients(parameters.output_directory+"bulk-source-failure-memory.csv");
            coefficients<<std::setprecision(17)<<"component,u,v,previous,next,increment\n";
            const auto &support=finite_element.get_unit_support_points();
            for(unsigned int i=0;i<dpc;++i)
              {
                const auto c=finite_element.system_to_component_index(i).first;
                if(c<dim)coefficients<<c<<','<<support[i][0]<<','<<support[i][1]<<','
                  <<parent->native_history[ids[i]]<<','<<next->native_history[ids[i]]<<','<<x[ids[i]]<<'\n';
              }
            std::ofstream vertices(parameters.output_directory+"bulk-source-failure-vertices.csv");
            vertices<<std::setprecision(17)<<"x,y\n";
            for(const auto v:cell->vertex_indices())vertices<<cell->vertex(v)[0]<<','<<cell->vertex(v)[1]<<'\n';
            const Quadrature<dim> points(next->tensor->space->element.get_unit_support_points());
            const auto a=model.evaluate_dg_trial_tensor(*parent->tensor,entry.first,points);
            const auto b=model.evaluate_dg_trial_tensor(*next->tensor,entry.first,points);
            const auto c=model.evaluate_dg_trial_tensor(*parent->native_projection,entry.first,points);
            const auto d=model.evaluate_dg_trial_tensor(*next->native_projection,entry.first,points);
            std::ofstream tensors(parameters.output_directory+"bulk-source-failure-tensors.csv");
            tensors<<std::setprecision(17)<<"u,v,old_DGxx,old_DGyy,old_DGxy,new_DGxx,new_DGyy,new_DGxy,old_Pxx,old_Pyy,old_Pxy,new_Pxx,new_Pyy,new_Pxy\n";
            for(unsigned int q=0;q<points.size();++q)
              {tensors<<points.point(q)[0]<<','<<points.point(q)[1];for(const auto &v:{a[q],b[q],c[q],d[q]})tensors<<','<<v[0][0]<<','<<v[1][1]<<','<<v[0][1];tensors<<'\n';}
            throw;
          }
      }
    accepted.projection_error=Utilities::MPI::max(accepted.projection_error,mpi_communicator);
    history_seconds=seconds(history_start);
    AssertThrow(accepted.projection_error<=1e-12,ExcMessage("Bulk native DG source projection fails its independent QP check."));
    for(unsigned int i=0;i<np;++i)next->phase[i]=x[po+i];
    double phase_work=0.;
    for(unsigned int i=0;i<np;++i)phase_work+=Kf*accepted.raw[po+i]*(next->phase[i]-parent->phase[i]);
    const double energy_change=accepted.storage_change+accepted.fracture-fracture.energy;
    const double physical_dissipation=accepted.Db+accepted.Dc+accepted.Df;
    const double discrete_remainder=accepted.stress_loss+accepted.traction_loss+accepted.phase_remainder+accepted.fracture_loss+accepted.gradient_loss;
    const double mechanical_work=accepted.internal_work-accepted.external_work;
    const double ledger_error=accepted.external_work-energy_change-physical_dissipation
                             -discrete_remainder+phase_work+mechanical_work;
    AssertThrow(std::abs(ledger_error)<=1e-10*std::max({1.,std::abs(accepted.external_work),accepted.bulk+accepted.cohesive+accepted.fracture})
                && accepted.Db>=0. && accepted.Dc>=0. && accepted.Df>=0.,
                ExcMessage("Bulk constitutive energy ledger or physical dissipation failed before publication."));
    if(benchmark_endpoint_precommit)benchmark_endpoint_precommit();
    AssertThrow(!pending_dg_endpoint_publication,ExcInternalError());
    LinearAlgebra::BlockVector owned_solution;
    owned_solution.reinit(introspection.index_sets.system_partitioning,mpi_communicator);owned_solution=solution;
    for(const auto i:dof_handler.locally_owned_dofs())
      {if(i<nv)owned_solution[i]=x[i]/dt;
       else if(i>=pressure_begin && i<pressure_end)owned_solution[i]=G*x[i];
       else if(i>=po && i<po+np)owned_solution[i]=next->phase[i-po];}
    owned_solution.compress(VectorOperation::insert);solution=owned_solution;
    current_linearization_point=solution;
    accepted.points.clear(); // The prepared owner retains the histories; do
                            // not duplicate full trial records in the callback.
    const double solve_seconds=seconds(started);
    if(rank==0)
      {
        std::ofstream profile(parameters.output_directory+"bulk-profile.csv",std::ios::app);
        if(profile.tellp()==0)profile<<"step,time,ranks,evaluations,linear_solves,assembly,communication,linear,history,total,open_normal,stability,stability_retries\n";
        profile<<std::setprecision(17)<<timestep_number<<','<<time<<','<<ranks<<','<<evaluations<<','<<linear_solves<<','<<assembly_seconds<<','<<communication_seconds<<','<<linear_seconds<<','<<history_seconds<<','<<solve_seconds<<','<<open_normal<<','<<stability_seconds<<','<<stability_retries<<'\n';
      }
    pending_dg_endpoint_publication=[this,&model,parent,next,accepted,Kf,physical_mass,phase_work,ledger_error,energy_change](const bool accept)
      {
        AssertThrow(model.bulk_cohesive_state==parent && next->publications==parent->publications+1
                    && next->step==timestep_number,ExcMessage("Bulk publication parent/clock changed or published twice."));
        if(!accept)return;
        model.bulk_cohesive_state=next;
        if(Utilities::MPI::this_mpi_process(mpi_communicator)!=0)return;
        if(next->slip_length>0.)
          {
            std::ofstream out(parameters.output_directory+"slip-gradient.csv",std::ios::app);
            if(out.tellp()==0)out<<"step,time,length,points,coordinate_error,energy,BE_loss,microforce_error\n";
            out<<std::setprecision(17)<<next->step<<','<<next->time<<','<<next->slip_length<<','<<next->slip_space->points.size()<<','<<next->slip_space->coordinate_error<<','<<accepted.gradient_energy<<','<<accepted.gradient_loss<<','<<accepted.micro<<'\n';
          }
        std::ofstream phase_out(parameters.output_directory+"phase-forces-"+Utilities::int_to_string(next->step,4)+".csv");
        phase_out<<std::setprecision(17)<<"dof,phi,obstacle,mass,fracture_gradient,mechanical_gradient,residual\n";
        const auto offset=solution.get_block_indices().block_start(introspection.variable("phase_field").block_index);
        for(unsigned int i=0;i<next->phase.size();++i)
          phase_out<<i<<','<<next->phase[i]<<','<<parent->phase[i]<<','<<physical_mass[i]<<','<<accepted.fracture_gradient[i]<<','<<Kf*accepted.raw[offset+i]-accepted.fracture_gradient[i]<<','<<Kf*accepted.raw[offset+i]<<'\n';
        std::ofstream out(parameters.output_directory+"bulk-accepted.csv",std::ios::app);out<<std::setprecision(17);
        if(out.tellp()==0)out<<"step,time,dt,publications,bulk,cohesive,fracture,Wext,Db,Dc,Df,stress_loss,traction_loss,phase_remainder,fracture_loss,phase_work,mechanical_work,pressure_work,mechanical_residual,phase_KKT,projection_error,energy_change,ledger_error\n";
        out<<next->step<<','<<next->time<<','<<time_step<<','<<next->publications<<','<<accepted.bulk<<','<<accepted.cohesive<<','<<accepted.fracture<<','<<accepted.external_work<<','
           <<accepted.Db<<','<<accepted.Dc<<','<<accepted.Df<<','<<accepted.stress_loss<<','<<accepted.traction_loss<<','<<accepted.phase_remainder<<','<<accepted.fracture_loss<<','
           <<phase_work<<','<<accepted.internal_work-accepted.external_work<<','<<accepted.pressure_work<<','<<accepted.mechanical<<','<<accepted.kkt<<','<<accepted.projection_error<<','<<energy_change<<','<<ledger_error<<'\n';
      };
  }

#define INSTANTIATE(dim) template void Simulator<dim>::solve_bulk_cohesive_phase();
  ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
}
