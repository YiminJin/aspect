/* Bounded bulk-tip diagnostics; reuse the existing benchmark friendship only. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/termination_criteria/interface.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/phase_field.h>
#include <aspect/geometry_model/box.h>
#include <aspect/simulator/solver/bulk_tip.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <aspect/reconstructed_fault/endpoint_phase_constraints.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <cstdlib>

namespace aspect
{
  namespace { bool tip_remote_margin_reached=false; }
  namespace Postprocess
  {
    template<int dim> class MomentCycle:public Interface<dim>,public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      bool rejected=false;
      unsigned long inner_stokes=0, inner_A=0, inner_S=0;
      Timer step_timer;

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
        if(model().uses_integrated_bulk_tip())
          for(const auto i:s.tip_phase.locally_owned_elements())a.push_back(s.tip_phase[i]);
        else a.insert(a.end(),s.phase.begin(),s.phase.end());
        for(const auto &v:s.sections)for(double x:{v.coordinate,v.weight,v.age,v.slip,v.rate,v.force})a.push_back(x);
        for(const auto &cell:s.history)for(const auto &p:cell.second)
          {for(const auto *t:{&p.stress,&p.A})for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)a.push_back((*t)[i][j]);
           for(double x:{p.rate,p.q,p.signed_shear,p.absolute_shear})a.push_back(x);}
        return a;
      }
      /** Read-only benchmark map from scalar phase rows to native cell shapes. */
      void dump_phase_row_map() const
      {
        const auto dump=std::getenv("ASPECT_PHASE_OBSTACLE_DUMP");
        if (!dump)return;
        const auto &element=this->get_fe();
        const auto pc=this->introspection().phase_field().first_component_index;
        const auto count=element.base_element(element.component_to_base_index(pc).first).n_dofs_per_cell();
        std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
        std::ofstream out(std::string(dump)+"/phase-map-resume-r"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv");
        out<<"index,cell,i,x,y\n"<<std::setprecision(17);
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(ids);
              for (unsigned int i=0;i<count;++i)
                out<<this->get_solution().get_block_indices().global_to_local(ids[element.component_to_system_index(pc,i)]).second
                   <<','<<cell->id().to_string()<<','<<i<<','<<cell->center()[0]<<','<<cell->center()[1]<<'\n';
            }
      }

      /** Narrow same-mesh benchmark import, never a production restart. All
       * components are keyed by CellId/component/local shape; physical history
       * uses its existing material archive and validates geometry/law on restore.
       */
      void export_parent(const std::string &file)
      {
        gate("parent export serial nodal",dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1
             && !this->get_parameters().use_bernstein_phase_basis);
        std::ofstream stream(file,std::ios::binary);AssertThrow(stream,ExcMessage("Cannot create parent export."));
        aspect::oarchive archive(stream);
        std::string signature="tip benchmark parent v1: cell-component Q2 Bernstein phase";
        archive &signature &sim().time &sim().time_step &sim().old_time_step &sim().timestep_number;
        const auto &fe=this->get_fe();const auto pc=this->introspection().phase_field().first_component_index;
        PhaseField::Q2Bounds<dim> basis(fe.base_element(fe.component_to_base_index(pc).first));
        std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
        for(const auto *field:{&sim().solution,&sim().old_solution,&sim().old_old_solution})
          {
            std::map<std::string,std::vector<double>> cells;
            for(const auto &cell:this->get_dof_handler().active_cell_iterators())
              {
                cell->get_dof_indices(ids);auto &row=cells[cell->id().to_string()];
                for(unsigned int c=0;c<fe.n_components();++c)
                  {
                    const auto n=fe.base_element(fe.component_to_base_index(c).first).n_dofs_per_cell();
                    Vector<double> values(n);
                    for(unsigned int j=0;j<n;++j)values[j]=(*field)[ids[fe.component_to_system_index(c,j)]];
                    if(c==pc)values=basis.coefficients(values);
                    row.insert(row.end(),values.begin(),values.end());
                  }
              }
            archive &cells;
          }
        model().serialize_bulk_cohesive_history(archive);
        fixed_parent_action();
      }

      void import_parent(const std::string &file)
      {
        gate("parent import serial Bernstein",dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1
             && this->get_parameters().use_bernstein_phase_basis && !model().bulk_cohesive_state);
        std::ifstream stream(file,std::ios::binary);AssertThrow(stream,ExcMessage("Cannot read parent export."));
        aspect::iarchive archive(stream);std::string signature;
        archive &signature;
        AssertThrow(signature=="tip benchmark parent v1: cell-component Q2 Bernstein phase",ExcMessage("Unexpected benchmark parent format."));
        archive &sim().time &sim().time_step &sim().old_time_step &sim().timestep_number;
        const auto &fe=this->get_fe();
        const auto &sets=this->introspection().index_sets;const auto comm=this->get_mpi_communicator();
        std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
        for(auto *field:{&sim().solution,&sim().old_solution,&sim().old_old_solution})
          {
            std::map<std::string,std::vector<double>> cells;archive &cells;
            AssertDimension(cells.size(),this->get_triangulation().n_global_active_cells());
            LinearAlgebra::BlockVector owned(sets.system_partitioning,comm);owned=0.;
            for(const auto &cell:this->get_dof_handler().active_cell_iterators())
              {
                const auto &row=cells.at(cell->id().to_string());unsigned int k=0;cell->get_dof_indices(ids);
                for(unsigned int c=0;c<fe.n_components();++c)
                  for(unsigned int j=0;j<fe.base_element(fe.component_to_base_index(c).first).n_dofs_per_cell();++j)
                    owned[ids[fe.component_to_system_index(c,j)]]=row.at(k++);
                AssertDimension(k,row.size());
              }
            owned.compress(VectorOperation::insert);*field=owned;
          }
        // Convert only the explicit accepted-phase copy in the material record.
        // Every quadrature history/section/clock/material byte retains its meaning.
        model().serialize_bulk_cohesive_history(archive);
        auto &record=*model().bulk_cohesive_archive;
        const FE_DGQ<dim> nodal(2);const PhaseField::Q2Bounds<dim> basis(nodal);
        const unsigned int geometry_size=GeometryInfo<dim>::vertices_per_cell*dim;
        const unsigned int stride=geometry_size+nodal.n_dofs_per_cell()+16*model().mechanical_quadrature().size();
        AssertDimension(record.local.size(),stride*record.cells.size());
        double phase_error=0.;
        const auto parent_quadrature=model().mechanical_quadrature();
        for(unsigned int cell=0;cell<record.cells.size();++cell)
          {
            Vector<double> a(nodal.n_dofs_per_cell());
            for(unsigned int j=0;j<a.size();++j)a[j]=record.local[cell*stride+geometry_size+j];
            const auto b=basis.coefficients(a);
            for(const auto &point:parent_quadrature.get_points())
              {
                double old_value=0.,new_value=0.;
                for(unsigned int j=0;j<a.size();++j)
                  {old_value+=nodal.shape_value(j,point)*a[j];new_value+=basis.basis_value(j,point)*b[j];}
                phase_error=std::max(phase_error,std::abs(old_value-new_value));
              }
            for(unsigned int j=0;j<a.size();++j)record.local[cell*stride+geometry_size+j]=b[j];
          }
        const auto accepted_step=record.step;const auto accepted_time=record.time;
        if(const auto value=std::getenv("ASPECT_TIP_RESTART_DT"))
          {sim().time_step=std::stod(value);sim().time=accepted_time+sim().time_step;}
        model().restore_bulk_cohesive_history();
        sim().current_linearization_point=sim().solution;
        gate("import phase polynomial preserved",phase_error<2e-14);
        const auto pb=this->introspection().phase_field().block_index;
        auto difference=model().bulk_cohesive_state->tip_phase;difference-=sim().solution.block(pb);
        gate("import accepted obstacle equals current phase",difference.linfty_norm()==0.);
        std::ofstream report(this->get_output_directory()+"parent-import.csv");
        report<<std::setprecision(17)<<"accepted_step,accepted_time,pending_time,dt,phase_error\n"
              <<accepted_step<<','<<accepted_time<<','<<sim().time<<','<<sim().time_step<<','<<phase_error<<'\n';
        fixed_parent_action();
      }

      void fixed_parent_action()
      {
        sim().compute_current_constraints();
        internal::BulkTipCoupling<dim> op(sim(),model(),sim().current_constraints);
        auto x=op.initial_guess(sim().solution);const auto response=op.evaluate(x,true);
        auto direction=x;direction=0.;
        const auto &fe=this->get_fe();const auto pc=this->introspection().phase_field().first_component_index;
        std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          {
            cell->get_dof_indices(ids);
            for(unsigned int c=0;c<fe.n_components();++c)
              {
                const auto &element=fe.base_element(fe.component_to_base_index(c).first);
                for(unsigned int j=0;j<element.n_dofs_per_cell();++j)
                  {
                    const auto p=c==pc ? cell->center() : this->get_mapping().transform_unit_to_real_cell(cell,element.get_generalized_support_points()[j]);
                    direction[ids[fe.component_to_system_index(c,j)]]=std::sin(.7*p[0]+.3*p[1]+c+(c==pc?.1*j:0.));
                  }
              }
          }
        direction.compress(VectorOperation::insert);auto action=x;op.vmult(action,direction);
        std::ofstream out(this->get_output_directory()+"fixed-parent-action.csv");
        out<<std::setprecision(17)<<"cell,component,shape,coordinate,residual,action\n";
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          {
            cell->get_dof_indices(ids);
            for(unsigned int c=0;c<fe.n_components();++c)
              for(unsigned int j=0;j<fe.base_element(fe.component_to_base_index(c).first).n_dofs_per_cell();++j)
                {const auto i=ids[fe.component_to_system_index(c,j)];out<<cell->id()<<','<<c<<','<<j<<','<<x[i]<<','<<response->raw[i]<<','<<action[i]<<'\n';}
          }
      }

      void compare_linear_oracle() const
      {
        gate("serial direct oracle only",Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1);
        auto &op=*sim().bulk_tip_coupling;const auto pb=op.phase_block();
        const auto &x=op.accepted_coordinates;const auto &lower=op.obstacle();
        const auto offset=x.get_block_indices().block_start(pb);
        std::vector<types::global_dof_index> free;
        auto exact=x;exact=0.;
        for(unsigned int b:{0u,1u,pb})for(const auto i:x.block(b).locally_owned_elements())
          {
            const auto global=x.get_block_indices().block_start(b)+i;
            const bool bounded=b==pb && (x.block(pb)[i]<=lower[i]+1e-8 || x.block(pb)[i]>=op.upper_bound()[i]-1e-8);
            if(!op.constraints().is_constrained(global) && !bounded)
              {
                free.push_back(global);
                exact[global]=1e-6*std::sin(.7*global)*(b==pb ? (x.block(pb)[i]-lower[i])*(op.upper_bound()[i]-x.block(pb)[i]) : 1.);
              }
          }
        exact.compress(VectorOperation::insert);
        auto rhs=exact;op.vmult(rhs,exact);
        auto residual=rhs;residual*=-1.;
        for(const auto i:x.block(pb).locally_owned_elements())
          if(std::find(free.begin(),free.end(),offset+i)==free.end())
            residual.block(pb)[i]+=(x.block(pb)[i]>=op.upper_bound()[i]-1e-8?-1.:1.)*op.phase_mass()[i];
        residual.compress(VectorOperation::insert);
        DynamicSparsityPattern pattern(free.size());
        for(unsigned int i=0;i<free.size();++i)for(unsigned int j=0;j<free.size();++j)pattern.add(i,j);
        SparsityPattern sparsity;sparsity.copy_from(pattern);SparseMatrix<double> matrix(sparsity);
        auto column=exact,action=exact;column=0.;
        for(unsigned int j=0;j<free.size();++j)
          {
            column=0.;column[free[j]]=1.;column.compress(VectorOperation::insert);op.vmult(action,column);
            for(unsigned int i=0;i<free.size();++i)matrix.set(i,j,action[free[i]]);
          }
        double skew=0.,weighted_norm=0.;
        const auto row_weight=[&](const unsigned int i)
        {return free[i]>=offset ? op.fracture_scale()/(op.relaxation()*op.shear_modulus()) : 1.;};
        for(unsigned int i=0;i<free.size();++i)for(unsigned int j=0;j<free.size();++j)
          {
            const double value=row_weight(i)*matrix(i,j);
            skew+=std::pow(value-row_weight(j)*matrix(j,i),2);weighted_norm+=value*value;
          }
        const double symmetry=std::sqrt(skew/std::max(1e-30,weighted_norm));
        gate("functional weighted tangent symmetry",symmetry<1e-10);
        dealii::Vector<double> load(free.size()),answer(free.size());
        for(unsigned int i=0;i<free.size();++i)load[i]=rhs[free[i]];
        SparseDirectUMFPACK direct;direct.initialize(matrix);direct.vmult(answer,load);
        auto direction=exact;
        const AffineConstraints<double> original_constraints(sim().current_constraints);
        const bool original_newton=sim().assemble_newton_stokes_system;
        for(const auto &line:original_constraints.get_lines())sim().current_constraints.set_inhomogeneity(line.index,0.);
        sim().current_constraints.close();sim().assemble_newton_stokes_system=false;
        bool solved=false;
        try {solved=sim().compute_bulk_tip_search_direction(x,residual,0.,direction);}
        catch(...){sim().current_constraints.copy_from(original_constraints);sim().assemble_newton_stokes_system=original_newton;throw;}
        sim().current_constraints.copy_from(original_constraints);sim().assemble_newton_stokes_system=original_newton;
        gate("coupled manufactured bound direction",solved);
        double difference=0.,reference=0.;
        for(unsigned int i=0;i<free.size();++i)
          {difference+=std::pow(direction[free[i]]-answer[i],2);reference+=answer[i]*answer[i];}
        const double relative=std::sqrt(difference/std::max(1e-30,reference));
        gate("distributed Krylov versus direct oracle",relative<2e-5);
        direction-=exact;gate("independent manufactured direction",direction.l2_norm()<2e-5*exact.l2_norm());
        std::ofstream out(this->get_output_directory()+"tip-linear-oracle.csv");
        out<<"free_unknowns,relative_krylov_direct,relative_manufactured,relative_weighted_symmetry\n"<<std::setprecision(17)
           <<free.size()<<','<<relative<<','<<direction.l2_norm()/exact.l2_norm()<<','<<symmetry<<'\n';
      }
      void check_rheology() const
      {
        using Law=MaterialModel::Rheology::FaultFriction<dim>;
        using Method=typename Law::AgingMethod;
        const auto &law=model().fault_friction;
        const double Dc=law.get_characteristic_slip_distance(),old=.5;
        for (const auto method:{Method::backward_euler,Method::constant_rate_exact})
          for (const double speed:{0.,1e-320,1e-20,.001,.01,.1})
            for (const double dt:{0.,.0005,.01})
              {
                const auto result=law.evaluate_age(speed,old,dt,method);
                double reference=(old+dt)/(1.+speed*dt/Dc);
                if (method==Method::constant_rate_exact)
                  {
                    // Independent Simpson integral of the exact forcing, not
                    // a call to the production exponential-kernel expression.
                    double integral=0.;
                    for (unsigned int j=0;j<=1000;++j)
                      integral+=(j==0 || j==1000 ? 1. : j%2 ? 4. : 2.)*std::exp(-speed*dt*j/(1000.*Dc));
                    reference=old*std::exp(-speed*dt/Dc)+dt*integral/3000.;
                    const double half=law.evaluate_age(speed,old,.5*dt,method).age;
                    gate("exact aging semigroup",std::abs(law.evaluate_age(speed,half,.5*dt,method).age-result.age)<1e-12);
                  }
                gate("aging value independent reference",std::abs(result.age-reference)<1e-11);
                const double eps=1e-7*Dc/std::max(dt,.01);
                const auto value=[&](const double v){return law.evaluate_age(v,old,dt,method).age;};
                const double derivative=speed<eps ? (-3.*value(speed)+4.*value(speed+eps)-value(speed+2.*eps))/(2.*eps)
                                                  : (value(speed+eps)-value(speed-eps))/(2.*eps);
                gate("aging derivative including zero",std::abs(derivative-result.speed_derivative)<2e-6*std::max(1.,std::abs(result.speed_derivative)));
              }
        std::vector<double> fractions(model().elastic_shear_moduli.size(),0.);fractions[0]=1.;
        // Copies retain only read-only simulator access. Parsing here configures
        // independent test laws; it never changes the model's physical law.
        for (const bool regularized:{false,true})
          {
            auto test=law;
            ParameterHandler prm;Law::declare_parameters(prm);
            prm.set("Use regularized formulation",regularized ? "true" : "false");
            prm.set("Direct effect parameters",regularized ? ".015" : "0");
            prm.set("Evolution effect parameters",".02");
            test.parse_parameters(prm);
            const double speed=.001,age=.5,eps=1e-6*age;
            const double partial=test.friction_coefficient_derivative_wrt_state(fractions,speed,age);
            const double fd=(test.friction_coefficient(fractions,speed,age+eps)-test.friction_coefficient(fractions,speed,age-eps))/(2.*eps);
            gate("fixed-state partial including nonregularized a=0",std::abs(partial-fd)<1e-8);
            if (!regularized)
              {
                gate("a=0 state derivative is b/theta",std::abs(partial-.02/age)<1e-14);
                auto constant=test;prm.set("Evolution effect parameters","0");constant.parse_parameters(prm);
                gate("constant friction has no state splitting timestep restriction",
                     constant.compute_time_step(fractions,speed,.5,true)==std::numeric_limits<double>::max());
              }
            const double rate_eps=1e-6*speed;
            const double rate_fd=(test.friction_coefficient(fractions,speed+rate_eps,age)-test.friction_coefficient(fractions,speed-rate_eps,age))/(2.*rate_eps);
            gate("fixed-age rate partial",std::abs(rate_fd-test.friction_coefficient_derivative_wrt_slip_rate(fractions,speed,age))<1e-5);
            if (fractions.size()>1)
              {
                prm.set("Reference friction coefficients",".6,.4");
                prm.set("Direct effect parameters",regularized ? ".015,.025" : "0,.025");
                prm.set("Evolution effect parameters",".02,.01");
                test.parse_parameters(prm);
                for (const double background:{0.,.3,1.})
                  {
                    fractions[0]=background;fractions[1]=1.-background;
                    const double reference=.6*background+.4*(1.-background);
                    const double direct=(regularized ? .015 : 0.)*background+.025*(1.-background);
                    const double evolution=.02*background+.01*(1.-background);
                    const double V0=test.get_reference_slip_rate(),length=test.get_characteristic_slip_distance();
                    const double expected=regularized
                      ? direct*std::asinh(speed/(2.*V0)*std::exp((reference+evolution*std::log(age*V0/length))/direct))
                      : reference+direct*std::log(speed/V0)+evolution*std::log(age*V0/length);
                    gate("distinct and inactive mixture parameters",std::abs(test.friction_coefficient(fractions,speed,age)-expected)<2e-13);
                    const double recovered=test.initial_state_for_friction_coefficient(fractions,speed,expected);
                    gate("mixture initial state inversion",std::abs(recovered-age)<1e-11);
                    const double expected_dt=direct>evolution ? std::numeric_limits<double>::max() : .5*direct*length/(evolution*speed);
                    const double actual_dt=test.compute_time_step(fractions,speed,.5,true);
                    gate("mixture state splitting timestep",expected_dt==std::numeric_limits<double>::max()
                         ? actual_dt==expected_dt : std::abs(actual_dt-expected_dt)<1e-12*std::max(1.,expected_dt));
                  }
                fractions[0]=1.;fractions[1]=0.;
              }
          }
        auto stateless=law;ParameterHandler prm;Law::declare_parameters(prm);
        prm.set("Friction law","rate dependent");stateless.parse_parameters(prm);
        bool rejected=false;
        try {stateless.friction_coefficient_derivative_wrt_state(fractions,.001,.5);}
        catch (const std::exception &) {rejected=true;}
        gate("state partial rejects stateless law before indexing",rejected);
        rejected=false;
        try {law.friction_coefficient_derivative_wrt_state(std::vector<double>(fractions.size()+1,0.),.001,.5);}
        catch (const std::exception &) {rejected=true;}
        gate("state partial rejects wrong mixture size",rejected);
      }

      void export_step()
      {
        const auto &s=*model().bulk_cohesive_state;
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        const auto directory=this->get_output_directory();
        std::ofstream points(directory+"tip-points-"+Utilities::int_to_string(s.step,4)+"-r"+std::to_string(rank)+".csv");
        points<<std::setprecision(17)<<"cell,q,x,y,weight,phi,tauxx,tauyy,tauzz,tauxy,Axx,Ayy,Azz,Axy,qcoh,w,signed_shear,absolute_shear,section,H,Y,fracture_density,Y_endpoint,stage_phi,stage_tauxy,stage_Axy,stage_pressure,stage_normal_x,stage_normal_y\n";
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),model().mechanical_quadrature(),update_values|update_gradients|update_quadrature_points|update_JxW_values);
        std::vector<double> phase(fe.n_quadrature_points),pressure(fe.n_quadrature_points);
        std::vector<Tensor<1,dim>> phase_gradient(fe.n_quadrature_points);
        const double ell=this->get_phase_field_handler().get_length_scale();
        const double Kf=model().critical_energy_release_rates[0]/((8./3.)*ell);
        const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
        const auto origin=box.get_origin(),extent=box.get_extents();
        const double center_x=origin[0]+.5*extent[0],center_y=origin[1]+.5*extent[1];
        // Synthetic masks match the independent 8x4 analysis. Other retained
        // controls keep their dimensionless old x-only remote marker.
        const bool synthetic=std::abs(extent[0]-8.)<1e-12 && std::abs(extent[1]-4.)<1e-12;
        double remote_driving=0.,remote_phase=0.,boundary_driving=0.,boundary_phase=0.;
        for(const auto &entry:s.native_cells)
          {
            fe.reinit(entry.second);fe[FEValuesExtractors::Scalar(this->introspection().phase_field().first_component_index)].get_function_values(this->get_solution(),phase);
            fe[FEValuesExtractors::Scalar(this->introspection().phase_field().first_component_index)].get_function_gradients(this->get_solution(),phase_gradient);
            fe[this->introspection().extractors.pressure].get_function_values(this->get_solution(),pressure);
            for(unsigned int q=0;q<phase.size();++q)
              {
                const auto &h=s.history.at(entry.first)[q];const auto p=fe.quadrature_point(q);
                const double phi=phase[q];
                const double c=s.parameters.m*phi*(1.+phi),d=(1.-phi)*(1.-phi);
                const double fraction=c/(c+d);
                const double driving=sim().bulk_tip_coupling && sim().bulk_tip_coupling->accepted_evaluation
                    ? sim().bulk_tip_coupling->accepted_evaluation->points.at(entry.first)[q].driving : 0.;
                if(std::min({p[0]-origin[0],origin[0]+extent[0]-p[0],p[1]-origin[1],origin[1]+extent[1]-p[1]})<=.25)
                  {boundary_driving=std::max(boundary_driving,driving);boundary_phase=std::max(boundary_phase,phi);}
                if(synthetic ? (std::abs(p[0]-center_x)>3. || std::abs(p[1]-center_y)>1.) : std::abs(p[0]-center_x)>.325*extent[0])
                  { remote_driving=std::max(remote_driving,driving); remote_phase=std::max(remote_phase,phi); }
                points<<entry.first<<','<<q<<','<<p[0]<<','<<p[1]<<','<<fe.JxW(q)<<','<<phase[q]<<','
                      <<h.stress[0][0]<<','<<h.stress[1][1]<<','<<h.stress[2][2]<<','<<h.stress[0][1]<<','
                      <<h.A[0][0]<<','<<h.A[1][1]<<','<<h.A[2][2]<<','<<h.A[0][1]<<','<<h.q<<','<<h.rate<<','
                      <<h.signed_shear<<','<<h.absolute_shear<<','<<s.section_ids.at(entry.first)[q]<<','<<fraction<<','<<driving<<','<<Kf*(phi+ell*ell*phase_gradient[q].norm_square())<<','
                      <<(phi>0. ? 2.*s.parameters.G*h.A[0][1]*h.A[0][1]*s.parameters.m*(1.+3.*phi)*(1.-phi)/(c*c) : std::numeric_limits<double>::quiet_NaN())<<',';
                if(sim().bulk_tip_coupling && sim().bulk_tip_coupling->accepted_evaluation)
                  {
                    const auto &stage=sim().bulk_tip_coupling->accepted_evaluation->points.at(entry.first)[q];
                    points<<stage.state.phi<<','<<stage.state.stress[0][1]<<','<<stage.state.cohesive_strain[0][1]<<','<<pressure[q]<<','<<stage.state.stress[0][0]-pressure[q]<<','<<stage.state.stress[1][1]-pressure[q];
                  }
                else points<<phi<<','<<h.stress[0][1]<<','<<h.A[0][1]<<','<<pressure[q]<<','<<h.stress[0][0]-pressure[q]<<','<<h.stress[1][1]-pressure[q];
                points<<'\n';
              }
          }
        remote_driving=Utilities::MPI::max(remote_driving,this->get_mpi_communicator());
        remote_phase=Utilities::MPI::max(remote_phase,this->get_mpi_communicator());
        boundary_driving=Utilities::MPI::max(boundary_driving,this->get_mpi_communicator());
        boundary_phase=Utilities::MPI::max(boundary_phase,this->get_mpi_communicator());
        if(std::getenv("ASPECT_TIP_REMOTE_MARGIN") && s.step>0)
          {
            const double Kf=sim().bulk_tip_coupling->fracture_scale();
            tip_remote_margin_reached=std::getenv("ASPECT_TIP_INTEGRITY_STOP")
              ? remote_phase>1e-6 || boundary_phase>1e-6
              : remote_driving/Kf>=.95 || remote_phase>1e-12
                || (std::getenv("ASPECT_TIP_BOUNDARY_STOP") && boundary_phase>1e-12);
            if(rank==0)
              {
                std::ofstream remote(directory+"tip-remote.csv",std::ios::app);
                if(remote.tellp()==0)remote<<"step,time,driving_ratio,max_phase,stop,boundary_driving_ratio,boundary_phase\n";
                remote<<std::setprecision(17)<<s.step<<','<<s.time<<','<<remote_driving/Kf<<','<<remote_phase<<','<<tip_remote_margin_reached<<','<<boundary_driving/Kf<<','<<boundary_phase<<'\n';
              }
          }
        if(s.step==0)
          {
            LinearAlgebra::Vector gradient;
            const double energy=this->get_phase_field_handler().assemble_dg_fracture(this->get_solution(),gradient);
            if(std::getenv("ASPECT_TIP_REMOTE_MARGIN") || std::getenv("ASPECT_TIP_PHASE_BALANCE"))
              {
                // Native seed coordinates and the authoritative SIPG derivative
                // permit an exact quadratic energy-change partition afterwards.
                // This is an observation, not a second equilibration or operator.
                const auto &element=this->get_fe();
                const auto pc=this->introspection().phase_field().first_component_index;
                const auto pb=this->introspection().phase_field().block_index;
                const auto offset=this->get_solution().get_block_indices().block_start(pb);
                PhaseField::Q2Bounds<dim> basis(element.base_element(element.component_to_base_index(pc).first));
                std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
                std::ofstream seed(directory+"tip-seed-fracture-r"+std::to_string(rank)+".csv");
                seed<<std::setprecision(17)<<"cell,i,B,fracture_gradient\n";
                for(const auto &entry:s.native_cells)
                  {
                    entry.second->get_dof_indices(ids);
                    dealii::Vector<double> nodal(basis.inverse.n()),coefficients(basis.inverse.m());
                    for(unsigned int i=0;i<nodal.size();++i)nodal[i]=this->get_solution()[ids[element.component_to_system_index(pc,i)]];
                    basis.inverse.vmult(coefficients,nodal);
                    for(unsigned int j=0;j<coefficients.size();++j)
                      {
                        double force=0.;
                        for(unsigned int i=0;i<basis.expansion.m();++i)
                          force+=basis.expansion(i,j)*gradient[ids[element.component_to_system_index(pc,i)]-offset];
                        seed<<entry.first<<','<<j<<','<<coefficients[j]<<','<<force<<'\n';
                      }
                  }
              }
            if(rank==0){std::ofstream initial(directory+"tip-initial.csv");initial<<"fracture,initial_age\n"<<std::setprecision(17)<<energy<<','<<s.sections.front().age<<'\n';}
            return;
          }
        if(!sim().bulk_tip_coupling || !sim().bulk_tip_coupling->accepted_evaluation)return;
        auto &op=*sim().bulk_tip_coupling;const auto &e=*op.accepted_evaluation;
        if(std::getenv("ASPECT_TIP_FIXED_PHASE"))
          {
            auto difference=s.tip_phase;difference-=op.accepted_parent()->tip_phase;
            gate("fixed screen phase invariant",difference.linfty_norm()==0.);
          }
        if(rank==0)
          {
            std::ofstream profile(directory+"tip-profile.csv",std::ios::app);
            if(profile.tellp()==0)profile<<"step,stokes_iterations,velocity_iterations,pressure_iterations,active_passes,active_repeats,exact_attempts,search_failures,line_trials,evaluations,tangent_assemblies,step_seconds\n";
            profile<<s.step<<','<<inner_stokes<<','<<inner_A<<','<<inner_S<<','<<op.active_passes<<','<<op.active_repeats<<','<<op.exact_attempts<<','<<op.search_failures<<','<<op.line_trials<<','<<op.evaluations<<','<<op.tangent_assemblies<<','<<step_timer.wall_time()<<'\n';
          }
        if(std::getenv("ASPECT_TIP_LINEAR_ORACLE") && s.step==static_cast<unsigned int>(std::getenv("ASPECT_TIP_LINEAR_ORACLE_STEP")?std::atoi(std::getenv("ASPECT_TIP_LINEAR_ORACLE_STEP")):1))compare_linear_oracle();
        if(std::getenv("ASPECT_BULK_CHECK_TANGENT") && s.step==static_cast<unsigned int>(std::getenv("ASPECT_TIP_TANGENT_STEP")?std::atoi(std::getenv("ASPECT_TIP_TANGENT_STEP")):8))
          {
            auto direction=op.accepted_coordinates;direction=0.;
            const auto pb=op.phase_block();
            for(unsigned int b:{0u,1u,pb})for(const auto i:direction.block(b).locally_owned_elements())
              direction.block(b)[i]=b==pb ? .1*op.accepted_coordinates.block(pb)[i]*std::sin(.7*i) : .01*std::sin(.7*i);
            direction.compress(VectorOperation::insert);op.constraints().set_zero(direction);
            auto action=direction;op.vmult(action,direction);
            const double eps=1e-7;auto plus=op.accepted_coordinates;plus.add(eps,direction);
            auto minus=op.accepted_coordinates;minus.add(-eps,direction);
            const auto ep=op.evaluate(plus,false),em=op.evaluate(minus,false);
            auto difference=ep->residual;difference-=em->residual;difference*=.5/eps;difference-=action;
            const double error=difference.l2_norm()/std::max(1e-30,action.l2_norm());
            gate("distributed coupled tangent",error<2e-5);
            const double multiplier_scale=op.relaxation()*op.shear_modulus();
            const double lp=ep->functional+multiplier_scale*(plus.block(1)*ep->raw.block(1));
            const double lm=em->functional+multiplier_scale*(minus.block(1)*em->raw.block(1));
            const double expected_lagrangian=op.directional_work(e,direction)
                                            +multiplier_scale*(e.residual.block(1)*direction.block(1));
            const double lagrangian_error=std::abs((lp-lm)/(2*eps)-expected_lagrangian)/std::max(1.,std::abs(expected_lagrangian));
            gate("scaled mechanical phase Lagrangian derivative",lagrangian_error<2e-5);
            direction=0.;
            for(const auto i:direction.block(pb).locally_owned_elements())
              {const double phi=op.accepted_coordinates.block(pb)[i];direction.block(pb)[i]=phi*(1.-phi)*std::sin(.7*i);}
            direction.compress(VectorOperation::insert);
            plus=op.accepted_coordinates;plus.add(eps,direction);
            minus=op.accepted_coordinates;minus.add(-eps,direction);
            const auto fp=op.evaluate(plus,false),fm=op.evaluate(minus,false);
            const double expected=op.directional_work(e,direction);
            const double functional_error=std::abs((fp->functional-fm->functional)/(2*eps)-expected)/std::max(1.,std::abs(expected));
            gate("restricted functional phase derivative",functional_error<2e-5);
            // Functional derivative is tested only for divergence-free variations
            // separately; arbitrary saddle-point directions cannot certify it.
            if(rank==0){std::ofstream out(directory+"bulk-tangent.csv");out<<"time,relative_action_error,functional_directional_reference,functional_derivative_error,lagrangian_derivative_error\n"<<std::setprecision(17)<<this->get_time()<<','<<error<<','<<expected<<','<<functional_error<<','<<lagrangian_error<<'\n';}
          }
        if(std::getenv("ASPECT_TIP_REMOTE_MARGIN") || std::getenv("ASPECT_TIP_PHASE_BALANCE"))
          {
            LinearAlgebra::Vector gradient;
            this->get_phase_field_handler().assemble_dg_fracture(this->get_solution(),gradient);
            LinearAlgebra::Vector stage_gradient(op.lower);
            op.fracture_matrix.vmult(stage_gradient,op.accepted_coordinates.block(op.phase_block()));
            stage_gradient+=op.fracture_linear;
            const auto pc=this->introspection().phase_field().first_component_index;
            const auto &element=this->get_fe();
            PhaseField::Q2Bounds<dim> basis(element.base_element(element.component_to_base_index(pc).first));
            const auto pb=op.phase_block();const auto offset=op.accepted_coordinates.get_block_indices().block_start(pb);
            std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
            std::ofstream balance(directory+"tip-balance-"+Utilities::int_to_string(s.step,4)+"-r"+std::to_string(rank)+".csv");
            balance<<std::setprecision(17)<<"cell,i,x,y,B,lower,mass,fracture_gradient,driving_load,residual,stage_B,stage_fracture_gradient\n";
            std::ofstream row_map;
            if (const auto dump=std::getenv("ASPECT_PHASE_OBSTACLE_DUMP"))
              {
                row_map.open(std::string(dump)+"/phase-map-r"+std::to_string(rank)+".csv");
                row_map<<"index,cell,i,x,y\n"<<std::setprecision(17);
              }
            for(const auto &entry:s.native_cells)
              {
                const auto cell=entry.second;cell->get_dof_indices(ids);
                for(unsigned int j=0;j<basis.expansion.n();++j)
                  {
                    double fracture=0.;
                    for(unsigned int i=0;i<basis.expansion.m();++i)
                      fracture+=basis.expansion(i,j)*gradient[ids[element.component_to_system_index(pc,i)]-offset];
                    const auto index=ids[element.component_to_system_index(pc,j)]-offset;
                    if (row_map)row_map<<index<<','<<entry.first<<','<<j<<','<<cell->center()[0]<<','<<cell->center()[1]<<'\n';
                    const double residual=op.fracture_scale()*e.raw.block(pb)[index];
                    balance<<entry.first<<','<<j<<','<<cell->center()[0]<<','<<cell->center()[1]<<','<<this->get_solution().block(pb)[index]<<','<<op.obstacle()[index]<<','<<op.phase_mass()[index]<<','<<fracture<<','<<stage_gradient[index]-residual<<','<<residual<<','<<op.accepted_coordinates.block(pb)[index]<<','<<stage_gradient[index]<<'\n';
                  }
              }
          }
        {
          // Constrained velocity rows are nodal reaction forces in the same
          // JxW/face measures as mechanics. Corner forces are not split into
          // fictitious per-wall tractions; coordinates identify shared corners.
          const auto &element=this->get_fe();
          std::vector<types::global_dof_index> ids(element.n_dofs_per_cell());
          std::map<types::global_dof_index,std::pair<Point<dim>,unsigned int>> nodes;
          for(const auto &entry:s.native_cells)
            {
              const auto cell=entry.second;cell->get_dof_indices(ids);
              for(unsigned int component=0;component<dim;++component)
                {
                  const auto &base=element.base_element(element.component_to_base_index(component).first);
                  for(unsigned int j=0;j<base.n_dofs_per_cell();++j)
                    {
                      const auto index=ids[element.component_to_system_index(component,j)];
                      if(op.boundary_constraints().is_constrained(index) && e.raw.locally_owned_elements().is_element(index))
                        nodes[index]={this->get_mapping().transform_unit_to_real_cell(cell,base.get_unit_support_points()[j]),component};
                    }
                }
            }
          std::ofstream reaction(directory+"tip-reactions-"+Utilities::int_to_string(s.step,4)+"-r"+std::to_string(rank)+".csv");
          reaction<<std::setprecision(17)<<"dof,x,y,component,force,stage_velocity,physical_work\n";
          for(const auto &node:nodes)
            {
              const double force=op.shear_modulus()*e.raw[node.first],velocity=this->get_solution()[node.first];
              reaction<<node.first<<','<<node.second.first[0]<<','<<node.second.first[1]<<','<<node.second.second<<','<<force<<','<<velocity<<','<<op.timestep()*force*velocity<<'\n';
            }
        }
        if(rank!=0)return;
        {
          std::ofstream timers(directory+"tip-timers.csv");timers<<"section,seconds,calls\n";
          const auto wall=this->get_computing_timer().get_summary_data(TimerOutput::total_wall_time);
          const auto calls=this->get_computing_timer().get_summary_data(TimerOutput::n_calls);
          std::ofstream step_timers(directory+"tip-step-timers.csv",std::ios::app);
          if(step_timers.tellp()==0)step_timers<<"step,section,seconds,calls\n";
          for(const auto &entry:wall)step_timers<<s.step<<','<<entry.first<<','<<std::setprecision(17)<<entry.second<<','<<calls.at(entry.first)<<'\n';
          for(const auto &entry:wall)timers<<entry.first<<','<<std::setprecision(17)<<entry.second<<','<<calls.at(entry.first)<<'\n';
        }
        std::ofstream location(directory+"tip-time-location.csv",std::ios::app);
        if(location.tellp()==0)location<<"step,parent_time,stage_time,endpoint_time,midpoint\n";
        location<<std::setprecision(17)<<s.step<<','<<op.accepted_parent()->time<<','<<s.time-(op.uses_midpoint()?.5*op.timestep():0.)<<','<<s.time<<','<<op.uses_midpoint()<<'\n';
        std::ofstream sections(directory+"tip-sections-"+Utilities::int_to_string(s.step,4)+".csv");
        sections<<std::setprecision(17)<<"s,ds,age,slip,V,force,integral_abs_w,positive_w,negative_w,resolved_V,integral_rate_defect,RSF_of_integral\n";
        std::vector<double> fractions(model().elastic_shear_moduli.size(),0.);fractions[0]=1.;
        for(unsigned int k=0;k<s.sections.size();++k)
          {
            const auto &v=s.sections[k];const auto raw=model().fault_friction.signed_endpoint(fractions,v.rate,op.accepted_parent()->sections[k].age,
                op.stage_timestep(),this->get_parameters().surface_pressure,model().radiation_damping_coefficients[0],op.uses_midpoint() ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                                   : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
            sections<<v.coordinate<<','<<v.weight<<','<<v.age<<','<<v.slip<<','<<v.rate<<','<<v.force<<','<<e.absolute_rate[k]<<','<<e.positive_rate[k]<<','<<e.negative_rate[k]<<','<<e.laws[k].rate<<','<<v.rate-e.laws[k].rate<<','<<raw.traction<<'\n';
          }
        std::ofstream work(directory+"tip-power.csv",std::ios::app);
        if(work.tellp()==0)work<<"step,power_adjoint_error,friction,radiation,section_residual_work\n";
        work<<std::setprecision(17)<<s.step<<','<<e.adjoint_error<<','<<e.friction<<','<<e.radiation<<','<<e.Df-e.friction-e.radiation<<'\n';
        std::ofstream out(directory+"bulk-accepted.csv",std::ios::app);
        if(out.tellp()==0)out<<"step,time,dt,publications,bulk,cohesive,fracture,Wext,Db,Dc,Df,stress_loss,traction_loss,phase_remainder,fracture_loss,phase_work,mechanical_work,mechanical_residual,phase_KKT,energy_change,ledger_error\n";
        out<<std::setprecision(17)<<s.step<<','<<s.time<<','<<op.timestep()<<','<<s.publications<<','<<e.bulk<<','<<e.cohesive<<','<<e.fracture<<','<<e.external_work<<','
           <<e.Db<<','<<e.Dc<<','<<e.Df<<','<<e.stress_loss<<','<<e.traction_loss<<','<<e.compliance_remainder<<','<<e.fracture_loss<<','<<e.phase_work<<','<<e.internal_work-e.external_work<<','
           <<e.mechanical<<','<<e.kkt<<','<<e.storage_change<<','<<e.ledger_error<<'\n';
        std::ofstream counts(directory+"tip-solver.csv",std::ios::app);
        if(counts.tellp()==0)counts<<"step,nonlinear_iterations,krylov_iterations,stokes_solves,active_releases,search_shifts,sparse_entries,owned_phase_rows\n";
        counts<<s.step<<','<<op.newton_iterations<<','<<op.linear_iterations<<','<<op.stokes_solves<<','<<op.active_releases<<','<<op.search_shifts<<','<<op.sparse_entries()<<','<<op.obstacle().locally_owned_size()<<'\n';
      }
      void recovery()
      {
        const auto before=persistent();const auto owner=model().bulk_cohesive_state;
        const auto original=sim().solution;
        sim().compute_current_constraints();
        // start_timestep emits its signal before ordinary matrix setup. A
        // first-pending-step restart probe must prepare those existing owners
        // before invoking a trial; later probes already have their matrices.
        if(sim().rebuild_sparsity_and_matrices)
          {
            sim().setup_system_matrix(this->introspection().index_sets.system_partitioning);
            sim().setup_system_preconditioner(this->introspection().index_sets.system_partitioning);
            sim().rebuild_sparsity_and_matrices=false;
            sim().rebuild_stokes_matrix=sim().rebuild_stokes_preconditioner=true;
          }
        sim().solve_timestep();
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
        this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&)
          { if(const auto file=std::getenv("ASPECT_TIP_IMPORT_PARENT"))import_parent(file); });
        this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim>&,double &dt)
          {
            dump_phase_row_map();
            if(const auto file=std::getenv("ASPECT_TIP_EXPORT_PARENT"))export_parent(file);
            if(const auto value=std::getenv("ASPECT_TIP_RESTART_DT"))dt=std::stod(value);
          });
        this->get_signals().set_assemblers.connect([this](const SimulatorAccess<dim>&,Assemblers::Manager<dim>&)
          {
            if(std::getenv("ASPECT_TIP_FIXED_PHASE") && sim().bulk_tip_coupling)
              sim().bulk_tip_coupling->upper=sim().bulk_tip_coupling->lower;
          });
        this->get_signals().post_stokes_solver.connect([this](const SimulatorAccess<dim>&,const unsigned int ns,const unsigned int na,const SolverControl &cheap,const SolverControl &expensive)
          {
            inner_S+=ns; inner_A+=na;
            if(cheap.last_step()!=numbers::invalid_unsigned_int)inner_stokes+=cheap.last_step();
            if(expensive.last_step()!=numbers::invalid_unsigned_int)inner_stokes+=expensive.last_step();
          });
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&){inner_stokes=inner_A=inner_S=0;step_timer.restart();});
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&)
          {if(std::getenv("ASPECT_TIP_RECOVERY") && !rejected && this->get_timestep_number()==static_cast<unsigned int>(std::getenv("ASPECT_TIP_RECOVERY_STEP")?std::atoi(std::getenv("ASPECT_TIP_RECOVERY_STEP")):2))recovery();});
      }
      std::pair<std::string,std::string> execute(TableHandler&)override
      {
        const auto &s=*model().bulk_cohesive_state;
        gate("single publication",s.step==this->get_timestep_number() && s.publications==s.step);
        if(model().uses_integrated_bulk_tip())export_step();
        if(s.step>0)return {"Bulk tip","accepted histories checked"};
        if(std::getenv("ASPECT_RHEOLOGY_AUDIT"))check_rheology();
        std::vector<double> fractions(model().elastic_shear_moduli.size(),0.);fractions[0]=1.;
        const auto &law=model().fault_friction;
        const double test_age=model().bulk_tip_initial_age, test_dt=model().initial_time_step;
        const double test_pressure=this->get_parameters().surface_pressure;
        const double test_radiation=model().radiation_damping_coefficients[0];
        const double test_Dc=law.get_characteristic_slip_distance();
        for(bool exact:{false,true}) for(double v:{0.,1e-20,-1e-20,.001,-.001,.01,-.01})
          {
            const auto r=law.signed_endpoint(fractions,v,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
            // Resolve the configured zero-rate asinh scale; a fixed rate probe
            // can leave its linear range when age or Dc changes.
            const double eps=v==0.?1e-6/std::max(1.,std::abs(r.derivative)):std::abs(v)*1e-5;
            const auto a=law.signed_endpoint(fractions,v+eps,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
            const auto b=law.signed_endpoint(fractions,v-eps,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
            gate("signed RSF derivative",std::abs((a.traction-b.traction)/(2*eps)-r.derivative)<2e-6*std::max(1.,std::abs(r.derivative)));
            gate("positive age and signed dissipation",r.age>0. && r.traction*v>=0.);
            if(exact)
              {
                // Independent Simpson integration of the positive aging kernel.
                double integral=0.;const double h=test_dt/1000.,k=std::abs(v)/test_Dc;
                for(unsigned int j=0;j<=1000;++j) integral+=(j==0 || j==1000 ? 1. : j%2 ? 4. : 2.)*std::exp(-k*j*h);
                const double expected=test_age*std::exp(-k*test_dt)+h*integral/3.;
                gate("exact aging independent quadrature",std::abs(r.age-expected)<2e-13);
              }
            if(v!=0.)
              {
                const double primitive_plus=law.section_primitive(fractions,v+eps,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
                const double primitive_minus=law.section_primitive(fractions,v-eps,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
                gate("section search primitive derivative",std::abs((primitive_plus-primitive_minus)/(2*eps)-r.traction)<2e-6*std::max(1.,std::abs(r.traction)));
              }
          }
        for(bool exact:{false,true}) for(double free:{0.,.001,-.001,.1,-.1})for(double mobility:{0.,1e-7,1e-4})
          {
            const auto r=law.section_endpoint(fractions,free,mobility,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
            gate("implicit section balance",std::abs(r.rate+mobility*r.law.traction-free)<1e-14);
            if(free!=0.)
              {
                const double eps=1e-6*std::abs(free);
                const auto a=law.section_endpoint(fractions,free+eps,mobility,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
                const auto b=law.section_endpoint(fractions,free-eps,mobility,test_age,test_dt,test_pressure,test_radiation,exact ? MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                               : MaterialModel::Rheology::FaultFriction<dim>::AgingMethod::backward_euler);
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

namespace aspect
{
  namespace TerminationCriteria
  {
    /** Benchmark observation only: preserve the accepted endpoint exhausting
     * the remote-intact margin, then stop through ordinary termination. */
    template <int dim>
    class TipRemoteMargin : public Interface<dim>
    {
      public:
        bool execute() override { return tip_remote_margin_reached; }
    };
    ASPECT_REGISTER_TERMINATION_CRITERION(TipRemoteMargin,"tip remote margin",
      "Stop the localized diagnostic after its accepted endpoint exhausts the remote-intact margin.")
  }
}
