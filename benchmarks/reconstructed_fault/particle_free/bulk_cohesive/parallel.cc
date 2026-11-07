/* Fixed-mesh bulk-model MPI/BC fixture; owns inputs and diagnostics only. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_velocity/interface.h>
#include <aspect/time_stepping/interface.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_mpi.h>
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <aspect/particle/manager.h>
#include <sys/resource.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <set>

namespace aspect
{
  namespace BoundaryVelocity
  {
    template<int dim> class BulkParallelLoading:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &x)const override
        {const double t=this->get_time();Tensor<1,dim> v;v[0]=(t>.05+1e-12 && t<=.175+1e-12?.018:0.)*(x[1]-.5);return v;}
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(BulkParallelLoading,"bulk cohesive loading","Same finite-seed load and hold; components selected by native boundary masks.")
  }
  namespace TimeStepping
  {
    template<int dim> class BulkParallelSteps:public Interface<dim>,public SimulatorAccess<dim>
    {public:double execute()override{return this->get_parameters().maximum_time_step;}};
    ASPECT_REGISTER_TIME_STEPPING_MODEL(BulkParallelSteps,"bulk cohesive steps","Prescribed bounded bulk-model timestep.")
  }
  namespace Postprocess
  {
    template<int dim> class MomentCycle:public Interface<dim>,public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      unsigned int publications=0;bool recovered=false,test_recovery_done=false;
      Model &model()const{return const_cast<Model&>(Plugins::get_plugin_as_type<const Model>(this->get_material_model()));}
      Simulator<dim> &sim()const{return const_cast<Simulator<dim>&>(this->get_simulator());}
      bool root()const{return Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0;}
      std::ofstream output(const std::string &name)const
      {std::ofstream out;if(root())out.open(this->get_output_directory()+name);out<<std::setprecision(17);return out;}
      void check(const std::string &label,const bool local)const
      {
        const bool pass=Utilities::MPI::min(static_cast<unsigned int>(local),this->get_mpi_communicator())!=0;
        if(root()){std::ofstream out(this->get_output_directory()+"gates.csv",std::ios::app);if(out.tellp()==0)out<<"gate,passed\n";out<<label<<','<<pass<<std::endl;}
        AssertThrow(pass,ExcMessage("Bulk parallel gate: "+label));
      }
      std::vector<double> persistent()const
      {
        const auto &s=*model().bulk_cohesive_state;std::vector<double> values;
        for(const auto *p:{s.tensor.get(),s.native_projection.get()})for(const auto &v:p->coefficients)
          for(const auto i:v.locally_owned_elements())values.push_back(v[i]);
        for(const auto &entry:s.history)for(const auto &h:entry.second)
          {for(double v:{h.a,h.signed_shear,h.absolute_shear,h.q,h.rate})values.push_back(v);
           if(s.tensor_flow()){for(const auto *t:{&h.A,&h.K,&h.Q,&h.W})for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)values.push_back((*t)[i][j]);values.push_back(h.L);}}
        for(const auto i:s.native_history.block(this->introspection().block_indices.velocities).locally_owned_elements())values.push_back(s.native_history[i]);
        values.insert(values.end(),s.phase.begin(),s.phase.end());values.push_back(s.step);values.push_back(s.time);values.push_back(s.publications);return values;
      }
      static bool same(const std::vector<double> &a,const std::vector<double> &b)
      {return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0;}
      virtual std::array<double,3> seed_parameters()const{return {{.15625,128.,.5}};}
      virtual double initial_phase(const Point<dim> &x,const PhaseField::PhaseFieldProfile &profile,const double tip)const
      {
#ifdef ASPECT_COULOMB_BLOCK_FIXTURE
        (void)x;(void)profile;(void)tip;
        return std::getenv("ASPECT_COULOMB_BLOCK_PHASE")?std::stod(std::getenv("ASPECT_COULOMB_BLOCK_PHASE")):.2;
#else
        return profile.value(std::hypot(std::max(0.,x[0]-tip),x[1]-.5));
#endif
      }
      void seed()
      {
        const auto parameters=seed_parameters();
        const PhaseField::GeometricFunction a(parameters[0],1.,8./3.);
        const PhaseField::DegradationFunction g(1.,parameters[1]);
        const PhaseField::PhaseFieldProfile profile(a,g,.6);
        LinearAlgebra::BlockVector owned;owned.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());owned=sim().solution;
        const auto pc=this->introspection().phase_field().first_component_index;
        const auto &partition=this->get_dof_handler().locally_owned_dofs();
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(!cell->is_artificial())for(const auto v:cell->vertex_indices())
          {const auto i=cell->vertex_dof_index(v,pc);if(partition.is_element(i)){const auto x=cell->vertex(v);owned[i]=initial_phase(x,profile,parameters[2]);}}
        #ifdef ASPECT_COULOMB_BLOCK_FIXTURE
        owned.block(this->introspection().block_indices.pressure)=0.;
#endif
        this->get_current_constraints().distribute(owned);owned.compress(VectorOperation::insert);sim().solution=owned;
      }
      void recovery()
      {
        const auto before=persistent();const auto original=sim().solution;
        sim().compute_current_constraints();sim().solve_timestep();
        check("trial does not publish",same(before,persistent()) && bool(sim().pending_dg_endpoint_publication));
        const auto trial=sim().solution;
        auto reject=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};reject(false);
        check("rejection restores all accepted owners",same(before,persistent()) && sim().solution==original);
        sim().solve_timestep();check("deterministic retry",same(before,persistent()) && sim().solution==trial);
        auto discard=std::move(sim().pending_dg_endpoint_publication);sim().pending_dg_endpoint_publication={};discard(false);
        check("second rejection unchanged",same(before,persistent()) && sim().solution==original);recovered=true;
      }
      void export_state()
      {
        namespace BC=MaterialModel::BulkCohesive;
        const auto comm=this->get_mpi_communicator();const auto &m=model();const auto &s=*m.bulk_cohesive_state;
        const auto label=Utilities::int_to_string(s.step,4);const auto &fe_element=this->get_fe();
        const auto pc=this->introspection().phase_field().first_component_index;
        const auto po=sim().solution.get_block_indices().block_start(this->introspection().phase_field().block_index);
        const auto nv=sim().solution.block(this->introspection().block_indices.velocities).size();
        const auto constraints=BC::global_constraints(this->get_current_constraints(),comm);
        std::map<unsigned int,Point<dim>> positions,local_positions;
        std::vector<types::global_dof_index> ids(fe_element.n_dofs_per_cell()),dgids(9);
        const auto quad=m.mechanical_quadrature();
        FEValues<dim> fe(this->get_mapping(),fe_element,quad,update_values|update_gradients|update_quadrature_points|update_JxW_values);
        std::vector<double> phi(quad.size()),pressure(quad.size());Vector<double> reactions(nv);
        std::vector<double> reaction_coordinates(nv*(dim+1),0.);std::vector<double> counts(nv,0.);
        std::map<std::string,std::vector<double>> local_records,records,local_fields,fields,local_means,means;
        const bool detailed=std::getenv("ASPECT_BULK_PRESSURE_DIAGNOSTIC");
        double pressure_integral=0.,volume=0.,p_min=std::numeric_limits<double>::max(),p_max=-p_min;
        const bool small=this->get_triangulation().n_global_active_cells()<=1000;
        const bool dense=!s.tensor_flow() && !std::getenv("ASPECT_BULK_COMPACT_OUTPUT") && ((small && s.step%4==0) || s.step==0 || std::abs(s.time-.12)<1e-12 || std::abs(s.time-.15)<1e-12
             || std::abs(s.time-.15125)<1e-12 || std::abs(s.time-.155)<1e-12 || std::abs(s.time-.16)<1e-12
             || std::abs(s.time-.17375)<1e-12 || std::abs(s.time-.175)<1e-12 || std::abs(s.time-.185)<1e-12);
        for(const auto &entry:s.native_cells)
          {
            const auto cell=entry.second;const auto key=entry.first.to_string();cell->get_dof_indices(ids);
            for(const auto v:cell->vertex_indices())local_positions[cell->vertex_dof_index(v,pc)-po]=cell->vertex(v);
            auto &record=local_records[key];
            s.tensor->space->cells.at(entry.first)->get_dof_indices(dgids);
            for(const auto *p:{s.tensor.get(),s.native_projection.get()})for(const auto &v:p->coefficients)for(const auto i:dgids)record.push_back(v[i]);
            for(unsigned int i=0;i<ids.size();++i)if(fe_element.system_to_component_index(i).first<dim)record.push_back(s.native_history[ids[i]]);
            for(const auto &h:s.history.at(entry.first))
              {for(double v:{h.a,h.signed_shear,h.absolute_shear,h.q,h.rate})record.push_back(v);
               if(s.tensor_flow()){for(const auto *t:{&h.A,&h.K,&h.Q,&h.W})for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)record.push_back((*t)[i][j]);record.push_back(h.L);}}
            fe.reinit(cell);fe[FEValuesExtractors::Scalar(pc)].get_function_values(sim().solution,phi);
            fe[this->introspection().extractors.pressure].get_function_values(sim().solution,pressure);
            const auto T=s.evaluate_tensor(m,entry.first,quad);const auto DG=m.evaluate_history_tensor(*s.tensor,entry.first,quad);
            std::vector<bool> boundary_dof(ids.size(),false);
            std::set<types::global_dof_index> boundary_ids;
            if(detailed)for(const unsigned int face:cell->face_indices())
              if(cell->face(face)->at_boundary())
                {
                  std::vector<types::global_dof_index> face_ids(fe_element.n_dofs_per_face(face));
                  cell->face(face)->get_dof_indices(face_ids);
                  boundary_ids.insert(face_ids.begin(),face_ids.end());
                }
            if(cell->at_boundary())for(unsigned int i=0;i<ids.size();++i)if(ids[i]<nv)
              {
                boundary_dof[i]=(constraints.is_constrained(ids[i]) && constraints.get_constraint_entries(ids[i])->empty())
                  || (detailed && fe_element.system_to_component_index(i).first==0 && boundary_ids.count(ids[i]));
              }
            auto &mean=local_means[key];mean.assign(s.tensor_flow()?19:(detailed?10:9),0.);mean[0]=cell->center()[0];mean[1]=cell->center()[1];
            for(unsigned int q=0;q<quad.size();++q)
              {
                const auto &h=s.history.at(entry.first)[q];const auto c=BC::compliance(phi[q],s.parameters.m);
                BC::Tensor full_T;full_T[0][0]=T[q][0][0];full_T[1][1]=T[q][1][1];full_T[2][2]=-T[q][0][0]-T[q][1][1];full_T[0][1]=T[q][0][1];
                const auto tau=s.tensor_flow()?full_T-2.*s.parameters.G*h.A:full_T;
                const double speed=s.tensor_flow()?std::sqrt(2.*(h.W*h.W)):std::abs(h.rate);
                const double xy=s.tensor_flow()?tau[0][1]:T[q][0][1]-s.parameters.G*h.a;
                const double drive=s.tensor_flow()?c.first*(h.Q*h.Q)/(4*s.parameters.G):c.first*h.q*h.q/(2*s.parameters.G),w=fe.JxW(q);
                const auto x=fe.quadrature_point(q);
                mean[2]+=w;mean[3]+=w*xy;mean[4]+=w*drive;mean[5]+=w*speed;mean[6]=std::max(mean[6],speed);mean[7]+=w*(s.tensor_flow()?tau[0][0]:T[q][0][0]);mean[8]+=w*(s.tensor_flow()?tau[1][1]:T[q][1][1]);
                if(s.tensor_flow()){mean[9]+=w*phi[q];mean[10]+=w*h.W[0][0];mean[11]+=w*h.W[0][1];mean[12]+=w*(tau*tau)/(4*s.parameters.G);mean[13]+=w*c.h*(h.Q*h.Q)/(4*s.parameters.G);mean[14]+=w*h.A[0][0];mean[15]+=w*h.A[0][1];mean[16]+=w*h.K[0][0];mean[17]+=w*h.K[0][1];mean[18]+=w*h.L;}
                if(detailed && !s.tensor_flow()){mean[9]+=w*pressure[q];pressure_integral+=w*pressure[q];volume+=w;p_min=std::min(p_min,pressure[q]);p_max=std::max(p_max,pressure[q]);}
                if(dense)
                  {
                    auto &f=local_fields[key];
                    const double norm=T[q][0][0]*T[q][0][0]+T[q][1][1]*T[q][1][1]+std::pow(T[q][0][0]+T[q][1][1],2)+2*xy*xy;
                    for(double value:{x[0],x[1],w,phi[q],T[q][0][0],T[q][1][1],T[q][0][1],T[q][0][0],T[q][1][1],xy,h.a,h.q,h.signed_shear,h.absolute_shear,drive,norm/(4*s.parameters.G),c.h*h.q*h.q/(2*s.parameters.G),DG[q][0][0],DG[q][1][1],DG[q][0][1],T[q][0][0]-DG[q][0][0],T[q][1][1]-DG[q][1][1],T[q][0][1]-DG[q][0][1]})f.push_back(value);
                    if(detailed)f.push_back(pressure[q]);
                  }
                if(cell->at_boundary())for(unsigned int i=0;i<ids.size();++i)
                  if(boundary_dof[i])
                    {
                      const auto raw=fe[this->introspection().extractors.velocities].symmetric_gradient(i,q);
                      const auto E=Utilities::Tensors::consistent_deviator(raw);
                      reactions[ids[i]]+=w*((s.tensor_flow()?tau[0][0]:T[q][0][0])*E[0][0]+(s.tensor_flow()?tau[1][1]:T[q][1][1])*E[1][1]+(s.tensor_flow()?-tau[2][2]:T[q][0][0]+T[q][1][1])*(E[0][0]+E[1][1])+2*xy*E[0][1]-(pressure[q]+(s.flow_rule==Model::BulkFlowRule::coulomb_fixed_phase?s.background_pressure:0.))*trace(raw));
                    }
              }
            for(unsigned int k=3;k<mean.size();++k)if(k!=6)mean[k]/=mean[2];
            for(unsigned int i=0;i<ids.size();++i)if(boundary_dof[i])
              {
                const auto x=this->get_mapping().transform_unit_to_real_cell(cell,fe_element.get_unit_support_points()[i]);
                counts[ids[i]]+=1.;for(unsigned int d=0;d<dim;++d)reaction_coordinates[(dim+1)*ids[i]+d]+=x[d];
                reaction_coordinates[(dim+1)*ids[i]+dim]+=fe_element.system_to_component_index(i).first;
              }
          }
        auto gather=[&](const auto &local,auto &global)
          {for(const auto &part:Utilities::MPI::gather(comm,local,0))for(const auto &entry:part)AssertThrow(global.emplace(entry).second,ExcMessage("Duplicate diagnostic cell owner."));};
        if(small || std::getenv("ASPECT_BULK_SAVE_RECORDS"))gather(local_records,records);
        gather(local_means,means);if(dense)gather(local_fields,fields);
        for(const auto &part:Utilities::MPI::gather(comm,local_positions,0))for(const auto &entry:part)
          {auto inserted=positions.emplace(entry);AssertThrow(inserted.second || inserted.first->second==entry.second,ExcInternalError());}
        BC::sum(reactions,comm);internal::endpoint_sum(reaction_coordinates,comm);internal::endpoint_sum(counts,comm);
        const auto velocity=internal::endpoint_values(sim().solution.block(this->introspection().block_indices.velocities),comm);
        auto node=output("phase-"+label+".csv");node<<"dof,x,y,phi\n";
        for(const auto &entry:positions)node<<entry.first<<','<<entry.second[0]<<','<<entry.second[1]<<','<<s.phase[entry.first]<<'\n';
        auto reaction=output("boundary-reactions-"+label+".csv");reaction<<"dof,component,x,y,force,increment,prescribed,actual_increment\n";
        double free_norm=0.;
        for(unsigned int i=0;i<nv;++i)if(counts[i]>0.)
          {
            const bool prescribed=constraints.is_constrained(i);
            const double du=s.step?this->get_timestep()*velocity[i]:0.;
            if(!prescribed)free_norm+=reactions[i]*reactions[i];
            if(root())reaction<<i<<','<<reaction_coordinates[(dim+1)*i+dim]/counts[i]<<','<<reaction_coordinates[(dim+1)*i]/counts[i]<<','<<reaction_coordinates[(dim+1)*i+1]/counts[i]<<','<<reactions[i]<<','<<(prescribed?du:0.)<<','<<prescribed<<','<<du<<'\n';
          }
        if(detailed)
          {
            pressure_integral=Utilities::MPI::sum(pressure_integral,comm);volume=Utilities::MPI::sum(volume,comm);
            p_min=Utilities::MPI::min(p_min,comm);p_max=Utilities::MPI::max(p_max,comm);
            check("natural free-side weak residual",s.step==0 || std::sqrt(free_norm)/s.parameters.G<=this->get_parameters().nonlinear_tolerance);
            if(root()){std::ofstream out(this->get_output_directory()+"pressure.csv",std::ios::app);if(out.tellp()==0)out<<"step,time,mean,min,max,free_side_force_L2,free_side_scaled_residual\n";out<<std::setprecision(17)<<s.step<<','<<s.time<<','<<pressure_integral/volume<<','<<p_min<<','<<p_max<<','<<std::sqrt(free_norm)<<','<<std::sqrt(free_norm)/s.parameters.G<<'\n';}
          }

        auto mean=output("means-"+label+".csv");mean<<"cell,x,y,area,tauxy,Y,abs_rate,max_rate,tauxx,tauyy";if(s.tensor_flow())mean<<",phi,Wxx,Wxy,bulk,cohesive,Axx,Axy,Kxx,Kxy,L";else if(detailed)mean<<",pressure";mean<<'\n';
        for(const auto &entry:means){mean<<entry.first;for(double v:entry.second)mean<<','<<v;mean<<'\n';}
        auto binary=output("record-"+label+".bin");for(const auto &entry:records)binary.write(reinterpret_cast<const char*>(entry.second.data()),entry.second.size()*sizeof(double));
        if(dense)
          {
            auto out=output("bulk-"+label+".csv");out<<"cell,x,y,JxW,phi,Txx,Tyy,Txy,tauxx,tauyy,tauxy,a,q,k,L,driving,bulk,cohesive,DGxx,DGyy,DGxy,Rxx,Ryy,Rxy";if(detailed)out<<",pressure";out<<'\n';
            for(const auto &entry:fields)for(unsigned int q=0;q<quad.size();++q){out<<entry.first;for(unsigned int k=0;k<(detailed?24U:23U);++k)out<<','<<entry.second[(detailed?24:23)*q+k];out<<'\n';}
          }
        if(s.step==0)
          {
#ifdef ASPECT_COULOMB_BLOCK_FIXTURE
            if(std::getenv("ASPECT_COULOMB_LOCALIZATION"))
              {
                std::map<std::string,std::vector<double>> basis_local,basis_global;
                for(const auto &entry:s.native_cells)
                  {
                    entry.second->get_dof_indices(ids);auto &v=basis_local[entry.first.to_string()];
                    for(unsigned int i=0;i<ids.size();++i)
                      {
                        const unsigned int c=fe_element.system_to_component_index(i).first;
                        if(c<dim || c==this->introspection().component_indices.pressure)
                          {v.push_back(c);v.push_back(fe_element.get_unit_support_points()[i][0]);v.push_back(fe_element.get_unit_support_points()[i][1]);v.push_back(ids[i]);}
                      }
                  }
                gather(basis_local,basis_global);auto basis=output("localization-cell-dofs.csv");basis<<"cell,component,u,v,dof\n";
                for(const auto &entry:basis_global)for(unsigned int i=0;i<entry.second.size();i+=4)
                  basis<<entry.first<<','<<entry.second[i]<<','<<entry.second[i+1]<<','<<entry.second[i+2]<<','<<entry.second[i+3]<<'\n';
              }
#endif
            auto map=output("native-cell-dofs.csv");map<<"cell,component,u,v,dof\n";
            std::map<std::string,std::vector<double>> local,global;
            for(const auto &entry:s.native_cells)
              {entry.second->get_dof_indices(ids);auto &v=local[entry.first.to_string()];for(unsigned int i=0;i<ids.size();++i)if(fe_element.system_to_component_index(i).first<dim)
                {v.push_back(fe_element.system_to_component_index(i).first);v.push_back(fe_element.get_unit_support_points()[i][0]);v.push_back(fe_element.get_unit_support_points()[i][1]);v.push_back(ids[i]);}}
            gather(local,global);for(const auto &entry:global)for(unsigned int i=0;i<entry.second.size();i+=4)map<<entry.first<<','<<entry.second[i]<<','<<entry.second[i+1]<<','<<entry.second[i+2]<<','<<entry.second[i+3]<<'\n';
            local.clear();global.clear();
            for(const auto &entry:s.native_cells)
              {const auto cell=s.tensor->space->cells.at(entry.first);cell->get_dof_indices(dgids);auto &v=local[entry.first.to_string()];
               for(unsigned int i=0;i<9;++i){v.push_back(s.tensor->space->element.get_unit_support_points()[i][0]);v.push_back(s.tensor->space->element.get_unit_support_points()[i][1]);v.push_back(dgids[i]);}}
            gather(local,global);auto dgmap=output("dg-cell-dofs.csv");dgmap<<"cell,u,v,dof\n";
            for(const auto &entry:global)for(unsigned int i=0;i<entry.second.size();i+=3)dgmap<<entry.first<<','<<entry.second[i]<<','<<entry.second[i+1]<<','<<entry.second[i+2]<<'\n';

          }
        check("unique distributed cell history ownership",Utilities::MPI::sum(s.history.size(),comm)==this->get_triangulation().n_global_active_cells());
      }
#ifdef ASPECT_COULOMB_BLOCK_FIXTURE
      void export_coulomb_block()const
      {
        namespace CR=MaterialModel::BulkCohesive::CoulombReturn;
        const auto &m=model();const auto &s=*m.bulk_cohesive_state;
        AssertThrow(s.flow_rule==Model::BulkFlowRule::coulomb_fixed_phase,ExcInternalError());
        const auto quad=m.mechanical_quadrature();
        std::vector<MaterialModel::BulkCohesive::Tensor> X;
        if(s.slip_length>0. && s.slip_space)
          {
            auto K=std::vector<MaterialModel::BulkCohesive::Tensor>(s.slip_space->points.size());
            for(const auto &cell:s.slip_space->ids)for(unsigned int q=0;q<cell.second.size();++q)K[cell.second[q]]=s.history.at(cell.first)[q].K;
            X=s.slip_space->apply(K,s.parameters.G,s.slip_length);
          }
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),quad,update_values|update_JxW_values|update_quadrature_points);
        std::vector<double> pressure(quad.size());
        std::map<std::string,std::vector<double>> local,global;
        for(const auto &entry:s.native_cells)
          {
            fe.reinit(entry.second);fe[this->introspection().extractors.pressure].get_function_values(sim().solution,pressure);
            const auto T=s.evaluate_tensor(m,entry.first,quad);auto &values=local[entry.first.to_string()];
            for(unsigned int q=0;q<quad.size();++q)
              {
                const auto &history=s.history.at(entry.first)[q];MaterialModel::BulkCohesive::Tensor tau;
                tau[0][0]=T[q][0][0];tau[1][1]=T[q][1][1];tau[2][2]=-tau[0][0]-tau[1][1];tau[0][1]=T[q][0][1];tau-=2*s.parameters.G*history.A;
                const double total=s.background_pressure+pressure[q],mean=.5*(tau[0][0]+tau[1][1]);
                const double minimum=total-mean-std::hypot(.5*(tau[0][0]-tau[1][1]),tau[0][1]);
                const auto backstress=X.empty()?MaterialModel::BulkCohesive::Tensor():X[s.slip_space->ids.at(entry.first)[q]];
                const auto planes=CR::planes(tau,history.Q+backstress,total,s.coulomb_mu,0.);
                double plus=0.,minus=0.;
                if(history.W.norm()>1e-14)
                  {
                    const double px=-std::sin(2*planes[0].theta),py=std::cos(2*planes[0].theta);
                    const double mx=std::sin(2*planes[1].theta),my=-std::cos(2*planes[1].theta),det=px*my-py*mx;
                    AssertThrow(std::abs(det)>1e-12,ExcMessage("Degenerate active-plane decomposition."));
                    plus=2*(history.W[0][0]*my-history.W[0][1]*mx)/det;
                    minus=2*(px*history.W[0][1]-py*history.W[0][0])/det;
                    AssertThrow(plus>=-1e-12 && minus>=-1e-12,ExcMessage("Negative accepted system slip speed."));
                  }
                const auto x=fe.quadrature_point(q);
                for(double v:{x[0],x[1],fe.JxW(q),s.background_pressure,pressure[q],total,minimum,
                              planes[0].defined?planes[0].normal:total-mean,planes[1].defined?planes[1].normal:total-mean,
                              planes[0].defined?planes[0].theta:std::numeric_limits<double>::quiet_NaN(),
                              planes[1].defined?planes[1].theta:std::numeric_limits<double>::quiet_NaN(),
                              planes[0].margin,planes[1].margin,plus,minus,
                              tau[0][0],tau[1][1],tau[2][2],tau[0][1],history.Q[0][0],history.Q[0][1],
                              history.A[0][0],history.A[0][1],history.K[0][0],history.K[0][1],history.L,
                              history.W[0][0],history.W[0][1],static_cast<double>(planes[0].defined),static_cast<double>(planes[1].defined)})values.push_back(v);
                if(s.slip_length>0.){values.push_back(backstress[0][0]);values.push_back(backstress[0][1]);}
              }
          }
        for(const auto &part:Utilities::MPI::gather(this->get_mpi_communicator(),local,0))
          for(const auto &entry:part)AssertThrow(global.emplace(entry).second,ExcMessage("Duplicate Coulomb output owner."));
        auto out=output("coulomb-"+Utilities::int_to_string(s.step,4)+".csv");
        out<<"cell,q,x_m,y_m,JxW_m2,p0_Pa,pprime_Pa,total_pressure_Pa,min_plane_normal_Pa,pn_plus_Pa,pn_minus_Pa,theta_plus_rad,theta_minus_rad,f_plus_Pa,f_minus_Pa,w_plus_s-1,w_minus_s-1,tauxx_Pa,tauyy_Pa,tauzz_Pa,tauxy_Pa,Qxx_Pa,Qxy_Pa,Axx,Axy,Kxx,Kxy,L,Wxx_s-1,Wxy_s-1,plane_plus_defined,plane_minus_defined";
        if(s.slip_length>0.)out<<",Xxx_Pa,Xxy_Pa";out<<'\n';
        for(const auto &entry:global)for(unsigned int q=0;q<quad.size();++q)
          {const unsigned int stride=s.slip_length>0.?32:30;out<<entry.first<<','<<q;for(unsigned int i=0;i<stride;++i)out<<','<<entry.second[stride*q+i];out<<'\n';}
      }
#endif
      public:
        void initialize()override
        {
          AssertThrow(model().uses_bulk_cohesive_phase(),ExcMessage("Bulk MPI fixture requires bulk cohesive mode."));
          this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&){seed();});
#ifdef ASPECT_COULOMB_BLOCK_FIXTURE
          this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim>&,double &dt)
            {if(std::getenv("ASPECT_COULOMB_LOCALIZATION"))dt=std::min(dt,this->get_parameters().maximum_time_step);});
#endif
          this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim>&)
            {
              // Replay the existing outer rejection test even if its saved
              // checkpoint has already exercised it. No production parameter
              // or persistent physical history is changed by this test hook.
              const bool forced=std::getenv("ASPECT_BULK_TEST_RECOVERY") && !test_recovery_done;
              if((forced || (!recovered && std::getenv("ASPECT_BULK_RECOVERY"))) && this->get_time()>(std::getenv("ASPECT_BULK_RECOVERY_TIME")?std::atof(std::getenv("ASPECT_BULK_RECOVERY_TIME")):.139))
                {recovery();test_recovery_done=true;}
            });
        }
        std::pair<std::string,std::string> execute(TableHandler&)override
        {
          const auto start=std::chrono::steady_clock::now();const auto &s=*model().bulk_cohesive_state;
          check("single accepted publication",s.publications==this->get_timestep_number() && (!s.publications || s.publications==publications+1));publications=s.publications;
          export_state();
#ifdef ASPECT_COULOMB_BLOCK_FIXTURE
          export_coulomb_block();
#endif
          struct rusage usage;getrusage(RUSAGE_SELF,&usage);
          const auto peaks=Utilities::MPI::gather(this->get_mpi_communicator(),static_cast<double>(usage.ru_maxrss),0);
          const auto &particles=this->get_particle_manager(0).get_particle_handler();
          std::uint64_t fingerprint=0;
          for(const auto &particle:particles)
            {
              std::uint64_t h=1469598103934665603ULL;
              auto add=[&](const std::uint64_t word){h^=word;h*=1099511628211ULL;};
              add(particle.get_id());for(unsigned int d=0;d<dim;++d){std::uint64_t bits;const double x=particle.get_location()[d];std::memcpy(&bits,&x,sizeof(bits));add(bits);}fingerprint+=h;
            }
          const auto global_fingerprint=Utilities::MPI::sum(fingerprint,this->get_mpi_communicator());
          if(root())
            {std::ofstream out(this->get_output_directory()+"resources.csv",std::ios::app);if(out.tellp()==0)out<<"step,time,particles,particle_fingerprint,max_rank_KiB,sum_rank_peaks_KiB,freeze\n";
             out<<std::setprecision(17)<<s.step<<','<<s.time<<','<<particles.n_global_particles()<<','<<global_fingerprint<<','<<*std::max_element(peaks.begin(),peaks.end())<<','<<std::accumulate(peaks.begin(),peaks.end(),0.)<<','<<(std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION")?std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION"):"unset")<<'\n';}
          if(root())
            {std::ofstream out(this->get_output_directory()+"output-profile.csv",std::ios::app);if(out.tellp()==0)out<<"step,time,seconds\n";out<<std::setprecision(17)<<s.step<<','<<s.time<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';}
          return {"Bulk parallel fixture","accepted"};
        }
        void save(std::map<std::string,std::string> &status)const override
        {status["bulk parallel publications"]=std::to_string(publications);status["bulk parallel recovered"]=std::to_string(recovered);}
        void load(const std::map<std::string,std::string> &status)override
        {publications=std::stoul(status.at("bulk parallel publications"));recovered=std::stoul(status.at("bulk parallel recovered"));}
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"bulk cohesive checks","Fixed-mesh distributed ownership, lifecycle and physical-field observer.")
  }
}
