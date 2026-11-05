#include "../../../../../tests/phase_field_fault_test_access.h"
#include "spatial_relaxation.h"
/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/history/tensor_history.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include <deal.II/fe/fe_values.h>
#include <aspect/time_stepping/interface.h>
#include <aspect/mesh_refinement/interface.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace {int transfer_direction=0;}
  namespace MeshRefinement
  {
    template<int dim> class ObjectiveDGMesh : public Interface<dim>,public SimulatorAccess<dim>
    {
    public:
      void execute(Vector<float> &indicator) const override {indicator=0.;}
      void tag_additional_cells() const override
      {
        if(!transfer_direction)return;
        for(const auto &cell:this->get_triangulation().active_cell_iterators())
          if(cell->is_locally_owned())
            {
              cell->clear_refine_flag();cell->clear_coarsen_flag();
              if(transfer_direction>0)cell->set_refine_flag();
              else if(cell->level()>0)cell->set_coarsen_flag();
            }
      }
    };
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(ObjectiveDGMesh,"objective DG transfer","Bounded complete tensor/residual mesh-transfer fixture.")
  }
}

namespace aspect { namespace Postprocess
{
  template<int dim> class MomentCycle : public Interface<dim>,public SimulatorAccess<dim>
  {
    using Model=MaterialModel::PhaseFieldFault<dim>;
    using History=typename Model::DGHistory;
    using Tensor=SymmetricTensor<2,dim>;
    std::shared_ptr<History> incoming;
    std::vector<double> Z,I,coefficients;
    LinearAlgebra::BlockVector lagged;
    unsigned int captured=numbers::invalid_unsigned_int,publications=0,retries=0;
    std::vector<std::vector<double>> surface;
    static double norm3(const Tensor &a)
    {return std::sqrt(a*a+std::pow(a[0][0]+a[1][1],2));}

    void transfer_check()
    {
      auto &sim=const_cast<Simulator<dim> &>(this->get_simulator());
      auto &model=Plugins::get_plugin_as_type<Model>(*sim.material_model);
      const auto Z0=model.fe_strain_state->Z,I0=model.fe_strain_state->I;
      const auto publications0=model.fe_strain_state->accepted_publications;
      const auto records=[&]()
        {
          std::map<std::string,std::vector<double>> local,all;
          std::vector<types::global_dof_index> ids(9);
          for(const auto &entry:model.fe_strain_state->cells)
            {
              const auto &h=*model.fe_strain_state->dg_history;
              h.space->cells.at(entry.first)->get_dof_indices(ids);
              auto &row=local[entry.first.to_string()];
              for(unsigned int c=0;c<3;++c)for(const auto i:ids)
                {row.push_back(h.coefficients[c][i]);row.push_back(h.residual->coefficients[c][i]);}
            }
          for(const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))
            for(const auto &entry:part)AssertThrow(all.emplace(entry).second,ExcMessage("Duplicate physical tensor owner in AMR audit."));
          return all;
        };
      const auto before=records();
      std::map<std::string,std::array<double,4>> boxes,local_boxes;
      for(const auto &entry:model.fe_strain_state->cells)
        local_boxes[entry.first.to_string()]={{entry.second->vertex(0)[0],entry.second->vertex(0)[1],
                                             entry.second->vertex(3)[0],entry.second->vertex(3)[1]}};
      for(const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local_boxes))
        boxes.insert(part.begin(),part.end());
      const auto cells0=this->get_triangulation().n_global_active_cells();
      for(const int direction:{1,-1})
        {
          transfer_direction=direction;
          sim.refine_mesh(this->get_triangulation().n_global_levels()+1);
          transfer_direction=0;
          sim.compute_current_constraints();
        }
      double error=0.,scale=1.;
      const QGauss<dim> rule(8);
      const auto &element=model.fe_strain_state->dg_history->space->element;
      for(const auto &entry:model.fe_strain_state->cells)
        {
          const auto T=model.evaluate_history_tensor(*model.fe_strain_state->dg_history,entry.first,rule);
          const auto R=model.evaluate_history_tensor(*model.fe_strain_state->dg_history->residual,entry.first,rule);
          for(unsigned int q=0;q<rule.size();++q)
            {
              const auto x=this->get_mapping().transform_unit_to_real_cell(entry.second,rule.point(q));
              auto old=std::find_if(boxes.begin(),boxes.end(),[&](const auto &b)
                {return x[0]>=b.second[0] && x[0]<=b.second[2] && x[1]>=b.second[1] && x[1]<=b.second[3];});
              AssertThrow(old!=boxes.end(),ExcMessage("Transfer audit lost original physical-cell coverage."));
              const auto &b=old->second;Point<dim> u;u[0]=(x[0]-b[0])/(b[2]-b[0]);u[1]=(x[1]-b[1])/(b[3]-b[1]);
              Tensor expected_T,expected_R;
              const auto &row=before.at(old->first);
              for(unsigned int c=0;c<3;++c)
                {
                  const auto ij=Tensor::unrolled_to_component_indices(c);
                  for(unsigned int i=0;i<9;++i)
                    {expected_T[ij]+=row[2*(9*c+i)]*element.shape_value(i,u);expected_R[ij]+=row[2*(9*c+i)+1]*element.shape_value(i,u);}
                }
              error=std::max({error,norm3(T[q]-expected_T),norm3(R[q]-expected_R)});
              scale=std::max({scale,norm3(expected_T),norm3(expected_R)});
            }
        }
      error=Utilities::MPI::max(error,this->get_mpi_communicator());
      scale=Utilities::MPI::max(scale,this->get_mpi_communicator());
      AssertThrow(error<2e-11*scale && publications0==model.fe_strain_state->accepted_publications
                  && Z0==model.fe_strain_state->Z && I0==model.fe_strain_state->I,
                  ExcMessage("Objective DG AMR roundtrip changed accepted physical history/publication."));
      if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
        {
          std::ofstream out(this->get_output_directory()+"objective-dg-transfer.csv");
          out<<"cells_before,cells_after,field_error_Pa,scale_Pa,publications\n"<<std::setprecision(17)
             <<cells0<<','<<this->get_triangulation().n_global_active_cells()<<','<<error<<','<<scale<<','<<publications0<<'\n';
        }
      // Destructive diagnostic transfer has ended; ordinary execution rebuilds
      // the matrices at the next timestep. It has not advanced a physical clock.
    }

    void transport_check() const
    {
      const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
      const auto accepted=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::dg_history_values(model);
      const auto parent=model.fe_strain_state->dg_history;
      struct Flow : Function<dim>
      {
        Flow(unsigned int n,double c,double cy):Function<dim>(n),c(c),cy(cy){}
        double c,cy;
        double value(const Point<dim> &,unsigned int component) const override
        {return component==0 ? c : (component==1 ? cy : 0.);}
      };
      for(const double direction:{-1.,1.})
        {
          const bool layered=std::getenv("ASPECT_OBJECTIVE_DG_LAYERED_CHECK");
          const double c=layered ? 0. : 200*direction,cy=layered ? 200*direction : 0.,finish=2.;
          LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),Flow(this->get_fe().n_components(),c,cy),owned);
          LinearAlgebra::BlockVector velocity(this->get_solution());velocity=owned;
          const auto op=model.make_dg_transport(velocity,8,false);
          ParticleFreeBenchmark::SpatialRelaxation relaxation;
          if(layered)relaxation.template attach<typename Model::DGTransport,dim>(*op,[](const Point<dim> &x){return .5+.3*std::tanh((x[1]-637.25)/120.);},this->get_mapping(),this->get_mpi_communicator());
          const auto exact=[c,cy,layered](const Point<dim> &x,double time)
            {
              const double u=(x[0]-c*time+200)/180.,v=(x[1]-cy*time-637.25)/160.;
              const double integrated_rate=layered ? .5*time+.3*120/cy
                *(std::log(std::cosh((x[1]-637.25)/120.))-std::log(std::cosh((x[1]-cy*time-637.25)/120.))) : 0.;
              const double f=1e6*std::exp(-.5*(u*u+v*v)-integrated_rate);
              Tensor a;a[0][0]=f*(1+.15*std::sin(v));a[1][1]=-.3*f*std::cos(.5*u);
              a[0][1]=.45*f*(1+.2*std::sin(u));return a;
            };
          const auto zero=[](const Point<dim> &,double){return Tensor();};
          auto history=model.project_dg_transport_field(*op,exact,0.,8);
          const unsigned int steps=std::max(1u,static_cast<unsigned int>(std::ceil(finish*(op->inverse_advection_time+op->relaxation_rate)/.1)));
          const double dt=finish/steps;
          for(unsigned int k=0;k<steps;++k)
            history=model.advance_dg_transport(*op,*history,exact,zero,k*dt,dt);
          const QGauss<dim> rule(10);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_quadrature_points|update_JxW_values);
          double error=0.,reference=0.,maximum=0.;
          for(const auto &entry:model.fe_strain_state->cells)
            {
              fe.reinit(entry.second);
              const auto actual=model.evaluate_history_tensor(*history,entry.first,rule);
              for(unsigned int q=0;q<rule.size();++q)
                {
                  const auto expected=exact(fe.quadrature_point(q),finish);
                  error+=fe.JxW(q)*std::pow(norm3(actual[q]-expected),2);
                  reference+=fe.JxW(q)*std::pow(norm3(expected),2);
                  maximum=std::max(maximum,norm3(actual[q]-expected));
                }
            }
          const auto comm=this->get_mpi_communicator();
          error=Utilities::MPI::sum(error,comm);reference=Utilities::MPI::sum(reference,comm);
          maximum=Utilities::MPI::max(maximum,comm);
          AssertThrow(parent==model.fe_strain_state->dg_history && accepted==MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::dg_history_values(model),
                      ExcMessage("Nonpublishing transport operator changed accepted history."));
          if(Utilities::MPI::this_mpi_process(comm)==0)
            {
              std::ofstream out(this->get_output_directory()+"objective-dg-transport.csv",direction<0 ? std::ios::out : std::ios::app);
              if(direction<0)out<<"direction,cells,steps,dt_s,distance_m,relative_L2,max_error_Pa,layered,max_rate_per_s\n";
              out<<std::setprecision(17)<<direction<<','<<this->get_triangulation().n_global_active_cells()<<','<<steps<<','<<dt<<",400,"<<std::sqrt(error/reference)<<','<<maximum<<','<<layered<<','<<op->relaxation_rate<<'\n';
            }
        }
    }

    static Tensor rotate_reference(const Tensor &a,double angle)
    {
      Tensor b;
      const double mean=.5*(a[0][0]+a[1][1]),difference=.5*(a[0][0]-a[1][1]);
      b[0][0]=mean+difference*std::cos(2*angle)-a[0][1]*std::sin(2*angle);
      b[1][1]=2*mean-b[0][0];
      b[0][1]=difference*std::sin(2*angle)+a[0][1]*std::cos(2*angle);
      return b;
    }
    void capture(bool temperature)
    {
      if (!temperature) return;
      const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
      if (!model.fe_strain_state) return;
      const auto &state=*model.fe_strain_state;
      if (captured==this->get_timestep_number())
        {
          const bool unchanged=publications==0 && incoming==state.dg_history
            && coefficients==MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::dg_history_values(model) && Z==state.Z && I==state.I;
          AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(unchanged),this->get_mpi_communicator())==1,
                      ExcMessage("Rejected objective DG trial modified accepted tensor/residual/scalar history."));
          ++retries;
        }
      captured=this->get_timestep_number();publications=0;
      incoming=state.dg_history;Z=state.Z;I=state.I;coefficients=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>::dg_history_values(model);
      lagged.reinit(this->get_old_solution());lagged=this->get_old_solution();
    }
  public:
    void initialize() override
    {
      this->get_signals().post_advection_solver.connect([this](const auto &,bool t,unsigned int,const auto &){capture(t);});
      this->get_signals().post_reconstructed_fault_solver.connect([this](unsigned int,unsigned int,double,const auto &){++publications;});
    }
    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
      const auto &state=*model.fe_strain_state;
      if (!this->get_timestep_number())
        {
          if(std::getenv("ASPECT_OBJECTIVE_DG_TRANSPORT_CHECK"))transport_check();
          return {"DG objective checks","initial accepted state"};
        }
      AssertThrow(incoming && incoming->residual && state.dg_history->residual && publications==1,
                  ExcMessage("Objective DG fixture lacks complete single-publication history."));
      const auto &intro=this->introspection();
      const QGauss<dim> rule(8);
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_gradients|update_values|update_quadrature_points|update_JxW_values);
      FEValues<dim> dg(this->get_mapping(),incoming->space->element,rule,update_values);
      std::vector<dealii::Tensor<2,dim>> gradients(rule.size());
      std::vector<SymmetricTensor<2,dim>> strains(rule.size());
      std::vector<double> phase(rule.size());
      const double dt=this->get_timestep(),beta=std::exp(-dt*state.G/state.eta),kappa=-state.eta*std::expm1(-dt*state.G/state.eta);
      double defect=0.,scale=1.,Rnorm=0.,angle=0.;
      for (const auto &entry:state.cells)
        {
          fe.reinit(entry.second);dg.reinit(incoming->space->cells.at(entry.first));
          fe[intro.extractors.velocities].get_function_gradients(lagged,gradients);
          fe[intro.extractors.velocities].get_function_symmetric_gradients(this->get_solution(),strains);
          fe[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)].get_function_values(state.displacement_history,phase);
          const auto T0=model.evaluate_history_tensor(*incoming,entry.first,rule);
          const auto R0=model.evaluate_history_tensor(*incoming->residual,entry.first,rule);
          const auto T1=model.evaluate_history_tensor(*state.dg_history,entry.first,rule);
          const auto R1=model.evaluate_history_tensor(*state.dg_history->residual,entry.first,rule);
          std::array<Tensor,9> moments_T,moments_R;
          const double volume=entry.second->measure();
          for (unsigned q=0;q<rule.size();++q)
            {
              const auto B=model.evaluate_mature_dg_memory(fe.quadrature_point(q),phase[q],Z,I);
              const double theta=.5*dt*(gradients[q][1][0]-gradients[q][0][1]);
              angle=std::max(angle,std::abs(theta));
              const auto expected_T=beta*rotate_reference(T0[q],theta)+2*kappa*Utilities::Tensors::consistent_deviator(strains[q]);
              const auto expected_R=beta*(rotate_reference(R0[q]+B,theta)-B);
              const auto change_T=T1[q]-expected_T,change_R=R1[q]-expected_R;
              // Independent monomial moments, not the production nodal mass solve.
              const auto u=rule.point(q);
              for (unsigned ix=0;ix<3;++ix)for(unsigned iy=0;iy<3;++iy)
                {
                  const double weight=fe.JxW(q)/volume*std::pow(u[0],ix)*std::pow(u[1],iy);
                  moments_T[3*ix+iy]+=weight*change_T;
                  moments_R[3*ix+iy]+=weight*change_R;
                }
              scale=std::max({scale,norm3(T0[q]),norm3(B),2*kappa*norm3(Utilities::Tensors::consistent_deviator(strains[q]))});
              Rnorm=std::max(Rnorm,norm3(R1[q]));
            }
          for (unsigned i=0;i<9;++i)
            defect=std::max({defect,norm3(moments_T[i]),norm3(moments_R[i])});
        }
      const auto comm=this->get_mpi_communicator();
      defect=Utilities::MPI::max(defect,comm);scale=Utilities::MPI::max(scale,comm);
      Rnorm=Utilities::MPI::max(Rnorm,comm);angle=Utilities::MPI::max(angle,comm);
      AssertThrow(defect<2e-11*scale,ExcMessage("Independent objective DG Galerkin recurrence failed."));
      if (Utilities::MPI::this_mpi_process(comm)==0)
        {
          std::ofstream out(this->get_output_directory()+"objective-dg-summary.csv",this->get_timestep_number()==1 ? std::ios::out : std::ios::app);
          if(this->get_timestep_number()==1)out<<"step,time_s,dt_s,moment_error_Pa,scale_Pa,residual_norm_Pa,max_angle_rad,publications,retries\n";
          out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<dt<<','<<defect<<','<<scale<<','<<Rnorm<<','<<angle<<','<<publications<<','<<retries<<'\n';
        }
      std::ofstream fields(this->get_output_directory()+"objective-dg-final-rank"+std::to_string(Utilities::MPI::this_mpi_process(comm))+".csv");
      fields<<std::setprecision(17)<<"cell,c,i,T_Pa,R_Pa\n";
      std::vector<types::global_dof_index> ids(9);
      for (const auto &entry:state.cells)
        {
          state.dg_history->space->cells.at(entry.first)->get_dof_indices(ids);
          for (unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)
            fields<<entry.first.to_string()<<','<<c<<','<<i<<','<<state.dg_history->coefficients[c][ids[i]]<<','<<state.dg_history->residual->coefficients[c][ids[i]]<<'\n';
        }
      if(std::getenv("ASPECT_OBJECTIVE_DG_TRANSFER_CHECK") && this->get_timestep_number()==3)transfer_check();
      return {"DG objective checks","independent tensor/residual moments and publication passed"};
    }
  };
  ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"objective DG checks","Bounded complete native objective DG history audit.")
}}

namespace aspect { namespace TimeStepping
{
  template<int dim> class ObjectiveDGRejection : public Interface<dim>,public SimulatorAccess<dim>
  {
    bool rejected=false;
  public:
    double execute() override {return std::numeric_limits<double>::max();}
    std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
    {
      if (this->get_timestep_number()==2 && !rejected)
        {rejected=true;return {Reaction::repeat_step,.5*this->get_timestep()};}
      return {Reaction::advance,std::numeric_limits<double>::max()};
    }
  };
  ASPECT_REGISTER_TIME_STEPPING_MODEL(ObjectiveDGRejection,"objective DG rejection",
    "Bounded test: reject one completed native objective DG step before publication.")
}}
