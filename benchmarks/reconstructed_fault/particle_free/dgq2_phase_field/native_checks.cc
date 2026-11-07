/* Native DG phase diagnostics. No solve, publication or accepted-state mutation. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/phase_field.h>
#include <aspect/phase_field/dg_operator.h>
#include <aspect/reconstructed_fault/endpoint_phase_constraints.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/reconstructed_fault/endpoint_mpi.h>
#include "../../../../tests/phase_field_fault_test_access.h"
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class DGPhaseChecks : public Interface<dim>, public SimulatorAccess<dim>
    {
        std::vector<double> previous;
        double previous_integral=0.;
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          using Model=MaterialModel::PhaseFieldFault<dim>;
          using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
          const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
          const auto &handler=this->get_phase_field_handler();
          AssertThrow(handler.uses_dg_phase(),ExcMessage("Native phase checks require DGQ2."));
          const auto comm=this->get_mpi_communicator();
          const auto b=this->introspection().variable("phase_field").block_index;
          const auto c=this->introspection().variable("phase_field").first_component_index;
          const auto &sets=this->introspection().index_sets;
          LinearAlgebra::BlockVector owned(sets.system_partitioning,comm),base(this->get_solution());
          owned=this->get_solution();
          const auto n=owned.block(b).size();
          AffineConstraints<double> empty;empty.close();
          internal::EndpointPhaseConstraints coordinates(empty,owned.get_block_indices().block_start(b),n);
          coordinates.set_discontinuous_q2(this->get_dof_handler(),c,owned.get_block_indices().block_start(b),comm);
          const auto physical=internal::endpoint_values(owned.block(b),comm);
          const auto coefficients=coordinates.select(physical);
          double increment=0.,minimum_gap=0.;
          if (!previous.empty())
            for (unsigned int i=0;i<n;++i)
              {increment=std::max(increment,coefficients[i]-previous[i]);minimum_gap=std::min(minimum_gap,coefficients[i]-previous[i]);}
          PhaseField::DGOperator<dim> op(this->get_mapping(),this->get_dof_handler(),c,owned.get_block_indices().block_start(b),
                                         sets.system_partitioning[b],sets.system_relevant_partitioning[b],comm);
          const double integral=op.integral(owned.block(b));
          const auto physical_fracture=handler.evaluate_cell_fracture(this->get_solution(),false);
          if (Utilities::MPI::this_mpi_process(comm)==0)
            {
              std::ofstream history(this->get_output_directory()+"dg-phase-growth.csv",std::ios::app);
              if (history.tellp()==0) history<<"step,time,integral,integral_increment,maximum_coefficient_growth,minimum_coefficient_growth,fracture_energy\n";
              history<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<integral<<','
                     <<(previous.empty()?0.:integral-previous_integral)<<','<<increment<<','<<minimum_gap<<','<<physical_fracture.energy<<'\n';
            }
          AssertThrow(minimum_gap>=-1e-12,ExcMessage("DG native growth healed the Bernstein material obstacle."));
          previous.assign(coefficients.begin(),coefficients.end());previous_integral=integral;
          std::vector<double> direction(n);
          for (unsigned int i=0;i<n;++i) direction[i]=std::sin(1.7*i)+.3*std::cos(.37*i);
          for (const auto i:sets.system_partitioning[b]) owned.block(b)[i]=.2+.01*std::sin(.7*i);
          owned.compress(VectorOperation::insert);base=owned;
          std::map<std::pair<unsigned int,unsigned int>,double> hessian;
          const auto fracture=handler.evaluate_cell_fracture(base,true,&hessian);
          std::vector<double> action(n,0.);
          double derivative=0.;
          for (unsigned int i=0;i<n;++i) derivative+=fracture.gradient[i]*direction[i];
          for (const auto &entry:hessian) action[entry.first.first]+=entry.second*direction[entry.first.second];
          const auto normalization=Access::fe_normalization_derivatives(model,base,direction);
          // Independent higher-order integration of the cell-clipped Q2 field,
          // before the same physical surface mass projection.
          const auto &normal_state=Access::fe_normalization_state(model);
          const auto &normal_cells=Access::normalization_cells(model);
          const auto &element=this->get_fe().base_element(this->get_fe().component_to_base_index(c).first);
          const QGauss<1> reference_rule(32);
          const auto reference_normalization=[&](const LinearAlgebra::BlockVector &field)
          {
            const auto phase=internal::endpoint_values(field.block(b),comm);
            std::vector<double> integrals(normal_cells.profiles.size(),0.);
            for (unsigned int k=0;k<normal_cells.profiles.size();++k)
              for (const auto &interval:normal_cells.profiles[k].intervals)
                for (unsigned int q=0;q<reference_rule.size();++q)
                  {
                    const double length=interval.upper-interval.lower;
                    const auto point=interval.reference_origin+(interval.lower+length*reference_rule.point(q)[0])*interval.reference_direction;
                    double phi=0.;
                    for (unsigned int j=0;j<interval.phase_dofs.size();++j)
                      phi+=element.shape_value(j,point)*phase[interval.phase_dofs[j]-field.get_block_indices().block_start(b)];
                    integrals[k]+=length*reference_rule.weight(q)*normal_state.m*phi*(1+phi)/((1-phi)*(1-phi));
                  }
            internal::endpoint_sum(integrals,comm);
            Vector<double> projected(normal_state.projection.m()),values(integrals.begin(),integrals.end());
            normal_state.projection.vmult(projected,values);
            return projected;
          };
          double quadrature_error=0.;
          for (const auto *field:std::array<const LinearAlgebra::BlockVector *,2>{{&base,&this->get_solution()}})
            {
              const auto reference=reference_normalization(*field);
              const auto native=Access::fe_normalization_only(model,*field);
              for (unsigned int i=0;i<native.size();++i)
                quadrature_error=std::max(quadrature_error,std::abs(reference[i]-native[i])/std::max(1.,std::abs(reference[i])));
            }
          this->get_pcout()<<"   DG normalization Gauss12/Gauss32 relative difference="<<quadrature_error<<std::endl;
          AssertThrow(quadrature_error<1e-10,ExcMessage("DG normal-ray quadrature reference check failed."));
          AssertThrow(normalization.admissible,ExcMessage("DG normalization diagnostic state is inadmissible."));
          std::vector<std::vector<double>> rates;
          for (unsigned int f=0;f<this->get_reconstructed_fault_manager().get_faults().size();++f)
            rates.push_back(this->get_reconstructed_fault_manager().get_slip_rate(f));
          const auto endpoint=model.make_fe_endpoint_trial(this->get_solution(),rates);
          const auto mechanical=model.evaluate_fe_endpoint_energy(base,endpoint);
          const auto mechanical_action=model.evaluate_fe_endpoint_action(base,endpoint,direction);
          double mechanical_derivative=0.;
          for (unsigned int i=0;i<n;++i) mechanical_derivative+=mechanical.gradient[i]*direction[i];
          std::ofstream report;
          if (Utilities::MPI::this_mpi_process(comm)==0)
            {
              report.open(this->get_output_directory()+"dg-phase-derivatives-"+std::to_string(this->get_timestep_number())+".csv");
              report<<std::setprecision(17)<<"epsilon,fracture_energy,fracture_tangent,normalization,mechanical_energy,mechanical_tangent\n";
            }
          double best_fracture=1.,best_tangent=1.,best_normalization=1.,best_mechanical=1.,best_mechanical_tangent=1.;
          for (const double epsilon:{1e-4,1e-5,1e-6})
            {
              typename PhaseFieldHandler<dim>::CellFractureEnergy f[2];
              typename Model::FEPhaseEnergy m[2];
              std::vector<double> norm[2];
              for (unsigned int side=0;side<2;++side)
                {
                  owned=base;
                  for (const auto i:sets.system_partitioning[b]) owned.block(b)[i]=base.block(b)[i]+(side==0?epsilon:-epsilon)*direction[i];
                  owned.compress(VectorOperation::insert);
                  LinearAlgebra::BlockVector trial(base);trial=owned;
                  f[side]=handler.evaluate_cell_fracture(trial,false);
                  norm[side]=Access::fe_normalization_only(model,trial);
                  m[side]=model.evaluate_fe_endpoint_energy(trial,endpoint);
                }
              const double energy_error=std::abs((f[0].energy-f[1].energy)/(2*epsilon)-derivative)/std::max(1.,std::abs(derivative));
              double tangent_error=0.,action_scale=1.,normalization_error=0.,normalization_scale=1.;
              double mechanics_error=std::abs((m[0].energy-m[1].energy)/(2*epsilon)-mechanical_derivative)/std::max(1.,std::abs(mechanical_derivative));
              double mechanics_tangent_error=0.,mechanics_scale=1.;
              for (unsigned int i=0;i<n;++i)
                {
                  tangent_error=std::max(tangent_error,std::abs((f[0].gradient[i]-f[1].gradient[i])/(2*epsilon)-action[i]));
                  action_scale=std::max(action_scale,std::abs(action[i]));
                  mechanics_tangent_error=std::max(mechanics_tangent_error,std::abs((m[0].gradient[i]-m[1].gradient[i])/(2*epsilon)-mechanical_action.gradient[i]));
                  mechanics_scale=std::max(mechanics_scale,std::abs(mechanical_action.gradient[i]));
                }
              for (unsigned int i=0;i<norm[0].size();++i)
                {
                  normalization_error=std::max(normalization_error,std::abs((norm[0][i]-norm[1][i])/(2*epsilon)-normalization.action[i]));
                  normalization_scale=std::max(normalization_scale,std::abs(normalization.action[i]));
                }
              tangent_error/=action_scale;normalization_error/=normalization_scale;mechanics_tangent_error/=mechanics_scale;
              best_fracture=std::min(best_fracture,energy_error);best_tangent=std::min(best_tangent,tangent_error);
              best_normalization=std::min(best_normalization,normalization_error);best_mechanical=std::min(best_mechanical,mechanics_error);
              best_mechanical_tangent=std::min(best_mechanical_tangent,mechanics_tangent_error);
              if (report) report<<epsilon<<','<<energy_error<<','<<tangent_error<<','<<normalization_error<<','<<mechanics_error<<','<<mechanics_tangent_error<<'\n';
            }
          AssertThrow(best_fracture<1e-7 && best_tangent<1e-8 && best_normalization<1e-7
                      && best_mechanical<1e-6 && best_mechanical_tangent<1e-6,
                      ExcMessage("Native DG phase derivative check failed; inspect retained CSV."));
          // A face-coincident sample chooses one physical trace before h(phi).
          owned=this->get_solution();
          std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell());
          for (const auto &cell:this->get_dof_handler().active_cell_iterators()) if (cell->is_locally_owned())
            {
              cell->get_dof_indices(ids);
              for (unsigned int i=0;i<ids.size();++i)
                if (this->get_fe().system_to_component_index(i).first==c)
                  owned[ids[i]]=.2+.001*cell->vertex(0)[0]+.002*cell->vertex(0)[1];
            }
          owned.compress(VectorOperation::insert);base=owned;
          Point<dim> point;point[0]=point[1]=.5;
          Utilities::MPI::RemotePointEvaluation<dim> lookup;
          lookup.reinit(handler.get_grid_cache(),std::vector<Point<dim>>{point});
          const double trace=handler.phase_point_values(lookup,base)[0];
          AssertThrow(std::abs(trace-.2015)<1e-14,ExcMessage("DG face-coincident trace convention failed."));
          this->get_pcout()<<"   DG native phase derivatives and face trace passed; phase DoFs="<<n<<std::endl;
          return {"DG phase", "SIPG, normalization, mechanics and explicit trace checked"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(DGPhaseChecks,"DG native phase checks","Noncommitting DG phase derivative and trace diagnostics.")
  }
}
