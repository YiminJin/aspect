/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/simulator.h>
#include <aspect/postprocess/interface.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/material_model/phase_field_fault/dg_history_internal.h>
#include <aspect/material_model/phase_field_fault/fe_strain_internal.h>
#include "../../../tests/phase_field_fault_test_access.h"
#include "transport_method_fields.h"
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/fe/fe_values.h>
#include <chrono>
#include <fstream>
#include <iomanip>
namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class MomentCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Model=MaterialModel::PhaseFieldFault<dim>;
        using BV=LinearAlgebra::BlockVector;
        using History=typename Model::DGHistory;
        using TensorField=SymmetricTensor<2,dim>;
        using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        std::ofstream output(const std::string &name) const
        {
          std::ofstream out(this->get_output_directory()+name);
          out<<std::setprecision(17);
          return out;
        }
        BV velocity(const bool source) const
        {
          struct Field : Function<dim>
          {
            Field(const unsigned int n,const bool s):Function<dim>(n),source(s) {}
            bool source;
            double value(const Point<dim> &p,const unsigned int c) const override
            {
              if (c>=dim) return 0.;
              return source?PF4Methods::velocity_source(p[0],p[1])[c]:(c==0?.4:0.);
            }
          } field(this->get_fe().n_components(),source);
          BV v(this->get_solution());
          VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),field,v,this->introspection().component_masks.velocities);
          return v;
        }
        void coefficients(const std::string &label,const History &h) const
        {
          auto out=output(label+".csv");
          out<<"i,xx,yy,xy\n";
          for (unsigned int i=0; i<h.space->dofs.n_dofs(); ++i)
            out<<i<<','<<h.trial_stress[0][i]<<','<<h.trial_stress[1][i]<<','<<h.trial_stress[2][i]<<'\n';
        }
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &sim=const_cast<Simulator<dim>&>(this->get_simulator());
          auto &model=Plugins::get_plugin_as_type<Model>(*sim.material_model);
          AssertThrow(dim==2 && this->get_timestep_number()==0,ExcMessage("Fresh native-history prescribed transport fixture required."));
          model.prepare_reconstructed_fault_mechanical_solve();
          const auto anchor=Access::dg_history_values(model);
          const BV phase_anchor(model.accepted_fe_strain_state());
          const auto Z_anchor=model.fe_strain_state->Z,I_anchor=model.fe_strain_state->I;
          const auto setup_begin=std::chrono::steady_clock::now();
          const auto op=model.make_dg_transport(velocity(false),4,false);
          const double setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-setup_begin).count();
          const auto &first=op->space->cells.begin()->second;
          const unsigned int nx=std::lround(1./(first->vertex(1)[0]-first->vertex(0)[0]));
          const auto maxwell=Model::compute_maxwell_coefficients(model.fe_strain_state->eta,model.fe_strain_state->G,.1);
          const auto source_begin=std::chrono::steady_clock::now();
          const auto initial=model.make_dg_history_trial(velocity(true),maxwell);
          const double source_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-source_begin).count();
          coefficients("initial-native_history",*initial);
          auto constants=output("constants.csv"),cells=output("cells.csv"),basis=output("basis.csv");
          constants<<"nx,ny,relaxation_rate,time,kappa\n"<<nx<<','<<2*nx<<",0,.2,"<<maxwell.eta_ve<<'\n';
          basis<<"i,r,s\n";
          const auto &support=op->space->element.get_unit_support_points();
          for (unsigned int i=0; i<9; ++i) basis<<i<<','<<support[i][0]<<','<<support[i][1]<<'\n';
          cells<<"cell,x0,y0,dx,dy,i0,i1,i2,i3,i4,i5,i6,i7,i8\n";
          unsigned int ci=0;
          std::map<CellId,unsigned int> cell_index;
          for (const auto &cell:op->cells)
            {
              const auto it=op->space->cells.at(cell.id);
              cell_index[cell.id]=ci;
              cells<<ci++<<','<<it->vertex(0)[0]<<','<<it->vertex(0)[1]<<','<<it->vertex(1)[0]-it->vertex(0)[0]<<','<<it->vertex(2)[1]-it->vertex(0)[1];
              for (const auto i:cell.indices) cells<<','<<i;
              cells<<'\n';
            }
          const auto exact=[nx](const Point<dim> &p,const double t)
          {
            TensorField out;
            const auto v=PF4Methods::native_history(p[0]-.4*t,p[1],nx);
            out[0][0]=v[0];
            out[1][1]=v[1];
            out[0][1]=v[2];
            return out;
          };
          const auto zero=[](const Point<dim> &,double)
          {
            return TensorField();
          };
          const QGauss<dim> check_rule(4);
          FEValues<dim> check(this->get_mapping(),op->space->element,check_rule,update_quadrature_points);
          double source_error=0.;
          for (const auto &entry:op->space->cells)
            {
              check.reinit(entry.second);
              const auto values=model.evaluate_dg_trial_tensor(*initial,entry.first,check_rule);
              for (unsigned int q=0; q<check_rule.size(); ++q)
                source_error=std::max(source_error,(values[q]-exact(check.quadrature_point(q),0.)).norm());
            }
          AssertThrow(source_error<1.e-9,ExcMessage("Independent Q2 strain reconstruction differs from native history."));
          // Export the actual native work-map QPs, associations, phase and I_h.
          // These rows apply L* to scratch transported T. Z=0 remains accepted.
          auto weights=output("weak-points.csv");
          weights<<"cell,x,y,segment,N0,N1,JxW,phi,I,chi,w,Sxx,Syy,Sxy,Nxx,Nyy,Nxy\n";
          auto &manager=this->get_reconstructed_fault_manager();
          manager.prepare_stokes_qp_projection_cache();
          const auto rule=model.mechanical_quadrature();
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_quadrature_points|update_JxW_values);
          std::vector<double> phase(rule.size());
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            {
              fe.reinit(cell);
              fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)]
              .get_function_values(phase_anchor,phase);
              const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),rule,fe.get_quadrature_points());
              for (unsigned int q=0; q<rule.size(); ++q)
                if (associations[q].active)
                  {
                    const auto &a=associations[q];
                    const double I=a.shape_0*I_anchor[a.segment_index]+a.shape_1*I_anchor[a.segment_index+1];
                    const double h=model.fe_strain_state->m*phase[q]*(1+phase[q])/std::pow(1-phase[q],2),chi=h/I;
                    const auto S=manager.get_shear_sense(a.fault_index)*symmetrize(outer_product(a.tangent,a.normal));
                    const auto N=symmetrize(outer_product(a.normal,a.normal));
                    weights<<cell_index.at(cell->id())<<','<<fe.quadrature_point(q)[0]<<','<<fe.quadrature_point(q)[1]<<','<<a.segment_index<<','<<a.shape_0<<','<<a.shape_1<<','
                           <<fe.JxW(q)<<','<<phase[q]<<','<<I<<','<<chi<<','<<fe.JxW(q)*chi<<','<<S[0][0]<<','<<S[1][1]<<','<<S[0][1]<<','<<N[0][0]<<','<<N[1][1]<<','<<N[0][1]<<'\n';
                  }
            }
          auto costs=output("costs.csv");
          costs<<"divisor,steps,dt,seconds,source_max_error,operator_seconds,source_seconds\n";
          const unsigned int base_steps=std::ceil(.2*op->inverse_advection_time/.025);
          for (const unsigned int divisor:
          {
            1u,2u,4u
          })
          {
            const unsigned int steps=base_steps*divisor;
            const double dt=.2/steps;
            auto state=initial;
            const auto begin=std::chrono::steady_clock::now();
            for (unsigned int step=0; step<steps; ++step)
              state=model.advance_dg_transport(*op,*state,exact,zero,step*dt,dt);
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            coefficients("field-native_history-"+std::to_string(divisor),*state);
            costs<<divisor<<','<<steps<<','<<dt<<','<<seconds<<','<<source_error<<','<<setup_seconds<<','<<source_seconds<<'\n';
            // Native reconstruction and work contraction, independent of the
            // Python coefficient evaluator used for all scalar methods.
            auto loads=output("native-loads-"+std::to_string(divisor)+".csv");
            std::vector<double> shear(Z_anchor.size()),normal(Z_anchor.size());
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              {
                fe.reinit(cell);
                fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(phase_anchor,phase);
                const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),rule,fe.get_quadrature_points());
                for (unsigned int q=0; q<rule.size(); ++q)
                  if (associations[q].active)
                    {
                      const auto &a=associations[q];
                      const double I=a.shape_0*I_anchor[a.segment_index]+a.shape_1*I_anchor[a.segment_index+1];
                      const double chi=model.fe_strain_state->m*phase[q]*(1+phase[q])/std::pow(1-phase[q],2)/I;
                      const auto stress=model.evaluate_dg_history_stress(*state,Z_anchor,I_anchor,cell->id(),fe.quadrature_point(q));
                      const auto S=manager.get_shear_sense(a.fault_index)*symmetrize(outer_product(a.tangent,a.normal));
                      const auto N=symmetrize(outer_product(a.normal,a.normal));
                      for (unsigned int i=0; i<2; ++i)
                        {
                          const double weight=fe.JxW(q)*chi*(i==0?a.shape_0:a.shape_1);
                          shear[a.segment_index+i]+=weight*(stress*S);
                          normal[a.segment_index+i]-=weight*(stress*N);
                        }
                    }
              }
            loads<<"node,shear,normal\n";
            for (unsigned int j=0; j<shear.size(); ++j) loads<<j<<','<<shear[j]<<','<<normal[j]<<'\n';
          }
          const bool unchanged=anchor==Access::dg_history_values(model) && Z_anchor==model.fe_strain_state->Z && I_anchor==model.fe_strain_state->I && phase_anchor==model.accepted_fe_strain_state();
          AssertThrow(unchanged,ExcMessage("Prescribed history comparison published state."));
          auto accepted=output("accepted.csv");
          accepted<<"accepted_unchanged\n"<<unchanged<<'\n';
          return {"PF4 native history methods","completed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(MomentCycle,"PF4 native history methods","Scratch transport of native independent history and its frozen fault work loads.")
  }
}
