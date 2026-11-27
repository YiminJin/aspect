/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/growing_fault_history.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/material_model/phase_field_fault/strip_material_history.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/simulator_signals.h>
#include <aspect/simulator/solver/midpoint_system.h>
#include <aspect/phase_field/bernstein_bounds.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_tools.h>
#include <fstream>

namespace aspect::Postprocess
{
  namespace
  {
    using namespace GrowingFault;
    void check_midpoint_tangent_and_work(const SimulatorAccess<2>      &sim,
                                         const StripMaterialHistory<2> &material,
                                         std::ostream                  &out)
    {
      MidpointSystem op(sim,material,sim.get_solution());
      if (!op.friction_pressures.stations.empty())
        {
          op.freeze_direction_neighborhood();
          // A two-point candidate with unequal stresses, support and phase.
          // The independent checker uses the pointwise score and its own RSF root.
          const auto a = onset_moments(.03,.8,.4,-70.,210.);
          const auto b = onset_moments(.02,.6,.7,155.,140.);
          std::array<double,9> moments;
          for (unsigned int j=0; j<9; ++j) moments[j]=a[j]+b[j];
          const double centered[3] = {a[8] *b[8]/moments[8]*225.*225.,
                                      -a[8] *b[8]/moments[8]*225.*70.,
                                      a[8] *b[8]/moments[8]*70.*70.
                                     };
          for (unsigned int k=0; k<op.eligible.size(); ++k)
            if (op.eligible[k])
              for (const double angle :
              {
                0.,.23
              })
          {
            const auto result=op.score_plane(k,moments.data(),centered,angle);
            out<<"pressure-score "<<k<<' '<<angle<<' '<<result.resistance<<' '<<result.score<<'\n';
          }
        }
      auto stage_fields=op.initial;
      std::fill(op.eligible.begin(),op.eligible.end(),1u);
      for (unsigned int i=0; i<op.nv; ++i) stage_fields[i]+=2.e-5*std::sin(1.+i);
      for (unsigned int i=op.phase_begin; i<stage_fields.size(); ++i) stage_fields[i]=.22+.02*std::sin(i+.7);
      const auto base=op.assemble_stage(stage_fields,true,true);
      for (unsigned int kind=0; kind<3; ++kind)
        {
          std::vector<double> direction(op.n,0.), action(op.n,0.), strip_force_change(op.angles.size(),0.);
          for (unsigned int i=0; i<op.n; ++i)
            if (!op.fixed.count(i))
              {
                if (i<op.nv && kind!=1) direction[i]=.0001*std::cos(i+.3);
                if (i>=op.phase_begin && kind!=0) direction[i]=.03*std::sin(i+.4);
                if (i>=op.nv && i<op.phase_begin && kind==2) direction[i]=.0002*std::cos(i);
              }
          // Eliminate the strip-force rows before pairing the condensed tangent
          // with the velocity-only, phase-only, or combined direction.
          for (const auto &[ij, value] : base.jacobian)
            if (ij.first >= op.n)
              strip_force_change[ij.first-op.n] -= value * direction[ij.second];
          Utilities::MPI::sum(strip_force_change,op.comm,strip_force_change);
          for (const auto &[ij, value] : base.jacobian)
            if (ij.first < op.n)
              action[ij.first] += value * (ij.second >= op.n ? strip_force_change[ij.second-op.n] : direction[ij.second]);
          Utilities::MPI::sum(action,op.comm,action);
          for (const double eps:
          {
            .001,.0003,.0001
          })
          {
            auto plus_fields=stage_fields,minus_fields=stage_fields;
            for (unsigned int i=0; i<stage_fields.size(); ++i)
              {
                plus_fields[i]+=eps*direction[i];
                minus_fields[i]-=eps*direction[i];
              }
            const auto plus=op.assemble_stage(plus_fields,false,false),minus=op.assemble_stage(minus_fields,false,false);
            for (unsigned int block=0; block<2; ++block)
              {
                double error=0.,scale=0.;
                for (unsigned int i=(block?op.phase_begin:0); i<(block?op.n:op.phase_begin); ++i)
                  if (!op.fixed.count(i))
                    {
                      const double central_difference=(plus.residual[i]-minus.residual[i])/(2.*eps);
                      error+=std::pow(central_difference-action[i],2);
                      scale+=action[i]*action[i];
                    }
                const double relative=std::sqrt(error/scale);
                out<<"direction "<<kind<<" block "<<block<<" epsilon "<<eps<<" relative "<<relative<<" norm "<<std::sqrt(scale)<<'\n';
                if (eps==.0001) AssertThrow(std::isfinite(relative) && relative<2.e-5,ExcMessage("Coupled tangent failed centered difference."));
              }
          }
        }
      // Undo the row normalization and test against v=(x,0), whose
      // gradient is constant. Include the pressure multiplier on this test.
      std::map<types::global_dof_index,double> test;
      double reference=0.,pressure=0.;
      const auto &system=sim.get_fe();
      const auto &intro=sim.introspection();
      const auto &solution=sim.get_solution();
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto &rule=material.sections().cells.at(cell->id());
            FEValues<2> values(sim.get_mapping(),system,rule.quadrature,update_values|update_JxW_values);
            values.reinit(cell);
            std::vector<types::global_dof_index> ids(system.n_dofs_per_cell());
            cell->get_dof_indices(ids);
            for (unsigned int i=0; i<ids.size(); ++i)
              {
                const auto [component,j]=system.system_to_component_index(i);
                if (component==intro.component_indices.velocities[0] && solution.block(0).locally_owned_elements().is_element(ids[i]))
                  {
                    const auto &base=system.base_element(system.component_to_base_index(component).first);
                    test[ids[i]]=sim.get_mapping().transform_unit_to_real_cell(cell,base.unit_support_point(j))[0];
                  }
              }
            for (unsigned int q=0; q<rule.quadrature.size(); ++q)
              {
                const auto &endpoint=base.candidate.cells.at(cell->id())[q];
                const auto &old=material.history().cells.at(cell->id())[q];
                reference+=values.JxW(q)*.5*(old.stress[0][0]+endpoint.stress[0][0]);
                for (unsigned int j=0; j<ids.size(); ++j)
                  if (ids[j]>=op.nv && ids[j]<op.phase_begin)
                    pressure+=values.JxW(q)*op.G*stage_fields[ids[j]]*values[intro.extractors.pressure].value(j,q);
              }
          }
      double work=0.;
      for (const auto &[i,v]:test) work+=op.G*v*base.residual[i];
      work=Utilities::MPI::sum(work+pressure,op.comm);
      reference=Utilities::MPI::sum(reference,op.comm);
      out<<"deviatoric work "<<work<<" reference "<<reference<<" opposite sign defect "<<std::abs(work+reference)<<'\n';
      AssertThrow(std::abs(reference)>1.e-4 && std::abs(work-reference)<1.e-9,ExcMessage("Stress virtual work failed."));
    }
  }

  template <int dim>
  class GrowingFaultSamples : public GrowingFaultHistory<dim>
  {
    public:
      std::pair<std::string,std::string> execute(TableHandler &statistics) override
      {
        if constexpr(dim==2)
          {
            const auto &m=dynamic_cast<const MaterialModel::PhaseFieldFault<2> &>(this->get_material_model()).strip_history();
            const auto c=this->introspection().component_indices.compositional_fields[m.phase_field()];
            const auto &fe=this->get_fe();
            std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
            unsigned int invalid=0;
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(ids);
                  PhaseField::BernsteinPatch<2> patch;
                  for (unsigned int j=0; j<patch.size(); ++j)
                    patch[j]=this->get_solution()[ids[fe.component_to_system_index(c,j)]];
                  const auto range=PhaseField::enclosure<2>(patch);
                  invalid |= !(range.first>=-2.e-14 && range.second<=1.+2.e-14);
                }
            AssertThrow(!Utilities::MPI::max(invalid,this->get_mpi_communicator()),ExcMessage("Endpoint polynomial violates physical bounds."));
          }
        return GrowingFaultHistory<dim>::execute(statistics);
      }

      void initialize() override
      {
        GrowingFaultHistory<dim>::initialize();
        if constexpr(dim==2)
          this->get_signals().pre_nonlinear_solver.connect([this](const SimulatorAccess<dim> &)
          {
            if (this->get_timestep_number()!=1) return;
            const auto &m=dynamic_cast<const MaterialModel::PhaseFieldFault<2> &>(this->get_material_model()).strip_history();
            const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            std::ofstream out(this->get_output_directory()+"operator-r"+std::to_string(rank)+".txt");
            out<<std::setprecision(17);
            check_midpoint_tangent_and_work(*this,m,out);
          });
      }
  };
  ASPECT_REGISTER_POSTPROCESSOR(GrowingFaultSamples,"growing fault samples","Observe the published midpoint bulk and strip histories.")

  /** Observe the accepted checkpoint on both sides of a restart, and the
   * ordinary solver-completion notification before the repeat decision. */
  template <int dim>
  class MidpointPublication : public GrowingFaultSamples<dim>
  {
    public:
      void initialize() override
      {
        this->get_signals().pre_nonlinear_solver.connect([this](const SimulatorAccess<dim> &)
        {
          if (this->get_timestep_number()==3) record_parent("pre-solve");
        });
        this->get_signals().post_nonlinear_solver.connect([this](const SolverControl &control)
        {
          const auto &m=dynamic_cast<const MaterialModel::PhaseFieldFault<dim> &>(this->get_material_model()).strip_history();
          if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
            std::ofstream(this->get_output_directory()+"completions",std::ios::app)
                <<this->get_timestep_number()<<' '<<m.history().step<<' '<<control.last_step()<<' '
                <<(control.last_check()==SolverControl::success)<<'\n';
        });
      }

      std::pair<std::string,std::string> execute(TableHandler &statistics) override
      {
        const auto &v=this->get_solution();
        const auto &intro=this->introspection();
        const auto comm=this->get_mpi_communicator();
        const auto &pressure=v.block(intro.block_indices.pressure);
        const double datum=Utilities::MPI::sum(pressure.locally_owned_elements().is_element(0)?pressure[0]:0.,comm);
        double wall_speed=0.;
        for (const auto i:DoFTools::extract_boundary_dofs(this->get_dof_handler(),intro.component_masks.velocities))
          if (v.locally_owned_elements().is_element(i)) wall_speed=std::max(wall_speed,std::abs(v[i]));
        wall_speed=Utilities::MPI::max(wall_speed,comm);
        if (Utilities::MPI::this_mpi_process(comm)==0)
          std::ofstream(this->get_output_directory()+"published-constraints",std::ios::app)
              <<std::setprecision(17)<<this->get_timestep_number()<<' '<<datum<<' '<<wall_speed<<'\n';
        if (this->get_timestep_number()==2) record_parent("published");
        return GrowingFaultSamples<dim>::execute(statistics);
      }
    private:
      void record_parent(const std::string &when) const
      {
        std::map<std::string,std::string> record;
        this->get_material_model().save(record);
        if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
          for (const auto &[name,data]:record)
            std::ofstream(this->get_output_directory()+when+"-"+name,std::ios::binary).write(data.data(),data.size());
      }
  };
  ASPECT_REGISTER_POSTPROCESSOR(MidpointPublication,"midpoint publication","Observe accepted history and solver completion.")

}

#include <aspect/time_stepping/interface.h>

namespace aspect::TimeStepping
{
  /** Reassemble the returned midpoint against the still-accepted parent at
   * the ordinary repeat decision. Check the Schur complement applied by the
   * direct solver at the final plane map on an admissible mixed direction. */
  template <int dim>
  class ReturnedMidpoint : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      double execute() override
      {
        return std::numeric_limits<double>::max();
      }
      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
      {
        if constexpr(dim==2)
          if (this->get_timestep_number()==3)
            {
              const auto &m=dynamic_cast<const MaterialModel::PhaseFieldFault<2> &>(this->get_material_model()).strip_history();
              const auto &v=this->get_solution();
              GrowingFault::MidpointSystem op(*this,m,v);
              op.freeze_direction_neighborhood();
              auto x=op.initial;
              std::fill(x.begin(),x.end(),0.);
              for (const unsigned int b:
              {
                0u,1u,op.phase_block
              })
              for (const auto j:v.block(b).locally_owned_elements())
                if (b==0) x[j]=op.h*v.block(b)[j];
                else if (b==1) x[op.nv+j]=v.block(b)[j]/op.G;
                else x[op.phase_begin+j]=.5*(v.block(b)[j]+op.lower[op.phase_begin+j]);
              Utilities::MPI::sum(x,op.comm,x);
              AssertThrow(!op.select_prospective_planes(x),ExcMessage("Returned endpoint would change the plane map."));
              const auto e=op.assemble_stage(x,true,false);
              if (m.friction().name()=="linear dashpot" && Utilities::MPI::this_mpi_process(op.comm)==0)
                {
                  std::ofstream slopes(this->get_output_directory()+"returned-linear-slopes");
                  slopes<<std::setprecision(17);
                  for (const auto &root:e.roots) slopes<<root.slope<<'\n';
                }
              AssertThrow(e.ledger.mechanical<=this->get_parameters().nonlinear_tolerance && e.ledger.kkt<=1.e-7,
                          ExcMessage("Returned fields fail the physical convergence gate."));
              std::vector<double> d(op.n,0.),action(op.n,0.),force(op.angles.size(),0.);
              for (unsigned int i=0; i<op.n; ++i)
                if (i<op.phase_begin && !op.fixed.count(i)) d[i]=2.e-5*std::sin(i+.3);
                else if (i>=op.phase_begin && x[i]>op.lower[i]+1.e-7 && x[i]<op.upper[i]-1.e-7) d[i]=.001*std::cos(i);
              for (const auto &[ij,a]:e.jacobian) if (ij.first>=op.n) force[ij.first-op.n]-=a*d[ij.second];
              Utilities::MPI::sum(force,op.comm,force);
              for (const auto &[ij,a]:e.jacobian) if (ij.first<op.n) action[ij.first]+=a*(ij.second>=op.n?force[ij.second-op.n]:d[ij.second]);
              Utilities::MPI::sum(action,op.comm,action);
              auto xp=x,xm=x;
              constexpr double eps=5.e-5;
              for (unsigned int i=0; i<op.n; ++i)
                {
                  xp[i]+=eps*d[i];
                  xm[i]-=eps*d[i];
                }
              const auto plus=op.assemble_stage(xp,false,false),minus=op.assemble_stage(xm,false,false);
              double defect=0.,norm=0.;
              for (unsigned int i=0; i<op.n; ++i)
                if (!op.fixed.count(i))
                  {
                    defect+=std::pow((plus.residual[i]-minus.residual[i])/(2.*eps)-action[i],2);
                    norm+=action[i]*action[i];
                  }
              const double relative=std::sqrt(defect/norm);
              AssertThrow(std::isfinite(relative) && relative<2.e-5,ExcMessage("Returned tangent fails the frozen action budget."));
              this->get_pcout()<<"   Returned midpoint tangent relative error "<<std::setprecision(12)<<relative<<std::endl;
            }
        return {Reaction::advance,std::numeric_limits<double>::max()};
      }
  };
  ASPECT_REGISTER_TIME_STEPPING_MODEL(ReturnedMidpoint,"check returned midpoint","Check the tangent at the returned growing fault solution before publication.")

  /** Reject a solved full step, then reject the half step once at unchanged
   * duration. Compare every accepted native value and locally owned FE value
   * across the two rollbacks before permitting publication. */
  template <int dim>
  class RetryGrowingFault : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      double execute() override
      {
        return this->get_timestep_number()==1?.01:.005;
      }
      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
      {
        if constexpr(dim==2)
          if (this->get_timestep_number()==2)
            {
              const auto &m=dynamic_cast<const MaterialModel::PhaseFieldFault<2> &>(this->get_material_model()).strip_history();
              AssertThrow(m.history().step==1,ExcMessage("Premature section publication."));
              std::map<std::string,std::string> state;
              this->get_material_model().save(state);
              const auto coefficients=[](const LinearAlgebra::BlockVector &v)
              {
                std::vector<double> data;
                for (unsigned int b=0; b<v.n_blocks(); ++b)
                  for (const auto j:v.block(b).locally_owned_elements()) data.push_back(v.block(b)[j]);
                return data;
              };
              const auto parent=coefficients(this->get_old_solution());
              if (attempt==0)
                {
                  accepted=state;
                  old=parent;
                }
              AssertThrow(accepted==state && parent==old,ExcMessage("Rejected attempt changed accepted fields or native history."));
              ++attempt;
              if (attempt==1) return {Reaction::repeat_step,.005};
              if (attempt==2)
                {
                  half=coefficients(this->get_solution());
                  return {Reaction::repeat_step,.005};
                }
              AssertThrow(attempt==3 && half==coefficients(this->get_solution()),ExcMessage("Repeated half step differs from identical trial."));
            }
        return {Reaction::advance,std::numeric_limits<double>::max()};
      }
    private:
      unsigned int attempt=0;
      std::map<std::string,std::string> accepted;
      std::vector<double> old,half;
  };
  ASPECT_REGISTER_TIME_STEPPING_MODEL(RetryGrowingFault,"retry growing fault","Exercise two solved midpoint rollbacks.")
}

namespace aspect::MaterialModel
{
  // Read parameter files produced by the original state-free benchmark.
  // These aliases only translate its old selector into the registered law.
  template <int dim>
  class LegacyDashpotSelector : public PhaseFieldFault<dim>
  {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        // The SDK plugin registers after the built-in materials. Each handler
        // owns these actions through parsing, including explicit RSF defaults.
        // Both legacy names register this same type on the parameter handler.
        if (const auto entry=selectors.find(&prm); entry!=selectors.end() && !entry->second.expired())
          return;
        for (auto it=selectors.begin(); it!=selectors.end();)
          if (it->second.expired())
            it=selectors.erase(it);
          else
            ++it;
        auto supplied=std::make_shared<std::set<std::string>>();
        selectors[&prm]=supplied;
        const std::string spellings[] = {"Phase field fault","Growing fault"};
        prm.enter_subsection("Material model");
        for (const auto &spelling : spellings)
          {
            prm.enter_subsection(spelling);
            prm.add_action("Resistance law",[supplied,spelling](const std::string &)
            {
              supplied->insert(spelling);
            },false);
            prm.leave_subsection();
          }
        prm.leave_subsection();
      }

      void parse_parameters(ParameterHandler &prm) override
      {
        const bool growing = prm.get({"Material model"}, "Model name") == "linear strip resistance";
        for (const auto &spelling : *selectors.at(&prm).lock())
          AssertThrow(prm.get({"Material model",spelling},"Resistance law")=="linear dashpot",
                      ExcMessage("Legacy dashpot selector conflicts with Material model / "+spelling+" / Resistance law."));
        prm.enter_subsection("Material model");
        prm.enter_subsection(growing ? "Growing fault" : "Phase field fault");
        prm.set("Resistance law","linear dashpot");
        prm.leave_subsection();
        prm.leave_subsection();
        using StripInputs = typename PhaseFieldFault<dim>::StripInputs;
        this->parse_fault_parameters(prm, growing ? StripInputs::historical : StripInputs::trace_only);
        AssertThrow(this->uses_growing_strips() == growing,
                    ExcMessage(growing ? "The linear strip resistance selector requires midpoint strip mechanics."
                               : "The linear trace resistance selector requires mature trace mechanics."));
      }

    private:
      inline static std::map<const ParameterHandler *,std::weak_ptr<std::set<std::string>>> selectors;
  };
  namespace
  {
    template <int dim>
    void register_legacy_selectors()
    {
      using aspect::internal::Plugins::RegisterHelper;
      RegisterHelper<Interface<dim>,LegacyDashpotSelector<dim>>(&register_material_model<dim>,"linear strip resistance","Legacy growing dashpot input.");
      RegisterHelper<Interface<dim>,LegacyDashpotSelector<dim>>(&register_material_model<dim>,"linear trace resistance","Legacy mature dashpot input.");
    }
    const auto legacy_registered=(register_legacy_selectors<2>(),register_legacy_selectors<3>(),true);
  }

}

namespace aspect::Postprocess
{
  template <int dim>
  class StatelessArchiveCheck : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void initialize() override
      {
        this->get_signals().pre_checkpoint_store_user_data.connect([this](auto &)
        {
          std::map<std::string,std::string> status;
          this->get_material_model().save(status);
          const auto &fault = dynamic_cast<const MaterialModel::PhaseFieldFault<dim> &>(this->get_material_model());
          if (fault.uses_growing_strips())
            {
              const auto entry=status.find("Native growing fault v2");
              if (entry==status.end()) return;
              std::istringstream bytes(entry->second);
              aspect::iarchive archive(bytes);
              std::vector<double> signature, edges, state;
              unsigned int order, step;
              double angle, time;
              std::string name;
              std::vector<GrowingFault::SectionHistory> strips;
              archive >> signature >> order >> angle >> edges >> step >> time >> name >> strips >> state;
              AssertThrow(name=="linear dashpot" && strips.size()+1==edges.size() && state.empty()
                          && fault.strip_history().friction().state().empty(), ExcMessage("State-free strip archive contains constitutive history."));
            }
          else
            {
              const auto entry=status.find("Material fault labels linear dashpot");
              if (entry==status.end()) return;
              std::istringstream bytes(entry->second);
              unsigned int count;
              bytes >> count;
              AssertThrow(count==fault.labels().size() && fault.label_ages().empty(),ExcInternalError());
              std::string row;
              std::getline(bytes,row);
              for (unsigned int i=0; i<count; ++i)
                {
                  std::getline(bytes,row);
                  std::istringstream fields(row);
                  std::vector<double> values {std::istream_iterator<double>(fields),std::istream_iterator<double>()};
                  // x,y,width,background shear,rate,slip,Maxwell memory.
                  AssertThrow(values.size()==7 && values[4]==fault.labels()[i].rate &&
                              values[6]==fault.label_memory()[i],ExcMessage("State-free vertex archive contains additional history."));
                }
            }
        });
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        return {};
      }
  };
  ASPECT_REGISTER_POSTPROCESSOR(StatelessArchiveCheck,"stateless fault archive check","Check the wire format of the two state-free fault archives.")
}
