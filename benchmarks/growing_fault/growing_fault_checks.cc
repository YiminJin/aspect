/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <aspect/simulator/solver/midpoint_system.h>
#include <aspect/phase_field/bernstein_bounds.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>

namespace aspect::Postprocess
{
  namespace
  {
    using namespace GrowingFault;
    void check_operator(const Material<2> &material, std::ostream &out)
    {
      MidpointSystem op(material,material.get_solution());
      auto x=op.initial;
      std::fill(op.eligible.begin(),op.eligible.end(),1u);
      for (unsigned int i=0; i<op.nv; ++i) x[i]+=2.e-5*std::sin(1.+i);
      for (unsigned int i=op.phase_begin; i<x.size(); ++i) x[i]=.22+.02*std::sin(i+.7);
      const auto e=op.assemble_stage(x,true,true);
      for (unsigned int kind=0; kind<3; ++kind)
        {
          std::vector<double> d(op.n,0.), action(op.n,0.), force(op.angles.size(),0.);
          for (unsigned int i=0; i<op.n; ++i)
            if (!op.fixed.count(i))
              {
                if (i<op.nv && kind!=1) d[i]=.0001*std::cos(i+.3);
                if (i>=op.phase_begin && kind!=0) d[i]=.03*std::sin(i+.4);
                if (i>=op.nv && i<op.phase_begin && kind==2) d[i]=.0002*std::cos(i);
              }
          for (const auto &[ij,v]:e.jacobian) if (ij.first>=op.n) force[ij.first-op.n]-=v*d[ij.second];
          Utilities::MPI::sum(force,op.comm,force);
          for (const auto &[ij,v]:e.jacobian) if (ij.first<op.n)
              action[ij.first]+=v*(ij.second>=op.n?force[ij.second-op.n]:d[ij.second]);
          Utilities::MPI::sum(action,op.comm,action);
          for (const double eps:
          {
            .001,.0003,.0001
          })
          {
            auto xp=x,xm=x;
            for (unsigned int i=0; i<x.size(); ++i)
              {
                xp[i]+=eps*d[i];
                xm[i]-=eps*d[i];
              }
            const auto plus=op.assemble_stage(xp,false,false),minus=op.assemble_stage(xm,false,false);
            for (unsigned int block=0; block<2; ++block)
              {
                double error=0.,scale=0.;
                for (unsigned int i=(block?op.phase_begin:0); i<(block?op.n:op.phase_begin); ++i)
                  if (!op.fixed.count(i))
                    {
                      const double fd=(plus.residual[i]-minus.residual[i])/(2.*eps);
                      error+=std::pow(fd-action[i],2);
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
      const auto &system=material.get_fe();
      const auto &intro=material.introspection();
      const auto &solution=material.get_solution();
      for (const auto &cell:material.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto &rule=material.sections().cells.at(cell->id());
            FEValues<2> values(material.get_mapping(),system,rule.quadrature,update_values|update_JxW_values);
            values.reinit(cell);
            std::vector<types::global_dof_index> ids(system.n_dofs_per_cell());
            cell->get_dof_indices(ids);
            for (unsigned int i=0; i<ids.size(); ++i)
              {
                const auto [component,j]=system.system_to_component_index(i);
                if (component==intro.component_indices.velocities[0] && solution.block(0).locally_owned_elements().is_element(ids[i]))
                  {
                    const auto &base=system.base_element(system.component_to_base_index(component).first);
                    test[ids[i]]=material.get_mapping().transform_unit_to_real_cell(cell,base.unit_support_point(j))[0];
                  }
              }
            for (unsigned int q=0; q<rule.quadrature.size(); ++q)
              {
                const auto &one=e.candidate.cells.at(cell->id())[q];
                const auto &old=material.history().cells.at(cell->id())[q];
                reference+=values.JxW(q)*.5*(old.stress[0][0]+one.stress[0][0]);
                for (unsigned int j=0; j<ids.size(); ++j)
                  if (ids[j]>=op.nv && ids[j]<op.phase_begin)
                    pressure+=values.JxW(q)*op.G*x[ids[j]]*values[intro.extractors.pressure].value(j,q);
              }
          }
      double work=0.;
      for (const auto &[i,v]:test) work+=op.G*v*e.residual[i];
      work=Utilities::MPI::sum(work+pressure,op.comm);
      reference=Utilities::MPI::sum(reference,op.comm);
      out<<"deviatoric work "<<work<<" reference "<<reference<<" opposite sign defect "<<std::abs(work+reference)<<'\n';
      AssertThrow(std::abs(reference)>1.e-4 && std::abs(work-reference)<1.e-9,ExcMessage("Stress virtual work failed."));
    }
  }

  template <int dim>
  class GrowingFaultSamples : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
      void initialize() override
      {
        if constexpr(dim==2)
          this->get_signals().pre_nonlinear_solver.connect([this](const SimulatorAccess<dim> &)
          {
            if (this->get_timestep_number()!=1) return;
            const auto &m=dynamic_cast<const GrowingFault::Material<2>&>(this->get_material_model());
            const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            std::ofstream out(this->get_output_directory()+"operator-r"+std::to_string(rank)+".txt");
            out<<std::setprecision(17);
            check_operator(m,out);
          });
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        if constexpr(dim==2)
          {
            const auto &m=dynamic_cast<const GrowingFault::Material<2>&>(this->get_material_model());
            const auto &history=m.history();
            const auto comm=this->get_mpi_communicator();
            const auto rank=Utilities::MPI::this_mpi_process(comm);
            const auto path=this->get_output_directory();
            const auto tag=Utilities::int_to_string(history.step,4);
            const auto &fe=this->get_fe();
            const auto &solution=this->get_solution();
            const auto c=this->introspection().component_indices.compositional_fields[m.phase_field()];
            std::ofstream points(path+"points-"+tag+"-r"+std::to_string(rank)+".csv");
            std::ofstream coefficients(path+"fields-"+tag+"-r"+std::to_string(rank)+".csv");
            points<<std::setprecision(17)<<"cell,q,x,y,weight,section,phi,tauxx,tauyy,tauzz,tauxy,Axx,Ayy,Azz,Axy,signed_shear,absolute_shear\n";
            coefficients<<std::setprecision(17)<<"cell,component,local,value\n";
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  const auto &rule=m.sections().cells.at(cell->id());
                  FEValues<2> values(this->get_mapping(),fe,rule.quadrature,update_values|update_JxW_values|update_quadrature_points);
                  values.reinit(cell);
                  std::vector<double> phase(rule.quadrature.size());
                  values[FEValuesExtractors::Scalar(c)].get_function_values(solution,phase);
                  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
                  cell->get_dof_indices(ids);
                  PhaseField::BernsteinPatch<2> patch;
                  for (unsigned int i=0; i<ids.size(); ++i)
                    {
                      const auto [component,j]=fe.system_to_component_index(i);
                      coefficients<<cell->id()<<','<<component<<','<<j<<','<<solution[ids[i]]<<'\n';
                      if (component==c) patch[j]=solution[ids[i]];
                    }
                  const auto range=PhaseField::enclosure<2>(patch);
                  AssertThrow(range.first>=-2.e-14 && range.second<=1.+2.e-14,ExcMessage("Endpoint polynomial violates physical bounds."));
                  for (unsigned int q=0; q<phase.size(); ++q)
                    {
                      const auto &v=history.cells.at(cell->id())[q];
                      const auto &p=values.quadrature_point(q);
                      points<<cell->id()<<','<<q<<','<<p[0]<<','<<p[1]<<','<<values.JxW(q)<<','<<rule.section[q]<<','<<phase[q];
                      for (const auto *tensor:
                      {
                        &v.stress,&v.cohesive_strain
                      })
                      points<<','<<(*tensor)[0][0]<<','<<(*tensor)[1][1]<<','<<(*tensor)[2][2]<<','<<(*tensor)[0][1];
                      points<<','<<v.signed_shear<<','<<v.absolute_shear<<'\n';
                    }
                }
            dealii::Utilities::System::MemoryStats memory;
            dealii::Utilities::System::get_memory_stats(memory);
            std::ofstream usage(path+"memory-"+tag+"-r"+std::to_string(rank)+".txt");
            usage<<memory.VmHWM<<" KiB peak RSS, "<<sizeof(GrowingFault::PointHistory)<<" bytes/native point history\n";
            if (rank==0)
              {
                std::ofstream strips(path+"sections-"+tag+".csv");
                strips<<std::setprecision(17)<<"section,xi,ds,beta,formed,age,slip,V,force\n";
                for (unsigned int k=0; k<history.sections.size(); ++k)
                  {
                    const auto &s=history.sections[k];
                    strips<<k<<','<<.5*(m.sections().edges[k]+m.sections().edges[k+1])<<','<<m.sections().length(k,s.angle)<<','<<s.angle
                          <<','<<s.formed<<','<<s.age<<','<<s.slip<<','<<s.rate<<','<<s.resistance<<'\n';
                  }
                // The initial seed has no completed step-work ledger.
                if (history.step>0)
                  {
                    std::ofstream book(path+"balance-"+tag+".csv");
                    const auto &b=m.balance();
                    book<<std::setprecision(17)<<"step,time,bulk,cohesive,fracture,Wext,Db,Dc,Df,phase_remainder,phase_work,energy_change,mechanical_residual,phase_KKT,adjoint_error,iterations\n"
                        <<history.step<<','<<history.time<<','<<b.bulk<<','<<b.cohesive<<','<<b.fracture<<','<<b.wall<<','<<b.bulk_loss<<','<<b.cohesive_loss
                        <<','<<b.friction_work<<','<<b.compliance_remainder<<','<<b.phase_work<<','<<b.storage_change<<','<<b.mechanical<<','<<b.kkt<<','<<b.adjoint_defect<<','<<b.iterations<<'\n';
                  }
              }

          }
        return {"Growing fault","native section and point history"};
      }
  };
  ASPECT_REGISTER_POSTPROCESSOR(GrowingFaultSamples,"growing fault samples","Observe the published midpoint bulk and strip histories.")
}
