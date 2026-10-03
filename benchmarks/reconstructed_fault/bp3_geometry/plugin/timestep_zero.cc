#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/simulator_signals.h>
#include <deal.II/dofs/dof_tools.h>
#include "../../../../tests/phase_field_fault_test_access.h"
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
  template <int dim>
  class TimestepZero : public Interface<dim>, public SimulatorAccess<dim>
  {
    using Surface=ReconstructedFaultSurfaceSystem<dim>;
    unsigned int observation=0,linear=0;
    std::vector<Point<dim>> points;
    std::vector<unsigned int> components;
    std::ofstream file(const std::string &name) const
    {std::ofstream o(this->get_output_directory()+name);o<<std::setprecision(17);return o;}
    void write_vector(const std::string &name,const LinearAlgebra::BlockVector &v) const
    {auto o=file(name);o<<"dof,value\n";for(unsigned int b=0;b<2;++b)for(unsigned int i=0;i<v.block(b).size();++i)o<<(b? v.block(0).size():0)+i<<','<<v.block(b)[i]<<'\n';}
    void observe(const ReconstructedFaultSurfaceResidual &r,const typename Surface::NormalTractionDiagnostic &d)
    {
      const auto &manager=this->get_reconstructed_fault_manager();const auto &fault=manager.get_fault(0);
      const unsigned int state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model());
      const auto &law=model.get_fault_friction();
      using Access=MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
      auto o=file("nodes_"+std::to_string(observation)+".csv");
      o<<"node,x,y,V,Theta,M_diag,M_right,K_diag,K_right,Mmu_diag,Mmu_right,R,shear,friction,damping,raw_normal,normal,z,p,dev,bg,dev_old,dev_strain,dev_slip\n";
      for(unsigned int i=0;i<fault.n_vertices();++i)
        o<<i<<','<<fault.vertex(i)[0]<<','<<fault.vertex(i)[1]<<','<<d.rates[0][i]<<','<<fault.get_properties(i)[state]
         <<','<<r.mass_diagonal[0][i]<<','<<(i+1<fault.n_vertices()?r.mass_off_diagonal[0][i]:0.)
         <<','<<d.filter_stiffness_diagonal[0][i]<<','<<(i+1<fault.n_vertices()?d.filter_stiffness_off_diagonal[0][i]:0.)
         <<','<<d.friction_mass_diagonal[0][i]<<','<<(i+1<fault.n_vertices()?d.friction_mass_off_diagonal[0][i]:0.)
         <<','<<r.values[0][i]<<','<<r.shear_traction[0][i]<<','<<r.friction_traction[0][i]<<','<<r.damping_traction[0][i]
         <<','<<r.raw_normal_traction[0][i]<<','<<r.normal_traction[0][i]<<','<<r.normal_filter_coefficients[0][i]
         <<','<<d.pressure_load[0][i]<<','<<d.deviatoric_load[0][i]<<','<<d.background_load[0][i]
         <<','<<d.deviatoric_component_loads[0][0][i]<<','<<d.deviatoric_component_loads[1][0][i]<<','<<d.deviatoric_component_loads[2][0][i]<<'\n';
      auto q=file("samples_"+std::to_string(observation)+".csv");
      q<<"cell,qp,x,y,segment,xi,weight,V,Theta,f0,mu,dmu,dmuTheta,mu_common,dmu_common,p,dev,bg,raw,filtered,along,length,extended,chi,eta_ve,nx,ny\n";
      for(const auto &s:d.samples)
        {
          const double v=(1-s.xi)*d.rates[0][s.segment]+s.xi*d.rates[0][s.segment+1];
          const double theta=(1-s.xi)*fault.get_properties(s.segment)[state]+s.xi*fault.get_properties(s.segment+1)[state];
          const auto fractions=Access::surface_material_state_at_projection(model,0,s.segment,s.xi).first;
          const auto tangent=(fault.vertex(s.segment+1)-fault.vertex(s.segment))/fault.vertex(s.segment).distance(fault.vertex(s.segment+1));
          const double length=fault.vertex(s.segment).distance(fault.vertex(s.segment+1));
          const double along=(s.position-fault.vertex(s.segment))*tangent;
          const bool extended=(s.segment==0 && along<0.) || (s.segment+1==fault.n_cells() && along>length);
          q<<s.cell<<','<<s.qp<<','<<s.position[0]<<','<<s.position[1]<<','<<s.segment<<','<<s.xi<<','<<s.weight<<','<<v<<','<<theta<<','<<fractions[0]
           <<','<<s.friction_coefficient<<','<<law.friction_coefficient_derivative_wrt_slip_rate(fractions,v,theta)
           <<','<<law.friction_coefficient_derivative_wrt_state(fractions,v,theta)
           <<','<<law.friction_coefficient(fractions,1e-9,theta)<<','<<law.friction_coefficient_derivative_wrt_slip_rate(fractions,1e-9,theta)
           <<','<<s.pressure<<','<<s.deviatoric<<','<<s.background<<','<<s.total<<','<<s.friction_normal
           <<','<<along<<','<<length<<','<<extended<<','<<s.chi<<','<<s.eta_ve<<','<<s.normal[0]<<','<<s.normal[1]<<'\n';
        }
      write_vector("iterate_"+std::to_string(observation)+".csv",this->get_current_linearization_point());
      write_vector("rhs_"+std::to_string(observation)+".csv",this->get_system_rhs());
      if(observation==0)
        {
          auto c=file("constraints.txt");this->get_current_constraints().print(c);
          const auto &matrix=this->get_system_matrix();auto a=file("A.csv");a<<"block_row,block_col,row,col,value\n";
          for(unsigned int br=0;br<2;++br)for(unsigned int bc=0;bc<2;++bc)
            for(auto i=matrix.block(br,bc).begin();i!=matrix.block(br,bc).end();++i)
              if(i->value()!=0.)a<<br<<','<<bc<<','<<i->row()<<','<<i->column()<<','<<i->value()<<'\n';
          points.resize(this->get_dof_handler().n_dofs());
          for(const auto &entry:DoFTools::map_dofs_to_support_points(this->get_mapping(),this->get_dof_handler()))points[entry.first]=entry.second;
          components.resize(points.size());std::vector<types::global_dof_index> dofs(this->get_fe().n_dofs_per_cell());
          for(const auto &cell:this->get_dof_handler().active_cell_iterators())
            {cell->get_dof_indices(dofs);for(unsigned int i=0;i<dofs.size();++i)components[dofs[i]]=this->get_fe().system_to_component_index(i).first;}
          auto m=file("dofs.csv");m<<"dof,component,x,y\n";
          const auto n=this->get_system_rhs().block(0).size()+this->get_system_rhs().block(1).size();
          for(unsigned int i=0;i<n;++i)m<<i<<','<<components[i]<<','<<points[i][0]<<','<<points[i][1]<<'\n';
        }
      ++observation;
    }
    public:
    void initialize() override
    {
      AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,ExcMessage("Serial diagnosis only."));
      this->get_signals().post_simulator_initialization.connect([this](const auto &){
        auto &s=this->get_reconstructed_fault_surface_system();
        const auto previous=s.normal_diagnostic_observer;
        s.set_normal_traction_diagnostic([](const Point<dim>&){return true;});
        s.normal_diagnostic_observer=[this,previous](const auto &r,const auto &d){if(previous)previous(r,d);observe(r,d);};
      });
      this->get_signals().post_reconstructed_fault_linear_solver.connect([this](const auto &,const auto &C,const auto &,const auto &,const auto &rhs,const auto &direction,double tolerance,unsigned int,double){
        auto residual=rhs;C(residual,direction);residual-=rhs;
        auto log=file("linear_"+std::to_string(linear)+".csv");log<<"fresh_norm,rhs_norm,tolerance\n"<<residual.l2_norm()<<','<<rhs.l2_norm()<<','<<tolerance<<'\n';
        if(linear++!=0)return;
        auto &s=this->get_reconstructed_fault_surface_system();const auto &f=this->get_reconstructed_fault_manager().get_fault(0);
        typename Surface::FaultVector e(1,std::vector<double>(f.n_vertices(),0.)),k;
        auto K=file("KV.csv"), B=file("B.csv");K<<"row,col,value\n";B<<"dof,col,value\n";
        for(unsigned int j=0;j<f.n_vertices();++j)
          {
            e[0][j]=1.;s.apply_surface_jacobian(e,k);auto b=this->get_system_rhs();
            this->get_reconstructed_fault_stokes_coupling().apply_B(e,b);
            for(unsigned int i=0;i<f.n_vertices();++i)K<<i<<','<<j<<','<<k[0][i]<<'\n';
            for(unsigned int i=0;i<b.block(0).size();++i)if(b.block(0)[i]!=0.)B<<i<<','<<j<<','<<b.block(0)[i]<<'\n';
            e[0][j]=0.;
          }
        auto G=file("G_probes.csv");G<<"probe,row,value\n";
        for(unsigned int probe=0;probe<9;++probe)
          {
            auto x=this->get_system_rhs();x=0.;
            for(unsigned int i=0;i<x.block(0).size()+x.block(1).size();++i)
              if(components[i]==probe/3)
                x[i]=(probe/3<2?1e-9:1.)*(probe%3==0?1.:probe%3==1?points[i][0]/50000.:points[i][1]/50000.);
            this->get_current_constraints().distribute(x);s.apply_G(x,k);
            for(unsigned int i=0;i<f.n_vertices();++i)G<<probe<<','<<i<<','<<k[0][i]<<'\n';
          }
      });
    }
    std::pair<std::string,std::string> execute(TableHandler &) override {return {};}
  };
  ASPECT_REGISTER_POSTPROCESSOR(TimestepZero,"timestep zero diagnosis","Read-only serial assembly/filter/friction diagnosis.")
}}
