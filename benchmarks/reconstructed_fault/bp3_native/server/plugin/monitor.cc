#include "runtime.h"
#include "profile.h"
#include "configuration.h"
#include "bp3_model.h"
#include "output_files.h"

#include <aspect/postprocess/interface.h>
#include <aspect/phase_field.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/surface_system.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <aspect/utilities.h>
#include <deal.II/fe/fe_values.h>

#include <fstream>
#include <iomanip>


// Stationary BP3 observations; no constitutive update is performed here.
#include <deal.II/base/quadrature_lib.h>
#include <numeric>

namespace aspect
{
  namespace BP3Restore
  {
    std::string filter_mode="helmholtz";
    double filter_length=20.;
    std::vector<double> incoming_theta;

    template <int dim>
    void before_mechanics(const SimulatorAccess<dim> &sim,bool temperature,unsigned int,const SolverControl &)
    {
      if (!temperature) return;
      auto &manager=sim.get_reconstructed_fault_manager();
      const auto &fault=manager.get_fault(0);
      const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
      if (BP3Benchmark::detailed_diagnostics)
        {
          incoming_theta.resize(fault.n_vertices());
          for(unsigned int i=0;i<fault.n_vertices();++i) incoming_theta[i]=fault.get_properties(i)[state];
        }
      if(sim.get_timestep_number()!=0) return;
      const auto &profile=BP3::loading_profile(sim);
      sim.get_pcout()<<"BP3 live stationary profile: support="<<profile.support
                     <<", full normal integral="<<2*profile.integral.back()
                     <<", primitive knots="<<profile.radius.size()<<std::endl;
    }
  }

  namespace Postprocess
  {
    template <int dim>
    class BP3RestoredMonitor : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");prm.enter_subsection("BP3 restored monitor");
        prm.declare_entry("Friction normal input","helmholtz",Patterns::Selection("raw|helmholtz"));
        prm.declare_entry("Normal filter length","20",Patterns::Double(0));
        prm.declare_entry("Bottom velocity constraint","full",Patterns::Selection("full|fault parallel"),
                          "Full Cartesian loading (default), or only its fault-parallel component. "
                          "The latter requires bottom absent from ordinary velocity boundary lists.");
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
        prm.declare_entry("Local state disturbance","0",Patterns::Double(0,.02));
#endif
        prm.declare_entry("Write detailed diagnostics","false",Patterns::Bool(),
                          "Write per-step fault, quadrature, particle and work-replay CSVs, plus the initial mesh. "
                          "The lightweight growth summary is always retained.");
        prm.leave_subsection();prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        BP3::configure_geometry(*this,prm);
        prm.enter_subsection("Postprocess");prm.enter_subsection("BP3 restored monitor");
        BP3Restore::filter_mode=prm.get("Friction normal input");
        BP3Restore::filter_length=prm.get_double("Normal filter length");
        BP3Restore::bottom_velocity_constraint=prm.get("Bottom velocity constraint");
#ifdef ASPECT_BP3_LOCAL_BOTTOM_TEST
        BP3::local_state_disturbance=prm.get_double("Local state disturbance");
#endif
        detailed_diagnostics=prm.get_bool("Write detailed diagnostics");
        BP3Benchmark::detailed_diagnostics=detailed_diagnostics;
        prm.leave_subsection();prm.leave_subsection();
        configuration=BP3::read_configuration(prm);
      }
      void initialize() override
      {
        using namespace BP3Restore;
        // Surface options attach after simulator construction. Loading prepares
        // its own live-profile cache and does not depend on this observer.
        this->get_signals().post_simulator_initialization.connect(
          [this](const SimulatorAccess<dim> &sim)
          {
            sim.get_pcout()<<"BP3 resolved model and runtime settings (not server qualification):\n"
                           <<configuration.resolved_settings;
            BP3::collective_root_write(sim.get_mpi_communicator(),[&]()
              {
                std::ofstream out(sim.get_output_directory()+"bp3_resolved_settings.json");
                out.exceptions(std::ios::failbit|std::ios::badbit);
                out<<configuration.resolved_settings;
                out.close();
              });
            auto &surface=sim.get_reconstructed_fault_surface_system();
            surface.set_normal_stress_filter(BP3Restore::filter_mode,BP3Restore::filter_length);
            if (detailed_diagnostics) surface.set_normal_traction_diagnostic(
              [](const Point<dim> &p)
              {
                const double s=BP3::down_dip(p[0],p[1]);
                return s<200. || s>BP3::geometry().length-200.
                       || std::abs(s-BP3::geometry().weakening_length)<100.
                       || std::abs(s-(BP3::geometry().weakening_length+3000.))<100.;
              });
          });
        this->get_signals().post_advection_solver.connect(&before_mechanics<dim>);
      }
      void save(std::map<std::string,std::string> &status) const override
      {status["BP3 restored model"]=identity();}
      void load(const std::map<std::string,std::string> &status) override
      {
        const auto entry=status.find("BP3 restored model");
        AssertThrow(entry!=status.end() && entry->second==identity(),
                    ExcMessage("BP3 restart requires the same geometry, material/profile, loading, refinement, particle policy and normal filter identity. Older checkpoints require their original plugin."));
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto &fault=manager.get_fault(0);
        const auto &w=this->get_reconstructed_fault_surface_system().get_linearization_residual();
        const auto &raw=w.raw_normal_traction.empty() ? w.normal_traction[0] : w.raw_normal_traction[0];
        const auto solve=[&](const auto &rhs){return ReconstructedFaultUtilities::solve_tridiagonal_system(
          w.mass_diagonal[0],w.mass_off_diagonal[0],rhs);};
        const auto raw_q1=detailed_diagnostics ? solve(raw) : std::vector<double>();
        const auto filtered=detailed_diagnostics ? solve(w.normal_traction[0]) : std::vector<double>();
        const auto &V=manager.get_timestep_committed_slip_rate(0);
        const auto state=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
        const auto step=this->get_timestep_number();
        const auto comm=this->get_mpi_communicator();
        if (detailed_diagnostics)
          {
            // These samples were captured in mechanics before history publication;
            // do not recompute current stress from already updated particles.
            const auto &diagnostic=this->get_reconstructed_fault_surface_system().get_normal_traction_diagnostic();
            AssertThrow(diagnostic.step==step,ExcMessage("Restored monitor has stale mechanical samples."));
            const auto rank=Utilities::MPI::this_mpi_process(comm);
            std::ofstream samples(this->get_output_directory()+"restored_raw_"+std::to_string(step)
                                  +"_rank"+std::to_string(rank)+".csv");
            samples<<std::setprecision(17)<<"cell,qp,segment,xi,x,y,s,r,cell_h,JxW,work_weight,V,p,deviatoric,sigma_raw,sigma_friction,mu,phi,chi,Ih,working_old_xx,working_old_yy,working_old_xy,particle_interpolated_xx,particle_interpolated_yy,particle_interpolated_xy,constitutive_dt\n";
            for(const auto &q:diagnostic.samples)
              {
                const double rate=manager.interpolate_slip_rate(q.fault,q.segment,q.xi);
                samples<<q.cell<<','<<q.qp<<','<<q.segment<<','<<q.xi<<','<<q.position[0]<<','<<q.position[1]
                       <<','<<BP3::down_dip(q.surface_position[0],q.surface_position[1])
                       <<','<<BP3::signed_normal(q.position[0],q.position[1])<<','<<q.cell_size<<','<<q.JxW<<','<<q.weight
                       <<','<<rate<<','<<q.pressure<<','<<q.deviatoric<<','<<q.total<<','<<q.friction_normal<<','<<q.friction_coefficient
                       <<','<<q.phase<<','<<q.chi<<','<<q.I_h
                       <<','<<q.incoming_stress[0][0]<<','<<q.incoming_stress[1][1]<<','<<q.incoming_stress[0][1]
                       <<','<<q.particle_interpolated_stress[0][0]<<','<<q.particle_interpolated_stress[1][1]<<','<<q.particle_interpolated_stress[0][1]
                       <<','<<q.stress_time_step<<'\n';
              }
            std::ofstream particles(this->get_output_directory()+"restored_incoming_particles_"+std::to_string(step)
                                    +"_rank"+std::to_string(rank)+".csv");
            particles<<std::setprecision(17)<<"cell,id,rank,x,y,old_xx,old_yy,old_xy\n";
            for(const auto &p:diagnostic.particles)
              particles<<p.cell<<','<<p.id<<','<<p.owner_rank<<','<<p.position[0]<<','<<p.position[1]
                       <<','<<p.stress[0][0]<<','<<p.stress[1][1]<<','<<p.stress[0][1]<<'\n';
            samples.close(); particles.close();
            AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(bool(samples) && bool(particles)),comm),
                        ExcMessage("Cannot write restored BP3 rank diagnostics."));
          }
        // Flux uses physical velocities on ALL boundaries, including the free
        // top; the three prescribed sides need not balance on their own.
        FEFaceValues<dim> face(this->get_mapping(),this->get_fe(),QGauss<dim-1>(3),
          update_values|update_normal_vectors|update_JxW_values);
        std::vector<Tensor<1,dim>> velocity(face.n_quadrature_points);
        double local_flux[4]={0,0,0,0};
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())
          if(cell->is_locally_owned())
            for(unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell;++f)
              if(cell->face(f)->at_boundary())
                {
                  face.reinit(cell,f);
                  face[this->introspection().extractors.velocities].get_function_values(this->get_solution(),velocity);
                  const auto id=cell->face(f)->boundary_id();
                  AssertThrow(id<4,ExcMessage("Restored BP3 requires Cartesian Box boundaries."));
                  for(unsigned int q=0;q<velocity.size();++q)
                    local_flux[id]+=velocity[q]*face.normal_vector(q)*face.JxW(q);
                }
        double flux[4]; for(unsigned int b=0;b<4;++b) flux[b]=Utilities::MPI::sum(local_flux[b],comm);
        const double minimum=w.raw_normal_traction.empty()?w.minimum_normal_traction:w.minimum_raw_normal_traction;
        AssertThrow(w.minimum_normal_traction>0.,ExcMessage("Restored BP3 friction normal input lost compression."));
        const double Dc=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
          this->get_material_model()).characteristic_fault_slip_distance();
        BP3::collective_root_write(comm,[&]()
          {
            std::ofstream out;
            if (detailed_diagnostics)
              {
                out.open(this->get_output_directory()+"restored_fault_"+std::to_string(step)+".csv");
                out.exceptions(std::ios::failbit|std::ios::badbit);
                out<<std::setprecision(17)<<"node,s_m,time_s,V,Theta_in,Theta_committed,Omega,raw_sigma_row,raw_sigma_Q1,friction_sigma_Q1,V_chord,raw_chord,filtered_chord,work_mass\n";
              }
            double max_chord=0.,mean_error=0.;
            const auto chord=[&](const auto &z,unsigned int i)
              {
                if(i==0 || i+1==z.size()) return 0.;
                const double left=(fault.vertex(i)-fault.vertex(i-1)).norm(),right=(fault.vertex(i+1)-fault.vertex(i)).norm();
                return z[i]-(right*z[i-1]+left*z[i+1])/(left+right);
              };
            for(unsigned int i=0;i<V.size();++i)
              {
                const double mass=w.mass_diagonal[0][i]+(i?w.mass_off_diagonal[0][i-1]:0.)
                                  +(i+1<V.size()?w.mass_off_diagonal[0][i]:0.);
                mean_error+=w.normal_traction[0][i]-raw[i];
                max_chord=std::max(max_chord,std::abs(chord(V,i)));
                const double theta=fault.get_properties(i)[state];
                if (detailed_diagnostics) out<<i<<','<<BP3::down_dip(fault.vertex(i)[0],fault.vertex(i)[1])<<','<<this->get_time()<<','<<V[i]
                   <<','<<BP3Restore::incoming_theta.at(i)<<','<<theta<<','<<V[i]*theta/Dc
                   <<','<<raw[i]/mass<<','<<raw_q1[i]<<','<<filtered[i]<<','<<chord(V,i)
                   <<','<<chord(raw_q1,i)<<','<<chord(filtered,i)<<','<<mass<<'\n';
              }
            if (detailed_diagnostics) out.close();
            const auto path=this->get_output_directory()+"restored_growth.csv";
            const bool header=BP3::needs_header(path);
            std::ofstream summary(path,std::ios::app);
            summary.exceptions(std::ios::failbit|std::ios::badbit);
            if(header) summary<<"step,time,dt,min_raw_sigma,min_friction_sigma,max_V_chord,filter_weak_mean_error,flux_left,flux_right,flux_bottom,flux_top\n";
            summary<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<minimum<<','<<w.minimum_normal_traction
                   <<','<<max_chord<<','<<mean_error;
            for(double f:flux) summary<<','<<f;
            summary<<'\n';
            summary.close();
          });
        return {"Restored BP3 monitor","raw/filtered traction and endpoint growth recorded"};
      }
    private:
      bool detailed_diagnostics = false;
      BP3::Configuration configuration;
      std::string identity() const
      {
        std::ostringstream out;
        const auto &profile=BP3::loading_profile(*this);
        out<<std::setprecision(17)<<BP3::geometry().identity<<" loading live stationary v4 "
           <<configuration.model_identity
           <<BP3Restore::filter_mode<<' '<<BP3Restore::filter_length;
        for(unsigned int i=0;i<profile.radius.size();++i)
          out<<' '<<profile.radius[i]<<' '<<profile.integral[i]<<' '<<profile.slope[i];
        if (BP3Restore::bottom_velocity_constraint!="full")
          out<<" bottom="<<BP3Restore::bottom_velocity_constraint;
        return out.str();
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(BP3RestoredMonitor,"BP3 restored monitor",
      "Accepted raw and filtered traction, unsmoothed slip-rate variation, state timing and boundary flux.")
  }
}
