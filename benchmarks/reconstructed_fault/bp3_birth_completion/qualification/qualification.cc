// Full production mesh and initial-field qualification, stopped before mechanics.
// All production input values and the maintained refinement plugin are retained.
#include "../../bp3/plugin/runtime.h"
#include "../../bp3/plugin/geometry.h"
#include "../../bp3/plugin/profile.h"
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include "../../bp3/plugin/bp3_model.h"
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <fstream>
#include <iomanip>
namespace aspect
{
  namespace
  {
    std::string mode;
    template <int dim> void mesh_check(const SimulatorAccess<dim> &sim)
    {
      BP3Benchmark::verify_paired_mesh(sim);
      auto &manager=sim.get_reconstructed_fault_manager();
      manager.prepare_boundary_contacts();
      const auto &faces=manager.get_boundary_faces();
      const auto &contacts=manager.get_boundary_contacts();
      AssertThrow(contacts.size()==2,ExcMessage("Expected both production boundary contacts."));
      const double tol=ReconstructedFaultUtilities::boundary_contact_tolerance(faces);
      const double R=BP3::loading_profile(sim).support;
      std::ofstream out;
      if(sim.get_pcout().is_active())
        { out.open(sim.get_output_directory()+"completion_footprints.csv");
          out<<std::setprecision(17)<<"contact,x,y,R,padding,half_width,h,faces,min_x,max_x\n"; }
      for(unsigned c=0;c<contacts.size();++c)
        {
          const auto &contact=contacts[c];
          double padding=0.;
          for(const auto &cell:sim.get_triangulation().active_cell_iterators())
            if(cell->is_locally_owned())
              {
                const auto b=cell->bounding_box().get_boundary_points();
                double width=0.;
                for(unsigned d=0;d<dim;++d)width+=(b.second[d]-b.first[d])*std::abs(contact.normal[d]);
                padding=std::max(padding,width);
              }
          padding=Utilities::MPI::max(padding,sim.get_mpi_communicator());
          const double half=(R+padding)/(contact.inward_boundary_normal*contact.inward_tangent);
          const auto cell=sim.get_triangulation().begin_active();
          const double base=std::ldexp(cell->diameter()/std::sqrt(double(dim)),cell->level());
          const double fine=std::ldexp(base,-int(sim.get_parameters().initial_global_refinement));
          unsigned count=0;double lo=1e100,hi=-1e100;
          for(const auto &f:faces)
            if(f.boundary_id==contact.boundary_id)
              {
                const double left=std::min(f.vertices[0][0],f.vertices[1][0]);
                const double right=std::max(f.vertices[0][0],f.vertices[1][0]);
                if(left>contact.position[0]+half || right<contact.position[0]-half)continue;
                ++count;lo=std::min(lo,left);hi=std::max(hi,right);
                for(unsigned d=0;d<dim;++d)
                  AssertThrow(std::abs(f.cell_upper[d]-f.cell_lower[d]-fine)<=tol
                    && std::abs((f.cell_lower[d]-BP3::geometry().origin[d])/fine
                      -std::round((f.cell_lower[d]-BP3::geometry().origin[d])/fine))<=tol/fine,
                    ExcMessage("Production completion footprint lacks its aligned fine boundary lattice."));
              }
          AssertThrow(count>0 && lo<=contact.position[0]-half && hi>=contact.position[0]+half,
                      ExcMessage("Production boundary footprint is not fully covered."));
          if(out)out<<c<<','<<contact.position[0]<<','<<contact.position[1]<<','<<R<<','<<padding
                    <<','<<half<<','<<fine<<','<<count<<','<<lo<<','<<hi<<'\n';
        }
      sim.get_pcout()<<"PRODUCTION_MESH_COMPLETION_FOOTPRINT_PASS"<<std::endl;
    }
  }
  template <int dim> void connect_completion_test(SimulatorSignals<dim> &signals)
  {
    signals.post_simulator_initialization.connect([](const SimulatorAccess<dim> &sim)
    {
      sim.get_signals().edit_parameters_pre_setup_dofs.connect([](const auto &sim,auto &)
      {
        mesh_check(sim);
        if(mode=="mesh")AssertThrow(false,ExcMessage("PRODUCTION_MESH_QUALIFICATION_COMPLETE intentional stop"));
      });
      sim.get_signals().post_set_initial_state.connect([](const SimulatorAccess<dim> &sim)
      {
        // Exercise the existing public material preparation on native initial
        // phase/particle data before allocating the production Stokes matrices.
        // Velocity-boundary verification belongs to the later normal callback.
        auto &model=const_cast<MaterialModel::PhaseFieldFault<dim> &>(
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(sim.get_material_model()));
        AssertThrow(model.is_mature_frictional_fault(),ExcMessage("Only the matrix-free mature phase lift is supported here."));
        // The normal mature phase entry only lifts essential data and returns;
        // it never accesses matrix/RHS. Invoke that exact operation, including
        // ghost publication, instead of inventing a field initialization.
        LinearAlgebra::BlockSparseMatrix unused_matrix;
        LinearAlgebra::BlockVector unused_rhs;
        auto &phase=const_cast<PhaseFieldHandler<dim> &>(sim.get_phase_field_handler());
        auto &solution=const_cast<LinearAlgebra::BlockVector &>(sim.get_solution());
        phase.evolve_phase_field(unused_matrix,unused_rhs,solution);
        auto &native=sim.get_reconstructed_fault_manager();
        native.reconstruct_initial_faults();
        native.set_shear_sense(0,BP3::geometry().shear_sense);
        native.set_prescribed_slip_rates(std::vector<std::map<unsigned int,double>>(1));
        native.initialize_slip_rate(0,std::vector<double>(native.get_fault(0).n_vertices(),BP3::Vinit));
        model.prepare_reconstructed_fault_mechanical_solve();
        const auto &manager=sim.get_reconstructed_fault_manager();
        AssertThrow(manager.get_boundary_contacts().size()==2,ExcMessage("Missing paired core completion."));
        for(const auto &c:manager.get_boundary_contacts())
          AssertThrow(c.unsupported_reason.empty() && c.transverse_extent>0.,ExcMessage("Core endpoint support not qualified."));
        const auto norm=manager.get_property_information()[manager.get_property_index("phase field fault previous I h")].position;
        std::ofstream values;
        if(sim.get_pcout().is_active())
          {values.open(sim.get_output_directory()+"completed_Ih.csv");values<<std::setprecision(17)<<"x,y,Ih\n";}
        for(const auto &f:manager.get_faults())for(unsigned v=0;v<f.n_vertices();++v)
          {
            const double I=f.get_properties(v)[norm];
            AssertThrow(std::isfinite(I) && I>0.,ExcMessage("Incomplete native normalization."));
            if(values)values<<f.vertex(v)[0]<<','<<f.vertex(v)[1]<<','<<I<<'\n';
          }
        const auto &pm=sim.get_phase_field_handler().get_associated_particle_manager();
        AssertThrow(pm.get_particle_handler().n_global_particles()
                    ==16*sim.get_triangulation().n_global_active_cells(),ExcMessage("Production population changed."));
        sim.get_pcout()<<"PRODUCTION_INITIAL_CORE_COMPLETION_PASS particles="
          <<pm.get_particle_handler().n_global_particles()<<std::endl;
        AssertThrow(false,ExcMessage("PRODUCTION_INITIAL_QUALIFICATION_COMPLETE intentional stop"));
      });
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_completion_test<2>,connect_completion_test<3>)
  namespace Postprocess
  {
    template <int dim> class CompletionQualification : public Interface<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        { prm.enter_subsection("Postprocess");prm.declare_entry("Production qualification mode","initial",Patterns::Selection("mesh|initial"));prm.leave_subsection(); }
        void parse_parameters(ParameterHandler &prm) override
        { prm.enter_subsection("Postprocess");mode=prm.get("Production qualification mode");prm.leave_subsection(); }
        std::pair<std::string,std::string> execute(TableHandler &) override
        { AssertThrow(false,ExcMessage("Qualification must stop before mechanics."));return {}; }
    };
    ASPECT_REGISTER_POSTPROCESSOR(CompletionQualification,"production completion qualification","Bounded actual production startup qualification.")
  }
}
