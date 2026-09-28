// Disposable frozen-position trials; never used by a BP5 trajectory.
#include <aspect/postprocess/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/simulator_signals.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <fstream>
#include <iomanip>
#include <set>
#include <cstring>
#include <cstdlib>
#include <map>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim> class StressCycle : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
      std::list<std::string> required_other_postprocessors() const override { return {"particles"}; }
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Postprocess");prm.enter_subsection("Stress cycle");
        prm.declare_entry("Timestep fraction","1",Patterns::Double(.25,1));
        prm.leave_subsection();prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Postprocess");prm.enter_subsection("Stress cycle");
        fraction=prm.get_double("Timestep fraction");
        prm.leave_subsection();prm.leave_subsection();
      }
      void initialize() override
      {
        AssertThrow(dim==2 && std::getenv("ASPECT_STRESS_CYCLE_TRACE")
                    && std::getenv("ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION"),
                    ExcMessage("Stress cycle requires 2D, ASPECT_STRESS_CYCLE_TRACE and ASPECT_DIAGNOSTIC_FREEZE_PARTICLE_ADVECTION."));
        this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim> &,double &dt)
          { original_dt=dt;dt*=fraction; });
        this->get_signals().start_timestep.connect([this](const SimulatorAccess<dim> &) { capture(); });
      }
      void capture()
      {
        AssertThrow(!started && original_dt>0. && this->get_timestep()==fraction*original_dt,
                    ExcMessage("Stress cycle permits exactly one restored pending step."));
        started=true;
        const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
        auto tangent=fault.vertex(1)-fault.vertex(0);tangent/=tangent.norm();
        Tensor<1,dim> normal;normal[0]=-tangent[1];normal[1]=tangent[0];
        const auto &vertex_cells=this->get_phase_field_handler().get_grid_cache().get_vertex_to_cell_map();
        // Small cell patches at the center of each existing diagnostic window,
        // plus every vertex-neighbor cell used by distance-weighted averaging.
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (!cell->is_artificial())
            {
              const auto x=cell->center();
              const double xd=(100000.-x[1])/.86602540378443864676;
              if (std::min(std::abs(xd-70500.),std::abs(xd-79500.))<cell->diameter()
                  && std::abs((x-fault.vertex(0))*normal)<cell->diameter())
                for (const auto v:cell->vertex_indices())
                  for (const auto &neighbour:vertex_cells[cell->vertex_index(v)])
                    if (!neighbour->is_artificial()) selected.insert(neighbour->id().to_string());
            }
        std::ofstream cells(this->get_output_directory()+"stress_trace_cells_rank"+rank()+".txt");
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned() && selected.count(cell->id().to_string())) cells<<cell->id()<<'\n';
        write_particles("before",true);
        if (this->get_pcout().is_active())
          {
            std::ofstream out(this->get_output_directory()+"stress_cycle_clock.csv");
            out<<std::setprecision(17)<<"step,time_s,original_dt,fraction,actual_dt\n"
               <<this->get_timestep_number()<<','<<this->get_time()<<','<<original_dt<<','<<fraction<<','<<this->get_timestep()<<'\n';
          }
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        write_particles("after",false);
        return {"Stress cycle","one frozen-position update captured"};
      }
      private:
      std::string rank() const { return std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())); }
      void write_particles(const std::string &stage,bool save)
      {
        const auto &pm=this->get_particle_manager(0);
        const unsigned int sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
        std::ofstream out(this->get_output_directory()+"stress_particles_"+stage+"_rank"+rank()+".csv");
        out<<std::setprecision(17)<<"step,time_s,dt,cell,id,x,y,ref_x,ref_y,tau_xx,tau_yy,tau_xy\n";
        unsigned long long hash_sum=0,count=0;
        for (const auto &p:pm.get_particle_handler())
          {
            if (save) positions.emplace(p.get_id(),p.get_location());
            else AssertThrow(positions.at(p.get_id())==p.get_location(),ExcMessage("Frozen trial moved a particle."));
            unsigned long long hash=1469598103934665603ULL;
            const auto mix=[&](const void *data,std::size_t size)
              { for (std::size_t i=0;i<size;++i) { hash^=static_cast<const unsigned char *>(data)[i];hash*=1099511628211ULL; } };
            const auto id=p.get_id();mix(&id,sizeof(id));
            for (unsigned int d=0;d<dim;++d) { const double x=p.get_location()[d];mix(&x,sizeof(x)); }
            for (const double v:p.get_properties()) mix(&v,sizeof(v));
            hash_sum+=hash;++count;
            const auto cell=p.get_surrounding_cell()->id().to_string();
            if (!selected.count(cell)) continue;
            const auto x=p.get_location(),r=p.get_reference_location();const auto v=p.get_properties();
            out<<this->get_timestep_number()<<','<<this->get_time()<<','<<this->get_timestep()<<','<<cell<<','<<p.get_id()
               <<','<<x[0]<<','<<x[1]<<','<<r[0]<<','<<r[1]<<','<<v[sp]<<','<<v[sp+1]<<','<<v[sp+2]<<'\n';
          }
        hash_sum=Utilities::MPI::sum(hash_sum,this->get_mpi_communicator());
        count=Utilities::MPI::sum(count,this->get_mpi_communicator());
        if (this->get_pcout().is_active())
          { std::ofstream inventory(this->get_output_directory()+"stress_particles_"+stage+"_inventory.txt");inventory<<count<<' '<<hash_sum<<'\n'; }
      }
      double fraction=1.,original_dt=0.;bool started=false;
      std::set<std::string> selected;
      std::map<types::particle_index,Point<dim>> positions;
    };
    ASPECT_REGISTER_POSTPROCESSOR(StressCycle,"stress cycle",
      "One restored frozen-position mechanical solve and actual particle update trace.")

    // Real Simulator::interpolate_particle_properties, not a reimplementation.
    // The first three components are distinct constants; the last three affine.
    template <int dim> class StressTransferTest : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
      std::list<std::string> required_other_postprocessors() const override { return {"particles"}; }
      void initialize() override
      {
        this->get_signals().pre_set_initial_state.connect([this](auto &tria)
          {
            std::ofstream out(this->get_output_directory()+"stress_trace_cells_rank"
                              +std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".txt");
            for (const auto &cell:tria.active_cell_iterators())
              if (cell->is_locally_owned()) out<<cell->id()<<'\n';
          });
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto &pm=this->get_particle_manager(0);
        for (const auto &particle:pm.get_particle_handler())
          {
            if (this->get_timestep_number()==0) initial_positions.emplace(particle.get_id(),particle.get_location());
            else AssertThrow(initial_positions.at(particle.get_id())==particle.get_location(),
                             ExcMessage("Frozen RK2 moved a manufactured-test particle."));
            for (const double v:particle.get_properties())
              AssertThrow(std::isfinite(v),ExcMessage("Manufactured particle value is nonfinite."));
          }
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),QGauss<dim>(3),update_values|update_quadrature_points);
        LinearAlgebra::BlockVector owned(this->get_solution());
        // Check both published FE data and the constrained working representation.
        LinearAlgebra::BlockVector working;
        working.reinit(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
        working=this->get_solution();this->get_current_constraints().distribute(working);owned=working;
        double error[4]={};
        unsigned int samples=0;
        std::vector<double> published(fe.n_quadrature_points),constrained(fe.n_quadrature_points);
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe.reinit(cell);
              for (unsigned int c=0;c<6;++c)
                {
                  fe[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(),published);
                  fe[this->introspection().extractors.compositional_fields[c]].get_function_values(owned,constrained);
                  for (unsigned int q=0;q<fe.n_quadrature_points;++q)
                    {
                      ++samples;
                      AssertThrow(std::isfinite(published[q]) && std::isfinite(constrained[q]),
                                  ExcMessage("Nonfinite transferred manufactured field: cell="+cell->id().to_string()
                                             +" component="+std::to_string(c)+" published="+std::to_string(published[q])
                                             +" constrained="+std::to_string(constrained[q])));
                      const auto x=fe.quadrature_point(q);
                      const double constant[]={3.,-7.,11.};
                      const double exact=c<3 ? constant[c] : constant[c-3]+(c-1)*x[0]-(c+1)*x[1];
                      const unsigned int j=c<3 ? 0:2;
                      error[j]=std::max(error[j],std::abs(published[q]-exact));
                      error[j+1]=std::max(error[j+1],std::abs(constrained[q]-exact));
                    }
                }
            }
        for (auto &e:error) e=Utilities::MPI::max(e,this->get_mpi_communicator());
        AssertThrow(Utilities::MPI::sum(samples,this->get_mpi_communicator())>0,
                    ExcMessage("Transfer test evaluated no quadrature points."));
        AssertThrow(std::max(error[0],error[1])<1e-11,ExcMessage("Constant stress transfer failed."));
        // DWA is not affine-exact; report its error rather than demanding that property.
        const bool least_squares=std::getenv("STRESS_TEST_EXPECT_AFFINE");
        if (least_squares) AssertThrow(std::max(error[2],error[3])<1e-10,ExcMessage("Unrestricted least squares lost affine reproduction."));
        if (this->get_pcout().is_active())
          {
            std::ofstream out(this->get_output_directory()+"transfer_test.csv");
            out<<std::setprecision(17)<<"constant_published,constant_constrained,linear_published,linear_constrained,step,dt\n"
               <<error[0]<<','<<error[1]<<','<<error[2]<<','<<error[3]<<','<<this->get_timestep_number()<<','<<this->get_timestep()<<'\n';
          }
        return {"Stress transfer","constant checked; affine errors recorded"};
      }
      private:
      std::map<types::particle_index,Point<dim>> initial_positions;
    };
    ASPECT_REGISTER_POSTPROCESSOR(StressTransferTest,"stress transfer test","Constant and affine actual particle-to-FE transfer checks.")
  }
}
