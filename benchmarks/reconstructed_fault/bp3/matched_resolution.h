// Benchmark-only mesh replay and a shared clock for two independent solutions.
#include <aspect/mesh_refinement/interface.h>
#include <aspect/time_stepping/convection_time_step.h>
#include <chrono>
#include <thread>
#include <set>

namespace aspect
{
  namespace MeshRefinement
  {
    template <int dim>
    class BP3SavedMesh : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override
        {
          const char *path=std::getenv("ASPECT_BP3_TARGET_MESH");
          AssertThrow(path,ExcMessage("BP3 saved mesh requires a target leaf list."));
          std::ifstream in(path);
          AssertThrow(in,ExcMessage("Cannot open BP3 mesh leaf list."));
          std::string id;
          while (in>>id)
            {
              leaves.insert(id);
              const auto split=id.find(':');
              const auto root=id.substr(0,id.find('_'));
              const auto children=id.substr(split+1);
              for (unsigned int n=0;n<children.size();++n)
                ancestors.insert(root+"_"+std::to_string(n)+":"+children.substr(0,n));
            }
          AssertThrow(!leaves.empty(),ExcMessage("Empty BP3 mesh leaf list."));
        }

        void tag_additional_cells() const override
        {
          // Reconstruct the saved tree, then its explicitly requested children.
          // deal.II retains responsibility for mandatory mesh grading.
          for (const auto &cell:this->get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->clear_refine_flag();
                cell->clear_coarsen_flag();
                if (ancestors.count(cell->id().to_string()))cell->set_refine_flag();
              }
        }
      private:
        std::set<std::string> leaves,ancestors;
    };
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(BP3SavedMesh,"BP3 saved mesh",
      "Recreate supplied BP3 leaf cells, with only mandatory mesh grading.")
  }

  namespace TimeStepping
  {
    template <int dim>
    class BP3SharedClock : public ConvectionTimeStep<dim>
    {
      public:
        double execute() override
        {
          // Use the actual production CFL and split-RSF computations. No
          // velocity, constitutive history, or mechanical field crosses jobs.
          const auto &model=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
            this->get_material_model());
          const double local=std::min(ConvectionTimeStep<dim>::execute(),
            model.compute_reconstructed_fault_time_step(this->get_parameters().CFL_number));
          const double limit=Utilities::MPI::min(local,this->get_mpi_communicator());
          double shared=0.;
          unsigned int failed=0;
          if (this->get_pcout().is_active())
            {
              try
                {
                  const std::filesystem::path directory(std::getenv("ASPECT_BP3_SHARED_CLOCK"));
                  const std::string label(std::getenv("ASPECT_BP3_PAIR_LABEL"));
                  AssertThrow(label=="coarse" || label=="refined",ExcMessage("Invalid BP3 pair label."));
                  const std::string step=std::to_string(this->get_timestep_number());
                  const auto own=directory/(label+"_"+step);
                  const auto peer=directory/((label=="coarse" ? "refined_" : "coarse_")+step);
                  AssertThrow(!std::filesystem::exists(own),ExcMessage("Refuse stale BP3 clock data."));
                  {
                    std::ofstream out(own.string()+".tmp");
                    out.exceptions(std::ios::failbit|std::ios::badbit);
                    out<<std::setprecision(17)<<this->get_time()<<' '<<limit<<'\n';
                  }
                  std::filesystem::rename(own.string()+".tmp",own);
                  const auto begin=std::chrono::steady_clock::now();
                  while (!std::filesystem::exists(peer))
                    {
                      AssertThrow(!std::filesystem::exists(directory/"abort"),ExcMessage("Other BP3 case failed."));
                      AssertThrow(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<600.,
                                  ExcMessage("BP3 shared-clock peer did not arrive within 600 s."));
                      std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                  double time,other;
                  std::ifstream in(peer);
                  AssertThrow(in>>time>>other,ExcMessage("Invalid BP3 peer timestep limit."));
                  AssertThrow(std::abs(time-this->get_time())<=1e-12*std::max(1.,std::abs(time)) && other>0.,
                              ExcMessage("BP3 paired states have different physical times."));
                  shared=.95*std::min(limit,other);
                  this->get_pcout()<<std::setprecision(17)<<"BP3 shared clock: step="<<step
                    <<", local limit="<<limit<<", peer limit="<<other<<", cap="<<shared<<std::endl;
                }
              catch (const std::exception &e)
                {this->get_pcout()<<e.what()<<std::endl;failed=1;}
            }
          failed=Utilities::MPI::max(failed,this->get_mpi_communicator());
          AssertThrow(!failed,ExcMessage("BP3 shared-clock exchange failed; no timestep override."));
          MPI_Bcast(&shared,1,MPI_DOUBLE,0,this->get_mpi_communicator());
          cap=shared;
          return shared;
        }

        std::pair<Reaction,double> determine_reaction(const TimeStepInfo &info) override
        {
          AssertThrow(info.next_time_step_size<=cap*(1+1e-14),ExcMessage("BP3 step exceeds paired controller cap."));
          this->get_pcout()<<std::setprecision(17)<<"BP3 shared clock selected: "<<info.next_time_step_size<<std::endl;
          return {Reaction::advance,std::numeric_limits<double>::max()};
        }
      private:
        double cap=0.;
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(BP3SharedClock,"BP3 shared clock",
      "Production CFL/RSF restrictions plus a common paired-run cap; manager limits remain unchanged.")
  }

  namespace BP3Benchmark
  {
    bool paired_history_ready=false;
    template <int dim>
    void verify_paired_mesh(const SimulatorAccess<dim> &sim)
    {
      if (!std::getenv("ASPECT_BP3_SHARED_CLOCK") && !std::getenv("ASPECT_BP3_EXACT_TARGET"))return;
      std::ifstream in(std::getenv("ASPECT_BP3_TARGET_MESH"));
      std::set<std::string> target;
      std::string id;
      while(in>>id)target.insert(id);
      unsigned int extra=0,invalid=0;
      const bool exact=std::getenv("ASPECT_BP3_EXACT_TARGET")
                       || std::string(std::getenv("ASPECT_BP3_PAIR_LABEL"))=="coarse";
      std::ofstream mesh(sim.get_output_directory()+"mesh_guard_rank"+
        std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".csv");
      mesh<<std::setprecision(17)<<"cell,x,y,h,xd,normal,extra,descendant,invalid\n";
      for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
        if(cell->is_locally_owned() && !target.count(cell->id().to_string()))
          {
            ++extra;
            id=cell->id().to_string();
            const auto root=id.substr(0,id.find('_'));
            auto path=id.substr(id.find(':')+1);
            bool descendant=false;
            while(!path.empty())
              {
                path.pop_back();
                if(target.count(root+"_"+std::to_string(path.size())+":"+path))
                  {descendant=true;break;}
              }
            const auto p=cell->center();
            const double xd=BP3::down_dip(p[0],p[1]);
            const bool bad=exact || !descendant || xd<34000 || xd>46000 || BP3::normal_distance(p[0],p[1])>3000;
            if(bad)
              ++invalid;
            mesh<<cell->id().to_string()<<','<<p[0]<<','<<p[1]<<','<<cell->diameter()/std::sqrt(2.)<<','
              <<xd<<','<<BP3::normal_distance(p[0],p[1])<<",1,"<<descendant<<','<<bad<<'\n';
          }
      mesh.close();
      extra=Utilities::MPI::sum(extra,sim.get_mpi_communicator());
      invalid=Utilities::MPI::sum(invalid,sim.get_mpi_communicator());
      sim.get_pcout()<<"BP3 mesh guard: extra leaves="<<extra<<", invalid leaves="<<invalid<<std::endl;
      AssertThrow(!invalid,ExcMessage("BP3 mesh differs from target outside permitted local grading."));
      sim.get_pcout()<<"BP3 target mesh verified before mechanics: cells="
        <<sim.get_triangulation().n_global_active_cells()<<", extra grading leaves="<<extra<<std::endl;
      if (const char *filename=std::getenv("ASPECT_BP3_EXPECTED_FAULT"))
        {
          std::ifstream input(filename);
          AssertThrow(input,ExcMessage("Cannot read the BP3 expected fault grid."));
          const auto &fault=sim.get_reconstructed_fault_manager().get_faults()[0];
          unsigned int vertex=0;
          double x,y,phi,error=0.;
          while(input>>x>>y>>phi)
            {
              AssertThrow(vertex<fault.n_vertices(),ExcMessage("BP3 fault grid lost an expected vertex."));
              error=std::max(error,fault.vertex(vertex).distance(Point<dim>(x,y)));
              ++vertex;
            }
          AssertThrow(input.eof() && vertex==fault.n_vertices() && error<1e-10,
                      ExcMessage("BP3 fault grid differs from the prescribed comparison grid."));
          sim.get_pcout()<<"BP3 exact fault-grid guard: vertices="<<vertex
            <<", coordinate error="<<error<<std::endl;
        }
      AssertThrow(!std::getenv("ASPECT_BP3_MESH_ONLY"),ExcMessage("Intentional mesh-only stop before mechanics."));
    }

    template <int dim>
    void paired_common_history(const SimulatorAccess<dim> &sim,AffineConstraints<double> &pending)
    {
      if (!std::getenv("ASPECT_BP3_SHARED_CLOCK") || !paired_history_ready
          || std::abs(sim.get_time()-2232176379.2516127)>1e-3)return;
      paired_history_ready=false;
      LinearAlgebra::BlockVector owned(sim.introspection().index_sets.system_partitioning,sim.get_mpi_communicator());
      owned=sim.get_solution();
      AffineConstraints<double> constraints(pending);constraints.close();constraints.distribute(owned);
      LinearAlgebra::BlockVector working(sim.introspection().index_sets.system_partitioning,
        sim.introspection().index_sets.system_relevant_partitioning,sim.get_mpi_communicator());
      working=owned;
      // Before Newton/history publication: these are the retained inputs to
      // this step, not newly committed stress or graphical postprocess fields.
      common_history_tests(sim,working);
    }
  }
}
