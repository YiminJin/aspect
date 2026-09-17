// Benchmark-only reconstruction and verification of a saved bulk/fault mesh.
#include <aspect/mesh_refinement/interface.h>
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


  namespace BP3Benchmark
  {
    template <int dim>
    void verify_paired_mesh(const SimulatorAccess<dim> &sim)
    {
      if (!std::getenv("ASPECT_BP3_EXACT_TARGET")) return;
      std::ifstream in(std::getenv("ASPECT_BP3_TARGET_MESH"));
      std::set<std::string> target;
      std::string id;
      while(in>>id)target.insert(id);
      unsigned int extra=0,invalid=0;
      const bool exact=true;
      std::ofstream cells;
      if (std::getenv("ASPECT_BP3_MESH_ONLY"))
        {
          cells.open(sim.get_output_directory()+"mesh_cells_rank"+
            std::to_string(Utilities::MPI::this_mpi_process(sim.get_mpi_communicator()))+".csv");
          cells<<std::setprecision(17)<<"cell,x,y,h\n";
          for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
            if(cell->is_locally_owned())
              cells<<cell->id().to_string()<<','<<cell->center()[0]<<','<<cell->center()[1]
                   <<','<<cell->diameter()/std::sqrt(2.)<<'\n';
          cells.close();
        }
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

  }
}
