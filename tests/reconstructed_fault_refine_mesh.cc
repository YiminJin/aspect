/* Generate one nested, graded bulk mesh; no FE field is initialized here. */
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <fstream>
#include <set>
#include <iostream>

int main(int argc,char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  AssertThrow(argc==3,ExcMessage("Expected coarse leaf file and refined output file."));
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,ExcMessage("Mesh preparation uses one rank."));
  const auto smoothing=static_cast<Triangulation<2>::MeshSmoothing>(
    Triangulation<2>::limit_level_difference_at_vertices|Triangulation<2>::smoothing_on_refinement|Triangulation<2>::smoothing_on_coarsening);
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD,smoothing);
  GridGenerator::subdivided_hyper_rectangle(tria,std::vector<unsigned int>{6,2},Point<2>(-100000,0),Point<2>(200000,100000),true);
  std::ifstream input(argv[1]);AssertThrow(input,ExcMessage("Missing coarse leaf tree."));
  std::set<std::string> leaves,ancestors;std::string id;
  while (input>>id)
    {
      leaves.insert(id);const auto root=id.substr(0,id.find('_')),digits=id.substr(id.find(':')+1);
      for (unsigned int n=0;n<digits.size();++n) ancestors.insert(root+"_"+std::to_string(n)+":"+digits.substr(0,n));
    }
  for (unsigned int level=0;level<10;++level)
    {
      bool changed=false;
      for (const auto &cell:tria.active_cell_iterators())
        if (ancestors.count(cell->id().to_string())) {cell->set_refine_flag();changed=true;}
      if (!changed) break;
      tria.execute_coarsening_and_refinement();
    }
  AssertThrow(tria.n_global_active_cells()==leaves.size(),ExcMessage("Coarse tree size mismatch."));
  unsigned int marked=0;
  for (const auto &cell:tria.active_cell_iterators())
    {
      AssertThrow(leaves.count(cell->id().to_string()),ExcMessage("Coarse tree changed."));
      const auto p=cell->center();const double h=cell->diameter()/std::sqrt(2.);
      const double sn=std::sqrt(3.)/2,xt=50000*(1+.5/sn);
      const double xd=(xt-p[0])*.5+(100000-p[1])*sn;
      const double normal=(xt-p[0])*sn-(100000-p[1])*.5;
      // The modal support is 15--18 km. This buffer refines its full transverse
      // source plus surrounding elastic bulk; mandatory grading is retained.
      if (xd+h>10000 && xd-h<23000 && std::abs(normal)<2500+h)
        {cell->set_refine_flag();++marked;}
    }
  tria.execute_coarsening_and_refinement();
  std::ofstream output(argv[2]);AssertThrow(output,ExcMessage("Cannot write refined leaf tree."));
  std::set<std::string> sorted;
  for (const auto &cell:tria.active_cell_iterators()) sorted.insert(cell->id().to_string());
  for (const auto &entry:sorted) output<<entry<<'\n';
  std::cout<<"Nested mesh: coarse="<<leaves.size()<<" marked="<<marked<<" fine="<<sorted.size()<<std::endl;
}
