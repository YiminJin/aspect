// Count and time the actual distributed fixed-mesh material geometry alone.
#include <aspect/material_model/phase_field_fault/tip_section_geometry.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <iostream>

int main(int argc,char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  AssertThrow(argc==4,ExcMessage("Expected nx, ny, and reference angle in degrees."));
  const auto comm=MPI_COMM_WORLD;
  const unsigned int nx=std::stoul(argv[1]),ny=std::stoul(argv[2]);
  parallel::distributed::Triangulation<2> mesh(comm);
  GridGenerator::subdivided_hyper_rectangle(mesh,{nx,ny},Point<2>(),Point<2>(8.,4.));
  std::map<CellId,Triangulation<2>::active_cell_iterator> cells;
  for (const auto &cell:mesh.active_cell_iterators())
    if (cell->is_locally_owned()) cells.emplace(cell->id(),cell);
  struct Station { double coordinate,weight,age,slip,rate,force; };
  std::vector<Station> stations;
  std::map<CellId,std::vector<unsigned int>> indices;
  std::map<CellId,std::vector<double>> normal;
  aspect::MaterialModel::BulkCohesive::Tip::SectionGeometry<2> geometry;
  MPI_Barrier(comm);
  const double start=MPI_Wtime();
  geometry.initialize(cells,comm,6,std::stod(argv[3]),false,1.,stations,indices,normal);
  const double seconds=Utilities::MPI::max(MPI_Wtime()-start,comm);
  unsigned long long count=0;
  for (const auto &entry:indices) count+=entry.second.size();
  count=Utilities::MPI::sum(count,comm);
  if (Utilities::MPI::this_mpi_process(comm)==0)
    std::cout<<"stations,points,max_rank_geometry_seconds\n"<<stations.size()<<','<<count<<','<<seconds<<'\n';
}
