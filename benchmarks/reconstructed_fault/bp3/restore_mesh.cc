// Actual deal.II hierarchy, including corner balance; no mechanics or particles.
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <fstream>
#include <iomanip>
#include <map>

int main(int argc,char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  AssertThrow(argc==3,ExcMessage("Expected output prefix and measured stationary support radius."));
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,ExcMessage("Prepare the mesh on one rank."));
  const double hmin=2000./512, radius=std::stod(argv[2]), rf=std::max(40.,radius+2*hmin);
  const double sn=std::sqrt(3.)/2;
  const std::string prefix=argv[1];
  const auto smoothing=static_cast<Triangulation<2>::MeshSmoothing>(
    Triangulation<2>::limit_level_difference_at_vertices|Triangulation<2>::smoothing_on_refinement|
    Triangulation<2>::smoothing_on_coarsening);
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD,smoothing);
  GridGenerator::subdivided_hyper_rectangle(mesh,std::vector<unsigned int>{75,25},
    Point<2>(-60000,0),Point<2>(90000,50000),true);
  mesh.refine_global(1);
  for(unsigned int pass=0;pass<9;++pass)
    {
      bool changed=false;
      for(const auto &cell:mesh.active_cell_iterators())
        {
          const auto p=cell->center();const double h=cell->vertex(1)[0]-cell->vertex(0)[0];
          const double distance=std::max(0.,std::abs(sn*p[0]+.5*(p[1]-50000))-h*(sn+.5)/2);
          const double target=distance<=rf?hmin:std::min(1000.,2*hmin+(distance-rf)/4);
          if(h>target*(1+1e-12)) {cell->set_refine_flag();changed=true;}
        }
      if(!changed) break;
      AssertThrow(pass<8,ExcMessage("Restored mesh failed to settle at level 9."));
      mesh.execute_coarsening_and_refinement();
    }
  std::ofstream targets(prefix+"target_cells.txt"),cells(prefix+"mesh.csv"),summary(prefix+"mesh_inventory.txt");
  AssertThrow(targets && cells && summary,ExcMessage("Cannot write restored mesh files."));
  cells<<"cell,level,x,y,h\n"<<std::setprecision(17);
  double area=0.,smallest=1e99,largest=0.;unsigned int cut=0;
  std::map<unsigned int,unsigned int> levels;
  for(const auto &cell:mesh.active_cell_iterators())
    {
      const auto p=cell->center();const double h=cell->vertex(1)[0]-cell->vertex(0)[0];
      const double d=std::max(0.,std::abs(sn*p[0]+.5*(p[1]-50000))-h*(sn+.5)/2);
      AssertThrow(std::abs(cell->measure()/(h*h)-1)<1e-12,ExcMessage("Mesh cells are not square."));
      if(d<=radius) {++cut;AssertThrow(h==hmin,ExcMessage("Underresolved localization support cell."));}
      targets<<cell->id().to_string()<<'\n';
      cells<<cell->id().to_string()<<','<<cell->level()<<','<<p[0]<<','<<p[1]<<','<<h<<'\n';
      area+=cell->measure();smallest=std::min(smallest,h);largest=std::max(largest,h);++levels[cell->level()];
    }
  AssertThrow(area==150000.*50000. && smallest==hmin && largest<=1000.,ExcMessage("Restored mesh geometry failed."));
  summary<<std::setprecision(17)<<"cells "<<mesh.n_global_active_cells()<<"\nplanned_particles "<<9ull*mesh.n_global_active_cells()
         <<"\nhmin "<<smallest<<"\nhmax "<<largest<<"\naspect_ratio 1\nsupport_radius "<<radius
         <<"\nfine_half_width "<<rf<<"\nsupport_cells "<<cut<<'\n';
  for(const auto &l:levels) summary<<"level "<<l.first<<" cells "<<l.second<<'\n';
  // Count the requested mixed element, without allocating particle domains or
  // Stokes matrices. This is a mesh inventory, not a mechanical memory claim.
  FESystem<2> fe(FE_Q<2>(2),2,FE_Q<2>(1),1,FE_Q<2>(2),6,FE_Q<2>(1),1);
  DoFHandler<2> dofs(mesh);dofs.distribute_dofs(fe);
  const auto counts=DoFTools::count_dofs_per_fe_component(dofs);
  summary<<"dofs_total "<<dofs.n_dofs()<<'\n';
  for(unsigned int i=0;i<counts.size();++i) summary<<"component "<<i<<" dofs "<<counts[i]<<'\n';
  Utilities::System::MemoryStats memory;Utilities::System::get_memory_stats(memory);
  summary<<"mesh_inventory_RSS_KiB "<<memory.VmRSS<<'\n';
}
