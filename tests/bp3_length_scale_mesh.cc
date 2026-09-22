/* Graded, fixed BP3 band mesh; never refines the entire box to band resolution. */
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <fstream>
#include <iostream>
#include <set>
#include <iomanip>

int main(int argc, char **argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  AssertThrow(argc==5, ExcMessage("Expected ell, finest h, local-reference flag and leaf output."));
  const double ell=std::stod(argv[1]), finest=std::stod(argv[2]);
  const unsigned int reference=std::stoul(argv[3]);
  AssertThrow(reference<=3, ExcMessage("Reference mode is 0 (none), 1 (interior), 2 (interior and endpoints), or 3 (BP5 long probe and endpoints)."));
  AssertThrow(ell>0 && finest>0 && Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,
              ExcMessage("Mesh preparation requires positive scales and one rank."));
  const auto smoothing=static_cast<Triangulation<2>::MeshSmoothing>(
    Triangulation<2>::limit_level_difference_at_vertices|Triangulation<2>::smoothing_on_refinement|
    Triangulation<2>::smoothing_on_coarsening);
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD,smoothing);
  GridGenerator::subdivided_hyper_rectangle(mesh,std::vector<unsigned int>{6,2},
    Point<2>(-100000,0),Point<2>(200000,100000),true);
  const double sn=std::sqrt(3.)/2, xt=50000*(1+.5/sn);
  // AT1 core .6, p=1 support radius. Use a cell enclosure, not center-only
  // tagging, so every potentially nonzero Q1 cell gets the target resolution.
  const double radius=1.977*ell;
  for (unsigned int level=0;level<18;++level)
    {
      bool changed=false;
      for (const auto &cell:mesh.active_cell_iterators())
        {
          const auto p=cell->center(); const double h=cell->diameter()/std::sqrt(2.);
          const double distance=std::max(0.,std::abs((xt-p[0])*sn-(100000-p[1])*.5)-h*(sn+.5)/2);
          double target=distance<=radius ? finest
            : std::min(12500.,2*finest+.5*(distance-radius));
          const double xd=(xt-p[0])*.5+(100000-p[1])*sn;
          // Halve both band and surrounding bulk-cell sizes in a buffered
          // interior patch. The physical profile and outer mesh are unchanged.
          const bool interior=xd+h>(reference==3?7000.:10000.) && xd-h<(reference==3?26000.:23000.);
          const bool endpoint=reference>=2 && (xd-h<5000 || xd+h>100000/sn-5000);
          if (reference && (interior || endpoint) && distance<2500+h)
            target*=.5;
          if (h>target*(1+1e-12)) {cell->set_refine_flag();changed=true;}
        }
      if (!changed) break;
      mesh.execute_coarsening_and_refinement();
      AssertThrow(level<17, ExcMessage("Length-scale mesh did not reach its target."));
    }
  std::set<std::string> leaves;
  double area=0.,hmin=1e100,hmax=0.;
  for (const auto &cell:mesh.active_cell_iterators())
    {
      leaves.insert(cell->id().to_string());area+=cell->measure();
      const double h=cell->diameter()/std::sqrt(2.);
      hmin=std::min(hmin,h);hmax=std::max(hmax,h);
    }
  AssertThrow(std::abs(area/3e10-1)<1e-12 && hmax<=12500.,ExcMessage("Box measure/outer size changed."));
  std::ofstream out(argv[4]);AssertThrow(out,ExcMessage("Cannot write target mesh."));
  for (const auto &id:leaves) out<<id<<'\n';
  std::cout<<std::setprecision(17)<<"cells="<<leaves.size()<<" particles="<<9*leaves.size()
           <<" hmin="<<hmin<<" hmax="<<hmax<<" levels="<<mesh.n_global_levels()<<std::endl;
}
