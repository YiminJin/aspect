#include "runtime.h"
#include "geometry.h"
#include "profile.h"
#include <aspect/phase_field.h>
#include <aspect/mesh_refinement/interface.h>

namespace aspect
{
  namespace
  {
#ifdef ASPECT_BP3_LOCAL_OSCILLATION_TEST
    std::string local_mesh_policy="A";
#endif
    struct Resolution
    {
      double h, fine, coarse, distance, target;
    };

    template <int dim, class Cell>
    Resolution resolution(const SimulatorAccess<dim> &sim,const Cell &cell)
    {
      AssertThrow(dim==2,ExcMessage("BP3 fault support refinement requires a two-dimensional Box."));
      Point<2> lo(cell->vertex(0)[0],cell->vertex(0)[1]),hi=lo;
      for(unsigned int v=1;v<GeometryInfo<dim>::vertices_per_cell;++v)
        for(unsigned int d=0;d<2;++d)
          {lo[d]=std::min(lo[d],cell->vertex(v)[d]);hi[d]=std::max(hi[d],cell->vertex(v)[d]);}
      const double h=hi[0]-lo[0];
      AssertThrow(std::abs((hi[1]-lo[1])/h-1.)<1e-12
                  && std::abs(cell->measure()/(h*h)-1.)<1e-12,
                  ExcMessage("BP3 fault support refinement requires axis-aligned square cells."));
      const auto &parameters=sim.get_parameters();
      const unsigned int maximum=parameters.initial_global_refinement+parameters.initial_adaptive_refinement;
      const double base=std::ldexp(h,cell->level());
      const double fine=std::ldexp(base,-static_cast<int>(maximum));
      const double coarse=std::ldexp(base,-static_cast<int>(parameters.min_grid_level));
      const double d=BP3::geometry().minimum_cell_distance((lo+hi)/2.,Point<2>((hi-lo)/2.));
      const double band=std::max(2*sim.get_phase_field_handler().get_length_scale(),
                                 BP3::loading_profile(sim).support+2*fine);
      double target=d<=band ? fine : std::min(coarse,2*fine+(d-band)/4.);
#ifdef ASPECT_BP3_LOCAL_OSCILLATION_TEST
      // Recover only the original exterior slope/cap (bp3_length_scale_mesh.cc).
      // Both diagnostic candidates retain the larger accepted support band.
      if(local_mesh_policy=="B" && d>band)
        target=std::min({coarse,12500.,2*fine+(d-band)/2.});
#endif
      // Protect the existing completion footprint using the configured upper
      // bound on every cell width. Keep this strip in production as well as
      // local comparisons; the exterior grading rule alone is insufficient.
      const auto &g=BP3::geometry();
      const double half_width=(BP3::loading_profile(sim).support
                                +coarse*(std::abs(g.normal[0])+std::abs(g.normal[1])))/g.sine;
      for(const auto &end : {g.upper,g.lower})
        if(lo[0]<=end[0]+half_width+fine && hi[0]>=end[0]-half_width-fine
           && lo[1]<=end[1]+2*fine && hi[1]>=end[1]-2*fine)
          target=fine;
      return {h,fine,coarse,d,target};
    }
  }

  namespace BP3Benchmark
  {
    template <int dim>
    void verify_paired_mesh(const SimulatorAccess<dim> &sim)
    {
      unsigned int cells=0,support_cells=0,invalid=0;
      double area=0.,min_h=std::numeric_limits<double>::infinity(),max_h=0.;
      const double support=BP3::loading_profile(sim).support;
      for(const auto &cell:sim.get_triangulation().active_cell_iterators())
        if(cell->is_locally_owned())
          {
            const auto r=resolution(sim,cell);
            ++cells;area+=cell->measure();
            min_h=std::min(min_h,r.h);max_h=std::max(max_h,r.h);
            if(r.distance<=support) ++support_cells;
            invalid+=r.h>r.target*(1+1e-12) || r.h<r.fine*(1-1e-12);
          }
      const auto comm=sim.get_mpi_communicator();
      cells=Utilities::MPI::sum(cells,comm);area=Utilities::MPI::sum(area,comm);
      support_cells=Utilities::MPI::sum(support_cells,comm);
      min_h=Utilities::MPI::min(min_h,comm);max_h=Utilities::MPI::max(max_h,comm);
      const auto &extent=BP3::geometry().extent;
      AssertThrow(Utilities::MPI::sum(invalid,comm)==0 && support_cells>0
                  && cells==sim.get_triangulation().n_global_active_cells()
                  && std::abs(area/(extent[0]*extent[1])-1.)<1e-12,
                  ExcMessage("BP3 generated mesh fails coverage/resolution invariants; check initial refinement passes."));
      sim.get_pcout()<<"BP3 mesh geometry: cells="<<cells<<", support cells="<<support_cells
                     <<", h range="<<min_h<<":"<<max_h<<", area="<<area<<std::endl;
    }
  }
  template void BP3Benchmark::verify_paired_mesh(const SimulatorAccess<2> &);
  template void BP3Benchmark::verify_paired_mesh(const SimulatorAccess<3> &);

  namespace MeshRefinement
  {
    template <int dim>
    class BP3FaultSupport : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
#ifdef ASPECT_BP3_LOCAL_OSCILLATION_TEST
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Mesh refinement");prm.enter_subsection("BP3 local comparison");
        prm.declare_entry("Policy","A",Patterns::Selection("A|B"));
        prm.leave_subsection();prm.leave_subsection();
      }
#endif
      void parse_parameters(ParameterHandler &prm) override
      {
        BP3::configure_geometry(*this,prm);
#ifdef ASPECT_BP3_LOCAL_OSCILLATION_TEST
        prm.enter_subsection("Mesh refinement");prm.enter_subsection("BP3 local comparison");
        local_mesh_policy=prm.get("Policy");prm.leave_subsection();prm.leave_subsection();
#endif
      }

      void initialize() override
      {
        const auto &p=this->get_parameters();
        AssertThrow(p.min_grid_level<=p.initial_global_refinement
                    && p.adaptive_refinement_interval==0 && p.additional_refinement_times.empty(),
                    ExcMessage("BP3 requires minimum level <= initial global level and a fixed mesh after startup."));
        // Native initial-global passes call tag_additional_cells before any
        // particle population exists. Unflagging exterior cells retains the
        // graded mesh; the global level is the finest permitted level here.
        AssertThrow(p.initial_adaptive_refinement==0,
                    ExcMessage("BP3 geometry-only startup uses Initial global refinement as the finest level, "
                               "Minimum refinement level as the coarsest, and Initial adaptive refinement = 0. "
                               "This prepares the complete mesh before initializing particle history."));
      }

      void tag_additional_cells() const override
      {
        for(const auto &cell:this->get_triangulation().active_cell_iterators())
          if(cell->is_locally_owned())
            {
              cell->clear_refine_flag();cell->clear_coarsen_flag();
              const auto r=resolution(*this,cell);
              if(r.h>r.target*(1+1e-12)) cell->set_refine_flag();
            }
      }
    };
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(BP3FaultSupport,"BP3 fault support",
      "Fixed startup mesh from the prescribed straight fault, live profile support and existing refinement levels. "
      "Retains the BP3 support band and dyadic exterior gradation with native mesh smoothing.")
  }
}
