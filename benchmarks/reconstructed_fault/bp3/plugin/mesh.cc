#include "runtime.h"
#include "geometry.h"
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <sstream>
// Fixed benchmark mesh: prescribed leaf tree, with deal.II grading.
#include <aspect/mesh_refinement/interface.h>
#include <set>

namespace aspect
{
  namespace BP3Benchmark
  {
    std::set<std::string> target_cells;

    template <int dim>
    void
    verify_paired_mesh (const SimulatorAccess<dim> &sim)
    {
      unsigned int invalid = 0;
      for (const auto &cell : sim.get_triangulation ().active_cell_iterators ())
        if (cell->is_locally_owned () && !target_cells.count (cell->id ().to_string ()))
          ++invalid;
      AssertThrow (Utilities::MPI::sum (invalid, sim.get_mpi_communicator ()) == 0
                       && sim.get_triangulation ().n_global_active_cells () == target_cells.size (),
                   ExcMessage ("BP3 mesh differs from its prescribed leaf tree."));
      const auto profiles=sim.get_phase_field_handler().get_phase_field_profiles(BP3::geometry().peak_phase);
      double support=0.;for(const auto &profile:profiles) support=std::max(support,profile->get_coordinate_values().back());
      unsigned int support_cells=0;
      double min_h=std::numeric_limits<double>::infinity(),max_h=0.;
      for (const auto &cell:sim.get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            Point<2> lo(cell->vertex(0)[0],cell->vertex(0)[1]),hi=lo;
            for(unsigned v=1;v<GeometryInfo<dim>::vertices_per_cell;++v)
              for(unsigned d=0;d<2;++d){lo[d]=std::min(lo[d],cell->vertex(v)[d]);hi[d]=std::max(hi[d],cell->vertex(v)[d]);}
            const double h=cell->diameter()/std::sqrt(2.);
            min_h=std::min(min_h,h);max_h=std::max(max_h,h);
            support_cells+=BP3::geometry().minimum_cell_distance((lo+hi)/2.,Point<2>((hi-lo)/2.))<=support;
          }
      const auto comm=sim.get_mpi_communicator();
      support_cells=Utilities::MPI::sum(support_cells,comm);
      min_h=Utilities::MPI::min(min_h,comm);max_h=Utilities::MPI::max(max_h,comm);
      sim.get_pcout()<<"BP3 mesh geometry: cells="<<sim.get_triangulation().n_global_active_cells()
                     <<", support cells="<<support_cells<<", h range="<<min_h<<":"<<max_h<<std::endl;

    }
  }

  template void BP3Benchmark::verify_paired_mesh(const SimulatorAccess<2> &);
  template void BP3Benchmark::verify_paired_mesh(const SimulatorAccess<3> &);

  namespace MeshRefinement
  {
    template <int dim> class BP3SavedMesh : public Interface<dim>, public SimulatorAccess<dim>
    {
    public:
      static void
      declare_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection ("Mesh refinement");
        prm.enter_subsection ("BP3 saved mesh");
        prm.declare_entry ("Target cells file", "", Patterns::Anything (),
                           "Fixed Box mesh leaf IDs matching the box coarse-cell layout.");
        prm.leave_subsection ();
        prm.leave_subsection ();
      }

      void
      parse_parameters (ParameterHandler &prm) override
      {
        BP3::configure_geometry(*this,prm);
        prm.enter_subsection ("Mesh refinement");
        prm.enter_subsection ("BP3 saved mesh");
        filename = prm.get ("Target cells file");
        prm.leave_subsection ();
        prm.leave_subsection ();
      }

      void
      initialize () override
      {
        AssertThrow (!filename.empty (), ExcMessage ("BP3 requires its fixed mesh input."));
        std::istringstream in (Utilities::read_and_distribute_file_content (
            Utilities::expand_ASPECT_SOURCE_DIR (filename), this->get_mpi_communicator ()));
        BP3Benchmark::target_cells.clear ();
        std::string id;
        while (in >> id)
          {
            AssertThrow (BP3Benchmark::target_cells.insert (id).second,
                         ExcMessage ("Duplicate BP3 target cell: " + id));
            const auto root = id.substr (0, id.find ('_'));
            const auto children = id.substr (id.find (':') + 1);
            for (unsigned int n = 0; n < children.size (); ++n)
              ancestors.insert (root + "_" + std::to_string (n) + ":" + children.substr (0, n));
          }
        AssertThrow (!BP3Benchmark::target_cells.empty (), ExcMessage ("Empty BP3 mesh leaf list."));
      }

      void
      tag_additional_cells () const override
      {
        for (const auto &cell : this->get_triangulation ().active_cell_iterators ())
          if (cell->is_locally_owned ())
            {
              cell->clear_refine_flag ();
              cell->clear_coarsen_flag ();
              if (ancestors.count (cell->id ().to_string ()))
                cell->set_refine_flag ();
            }
      }

    private:
      std::string filename;
      std::set<std::string> ancestors;
    };
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION (
        BP3SavedMesh, "BP3 saved mesh",
        "Recreate the prescribed fixed BP3 leaf tree, with mandatory mesh grading.")
  }
}
