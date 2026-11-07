/* Benchmark-only Cartesian-cell diagnosis, with an explicit distortion test. */
#include <aspect/simulator.h>
#include <aspect/simulator_signals.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_tools.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace aspect
{
  template <int dim>
  void connect_layered_mesh_audit(SimulatorSignals<dim> &signals)
  {
    signals.post_advection_solver.connect([](const SimulatorAccess<dim> &sim,
                                             bool temperature, unsigned,
                                             const SolverControl &)
    {
      if (!temperature || sim.get_timestep_number()!=0) return;
      if (std::getenv("ASPECT_CARTESIAN_AUDIT_DISTORT"))
        GridTools::transform([](const Point<dim> &p)
        {
          Point<dim> q=p;
          q[1]+=1.e-6*p[0];
          return q;
        }, const_cast<parallel::distributed::Triangulation<dim> &>(sim.get_triangulation()));
      double mapped_error=0., raw_error=0.;
      unsigned int bad_cells=0;
      std::ostringstream examples;
      examples << std::setprecision(17);
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const auto vertices=sim.get_mapping().get_vertices(cell);
            bool bad=false;
            for (const unsigned int v : cell->vertex_indices())
              for (unsigned int d=0; d<dim; ++d)
                {
                  const unsigned int corner=(v & (1u<<d)) ? vertices.size()-1 : 0;
                  mapped_error=std::max(mapped_error,std::abs(vertices[v][d]-vertices[corner][d]));
                  raw_error=std::max(raw_error,std::abs(cell->vertex(v)[d]-cell->vertex(corner)[d]));
                  bad |= vertices.back()[d]<=vertices.front()[d] || vertices[v][d]!=vertices[corner][d];
                }
            if (bad && bad_cells++<2)
              {
                examples << " cell=" << cell->id().to_string();
                for (const unsigned int v : cell->vertex_indices())
                  examples << " mapped[" << v << "]=" << vertices[v]
                           << " raw[" << v << "]=" << cell->vertex(v);
              }
          }
      std::ostringstream message;
      message << std::setprecision(17) << "Cartesian audit rank="
              << Utilities::MPI::this_mpi_process(sim.get_mpi_communicator())
              << " bad_cells=" << bad_cells << " mapped_error=" << mapped_error
              << " raw_error=" << raw_error << examples.str() << '\n';
      std::cerr << message.str();
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_layered_mesh_audit<2>,connect_layered_mesh_audit<3>)
}
