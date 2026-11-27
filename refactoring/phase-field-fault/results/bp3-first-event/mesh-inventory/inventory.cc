// Measurement-only DSO. Stop collectively before allocating the full DoF system.
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>

namespace aspect
{
  template <int dim>
  void connect_mesh_inventory(SimulatorSignals<dim> &signals)
  {
    signals.edit_parameters_pre_setup_dofs.connect([](const SimulatorAccess<dim> &sim, Parameters<dim> &)
    {
      std::vector<unsigned int> histogram(sim.get_triangulation().n_global_levels());
      unsigned int support=0,contacts=0,violations=0;
      double area=0.;
      const double sine=std::sqrt(.75), radius=39.5291641;
      for (const auto &cell:sim.get_triangulation().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            ++histogram[cell->level()];
            const auto a=cell->vertex(0), b=cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
            const double h=b[0]-a[0];
            const double d=std::abs(sine*(a[0]+b[0])/2.+.5*((a[1]+b[1])/2.-50000.))-.5*(sine+.5)*h;
            const bool positive=d<=radius;
            support+=positive;
            const double pad=(radius+1000.*(sine+.5))/sine+3.90625;
            bool contact=false;
            for (double y : {0.,50000.})
              {
                const double x=(50000.-y)/std::sqrt(3.);
                contact |= a[0]<=x+pad && b[0]>=x-pad && a[1]<=y+7.8125 && b[1]>=y-7.8125;
              }
            contacts+=contact;
            violations+=(h>1000. || ((positive || contact) && h!=3.90625));
            area+=cell->measure();
          }
      const auto comm=sim.get_mpi_communicator();
      support=Utilities::MPI::sum(support,comm);
      contacts=Utilities::MPI::sum(contacts,comm);
      violations=Utilities::MPI::sum(violations,comm);
      area=Utilities::MPI::sum(area,comm);
      sim.get_pcout()<<"Physical support cells "<<support<<", protected face cells "<<contacts
                     <<", invalid footprint cells "<<violations<<", area "<<area<<" m2"<<std::endl;
      AssertThrow(violations==0 && std::abs(area-7.5e9)<.01,ExcMessage("Historical mesh footprint differs."));
      Utilities::MPI::sum(histogram,sim.get_mpi_communicator(),histogram);
      for (unsigned int level=0; level<histogram.size(); ++level)
        sim.get_pcout() << "Historical mesh level " << level << " cells " << histogram[level] << std::endl;
      sim.get_pcout() << "Historical mesh total " << sim.get_triangulation().n_global_active_cells() << std::endl;
      AssertThrow(false,ExcMessage("Mesh inventory complete before distributing degrees of freedom."));
    });
  }
  ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_mesh_inventory<2>,connect_mesh_inventory<3>)
}
