/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/boundary_composition/interface.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "transport_method_fields.h"
namespace aspect
{
  namespace InitialComposition
  {
    template <int dim>
    class TransportMethods : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double initial_composition(const Point<dim> &p,const unsigned int c) const override
        {
          const unsigned int nx=1u<<this->get_parameters().initial_global_refinement;
          return PF4Methods::scalar(c+(this->introspection().n_compositional_fields==3?2:0),p[0],p[1],0.,nx);
        }
    };
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(TransportMethods,"PF4 transport methods","Prescribed scalar benchmark and native Q2 strain-history data.")
  }
  namespace BoundaryComposition
  {
    template <int dim>
    class TransportMethods : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double boundary_composition(const types::boundary_id,const Point<dim> &p,const unsigned int c) const override
        {
          const unsigned int nx=1u<<this->get_parameters().initial_global_refinement;
          return PF4Methods::scalar(c+(this->introspection().n_compositional_fields==3?2:0),p[0],p[1],this->get_time(),nx);
        }
    };
    ASPECT_REGISTER_BOUNDARY_COMPOSITION_MODEL(TransportMethods,"PF4 transport methods","Exact characteristic inflow for the bounded comparison.")
  }
  namespace Postprocess
  {
    template <int dim>
    class ScalarMethodChecks : public Interface<dim>, public SimulatorAccess<dim>
    {
        std::chrono::steady_clock::time_point start;
        unsigned int calls=0;
        std::ofstream output(const std::string &name) const
        {
          std::ofstream out(this->get_output_directory()+name);
          out<<std::setprecision(17);
          return out;
        }
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim==2 && !this->get_parameters().mesh_deformation_enabled && !this->get_parameters().enable_phase_field
                      && this->get_parameters().nonlinear_solver==Parameters<dim>::NonlinearSolver::single_Advection_no_Stokes
                      && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                      ExcMessage("Serial 2D prescribed transport comparison only."));
          if (calls++==0)
            {
              start=std::chrono::steady_clock::now();
              const unsigned int nx=1u<<this->get_parameters().initial_global_refinement;
              AssertThrow(this->get_triangulation().n_global_active_cells()==2*nx*nx,
                          ExcMessage("Fixed Cartesian comparison mesh required."));
              for (const auto &cell:this->get_dof_handler().active_cell_iterators())
                for (const auto face:cell->face_indices())
                  AssertThrow(!cell->has_periodic_neighbor(face) && (cell->at_boundary(face) ||
                                                                     (!cell->neighbor_is_coarser(face) && cell->neighbor(face)->is_active())),
                              ExcMessage("No periodic or hanging faces in this comparison."));
            }
          if (this->get_timestep_number()!=0 && this->get_time()<.2-1.e-12) return {"PF4 scalar methods","running"};
          const bool initial=this->get_timestep_number()==0;
          const FE_Q<dim> element(2);
          const Quadrature<dim> rule(element.get_unit_support_points());
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values);
          const auto first=this->get_triangulation().begin_active();
          const unsigned int nx=std::lround(1./(first->vertex(1)[0]-first->vertex(0)[0]));
          if (initial)
            {
              auto constants=output("constants.csv"),basis=output("basis.csv"),cells=output("cells.csv");
              constants<<"nx,ny,relaxation_rate,time\n"<<nx<<','<<2*nx<<",0,.2\n";
              basis<<"i,r,s\n";
              for (unsigned int i=0; i<9; ++i) basis<<i<<','<<rule.point(i)[0]<<','<<rule.point(i)[1]<<'\n';
              cells<<"cell,x0,y0,dx,dy,i0,i1,i2,i3,i4,i5,i6,i7,i8\n";
              unsigned int index=0;
              for (const auto &cell:this->get_dof_handler().active_cell_iterators())
                {
                  cells<<index<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]<<','
                       <<cell->vertex(1)[0]-cell->vertex(0)[0]<<','<<cell->vertex(2)[1]-cell->vertex(0)[1];
                  for (unsigned int i=0; i<9; ++i) cells<<','<<9*index+i;
                  cells<<'\n';
                  ++index;
                }
            }
          const std::string prefix=initial?"initial-":"field-";
          auto sharp=output(prefix+"signed_jump.csv"),finite=output(prefix+"finite_width.csv"),history=output(prefix+"native_history.csv");
          for (auto *out:
               {
                 &sharp,&finite,&history
               }) *out<<"i,xx,yy,xy\n";
          const unsigned int nfields=this->introspection().n_compositional_fields;
          std::vector<std::vector<double>> values(nfields,std::vector<double>(9));
          unsigned int index=0;
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            {
              fe.reinit(cell);
              for (unsigned int c=0; c<nfields; ++c)
                fe[this->introspection().extractors.compositional_fields[c]].get_function_values(this->get_solution(),values[c]);
              for (unsigned int q=0; q<9; ++q)
                {
                  if (nfields!=3) sharp<<index<<','<<1000*values[0][q]<<','<<-400*values[0][q]<<','<<700*values[0][q]<<'\n';
                  if (nfields!=3) finite<<index<<','<<1000*values[1][q]<<','<<-400*values[1][q]<<','<<700*values[1][q]<<'\n';
                  if (nfields>=3)
                    {
                      const unsigned int offset=nfields==3?0:2;
                      history<<index<<','<<values[offset][q]<<','<<values[offset+1][q]<<','<<values[offset+2][q]<<'\n';
                    }
                  ++index;
                }
            }
          auto metrics=output(initial?"initial-metadata.csv":"metadata.csv");
          metrics<<"time,dt,step,scalar_dofs,total_dofs,evolution_wall_seconds\n"<<this->get_time()<<','<<this->get_timestep()<<','<<this->get_timestep_number()<<','
                 <<this->introspection().index_sets.system_partitioning[this->introspection().block_indices.compositional_fields[0]].n_elements()<<','<<this->get_dof_handler().n_dofs()<<','
                 <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<'\n';
          auto timer=output(initial?"initial-timer.csv":"timer.csv");
          timer<<"category,seconds\n";
          for (const auto &entry:this->get_computing_timer().get_summary_data(TimerOutput::total_wall_time))
            timer<<entry.first<<','<<entry.second<<'\n';
          return {"PF4 scalar methods",initial?"initial exported":"final exported"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(ScalarMethodChecks,"PF4 scalar methods","Native scalar EV/SUPG field export; no history backend replacement.")
  }
}
