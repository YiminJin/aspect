/* Double-precision Q2 response fields for offline, cell-exact strip diagnostics. */
#ifndef aspect_test_reconstructed_fault_velocity_export_h
#define aspect_test_reconstructed_fault_velocity_export_h

namespace aspect
{
  namespace MechanicalModes
  {
    template <int dim>
    void export_velocity_cells(const SimulatorAccess<dim> &sim,
                               const LinearAlgebra::BlockVector &direction,
                               const std::string &mode,std::ostream &out)
    {
      if constexpr (dim==2)
        {
          AssertThrow(sim.get_parameters().stokes_velocity_degree==2,
                      ExcMessage("Response export requires the qualified Cartesian Q2 fixture."));
          // Tensor lexicographic points, not FESystem local numbering. These
          // nine values uniquely specify each continuous Q2 velocity polynomial.
          QIterated<dim> points(QTrapezoid<1>(),2);
          FEValues<dim> fe(sim.get_mapping(),sim.get_fe(),points,update_values);
          std::vector<Tensor<1,dim>> velocity(points.size());
          const auto &fault=sim.get_reconstructed_fault_manager().get_fault(0);
          const auto top=fault.vertex(fault.n_vertices()-1);
          Tensor<1,dim> tangent=top-fault.vertex(0);tangent/=tangent.norm();
          Tensor<1,dim> normal;normal[0]=-tangent[1];normal[1]=tangent[0];
          for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const auto offset=cell->center()-top;
                const double xd=-offset*tangent,z=offset*normal,h=cell->diameter()/std::sqrt(2.);
                // Include intact material and the entire identical integration
                // window; do not restrict velocity export to associated QPs.
                if (xd+h<12000. || xd-h>21000. || std::abs(z)>1600.+h) continue;
                fe.reinit(cell);fe[sim.introspection().extractors.velocities].get_function_values(direction,velocity);
                out<<mode<<','<<cell->id()<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]<<','<<h;
                for (const auto &u:velocity) out<<','<<u[0]<<','<<u[1];
                out<<'\n';
              }
        }
    }
  }
}
#endif
