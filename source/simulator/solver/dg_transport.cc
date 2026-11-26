/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/simulator/solver/dg_transport.h>
#include <aspect/phase_field/bernstein_bounds.h>
#include <aspect/fe_dgq_bernstein.h>
#include <numeric>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>

#ifdef ASPECT_USE_FP_EXCEPTIONS
#include <deal.II/base/scope_exit.h>
#include <cfenv>
#endif
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace Transport
  {
    namespace
    {
      template <typename Value>
      using Field = std::map<CellId,std::vector<Value>>;

      double &component_entry(double &value, unsigned int)
      {
        return value;
      }
      template <int tensor_dim>
      double &component_entry(SymmetricTensor<2,tensor_dim> &value, unsigned int c)
      {
        return value.access_raw_entry(c);
      }

      template <int dim>
      struct Exterior
      {
        std::function<void(types::boundary_id,const Point<dim>&,double,Vector<double>&)> evaluate;
        const char *diagnostic;
      };

      template <int dim>
      struct Inflow
      {
        types::boundary_id boundary;
        Point<dim> position;
        Vector<double> weights;
      };

      // Blocks of the spatial action on one owned cell, including M_K^{-1}.
      template <int dim>
      struct CellOperator
      {
        Vector<double> mean_weights;
        FullMatrix<double> self;
        std::vector<std::pair<CellId,FullMatrix<double>>> upstream;
        std::vector<Inflow<dim>> inflow;
      };

      template <int dim>
      struct SpatialOperator
      {
        std::map<CellId,CellOperator<dim>> cells;
        /** Global maximum of the two CFL rates including their safety factors (1/s). */
        double maximum_inverse_substep = 0.;
      };

      template <typename Value, std::size_t N, int dim>
      Field<Value> read_owned_coefficients(const SimulatorAccess<dim>       &sim,
                                           const std::array<unsigned int,N> &fields,
                                           const LinearAlgebra::BlockVector &accepted,
                                           const unsigned int                n)
      {
        Field<Value> result;
        const auto &system = sim.get_fe();
        const auto &components = sim.introspection().component_indices.compositional_fields;
        std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
        for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(indices);
              auto &values = result[cell->id()];
              values.resize(n);
              for (unsigned int c=0; c<fields.size(); ++c)
                for (unsigned int i=0; i<n; ++i)
                  component_entry(values[i],c) =
                    accepted[indices[system.component_to_system_index(components[fields[c]],i)]];
            }
        return result;
      }

      template <int dim>
      SpatialOperator<dim> assemble_spatial_operator(const SimulatorAccess<dim>       &sim,
                                                     const FiniteElement<dim>         &dg,
                                                     const LinearAlgebra::BlockVector &accepted,
                                                     const bool                        bounded)
      {
        const auto &system = sim.get_fe();
        const auto &velocity = sim.introspection().extractors.velocities;
        const unsigned int n = dg.n_dofs_per_cell();
        const Quadrature<dim> rule = bounded ? Quadrature<dim>(QGaussLobatto<dim>(5)) : Quadrature<dim>(QGauss<dim>(8));
        const Quadrature<dim-1> face_rule = bounded ? Quadrature<dim-1>(QGaussLobatto<dim-1>(5)) : Quadrature<dim-1>(QGauss<dim-1>(8));
        std::unique_ptr<FEValues<dim>> mass_fe;
        if (bounded)
          mass_fe = std::make_unique<FEValues<dim>>(sim.get_mapping(),dg,QGaussLobatto<dim>(4),update_values | update_JxW_values);
        FEValues<dim> flow(sim.get_mapping(),system,rule,update_values | update_gradients);
        FEValues<dim> fe(sim.get_mapping(),dg,rule,update_values | update_gradients | update_JxW_values);
        FEFaceValues<dim> face_flow(sim.get_mapping(),system,face_rule,update_values);
        FEFaceValues<dim> face(sim.get_mapping(),dg,face_rule,update_values | update_normal_vectors |
                               update_quadrature_points | update_JxW_values);
        std::vector<Tensor<1,dim>> u(rule.size()), face_u(face_rule.size());
        std::vector<double> divergence(rule.size());
        SpatialOperator<dim> op;
        for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const typename Triangulation<dim>::active_cell_iterator mesh_cell = cell;
              flow.reinit(cell);
              fe.reinit(mesh_cell);
              flow[velocity].get_function_values(accepted,u);
              flow[velocity].get_function_divergences(accepted,divergence);
              auto &row = op.cells[cell->id()];
              double mean_loss_rate = 0.;
              Tensor<1,dim> max_speed;
              FullMatrix<double> inverse_mass(n,n), local_action(n,n);
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  for (unsigned int d=0; d<dim; ++d)
                    max_speed[d] = std::max(max_speed[d],std::abs(u[q][d]));
                  // Integrate -u.grad(s) by parts. Discrete incompressibility
                  // does not imply pointwise zero divergence.
                  for (unsigned int i=0; i<n; ++i)
                    for (unsigned int j=0; j<n; ++j)
                      {
                        const double trial = fe.shape_value(j,q)*fe.JxW(q);
                        if (!bounded)
                          inverse_mass(i,j) += fe.shape_value(i,q)*trial;
                        local_action(i,j) += (u[q]*fe.shape_grad(i,q) + divergence[q]*fe.shape_value(i,q))*trial;
                      }
                }
              if (bounded)
                {
                  // Exact physical mass and cell mean on the Cartesian Bernstein element.
                  mass_fe->reinit(mesh_cell);
                  row.mean_weights.reinit(n);
                  for (unsigned int q=0; q<mass_fe->n_quadrature_points; ++q)
                    for (unsigned int i=0; i<n; ++i)
                      {
                        const double w=mass_fe->shape_value(i,q)*mass_fe->JxW(q);
                        row.mean_weights[i]+=w;
                        for (unsigned int j=0; j<n; ++j)
                          inverse_mass(i,j)+=w*mass_fe->shape_value(j,q);
                      }
                  row.mean_weights /= cell->measure();
                  mean_loss_rate = std::max(0.,-*std::min_element(divergence.begin(),divergence.end()));
                }
              inverse_mass.gauss_jordan();
              for (const unsigned int f : cell->face_indices())
                {
                  AssertThrow(!cell->has_periodic_neighbor(f) &&
                              (cell->at_boundary(f) || (cell->neighbor(f)->is_active() && cell->neighbor(f)->level()==cell->level())),
                              ExcMessage("Explicit DG transport requires conforming, nonperiodic Cartesian box faces."));
                  face.reinit(mesh_cell,f);
                  face_flow.reinit(cell,f);
                  face_flow[velocity].get_function_values(accepted,face_u);
                  FullMatrix<double> upstream(n,n);
                  bool receives_from_neighbor = false;
                  double max_outflow = 0.;
                  for (unsigned int q=0; q<face_rule.size(); ++q)
                    {
                      for (unsigned int d=0; d<dim; ++d)
                        max_speed[d] = std::max(max_speed[d],std::abs(face_u[q][d]));
                      const double flux = face_u[q]*face.normal_vector(q);
                      if (bounded)
                        max_outflow = std::max(max_outflow,flux*face.JxW(q)/face_rule.weight(q));
                      const auto &point = face.quadrature_point(q);
                      Vector<double> test(n);
                      for (unsigned int i=0; i<n; ++i)
                        test[i] = -flux*face.JxW(q)*face.shape_value(i,q);
                      if (flux >= 0.)
                        for (unsigned int i=0; i<n; ++i)
                          for (unsigned int j=0; j<n; ++j)
                            local_action(i,j) += test[i]*face.shape_value(j,q);
                      else if (cell->at_boundary(f))
                        {
                          row.inflow.push_back({cell->face(f)->boundary_id(),point,Vector<double>(n)});
                          inverse_mass.vmult(row.inflow.back().weights,test);
                        }
                      else
                        {
                          const auto unit = sim.get_mapping().transform_real_to_unit_cell(cell->neighbor(f),point);
                          receives_from_neighbor = true;
                          for (unsigned int j=0; j<n; ++j)
                            {
                              const double trace = dg.shape_value(j,unit);
                              for (unsigned int i=0; i<n; ++i)
                                upstream(i,j) += test[i]*trace;
                            }
                        }
                    }
                  // Lobatto Q2 face trace bound: outward loss <= 6 J_face vn / J_cell.
                  if (bounded)
                    mean_loss_rate += 6.*max_outflow/cell->measure();
                  if (receives_from_neighbor)
                    {
                      row.upstream.emplace_back(cell->neighbor(f)->id(),FullMatrix<double>(n,n));
                      inverse_mass.mmult(row.upstream.back().second,upstream);
                    }
                }
              row.self.reinit(n,n);
              inverse_mass.mmult(row.self,local_action);
              double interior_speed_over_width = 0.;
              for (unsigned int d=0; d<dim; ++d)
                interior_speed_over_width += max_speed[d]/(cell->vertex(1u<<d)[d]-cell->vertex(0)[d]);
              // Even a solenoidal flow tangent to every face transports cell
              // moments. The mean-loss bound alone cannot resolve this motion.
              op.maximum_inverse_substep = std::max(op.maximum_inverse_substep,interior_speed_over_width/.1);
              if (bounded)
                op.maximum_inverse_substep = std::max(op.maximum_inverse_substep,mean_loss_rate/.8);
            }

        op.maximum_inverse_substep = Utilities::MPI::max(op.maximum_inverse_substep,sim.get_mpi_communicator());
        return op;
      }

      template <int dim>
      bool limit_cell_about_mean(std::vector<double> &values, const Vector<double> &weights)
      {
        const auto is_finite = [](const double value)
        {
          return std::isfinite(value);
        };
        if (!std::all_of(values.begin(), values.end(), is_finite))
          return false;
        const double mean=std::inner_product(values.begin(),values.end(),weights.begin(),0.);
        return PhaseField::limit_bernstein<dim>(values,mean);
      }

      // The substep parent and current stage contain only owned coefficients.
      // The halo belongs to this evaluation, even when current aliases parent.
      template <typename Value, std::size_t N, int dim>
      Field<Value> advance_rk_stage(const SimulatorAccess<dim>      &sim,
                                    const SpatialOperator<dim>      &op,
                                    const Field<Value>              &parent,
                                    const Field<Value>              &current,
                                    const Exterior<dim>             &exterior,
                                    const std::array<std::string,N> &names,
                                    const double                     time,
                                    const double                     dt,
                                    const double                     euler_weight,
                                    const unsigned int               stage_number)
      {
        using Coefficients = std::vector<Value>;
        Field<Value> halo;
        // Only owned cells have rows. Exchange this stage's coefficients before
        // applying off-diagonal blocks on partition faces.
        const auto pack = [&](const auto &cell)
        {
          return current.at(cell->id());
        };
        const auto unpack = [&](const auto &cell,const Coefficients &data)
        {
          halo.emplace(cell->id(),data);
        };
        GridTools::exchange_cell_data_to_ghosts<Coefficients,DoFHandler<dim>>(sim.get_dof_handler(),pack,unpack);
        Field<Value> next;
        Vector<double> value(N);
        // First N flags cover exterior data, the next N cover owned stage
        // coefficients. A bounded scalar also reports inadmissible cell means.
        constexpr unsigned int n_flags = 2*N+(N==1);
        unsigned int invalid[n_flags] = {};
#ifdef ASPECT_USE_FP_EXCEPTIONS
        // Collectively diagnose domain errors in the exterior function and
        // overflow in stage arithmetic even in builds trapping invalid FP.
        std::fenv_t environment;
        std::feholdexcept(&environment);
        ScopeExit restore_exceptions([&]()
        {
          std::fesetenv(&environment);
        });
#endif
        for (const auto &[id,row] : op.cells)
          {
            const unsigned int n = row.self.m();
            Coefficients rhs(n);
            const auto apply = [&](const FullMatrix<double> &block,const Coefficients &source)
            {
              for (unsigned int i=0; i<n; ++i)
                for (unsigned int j=0; j<n; ++j)
                  rhs[i] += block(i,j)*source[j];
            };
            const auto &local = current.at(id);
            apply(row.self,local);
            for (const auto &[neighbor,block] : row.upstream)
              {
                const auto owned = current.find(neighbor);
                apply(block,owned == current.end() ? halo.at(neighbor) : owned->second);
              }
            for (const auto &load : row.inflow)
              {
                exterior.evaluate(load.boundary,load.position,time,value);
                for (unsigned int c=0; c<N; ++c)
                  if (std::isfinite(value[c]) && (row.mean_weights.size()==0 || (value[c]>=0. && value[c]<=1.)))
                    for (unsigned int i=0; i<n; ++i)
                      component_entry(rhs[i],c) += load.weights[i]*value[c];
                  else
                    invalid[c] = 1;
              }
            const auto &base = parent.at(id);
            auto &result = next[id];
            result.resize(n);
            for (unsigned int i=0; i<n; ++i)
              {
                // Blend the forward Euler update of the current stage with
                // the same substep parent at every SSPRK3 stage.
                result[i] = base[i] + euler_weight * ((local[i]-base[i]) + dt*rhs[i]);
                for (unsigned int c=0; c<N; ++c)
                  if (!std::isfinite(component_entry(result[i],c)))
                    invalid[N+c] = 1;
              }
            if constexpr (N==1)
              if (row.mean_weights.size() && !limit_cell_about_mean<dim>(result,row.mean_weights))
                invalid[2*N]=1;
          }

        unsigned int counts[n_flags];
        const auto comm = sim.get_mpi_communicator();
        Utilities::MPI::sum(invalid,comm,counts);
        for (unsigned int i=0; i<n_flags; ++i)
          if (counts[i])
            {
              std::ostringstream message;
              if (i==2*N)
                message << "Inadmissible Bernstein phase mean";
              else
                message << (i<N ? exterior.diagnostic :
                            "Nonfinite owned DG stage coefficient ") << names[i%N];
              message << " on " << counts[i] << " of " << Utilities::MPI::n_mpi_processes(comm)
                      << " ranks at SSPRK stage " << stage_number << ", time " << std::setprecision(17) << time << " s.";
              AssertThrow(false,ExcMessage(message.str()));
            }
        return next;
      }

      template <typename Value, std::size_t N, int dim>
      Field<Value> advance(const SimulatorAccess<dim>      &sim,
                           const SpatialOperator<dim>      &op,
                           Field<Value>                     parent,
                           const Exterior<dim>             &exterior,
                           const std::array<std::string,N> &names,
                           const double                     start_time,
                           const double                     duration,
                           const bool                       bounded)
      {
        if constexpr (N==1)
          if (bounded)
            {
              bool valid=true;
              for (auto &[id,coefficients] : parent)
                if (!limit_cell_about_mean<dim>(coefficients,op.cells.at(id).mean_weights))
                  valid=false;
              const auto comm=sim.get_mpi_communicator();
              const auto bad_ranks=Utilities::MPI::sum(static_cast<unsigned int>(!valid),comm);
              AssertThrow(bad_ranks==0,ExcMessage("Inadmissible Bernstein phase mean at transport entry on "+
                                                  std::to_string(bad_ranks)+" of "+std::to_string(Utilities::MPI::n_mpi_processes(comm))+" ranks."));
            }
        if (op.maximum_inverse_substep==0. || duration==0.)
          return parent;
        const double count = std::ceil(duration*op.maximum_inverse_substep);
        AssertThrow(count<=(bounded ? 65536. : 4096.),ExcMessage("Explicit DG transport exceeds its CFL substep budget."));
        const unsigned int steps = std::max(1u,static_cast<unsigned int>(count));
        const double dt = duration/steps;
        for (unsigned int k=0; k<steps; ++k)
          {
            const double t = start_time+k*dt;
            auto trial = advance_rk_stage(sim,op,parent,parent,exterior,names,t,dt,1.,1);
            trial = advance_rk_stage(sim,op,parent,trial,exterior,names,t+dt,dt,.25,2);
            parent = advance_rk_stage(sim,op,parent,trial,exterior,names,t+.5*dt,dt,2./3.,3);
          }
        return parent;
      }
    }

    CellStress advect_stress(const SimulatorAccess<2>         &sim,
                             const std::array<unsigned int,3> &fields,
                             const LinearAlgebra::BlockVector &accepted,
                             Function<2>                      &exterior,
                             const double                      start_time,
                             const double                      duration)
    {
      AssertThrow(std::isfinite(start_time) && std::isfinite(duration) && duration>=0.,
                  ExcMessage("DG stress transport requires a finite clock and nonnegative duration."));
      const auto &dg = sim.get_fe().base_element(sim.introspection().base_elements.compositional_fields[fields[0]]);
      const auto op = assemble_spatial_operator(sim,dg,accepted,false);
      const auto exterior_at_stage = [&](types::boundary_id, const Point<2> &point, double time, Vector<double> &values)
      {
        if (time!=exterior.get_time())
          exterior.set_time(time);
        exterior.vector_value(point,values);
      };
      const Exterior<2> boundary {exterior_at_stage, "Nonfinite exterior parent stress "};
      return advance(sim, op,
                     read_owned_coefficients<SymmetricTensor<2,2>>(sim, fields, accepted, dg.n_dofs_per_cell()),
      boundary, std::array<std::string,3> {{"xx","yy","xy"}}, start_time, duration, false);
    }

    template <int dim>
    CellScalar advect_scalar(const SimulatorAccess<dim> &sim,
                             const unsigned int          field,
                             const ScalarInflow<dim>    &inflow,
                             const bool                  bounded)
    {
      const auto &accepted = sim.get_old_solution();
      const auto &dg = sim.get_fe().base_element(sim.introspection().base_elements.compositional_fields[field]);
      AssertThrow(dg.degree==2 && dynamic_cast<const FE_DGQ<dim>*>(&dg)!=nullptr &&
                  (dynamic_cast<const FE_DGQBernstein<dim>*>(&dg)!=nullptr)==bounded,
                  ExcMessage("Explicit scalar transport requires the selected nodal or Bernstein DGQ2 basis."));
      const auto op = assemble_spatial_operator(sim,dg,accepted,bounded);
      const auto exterior_at_stage = [&](types::boundary_id id, const Point<dim> &point, double time, Vector<double> &values)
      {
        values[0]=inflow(id,point,time);
      };
      const Exterior<dim> boundary {exterior_at_stage, "Invalid exterior composition "};
      return advance(sim, op,
      read_owned_coefficients<double>(sim, std::array<unsigned int,1> {{field}}, accepted, dg.n_dofs_per_cell()),
      boundary, std::array<std::string,1> {{sim.introspection().name_for_compositional_index(field)}},
      sim.get_time()-sim.get_timestep(), sim.get_timestep(), bounded);
    }
    template CellScalar advect_scalar(const SimulatorAccess<2> &, unsigned int, const ScalarInflow<2> &, bool);
    template CellScalar advect_scalar(const SimulatorAccess<3> &, unsigned int, const ScalarInflow<3> &, bool);
  }
}
