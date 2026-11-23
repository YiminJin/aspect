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

      double &entry(double &value, unsigned int)
      {
        return value;
      }
      double &entry(SymmetricTensor<2,2> &value, unsigned int c)
      {
        return value.access_raw_entry(c);
      }

      struct Exterior
      {
        std::function<void(types::boundary_id,const Point<2>&,double,Vector<double>&)> evaluate;
        const char *diagnostic;
      };

      struct Inflow
      {
        types::boundary_id boundary;
        Point<2> position;
        Vector<double> weights;
      };

      // Blocks of the spatial action on one owned cell, including M_K^{-1}.
      struct CellOperator
      {
        Vector<double> mean_weights;
        FullMatrix<double> self;
        std::vector<std::pair<CellId,FullMatrix<double>>> upstream;
        std::vector<Inflow> inflow;
      };

      struct SpatialOperator
      {
        std::map<CellId,CellOperator> cells;
        double inverse_time = 0.;
      };

      template <typename Value, std::size_t N>
      Field<Value> read_fields(const SimulatorAccess<2> &sim, const std::array<unsigned int,N> &fields,
                               const LinearAlgebra::BlockVector &accepted, const unsigned int n)
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
                  entry(values[i],c) =
                    accepted[indices[system.component_to_system_index(components[fields[c]],i)]];
            }
        return result;
      }

      SpatialOperator assemble(const SimulatorAccess<2> &sim, const FiniteElement<2> &dg,
                               const LinearAlgebra::BlockVector &accepted, const bool bounded)
      {
        const auto &system = sim.get_fe();
        const auto &velocity = sim.introspection().extractors.velocities;
        const unsigned int n = dg.n_dofs_per_cell();
        const Quadrature<2> rule = bounded ? Quadrature<2>(QGaussLobatto<2>(5)) : Quadrature<2>(QGauss<2>(8));
        const Quadrature<1> face_rule = bounded ? Quadrature<1>(QGaussLobatto<1>(5)) : Quadrature<1>(QGauss<1>(8));
        std::unique_ptr<FEValues<2>> mass_fe;
        if (bounded)
          mass_fe = std::make_unique<FEValues<2>>(sim.get_mapping(),dg,QGaussLobatto<2>(4),update_values | update_JxW_values);
        FEValues<2> flow(sim.get_mapping(),system,rule,update_values | update_gradients);
        FEValues<2> fe(sim.get_mapping(),dg,rule,update_values | update_gradients | update_JxW_values);
        FEFaceValues<2> face_flow(sim.get_mapping(),system,face_rule,update_values);
        FEFaceValues<2> face(sim.get_mapping(),dg,face_rule,update_values | update_normal_vectors |
                             update_quadrature_points | update_JxW_values);
        std::vector<Tensor<1,2>> u(rule.size()), face_u(face_rule.size());
        std::vector<double> divergence(rule.size());
        SpatialOperator op;
        for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              const Triangulation<2>::active_cell_iterator mesh_cell = cell;
              flow.reinit(cell);
              fe.reinit(mesh_cell);
              flow[velocity].get_function_values(accepted,u);
              flow[velocity].get_function_divergences(accepted,divergence);
              auto &row = op.cells[cell->id()];
              double mean_loss = 0.;
              Tensor<1,2> maximum;
              FullMatrix<double> inverse_mass(n,n), local_action(n,n);
              for (unsigned int q=0; q<rule.size(); ++q)
                {
                  for (unsigned int d=0; d<2; ++d)
                    maximum[d] = std::max(maximum[d],std::abs(u[q][d]));
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
                  mean_loss = std::max(0.,-*std::min_element(divergence.begin(),divergence.end()));
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
                      for (unsigned int d=0; d<2; ++d)
                        maximum[d] = std::max(maximum[d],std::abs(face_u[q][d]));
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
                    mean_loss += 6.*max_outflow/cell->measure();
                  if (receives_from_neighbor)
                    {
                      row.upstream.emplace_back(cell->neighbor(f)->id(),FullMatrix<double>(n,n));
                      inverse_mass.mmult(row.upstream.back().second,upstream);
                    }
                }
              row.self.reinit(n,n);
              inverse_mass.mmult(row.self,local_action);
              double cell_inverse_time = 0.;
              for (unsigned int d=0; d<2; ++d)
                cell_inverse_time += maximum[d]/(cell->vertex(1u<<d)[d]-cell->vertex(0)[d]);
              op.inverse_time = std::max(op.inverse_time,bounded ? mean_loss : cell_inverse_time);
            }

        op.inverse_time = Utilities::MPI::max(op.inverse_time,sim.get_mpi_communicator());
        return op;
      }

      bool limit_cell(std::vector<double> &values, const Vector<double> &weights)
      {
        if (!std::all_of(values.begin(),values.end(),[](double v)
        {
          return std::isfinite(v);
          }))
        return false;
        const double mean=std::inner_product(values.begin(),values.end(),weights.begin(),0.);
        return PhaseField::limit_bernstein(values,mean);
      }

      // The substep parent and current stage contain only owned coefficients.
      // The halo belongs to this evaluation, even when current aliases parent.
      template <typename Value, std::size_t N>
      Field<Value> rk_stage(const SimulatorAccess<2> &sim, const SpatialOperator &op, const Field<Value> &parent,
                            const Field<Value> &current, const Exterior &exterior, const std::array<std::string,N> &names, const double time,
                            const double dt, const double fraction, const unsigned int stage_number)
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
        GridTools::exchange_cell_data_to_ghosts<Coefficients,DoFHandler<2>>(sim.get_dof_handler(),pack,unpack);
        Field<Value> next;
        Vector<double> value(N);
        // Count affected ranks per component, before any rank can start a new exchange.
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
                      entry(rhs[i],c) += load.weights[i]*value[c];
                  else
                    invalid[c] = 1;
              }
            const auto &base = parent.at(id);
            auto &result = next[id];
            result.resize(n);
            for (unsigned int i=0; i<n; ++i)
              {
                result[i] = base[i]+fraction*((local[i]-base[i])+dt*rhs[i]);
                for (unsigned int c=0; c<N; ++c)
                  if (!std::isfinite(entry(result[i],c)))
                    invalid[N+c] = 1;
              }
            if constexpr (N==1)
              if (row.mean_weights.size() && !limit_cell(result,row.mean_weights))
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

      template <typename Value, std::size_t N>
      Field<Value> advance(const SimulatorAccess<2> &sim, const SpatialOperator &op, Field<Value> parent,
                           const Exterior &exterior, const std::array<std::string,N> &names,
                           const double start_time, const double duration, const bool bounded)
      {
        if constexpr (N==1)
          if (bounded)
            {
              bool valid=true;
              for (auto &[id,coefficients] : parent)
                if (!limit_cell(coefficients,op.cells.at(id).mean_weights))
                  valid=false;
              const auto comm=sim.get_mpi_communicator();
              const auto bad_ranks=Utilities::MPI::sum(static_cast<unsigned int>(!valid),comm);
              AssertThrow(bad_ranks==0,ExcMessage("Inadmissible Bernstein phase mean at transport entry on "+
                                                  std::to_string(bad_ranks)+" of "+std::to_string(Utilities::MPI::n_mpi_processes(comm))+" ranks."));
            }
        if (op.inverse_time==0. || duration==0.)
          return parent;
        const double count = std::ceil(duration*op.inverse_time/(bounded ? .8 : .1));
        AssertThrow(count<=(bounded ? 65536. : 4096.),ExcMessage("Explicit DG transport exceeds its CFL substep budget."));
        const unsigned int steps = std::max(1u,static_cast<unsigned int>(count));
        const double dt = duration/steps;
        for (unsigned int k=0; k<steps; ++k)
          {
            const double t = start_time+k*dt;
            auto trial = rk_stage(sim,op,parent,parent,exterior,names,t,dt,1.,1);
            trial = rk_stage(sim,op,parent,trial,exterior,names,t+dt,dt,.25,2);
            parent = rk_stage(sim,op,parent,trial,exterior,names,t+.5*dt,dt,2./3.,3);
          }
        return parent;
      }
    }

    CellStress advect_stress(const SimulatorAccess<2> &sim, const std::array<unsigned int,3> &fields,
                             const LinearAlgebra::BlockVector &accepted, Function<2> &exterior,
                             const double start_time, const double duration)
    {
      AssertThrow(std::isfinite(start_time) && std::isfinite(duration) && duration>=0.,
                  ExcMessage("DG stress transport requires a finite clock and nonnegative duration."));
      const auto &dg = sim.get_fe().base_element(sim.introspection().base_elements.compositional_fields[fields[0]]);
      const auto op = assemble(sim,dg,accepted,false);
      const Exterior boundary = { [&](types::boundary_id,const Point<2> &point,double time,Vector<double> &values)
      {
        if (time!=exterior.get_time())
          exterior.set_time(time);
        exterior.vector_value(point,values);
      }, "Nonfinite exterior parent stress "
                                };
      return advance(sim,op,read_fields<SymmetricTensor<2,2>>(sim,fields,accepted,dg.n_dofs_per_cell()),
      boundary,std::array<std::string,3> {{"xx","yy","xy"}},start_time,duration,false);
    }

    CellScalar advect_scalar(const SimulatorAccess<2> &sim, const unsigned int field,
                             const ScalarInflow &inflow, const bool bounded)
    {
      const auto &accepted=sim.get_old_solution();
      const auto &dg=sim.get_fe().base_element(sim.introspection().base_elements.compositional_fields[field]);
      AssertThrow(dg.degree==2 && dynamic_cast<const FE_DGQ<2>*>(&dg)!=nullptr &&
                  (dynamic_cast<const FE_DGQBernstein<2>*>(&dg)!=nullptr)==bounded,
                  ExcMessage("Explicit scalar transport requires the selected nodal or Bernstein DGQ2 basis."));
      const auto op=assemble(sim,dg,accepted,bounded);
      const Exterior boundary = { [&](types::boundary_id id,const Point<2> &point,double time,Vector<double> &values)
      {
        values[0]=inflow(id,point,time);
      }, "Invalid exterior composition "
                                };
      return advance(sim,op,read_fields<double>(sim,std::array<unsigned int,1> {{field}},accepted,dg.n_dofs_per_cell()),
      boundary,std::array<std::string,1> {{sim.introspection().name_for_compositional_index(field)}},
      sim.get_time()-sim.get_timestep(),sim.get_timestep(),bounded);
    }
  }
}
