/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/simulator/solver/dg_stress_transport.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>

namespace aspect
{
  namespace Transport
  {
    namespace
    {
      using Coefficients = CellStress::mapped_type;

      struct Inflow
      {
        Point<2> position;
        Vector<double> weights;
      };

      // Blocks of the spatial action on one owned cell, including M_K^{-1}.
      struct CellOperator
      {
        FullMatrix<double> self;
        std::vector<std::pair<CellId,FullMatrix<double>>> upstream;
        std::vector<Inflow> inflow;
      };
    }

    CellStress advect_stress(const SimulatorAccess<2> &sim,
                             const std::array<unsigned int,3> &fields,
                             const LinearAlgebra::BlockVector &accepted,
                             Function<2> &exterior,
                             const double start_time, const double duration)
    {
      AssertThrow(std::isfinite(start_time) && std::isfinite(duration) && duration>=0.,
                  ExcMessage("DG stress transport requires a finite clock and nonnegative duration."));
      const auto &intro = sim.introspection();
      const auto &system = sim.get_fe();
      const auto &dg = system.base_element(intro.base_elements.compositional_fields[fields[0]]);
      const auto &velocity = intro.extractors.velocities;
      const unsigned int n = dg.n_dofs_per_cell();
      const QGauss<2> rule(8);
      const QGauss<1> face_rule(8);
      FEValues<2> flow(sim.get_mapping(),system,rule,update_values | update_gradients);
      FEValues<2> fe(sim.get_mapping(),dg,rule,update_values | update_gradients | update_JxW_values);
      FEFaceValues<2> face_flow(sim.get_mapping(),system,face_rule,update_values);
      FEFaceValues<2> face(sim.get_mapping(),dg,face_rule,update_values | update_normal_vectors |
                           update_quadrature_points | update_JxW_values);
      std::vector<Tensor<1,2>> u(rule.size()), face_u(face_rule.size());
      std::vector<double> divergence(rule.size());
      std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
      CellStress initial;
      std::map<CellId,CellOperator> operations;
      double inverse_time = 0.;
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            const CellId id = cell->id();
            cell->get_dof_indices(indices);
            auto &coefficients = initial[id];
            coefficients.resize(n);
            for (unsigned int c=0; c<fields.size(); ++c)
              for (unsigned int i=0; i<n; ++i)
                coefficients[i].access_raw_entry(c) =
                  accepted[indices[system.component_to_system_index(intro.component_indices.compositional_fields[fields[c]],i)]];

            const Triangulation<2>::active_cell_iterator mesh_cell = cell;
            flow.reinit(cell);
            fe.reinit(mesh_cell);
            flow[velocity].get_function_values(accepted,u);
            flow[velocity].get_function_divergences(accepted,divergence);
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
                      inverse_mass(i,j) += fe.shape_value(i,q)*trial;
                      local_action(i,j) += (u[q]*fe.shape_grad(i,q) + divergence[q]*fe.shape_value(i,q))*trial;
                    }
              }
            inverse_mass.gauss_jordan();
            auto &op = operations[id];
            for (const unsigned int f : cell->face_indices())
              {
                AssertThrow(!cell->has_periodic_neighbor(f) &&
                            (cell->at_boundary(f) || (cell->neighbor(f)->is_active() && cell->neighbor(f)->level()==cell->level())),
                            ExcMessage("DG stress transport requires conforming, nonperiodic Cartesian box faces."));
                face.reinit(mesh_cell,f);
                face_flow.reinit(cell,f);
                face_flow[velocity].get_function_values(accepted,face_u);
                FullMatrix<double> upstream(n,n);
                bool receives_from_neighbor = false;
                for (unsigned int q=0; q<face_rule.size(); ++q)
                  {
                    for (unsigned int d=0; d<2; ++d)
                      maximum[d] = std::max(maximum[d],std::abs(face_u[q][d]));
                    const double flux = face_u[q]*face.normal_vector(q);
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
                        op.inflow.push_back({point,Vector<double>(n)});
                        inverse_mass.vmult(op.inflow.back().weights,test);
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
                if (receives_from_neighbor)
                  {
                    op.upstream.emplace_back(cell->neighbor(f)->id(),FullMatrix<double>(n,n));
                    inverse_mass.mmult(op.upstream.back().second,upstream);
                  }
              }
            op.self.reinit(n,n);
            inverse_mass.mmult(op.self,local_action);
            double cell_inverse_time = 0.;
            for (unsigned int d=0; d<2; ++d)
              cell_inverse_time += maximum[d]/(cell->vertex(1u<<d)[d]-cell->vertex(0)[d]);
            inverse_time = std::max(inverse_time,cell_inverse_time);
          }

      inverse_time = Utilities::MPI::max(inverse_time,sim.get_mpi_communicator());
      if (inverse_time==0. || duration==0.)
        return initial;
      const double count = std::ceil(duration*inverse_time/.1);
      AssertThrow(count<=4096.,ExcMessage("DG stress transport requires a smaller step (4096 CFL substeps exceeded)."));
      const unsigned int steps = std::max(1u,static_cast<unsigned int>(count));
      const double dt = duration/steps;
      const auto stage = [&](CellStress &current, const double time, const double fraction)
      {
        // Only owned cells have rows. Exchange this stage's coefficients before
        // applying off-diagonal blocks on partition faces.
        const auto pack = [&](const auto &cell)
        {
          return current.at(cell->id());
        };
        const auto unpack = [&](const auto &cell,const Coefficients &data)
        {
          current[cell->id()] = data;
        };
        GridTools::exchange_cell_data_to_ghosts<Coefficients,DoFHandler<2>>(sim.get_dof_handler(),pack,unpack);
        exterior.set_time(time);
        CellStress next;
        Vector<double> value(3);
        for (const auto &[id,op] : operations)
          {
            Coefficients rhs(n);
            const auto apply = [&](const FullMatrix<double> &block,const Coefficients &source)
            {
              for (unsigned int i=0; i<n; ++i)
                for (unsigned int j=0; j<n; ++j)
                  rhs[i] += block(i,j)*source[j];
            };
            const auto &local = current.at(id);
            apply(op.self,local);
            for (const auto &[neighbor,block] : op.upstream)
              apply(block,current.at(neighbor));
            for (const auto &load : op.inflow)
              {
                exterior.vector_value(load.position,value);
                for (unsigned int i=0; i<n; ++i)
                  for (unsigned int c=0; c<3; ++c)
                    rhs[i].access_raw_entry(c) += load.weights[i]*value[c];
              }
            const auto &base = initial.at(id);
            auto &result = next[id];
            result.resize(n);
            for (unsigned int i=0; i<n; ++i)
              result[i] = base[i]+fraction*((local[i]-base[i])+dt*rhs[i]);
          }
        return next;
      };
      for (unsigned int k=0; k<steps; ++k)
        {
          const double t = start_time+k*dt;
          auto trial = stage(initial,t,1.);
          trial = stage(trial,t+dt,.25);
          initial = stage(trial,t+.5*dt,2./3.);
        }
      return initial;
    }
  }
}
