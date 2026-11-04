/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_bulk_history.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace CohesiveInterface
  {
    template <int dim>
    BulkHistory<dim>::BulkHistory (const Simulator<dim> &sim, const double modulus,
                                   const double viscosity)
        : G (modulus), eta (viscosity), element (2)
    {
      this->initialize_simulator (sim);
      AssertThrow (dim == 2 && G > 0 && eta > 0
                       && this->get_parameters ().stokes_velocity_degree == 2,
                   ExcMessage ("Exact-jump Maxwell history requires 2-D Q2 native velocity and "
                               "positive constant G/eta."));
      native.reinit (this->get_solution ());
      native = 0.;
      for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
        if (cell->is_locally_owned ())
          {
            cells.emplace (cell->id (), cell);
            auto &points = vertices[cell->id ()];
            for (const auto v : cell->vertex_indices ())
              points.push_back (cell->vertex (v));
          }
    }

    template <int dim>
    typename BulkHistory<dim>::Tensor
    BulkHistory<dim>::dev3 (const SymmetricTensor<2, dim> &strain)
    {
      Tensor result;
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = i; j < dim; ++j)
          result[i][j] = strain[i][j];
      const double tr = trace (result) / 3.;
      for (unsigned int i = 0; i < 3; ++i)
        result[i][i] -= tr;
      return result;
    }

    template <int dim>
    void
    BulkHistory<dim>::check_layout () const
    {
      AssertThrow (tensor.size () == cells.size () && projection.size () == cells.size (),
                   ExcMessage ("Incomplete native Maxwell history."));
      for (const auto &entry : cells)
        {
          AssertThrow(!cut_geometry || cut_tensor.count(entry.first),
                      ExcMessage("Incomplete piece/side Maxwell tensor history."));
          AssertThrow (entry.second->is_locally_owned (),
                       ExcMessage ("Native history ownership changed without a transfer."));
          for (const auto v : entry.second->vertex_indices ())
            AssertThrow (entry.second->vertex (v) == vertices.at (entry.first)[v],
                         ExcMessage ("Exact-jump native history requires a fixed bulk mesh."));
        }
    }

    template <int dim>
    std::map<CellId, typename BulkHistory<dim>::Coefficients>
    BulkHistory<dim>::project_native (const LinearAlgebra::BlockVector &field) const
    {
      std::map<CellId, Coefficients> result;
      const QGauss<dim> rule (3);
      FEValues<dim> fe (this->get_mapping (), this->get_fe (), rule,
                        update_gradients | update_JxW_values);
      std::vector<SymmetricTensor<2, dim>> strain (rule.size ());
      for (const auto &entry : cells)
        {
          fe.reinit (entry.second);
          fe[this->introspection ().extractors.velocities].get_function_symmetric_gradients (
              field, strain);
          FullMatrix<double> mass (9, 9);
          std::array<Vector<double>, 3> rhs;
          for (auto &v : rhs)
            v.reinit (9);
          for (unsigned int q = 0; q < rule.size (); ++q)
            {
              const Tensor T = 2 * G * dev3 (strain[q]);
              const double components[3] = { T[0][0], T[1][1], T[0][1] };
              for (unsigned int i = 0; i < 9; ++i)
                {
                  const double w = fe.JxW (q) * element.shape_value (i, rule.point (q));
                  for (unsigned int j = 0; j < 9; ++j)
                    mass (i, j) += w * element.shape_value (j, rule.point (q));
                  for (unsigned int c = 0; c < 3; ++c)
                    rhs[c][i] += w * components[c];
                }
            }
          mass.gauss_jordan ();
          auto &coefficients = result[entry.first];
          for (unsigned int c = 0; c < 3; ++c)
            {
              Vector<double> value (9);
              mass.vmult (value, rhs[c]);
              for (unsigned int i = 0; i < 9; ++i)
                coefficients[9 * c + i] = value[i];
            }
        }
      return result;
    }

    template <int dim>
    void
    BulkHistory<dim>::initialize (const Function<dim> &prescribed,
                                  const LinearAlgebra::BlockVector *initial_native)
    {
      AssertThrow (
          tensor.empty () && prescribed.n_components == 3,
          ExcMessage (
              "Native Maxwell history initializes once from three physical tensor components."));
      if (initial_native)
        {
          LinearAlgebra::BlockVector owned (this->introspection ().index_sets.system_partitioning,
                                            this->get_mpi_communicator ());
          owned = 0.;
          const auto b = this->introspection ().block_indices.velocities;
          for (const auto i : owned.block (b).locally_owned_elements ())
            owned.block (b)[i] = initial_native->block (b)[i];
          owned.compress (VectorOperation::insert);
          native = owned;
        }
      projection = project_native (native);
      const QGauss<dim> rule (3);
      FEValues<dim> fe (this->get_mapping (), this->get_fe (), rule,
                        update_quadrature_points | update_JxW_values);
      for (const auto &entry : cells)
        {
          fe.reinit (entry.second);
          FullMatrix<double> mass (9, 9);
          std::array<Vector<double>, 3> rhs;
          for (auto &v : rhs)
            v.reinit (9);
          for (unsigned int q = 0; q < rule.size (); ++q)
            for (unsigned int i = 0; i < 9; ++i)
              {
                const double w = fe.JxW (q) * element.shape_value (i, rule.point (q));
                for (unsigned int j = 0; j < 9; ++j)
                  mass (i, j) += w * element.shape_value (j, rule.point (q));
                for (unsigned int c = 0; c < 3; ++c)
                  rhs[c][i] += w * prescribed.value (fe.quadrature_point (q), c);
              }
          mass.gauss_jordan ();
          auto &coefficients = tensor[entry.first];
          for (unsigned int c = 0; c < 3; ++c)
            {
              Vector<double> value (9);
              mass.vmult (value, rhs[c]);
              for (unsigned int i = 0; i < 9; ++i)
                coefficients[9 * c + i] = value[i];
            }
        }
      check_layout ();
    }

    template <int dim>
    void
    BulkHistory<dim>::initialize_native (const LinearAlgebra::BlockVector &field,
                                         const Function<dim> &independent)
    {
      initialize (independent, &field);
      for (auto &entry : tensor)
        for (unsigned int i = 0; i < 27; ++i)
          entry.second[i] += projection.at (entry.first)[i];
    }

    template <int dim>
    std::vector<typename BulkHistory<dim>::Tensor>
    BulkHistory<dim>::evaluate (const Cell &cell, const Quadrature<dim> &rule) const
    {
      return evaluate_sided(cell,rule,0);
    }

    template <int dim>
    std::vector<typename BulkHistory<dim>::Tensor>
    BulkHistory<dim>::evaluate_sided(const Cell &cell,const Quadrature<dim> &rule,const int side) const
    {
      const auto &dg = tensor.at (cell->id ()), &pi = projection.at (cell->id ());
      FEValues<dim> fe (this->get_mapping (), this->get_fe (), rule, update_gradients);
      fe.reinit (cell);
      std::vector<SymmetricTensor<2, dim>> strain (rule.size ());
      fe[this->introspection ().extractors.velocities].get_function_symmetric_gradients (native,
                                                                                         strain);
      std::vector<Tensor> result (rule.size ());
      for (unsigned int q = 0; q < rule.size (); ++q)
        {
          double t[3] = {};
          for (unsigned int c = 0; c < 3; ++c)
            for (unsigned int i = 0; i < 9; ++i)
              t[c] += (dg[9 * c + i] - pi[9 * c + i]) * element.shape_value (i, rule.point (q));
          if(cut_geometry)
            {
              const auto x=this->get_mapping().transform_unit_to_real_cell(cell,rule.point(q));
              const auto value=cut_value(cell->id(),rule.point(q),x,side);
              t[0]=value[0][0];t[1]=value[1][1];t[2]=value[0][1];
              for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)
                t[c]-=pi[9*c+i]*element.shape_value(i,rule.point(q));
            }
          result[q] = 2 * G * dev3 (strain[q]);
          result[q][0][0] += t[0];
          result[q][1][1] += t[1];
          result[q][0][1] += t[2];
          result[q][2][2] -= t[0] + t[1];
        }
      return result;
    }

    template <int dim>
    std::shared_ptr<BulkHistory<dim>>
    BulkHistory<dim>::rotated_anchor(const PhysicalField &physical,const GradientField &gradient,
                                    const double dt,const unsigned int order,
                                    const std::function<Quadrature<dim>(const Cell &)> &quadrature) const
    {
      AssertThrow(dim==2 && dt>0. && std::isfinite(dt) && order>=8,
                  ExcMessage("Projected complete cohesive rotation requires plane strain, positive dt and Gauss8."));
      check_layout();
      auto result=std::make_shared<BulkHistory>(this->get_simulator(),G,eta);
      result->tensor=tensor;result->native=native;result->projection=projection;
      for(const auto &entry:cells)
        {
          const Quadrature<dim> rule=quadrature ? quadrature(entry.second) : QGauss<dim>(order);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_JxW_values);
          fe.reinit(entry.second);
          const auto tau=physical(entry.second,rule);
          const auto L=gradient(entry.second,rule);
          AssertDimension(tau.size(),rule.size());AssertDimension(L.size(),rule.size());
          FullMatrix<double> mass(9,9);
          std::array<Vector<double>,3> rhs;
          for(auto &v:rhs)v.reinit(9);
          for(unsigned int q=0;q<rule.size();++q)
            {
              SymmetricTensor<2,2> in_plane;
              in_plane[0][0]=tau[q][0][0];in_plane[1][1]=tau[q][1][1];in_plane[0][1]=tau[q][0][1];
              SymmetricTensor<2,2> change;
              if constexpr(dim==2)
                change=ReconstructedFaultUtilities::rotate_plane_strain_stress_increment(in_plane,L[q],dt);
              const double components[3]={change[0][0],change[1][1],change[0][1]};
              AssertThrow(std::isfinite(change.norm()),ExcMessage("Nonfinite complete cohesive stress rotation."));
              for(unsigned int i=0;i<9;++i)
                {
                  const double weight=fe.JxW(q)*element.shape_value(i,rule.point(q));
                  for(unsigned int j=0;j<9;++j)mass(i,j)+=weight*element.shape_value(j,rule.point(q));
                  for(unsigned int c=0;c<3;++c)rhs[c][i]+=weight*components[c];
                }
            }
          mass.gauss_jordan();
          for(unsigned int c=0;c<3;++c)
            {
              Vector<double> correction(9);mass.vmult(correction,rhs[c]);
              for(unsigned int i=0;i<9;++i)result->tensor.at(entry.first)[9*c+i]+=correction[i];
            }
        }
      return result;
    }

    template <int dim>
    std::shared_ptr<BulkHistory<dim>>
    BulkHistory<dim>::advance (const LinearAlgebra::BlockVector &du, const double dt) const
    {
      AssertThrow (dt > 0 && std::isfinite (dt),
                   ExcMessage ("Native Maxwell advancement needs positive finite dt."));
      check_layout ();
      auto trial = std::make_shared<BulkHistory> (this->get_simulator (), G, eta);
      const double beta = 1 / (1 + dt * G / eta);
      trial->tensor = tensor;
      trial->cut_geometry=cut_geometry;
      trial->cut_tensor=cut_tensor;
      const auto source = project_native (du);
      for (auto &entry : trial->tensor)
        for (unsigned int i = 0; i < 27; ++i)
          entry.second[i] = beta * (entry.second[i] + source.at (entry.first)[i]);
      for(auto &entry:trial->cut_tensor)
        for(auto &region:entry.second)
          {
            const auto increment=restrict_polynomial(source.at(entry.first),region.second.bounds);
            for(unsigned int i=0;i<27;++i)
              region.second.coefficients[i]=beta*(region.second.coefficients[i]+increment[i]);
          }
      LinearAlgebra::BlockVector owned (this->introspection ().index_sets.system_partitioning,
                                        this->get_mpi_communicator ());
      owned = 0.;
      const auto b = this->introspection ().block_indices.velocities;
      // Ghosted native fields are readable inputs, never vector-add operands
      // for an owned-only Epetra map. Publish the completed owned vector once.
      for (const auto i : owned.block (b).locally_owned_elements ())
        owned.block (b)[i] = beta * (native.block (b)[i] + du.block (b)[i]);
      owned.compress (VectorOperation::insert);
      trial->native = owned;
      trial->projection = trial->project_native (trial->native);
      return trial;
    }

    template <int dim>
    std::pair<double, double>
    BulkHistory<dim>::check_advance (const BulkHistory &trial, const LinearAlgebra::BlockVector &du,
                                     const double dt, const Quadrature<dim> &rule) const
    {
      const double beta = 1 / (1 + dt * G / eta);
      double absolute = 0., scaled = 0.;
      std::ostringstream failure;
      FEValues<dim> fe (this->get_mapping (), this->get_fe (), rule,
                        update_gradients | update_quadrature_points);
      std::vector<SymmetricTensor<2, dim>> strain (rule.size ());
      for (const auto &entry : cells)
        {
          fe.reinit (entry.second);
          fe[this->introspection ().extractors.velocities].get_function_symmetric_gradients (
              du, strain);
          const auto old = evaluate (entry.second, rule),
                     actual = trial.evaluate (entry.second, rule);
          for (unsigned int q = 0; q < rule.size (); ++q)
            {
              const Tensor expected = beta * (old[q] + 2 * G * dev3 (strain[q]));
              const double defect = (actual[q] - expected).norm ();
              absolute = std::max (absolute, defect);
              const double relative = defect / std::max (1000., expected.norm ());
              if (relative > scaled)
                {
                  scaled = relative;
                  failure.str ("");
                  failure << std::setprecision (17) << "Native Maxwell check: cell=" << entry.first
                          << ", unit=" << rule.point (q) << ", x=" << fe.quadrature_point (q)
                          << ", error_Pa=" << defect << ", scaled=" << relative
                          << ", bound=1e-12, ratio=" << relative / 1e-12 << ", actual=" << actual[q]
                          << ", expected=" << expected;
                }
            }
        }
      const bool valid = Utilities::MPI::min (static_cast<unsigned int> (scaled <= 1e-12),
                                              this->get_mpi_communicator ())
                         == 1;
      if (!valid)
        {
          const auto messages
              = Utilities::MPI::all_gather (this->get_mpi_communicator (), failure.str ());
          std::string message;
          for (const auto &m : messages)
            message += m + '\n';
          AssertThrow (false, ExcMessage (message));
        }
      return { Utilities::MPI::max (absolute, this->get_mpi_communicator ()),
               Utilities::MPI::max (scaled, this->get_mpi_communicator ()) };
    }

    template <int dim>
    double
    BulkHistory<dim>::storage (const Quadrature<dim> &rule) const
    {
      double value = 0.;
      FEValues<dim> fe (this->get_mapping (), this->get_fe (), rule, update_JxW_values);
      for (const auto &entry : cells)
        {
          fe.reinit (entry.second);
          const auto stress = evaluate (entry.second, rule);
          for (unsigned int q = 0; q < rule.size (); ++q)
            value += fe.JxW (q) * (stress[q] * stress[q]) / (4 * G);
        }
      return Utilities::MPI::sum (value, this->get_mpi_communicator ());
    }

    template <int dim>
    typename BulkHistory<dim>::CellRecords
    BulkHistory<dim>::pack () const
    {
      check_layout ();
      CellRecords local, global;
      std::vector<types::global_dof_index> ids (this->get_fe ().n_dofs_per_cell ());
      for (const auto &entry : cells)
        {
          auto &row = local[entry.first.to_string ()];
          for (const auto &v : vertices.at (entry.first))
            for (unsigned int d = 0; d < dim; ++d)
              row.push_back (v[d]);
          row.insert (row.end (), tensor.at (entry.first).begin (), tensor.at (entry.first).end ());
          entry.second->get_dof_indices (ids);
          for (unsigned int j = 0; j < ids.size (); ++j)
            if (this->get_fe ().system_to_component_index (j).first < dim)
              row.push_back (native[ids[j]]);
        }
      for (const auto &part : Utilities::MPI::all_gather (this->get_mpi_communicator (), local))
        for (const auto &entry : part)
          AssertThrow (global.emplace (entry).second,
                       ExcMessage ("Duplicate Maxwell checkpoint CellId owner."));
      AssertThrow (global.size () == this->get_triangulation ().n_global_active_cells (),
                   ExcMessage ("Incomplete native Maxwell checkpoint coverage."));
      if(cut_geometry)pack_cut(global);
      return global;
    }

    template <int dim>
    void
    BulkHistory<dim>::restore (const CellRecords &record)
    {
      AssertThrow (
          tensor.empty () && record.size () == this->get_triangulation ().n_global_active_cells()
            +record.count("@objective_cut_geometry_v1"),
          ExcMessage ("Native Maxwell restore requires a fresh owner and matching cell coverage."));
      LinearAlgebra::BlockVector owned (this->introspection ().index_sets.system_partitioning,
                                        this->get_mpi_communicator ());
      owned = 0.;
      std::map<types::global_dof_index, double> duplicate;
      std::vector<types::global_dof_index> ids (this->get_fe ().n_dofs_per_cell ());
      for (const auto &entry : cells)
        {
          const auto &row = record.at (entry.first.to_string ());
          unsigned int at = 0;
          for (const auto &v : vertices.at (entry.first))
            for (unsigned int d = 0; d < dim; ++d)
              AssertThrow (row.at (at++) == v[d],
                           ExcMessage ("Native Maxwell checkpoint bulk geometry changed."));
          auto &coefficients = tensor[entry.first];
          for (auto &v : coefficients)
            v = row.at (at++);
          entry.second->get_dof_indices (ids);
          for (unsigned int j = 0; j < ids.size (); ++j)
            if (this->get_fe ().system_to_component_index (j).first < dim)
              {
                const double value = row.at (at++);
                const auto inserted = duplicate.emplace (ids[j], value);
                AssertThrow (
                    inserted.second || inserted.first->second == value,
                    ExcMessage ("Inconsistent shared native Maxwell checkpoint coefficient."));
                if (owned.locally_owned_elements ().is_element (ids[j]))
                  owned[ids[j]] = value;
              }
          AssertThrow(at==row.size() || record.count("@objective_cut_geometry_v1"),
                      ExcMessage("Unexpected Maxwell checkpoint extension without cut-space metadata."));
        }
      owned.compress (VectorOperation::insert);
      native = owned;
      projection = project_native (native);
      if(record.count("@objective_cut_geometry_v1"))restore_cut(record);
      check_layout ();
    }

    template class BulkHistory<2>;
  }
}
