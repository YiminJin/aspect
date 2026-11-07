/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_RECONSTRUCTED_FAULT_ENDPOINT_PHASE_CONSTRAINTS_H
#define ASPECT_RECONSTRUCTED_FAULT_ENDPOINT_PHASE_CONSTRAINTS_H

#include <aspect/global.h>
#include <aspect/phase_field/dg_bounds.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace aspect
{
  namespace internal
  {
    /** Coordinates for the bounded cohesive Q1 phase solve. Native closed,
     * homogeneous convex hanging rows define phi=C psi. Primal expansion uses
     * AffineConstraints::distribute; duals use C^T. No constrained-row diagonal
     * or multiplier is a master coordinate. This helper owns no history. */
    class EndpointPhaseConstraints
    {
      public:
        using Row = std::vector<std::pair<unsigned int,double>>;

        EndpointPhaseConstraints(const dealii::AffineConstraints<double> &native,
                                 const dealii::types::global_dof_index offset,
                                 const unsigned int size)
          : rows(size)
        {
          AssertThrow(native.is_closed(),dealii::ExcMessage("Endpoint phase requires closed native constraints."));
          std::vector<unsigned int> indices(size,dealii::numbers::invalid_unsigned_int);
          for (unsigned int i=0; i<size; ++i)
            if (!native.is_constrained(offset+i))
              {
                indices[i]=masters.size();
                masters.push_back(i);
              }
          for (unsigned int i=0; i<size; ++i)
            if (indices[i]!=dealii::numbers::invalid_unsigned_int)
              rows[i].emplace_back(indices[i],1.);
            else
              {
                const auto *entries=native.get_constraint_entries(offset+i);
                AssertThrow(entries && !entries->empty() && native.get_inhomogeneity(offset+i)==0.,
                            dealii::ExcMessage("Endpoint phase admits only homogeneous Q1 hanging constraints."));
                double sum=0.;
                constraints.add_line(i);
                for (const auto &entry:*entries)
                  {
                    AssertThrow(entry.first>=offset && entry.first<offset+size
                                && std::isfinite(entry.second) && entry.second>=0.,
                                dealii::ExcMessage("Endpoint phase requires convex within-phase hanging rows."));
                    const auto j=entry.first-offset;
                    AssertThrow(indices[j]!=dealii::numbers::invalid_unsigned_int,
                                dealii::ExcMessage("Closed phase rows must reference independent masters."));
                    rows[i].emplace_back(indices[j],entry.second);
                    constraints.add_entry(i,j,entry.second);
                    sum+=entry.second;
                  }
                AssertThrow(std::abs(sum-1.)<=32*std::numeric_limits<double>::epsilon(),
                            dealii::ExcMessage("Endpoint phase hanging rows must be convex partitions of unity."));
              }
          constraints.close();
        }

        /** Use cell-local Bernstein growth coordinates for native DGQ2.
         * Physical = E coordinates; gradients and Hessians use E^T and E^T H E.
         * This keeps the constrained equilibrium distinct from transport limiting.
         * Global sparse coordinate rows are replicated like the existing endpoint
         * search vectors; no dense global transformation or Hessian is allocated. */
        template <int dim>
        void set_discontinuous_q2(const dealii::DoFHandler<dim> &dofs,
                                  const unsigned int component,
                                  const dealii::types::global_dof_index offset,
                                  const MPI_Comm communicator)
        {
          AssertThrow(identity(),dealii::ExcMessage("DG phase coordinates cannot contain hanging/Dirichlet coefficient constraints."));
          const auto &fe=dofs.get_fe();
          const auto &element=fe.base_element(fe.component_to_base_index(component).first);
          const PhaseField::Q2Bounds<dim> basis(element);
          std::vector<std::vector<unsigned int>> local;
          std::vector<dealii::types::global_dof_index> indices(fe.n_dofs_per_cell());
          for (const auto &cell:dofs.active_cell_iterators())
            if (cell->is_locally_owned())
              {
                cell->get_dof_indices(indices);std::vector<unsigned int> phase;
                for (unsigned int i=0;i<element.n_dofs_per_cell();++i)
                  phase.push_back(indices[fe.component_to_system_index(component,i)]-offset);
                local.push_back(std::move(phase));
              }
          inverse_rows.resize(rows.size());
          for (const auto &part:dealii::Utilities::MPI::all_gather(communicator,local))
            for (const auto &cell:part)
              for (unsigned int i=0;i<cell.size();++i)
                {
                  rows[cell[i]].clear();
                  for (unsigned int j=0;j<cell.size();++j)
                    {
                      if (basis.expansion(i,j)!=0.) rows[cell[i]].emplace_back(cell[j],basis.expansion(i,j));
                      if (basis.inverse(i,j)!=0.) inverse_rows[cell[i]].emplace_back(cell[j],basis.inverse(i,j));
                    }
                }
          polynomial_coordinates=true;
        }

        bool identity() const { return !polynomial_coordinates && masters.size()==rows.size(); }
        bool uses_polynomial_coordinates() const { return polynomial_coordinates; }
        unsigned int physical_size() const { return rows.size(); }
        unsigned int size() const { return masters.size(); }
        unsigned int physical_index(const unsigned int i) const { return masters.at(i); }
        const std::vector<Row> &weights() const { return rows; }

        dealii::Vector<double> select(const dealii::Vector<double> &physical) const
        {
          AssertDimension(physical.size(),physical_size());
          if (identity()) return physical;
          dealii::Vector<double> result(size());
          if (polynomial_coordinates)
            for (unsigned int i=0;i<size();++i)
              for (const auto &entry:inverse_rows[i]) result[i]+=entry.second*physical[entry.first];
          else
            for (unsigned int i=0; i<size(); ++i) result[i]=physical[masters[i]];
          return result;
        }

        dealii::Vector<double> expand(const dealii::Vector<double> &master) const
        {
          AssertDimension(master.size(),size());
          if (identity()) return master;
          dealii::Vector<double> result(physical_size());
          if (polynomial_coordinates)
            for (unsigned int i=0;i<physical_size();++i)
              for (const auto &entry:rows[i]) result[i]+=entry.second*master[entry.first];
          else
            {
              for (unsigned int i=0; i<size(); ++i) result[masters[i]]=master[i];
              constraints.distribute(result);
            }
          return result;
        }

        dealii::Vector<double> transpose(const dealii::Vector<double> &physical) const
        {
          AssertDimension(physical.size(),physical_size());
          if (identity()) return physical;
          dealii::Vector<double> result(size());
          for (unsigned int i=0; i<physical_size(); ++i)
            for (const auto &entry:rows[i]) result[entry.first]+=entry.second*physical[i];
          return result;
        }

        /** Raw physical-node Hessian only: never pass a condensed FE matrix. */
        dealii::FullMatrix<double> pull_back(const dealii::FullMatrix<double> &physical) const
        {
          AssertDimension(physical.m(),physical_size());
          AssertDimension(physical.n(),physical_size());
          if (identity()) return physical;
          dealii::FullMatrix<double> result(size(),size());
          for (unsigned int i=0; i<physical_size(); ++i)
            for (unsigned int j=0; j<physical_size(); ++j)
              if (physical(i,j)!=0.)
                for (const auto &a:rows[i])
                  for (const auto &b:rows[j])
                    result(a.first,b.first)+=a.second*physical(i,j)*b.second;
          return result;
        }

      private:
        dealii::AffineConstraints<double> constraints;
        std::vector<unsigned int> masters;
        std::vector<Row> rows, inverse_rows;
        bool polynomial_coordinates=false;
    };
  }
}
#endif
