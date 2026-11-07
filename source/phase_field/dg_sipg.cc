/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/phase_field/dg_operator.h>
#include <aspect/simulator/assemblers/advection.h>
#include <numeric>
#include <aspect/phase_field/mapped_face.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

namespace aspect
{
  namespace PhaseField
  {
    template <int dim>
    void DGOperator<dim>::reinit_matrix(LinearAlgebra::SparseMatrix &matrix) const
    {
      TrilinosWrappers::SparsityPattern graph(owned, owned, relevant, communicator);
      for (const auto &cell : cells)
        {
          for (const auto i : cell.indices)
            for (const auto j : cell.indices) graph.add(i, j);
          for (const auto f : cell.iterator->face_indices())
            if (!cell.iterator->at_boundary(f))
              {
                std::vector<CellIterator> neighbors;
                if (cell.iterator->neighbor(f)->has_children())
                  for (unsigned int s = 0; s < GeometryInfo<dim>::max_children_per_face; ++s)
                    neighbors.push_back(cell.iterator->neighbor_child_on_subface(f, s));
                else neighbors.push_back(cell.iterator->neighbor(f));
                for (const auto &neighbor : neighbors)
                  {
                    const auto columns = indices(neighbor);
                    // The single face owner writes both rows. Register writable
                    // ghost-row entries too, before Trilinos freezes the graph.
                    for (const auto i : cell.indices)
                      for (const auto j : columns)
                        {
                          graph.add(i,j);
                          graph.add(j,i);
                        }
                    for (const auto i : columns)
                      for (const auto j : columns) graph.add(i,j);
                  }
              }
        }
      graph.compress();
      matrix.reinit(graph);
    }

    template <int dim>
    double DGOperator<dim>::assemble_sipg(LinearAlgebra::SparseMatrix &matrix,
                                          LinearAlgebra::Vector &rhs, const Coefficient &coefficient,
                                          const double penalty, const Coefficient &dirichlet) const
    {
      AssertThrow(penalty > 0. && std::isfinite(penalty), ExcMessage("SIPG phase penalty must be positive."));
      matrix = 0.;
      rhs = 0.;
      double constant = 0.;
      const QGauss<dim> volume_rule(quadrature.fracture_gauss);
      const QGauss<dim-1> face_rule(quadrature.fracture_gauss);
      FEValues<dim> volume(mapping,element,volume_rule,
                           update_values | update_gradients | update_JxW_values | update_quadrature_points);
      std::vector<unsigned int> shapes(element.n_dofs_per_cell());
      std::iota(shapes.begin(),shapes.end(),0);
      internal::Assembly::Scratch::ScalarFace<dim> face(mapping,element,face_rule,0,shapes);
      face.indices = [&](const auto &cell) { return indices(cell); };
      face.coefficients = [&](const auto &values,const auto &cell,unsigned int,bool subface)
      {
        typename internal::Assembly::Scratch::ScalarFace<dim>::Coefficients result(values.n_quadrature_points);
        const double jacobian = minimum_cell_jacobian(mapping,element,cell);
        for (unsigned int q = 0; q < result.size(); ++q)
          {
            auto &c = result[q];
            c.diffusivity = coefficient(values.quadrature_point(q));
            AssertThrow(c.diffusivity > 0. && std::isfinite(c.diffusivity),ExcMessage("SIPG requires a positive finite gradient coefficient."));
            const double h_inverse = values.JxW(q) / face_rule.weight(q)
                                     * (subface ? GeometryInfo<dim>::max_children_per_face : 1) / jacobian;
            c.penalty = penalty * (element.degree+1)*(element.degree+1) * c.diffusivity * h_inverse;
          }
        return result;
      };
      if (dirichlet)
        face.boundary_value = [&](types::boundary_id,const Point<dim> &x) { return dirichlet(x); };
      for (const auto &cell : cells)
        {
          volume.reinit(typename Triangulation<dim>::cell_iterator(cell.iterator));
          internal::Assembly::Scratch::ScalarCell<dim> scratch(volume);
          scratch.diffusion = 1.;
          scratch.diffusivity.resize(volume_rule.size());
          for (unsigned int q = 0; q < volume_rule.size(); ++q)
            {
              scratch.diffusivity[q] = coefficient(volume.quadrature_point(q));
              AssertThrow(scratch.diffusivity[q] > 0. && std::isfinite(scratch.diffusivity[q]),
                          ExcMessage("SIPG requires a positive finite gradient coefficient."));
            }
          internal::Assembly::CopyData::AdvectionSystem<dim> data(element,true);
          data.local_dof_indices = cell.indices;
          if (cell_assembler) cell_assembler->execute(scratch,data);
          else Assemblers::AdvectionSystem<dim>::assemble_spatial(scratch,data);
          face.cell = cell.iterator;
          assemble_faces(face,data,matrix);
          for (unsigned int i = 0; i < cell.indices.size(); ++i) rhs[cell.indices[i]] += data.local_rhs[i];
        }
      constant = face.boundary_energy;

      matrix.compress(VectorOperation::add);
      rhs.compress(VectorOperation::insert);
      return Utilities::MPI::sum(constant, communicator);
    }

    template <int dim>
    void DGOperator<dim>::set_assemblers(const Assemblers::Interface<dim> *cell,
                                         const Assemblers::Interface<dim> *interior,
                                         const Assemblers::Interface<dim> *boundary)
    {
      cell_assembler = cell;
      interior_assembler = interior;
      boundary_assembler = boundary;
    }

    template <int dim>
    void DGOperator<dim>::assemble_faces(internal::Assembly::Scratch::ScalarFace<dim> &scratch,
                                         internal::Assembly::CopyData::AdvectionSystem<dim> &data,
                                         LinearAlgebra::SparseMatrix &matrix) const
    {
      for (const auto f : scratch.cell->face_indices())
        {
          AssertThrow(!scratch.cell->has_periodic_neighbor(f),ExcMessage("Native phase periodic topology is not qualified."));
          scratch.face_number = f;
          if (scratch.cell->at_boundary(f))
            {
              if (boundary_assembler) boundary_assembler->execute(scratch,data);
              else Assemblers::AdvectionSystemBoundaryFace<dim>::assemble_spatial(scratch,data);
            }
          else
            {
              if (interior_assembler) interior_assembler->execute(scratch,data);
              else Assemblers::AdvectionSystemInteriorFace<dim>::assemble_spatial(scratch,data);
            }
        }
      for (unsigned int i = 0; i < data.local_dof_indices.size(); ++i)
        for (unsigned int j = 0; j < data.local_dof_indices.size(); ++j)
          matrix.add(data.local_dof_indices[i],data.local_dof_indices[j],data.local_matrix(i,j));
      Assemblers::distribute_scalar_dg_faces(data,matrix);
    }

#define INSTANTIATE(dim) \
  template void DGOperator<dim>::set_assemblers(const Assemblers::Interface<dim> *,const Assemblers::Interface<dim> *,const Assemblers::Interface<dim> *); \
  template void DGOperator<dim>::assemble_faces(internal::Assembly::Scratch::ScalarFace<dim> &,internal::Assembly::CopyData::AdvectionSystem<dim> &,LinearAlgebra::SparseMatrix &) const; \
  template void DGOperator<dim>::reinit_matrix(LinearAlgebra::SparseMatrix &) const; \
  template double DGOperator<dim>::assemble_sipg(LinearAlgebra::SparseMatrix &,LinearAlgebra::Vector &,const Coefficient &,double,const Coefficient &) const;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
