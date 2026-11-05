/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_BENCHMARK_SPATIAL_RELAXATION_H
#define ASPECT_BENCHMARK_SPATIAL_RELAXATION_H
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/material_model/phase_field_fault/tensor_history_internal.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
namespace aspect
{
namespace ParticleFreeBenchmark
{
/** Scratch prescribed relaxation for the standalone transport comparison.
 * Coupled mature reaction remains owned by the material, outside advection. */
struct SpatialRelaxation
{
  bool attached = false;
  template <class Transport, int dim>
  void
  attach (Transport &op,
          const std::function<double (const Point<dim> &)> &rate,
          const Mapping<dim> &mapping, const MPI_Comm comm)
  {
    AssertThrow (op.distributed && op.relaxation_rate == 0. && !attached,
                 ExcMessage ("Spatial relaxation may be attached once to an "
                             "unrelaxed distributed DG operator."));
    double maximum = 0.;
    std::map<types::global_dof_index,
             std::vector<const typename Transport::Load *>>
        samples;
    for (const auto &load : op.source)
      samples[load.indices.front ()].push_back (&load);
    const QGauss<dim> rule(op.quadrature_order);
    FEValues<dim> shapes(mapping,op.space->element,rule,update_values);
    for (const auto &cell : op.cells)
      {
        shapes.reinit(op.space->cells.at(cell.id));
        AssertDimension(samples.at(cell.indices.front()).size(),rule.size());
        FullMatrix<double> values (cell.indices.size (), cell.indices.size ());
        unsigned q=0;
        for (const auto p : samples.at (cell.indices.front ()))
          {
            const auto &load = *p;
            const double r = rate (load.point);
            AssertThrow (
                std::isfinite (r) && r >= 0.,
                ExcMessage ("Spatial Maxwell rate requires finite nonnegative "
                            "values and complete volume samples."));
            maximum = std::max (maximum, r);
            for (unsigned int i = 0; i < cell.indices.size (); ++i)
              for (unsigned int j = 0; j < cell.indices.size (); ++j)
                values (i, j) -= r * load.weights[i] * shapes.shape_value(j,q);
            ++q;
          }
        op.blocks.push_back (
            { cell.indices, cell.indices, std::move (values) });
      }
    op.relaxation_rate = Utilities::MPI::max (maximum, comm);
    attached = true;
  }
};
}
}
#endif
