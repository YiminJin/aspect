/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <aspect/material_model/phase_field_fault.h>

#include <deal.II/base/function.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>

namespace
{
  using TestAccess =
    aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;

  double
  degradation(const double phi)
  {
    const double intact_fraction = 1.0-phi;
    return dealii::Utilities::fixed_power<2>(intact_fraction)
           / (dealii::Utilities::fixed_power<2>(intact_fraction) + phi);
  }



  double
  analytic_infinite_integral(const double amplitude,
                             const double length_scale)
  {
    return 2.0 * length_scale * amplitude / (1.0-amplitude);
  }



  double
  analytic_finite_integral(const double amplitude,
                           const double length_scale,
                           const double half_width)
  {
    const double tail_amplitude =
      amplitude * std::exp(-half_width/length_scale);
    return 2.0 * length_scale
           * (amplitude/(1.0-amplitude)
              - tail_amplitude/(1.0-tail_amplitude));
  }



  double
  broadcast_owned_integral(const std::vector<double> &local_integrals,
                           const MPI_Comm communicator)
  {
    const unsigned int rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
    if (rank == 0)
      REQUIRE(local_integrals.size() == 1);
    else
      REQUIRE(local_integrals.empty());
    return dealii::Utilities::MPI::broadcast(
      communicator, rank == 0 ? local_integrals[0] : 0.0, 0);
  }



  double
  integrate_analytic_profile(const double amplitude,
                             const double length_scale,
                             const double quadrature_tolerance,
                             const double tail_tolerance,
                             const MPI_Comm communicator)
  {
    const unsigned int rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
    std::vector<dealii::Point<2>> origins;
    std::vector<dealii::Tensor<1,2>> normals;
    if (rank == 0)
      {
        origins.emplace_back(0.5, 0.5);
        dealii::Tensor<1,2> normal;
        normal[1] = 1.0;
        normals.push_back(normal);
      }

    const auto evaluate_points =
      [amplitude, length_scale](const std::vector<dealii::Point<2>> &points)
      {
        std::vector<TestAccess::PointSample> samples(points.size());
        for (unsigned int i = 0; i < points.size(); ++i)
          {
            samples[i].found = true;
            samples[i].phase_field =
              amplitude * std::exp(-std::abs(points[i][1]-0.5)/length_scale);
            samples[i].cell_diameter = length_scale;
          }
        return samples;
      };

    return broadcast_owned_integral(
      TestAccess::integrate_normalization_profiles(
        origins, normals, length_scale, quadrature_tolerance, tail_tolerance,
        communicator, evaluate_points, degradation),
      communicator);
  }



  class ExponentialPhaseField : public dealii::Function<2>
  {
    public:
      ExponentialPhaseField(const double amplitude,
                            const double length_scale)
        : amplitude(amplitude)
        , length_scale(length_scale)
      {}

      double value(const dealii::Point<2> &point,
                   const unsigned int = 0) const override
      {
        return amplitude * std::exp(-std::abs(point[1]-0.5)/length_scale);
      }

    private:
      const double amplitude;
      const double length_scale;
  };



  double
  independent_q1_integral(const unsigned int refinement_level,
                          const double amplitude,
                          const double length_scale)
  {
    const unsigned int n_intervals = 1u << refinement_level;
    const double interval_width = 1.0/n_intervals;
    const dealii::QGauss<1> quadrature(32);
    double integral = 0.0;
    for (unsigned int interval = 0; interval < n_intervals; ++interval)
      {
        const double y0 = interval * interval_width;
        const double y1 = y0 + interval_width;
        const double phi0 = amplitude * std::exp(-std::abs(y0-0.5)/length_scale);
        const double phi1 = amplitude * std::exp(-std::abs(y1-0.5)/length_scale);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            const double xi = quadrature.point(q)[0];
            const double phi = (1.0-xi)*phi0 + xi*phi1;
            integral += interval_width * quadrature.weight(q)
                        * (1.0/degradation(phi)-1.0);
          }
      }
    return integral;
  }



  double
  integrate_distributed_q1_profile(const unsigned int refinement_level,
                                   const double amplitude,
                                   const double length_scale,
                                   const MPI_Comm communicator)
  {
    dealii::parallel::distributed::Triangulation<2> triangulation(communicator);
    dealii::GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(refinement_level);

    const dealii::FE_Q<2> finite_element(1);
    dealii::DoFHandler<2> dof_handler(triangulation);
    dof_handler.distribute_dofs(finite_element);

    const dealii::IndexSet locally_relevant_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    dealii::LinearAlgebra::distributed::Vector<double> solution;
    solution.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs,
                    communicator);
    dealii::VectorTools::interpolate(
      dof_handler, ExponentialPhaseField(amplitude, length_scale), solution);
    solution.compress(dealii::VectorOperation::insert);
    solution.update_ghost_values();

    dealii::GridTools::Cache<2> grid_cache(triangulation);
    const double cell_diameter = std::sqrt(2.0) / (1u << refinement_level);
    const auto evaluate_points =
      [&](const std::vector<dealii::Point<2>> &points)
      {
        std::vector<TestAccess::PointSample> samples(points.size());
        std::vector<dealii::Point<2>> interior_points;
        std::vector<unsigned int> interior_indices;
        for (unsigned int i = 0; i < points.size(); ++i)
          if (points[i][0] >= 0.0 && points[i][0] <= 1.0
              && points[i][1] >= 0.0 && points[i][1] <= 1.0)
            {
              interior_indices.push_back(i);
              interior_points.push_back(points[i]);
            }

        dealii::Utilities::MPI::RemotePointEvaluation<2> cache;
        cache.reinit(grid_cache, interior_points);
        const std::vector<double> values =
          dealii::VectorTools::point_values<1>(
            cache, dof_handler, solution,
            dealii::VectorTools::EvaluationFlags::avg);
        REQUIRE(values.size() == interior_points.size());
        for (unsigned int j = 0; j < interior_points.size(); ++j)
          {
            REQUIRE(cache.point_found(j));
            samples[interior_indices[j]].found = true;
            samples[interior_indices[j]].phase_field = values[j];
            samples[interior_indices[j]].cell_diameter = cell_diameter;
          }
        return samples;
      };

    const unsigned int rank =
      dealii::Utilities::MPI::this_mpi_process(communicator);
    std::vector<dealii::Point<2>> origins;
    std::vector<dealii::Tensor<1,2>> normals;
    if (rank == 0)
      {
        origins.emplace_back(0.5, 0.5);
        dealii::Tensor<1,2> normal;
        normal[1] = 1.0;
        normals.push_back(normal);
      }

    const double integral = broadcast_owned_integral(
      TestAccess::integrate_normalization_profiles(
        origins, normals, length_scale, 1.e-10, 1.e-10, communicator,
        evaluate_points, degradation),
      communicator);
    solution.zero_out_ghost_values();
    return integral;
  }
}



TEST_CASE("Adaptive I_h kernel converges to an analytic profile integral",
          "[phase_field_fault_ih_accuracy]")
{
  const MPI_Comm communicator = MPI_COMM_WORLD;
  const double amplitude = 0.95;
  const double length_scale = 0.1;
  const double exact = analytic_infinite_integral(amplitude, length_scale);

  std::vector<double> quadrature_errors;
  for (const double tolerance : {1.e-4, 1.e-6, 1.e-8})
    quadrature_errors.push_back(std::abs(
      integrate_analytic_profile(amplitude, length_scale, tolerance, 1.e-10,
                                 communicator)-exact));
  CHECK(quadrature_errors.back()/exact <= 1.e-5);
  CHECK(quadrature_errors.back() <= quadrature_errors.front());
  CHECK(*std::min_element(quadrature_errors.begin()+1, quadrature_errors.end())
        < quadrature_errors.front());

  std::vector<double> tail_errors;
  for (const double tolerance : {1.e-3, 1.e-5, 1.e-7})
    tail_errors.push_back(std::abs(
      integrate_analytic_profile(amplitude, length_scale, 1.e-10, tolerance,
                                 communicator)-exact));
  CHECK(tail_errors.back()/exact <= 1.e-5);
  CHECK(tail_errors.back() <= tail_errors.front());
  CHECK(*std::min_element(tail_errors.begin()+1, tail_errors.end())
        < tail_errors.front());
}



TEST_CASE("Distributed Q1 I_h path agrees with an independent integral and refines",
          "[phase_field_fault_ih_accuracy]")
{
  const MPI_Comm communicator = MPI_COMM_WORLD;
  const double amplitude = 0.5;
  const double length_scale = 0.25;
  const double continuous_reference =
    analytic_finite_integral(amplitude, length_scale, 0.5);
  std::vector<double> discretization_errors;

  for (const unsigned int level : {6u, 7u, 8u})
    {
      const double computed = integrate_distributed_q1_profile(
        level, amplitude, length_scale, communicator);
      const double q1_reference = independent_q1_integral(
        level, amplitude, length_scale);
      CHECK(computed > 0.0);
      CHECK(std::abs(computed-q1_reference)
            / std::max(q1_reference, length_scale) <= 1.e-8);
      discretization_errors.push_back(std::abs(computed-continuous_reference));
    }

  CHECK(discretization_errors[1] < discretization_errors[0]);
  CHECK(discretization_errors[2] < discretization_errors[1]);
  const double final_order =
    std::log(discretization_errors[1]/discretization_errors[2]) / std::log(2.0);
  CHECK(final_order > 1.5);
}
