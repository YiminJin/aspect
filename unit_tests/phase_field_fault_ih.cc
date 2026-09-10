/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include "../tests/phase_field_fault_test_access.h"

#include <aspect/material_model/phase_field_fault.h>

#include <deal.II/base/function.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/base/quadrature_lib.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>

TEST_CASE("I_h lookup reuse samples fresh values and invalidates collectively",
          "[phase_field_fault_ih_cache]")
{
  using namespace dealii;
  using Cache = aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>::NormalizationPointLookupCache;
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global(2);
  GridTools::Cache<2> grid(triangulation);
  DoFHandler<2> dofs(triangulation);
  FE_Q<2> fe(1);
  dofs.distribute_dofs(fe);
  Vector<double> values(dofs.n_dofs());
  values = 3.0;
  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::vector<Point<2>> points;
  if (rank == 0)
    points = {{.2,.3}, {.5,.5}, {1.2,.5}};
  Cache cache;
  const auto compare = [&]()
  {
    const auto &batch = cache.get(grid, points);
    const auto &lookup = *batch.lookup;
    Utilities::MPI::RemotePointEvaluation<2> reference;
    reference.reinit(grid, points);
    const auto sampled = VectorTools::point_values<1>(lookup, dofs, values,
                                                     VectorTools::EvaluationFlags::avg);
    const auto expected = VectorTools::point_values<1>(reference, dofs, values,
                                                      VectorTools::EvaluationFlags::avg);
    std::vector<double> restored(points.size(), 0.0);
    std::vector<unsigned int> multiplicity(points.size(), 0);
    for (unsigned int q=0; q<batch.request_indices.size(); ++q)
      {
        restored[batch.request_indices[q]] = sampled[q];
        multiplicity[batch.request_indices[q]] = lookup.get_point_ptrs()[q+1]-lookup.get_point_ptrs()[q];
      }
    REQUIRE(restored == expected);
    for (unsigned int p=0; p<points.size(); ++p)
      {
        REQUIRE(multiplicity[p] == reference.get_point_ptrs()[p+1]-reference.get_point_ptrs()[p]);
        if (reference.point_found(p)) REQUIRE(restored[p] == Approx(values[0]));
      }
  };
  compare();
  REQUIRE(cache.rebuilds == 1);
  cache.next_batch = 0;
  values = 6.0;
  compare();
  REQUIRE(cache.hits == 1);

  // One requesting rank changes its coordinates; idle ranks must rebuild too.
  cache.next_batch = 0;
  if (rank == 0) points[0][0] += .01;
  compare();
  REQUIRE(cache.rebuilds == 2);
  cache.next_batch = 0;
  triangulation.refine_global(1);
  dofs.distribute_dofs(fe);
  values.reinit(dofs.n_dofs());
  values = 9.0;
  compare();
  REQUIRE(cache.rebuilds == 3);
}

TEST_CASE("I_h conservative rejection preserves boundary ownership and request indices",
          "[phase_field_fault_ih_cache]")
{
  using namespace dealii;
  using Cache = aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>::NormalizationPointLookupCache;
  parallel::distributed::Triangulation<2> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_rectangle(tria, Point<2>(0,.4921875), Point<2>(.0078125,.5));
  tria.refine_global(1);
  const MappingQ<2> q1(1), q2(2);
  const MappingCartesian<2> cartesian;
  const double h = 1.0/256;
  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  for (const Mapping<2> *mapping : std::vector<const Mapping<2>*>{&q1, &q2, &cartesian})
    {
      GridTools::Cache<2> grid(tria, *mapping);
      Cache cache;
      for (unsigned int pass=0; pass<2; ++pass)
        {
          std::vector<Point<2>> points = {
            {.00044024088038695236, .50000042146977086}, // captured missing probe
            {.002, .5+0.5e-6*h}, {.002, .5+2e-6*h},
            {h, .5-h}, {h, .5-.5*h}, {.002, .5}, {.002, .5-0.5e-6*h}};
          if (rank == 1) std::reverse(points.begin(), points.end());
          if (pass == 1 && rank == 0) points.clear();
          cache.next_batch = 0;
          const auto &batch = cache.get(grid, points);
          REQUIRE(cache.rejection_supported == (mapping != &q2));
          Utilities::MPI::RemotePointEvaluation<2> reference;
          reference.reinit(grid, points);
          const auto evaluate = [](const auto &lookup)
          {
            return lookup.template evaluate_and_process<double>(
              [](const ArrayView<double> &values,
                 const Utilities::MPI::RemotePointEvaluation<2>::CellData &cells)
              {
                for (const auto c : cells.cell_indices())
                  {
                    const auto view = cells.get_data_view(c,values);
                    const auto refs = cells.get_unit_points(c);
                    for (unsigned int q=0; q<view.size(); ++q)
                      view[q] = cells.get_active_cell_iterator(c)->active_cell_index()+refs[q][0];
                  }
              });
          };
          const auto expected = evaluate(reference);
          const auto sampled = evaluate(*batch.lookup);
          std::vector<unsigned int> counts(points.size(), 0);
          for (unsigned int q=0; q<batch.request_indices.size(); ++q)
            {
              const auto p = batch.request_indices[q];
              const auto begin = batch.lookup->get_point_ptrs()[q];
              counts[p] = batch.lookup->get_point_ptrs()[q+1]-begin;
              REQUIRE(counts[p] == reference.get_point_ptrs()[p+1]-reference.get_point_ptrs()[p]);
              for (unsigned int j=0; j<counts[p]; ++j)
                REQUIRE(sampled[begin+j] == expected[reference.get_point_ptrs()[p]+j]);
            }
          for (unsigned int p=0; p<points.size(); ++p)
            {
              REQUIRE((counts[p] > 0) == reference.point_found(p));
              if (points[p][0] == h)
                REQUIRE(counts[p] == (points[p][1] == .5-h ? 4 : 2));
              if (points[p][1] > .5+1e-6*h) REQUIRE(counts[p] == 0);
              else REQUIRE(counts[p] > 0);
            }
        }
    }
}

// Opt-in saved-state benchmark: only scalar Q1 FE data and the actual polyline
// are loaded. No mechanics, particle advection or convergence campaign runs.
TEST_CASE("Saved K2 fine-state I_h lookup benchmark", "[.fault_ih_performance]")
{
  using namespace dealii;
  using Access = aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;
  std::cout << "Saved I_h benchmark: loading scalar FE state" << std::endl;
  const char *directory = std::getenv("ASPECT_FAULT_PERFORMANCE_STATE");
  REQUIRE(directory != nullptr);
  REQUIRE(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1);
  const auto read_rows = [&](const std::string &name)
  {
    std::ifstream input(std::string(directory)+"/"+name+"_0.csv");
    REQUIRE(input.good());
    std::string line;
    std::getline(input,line);
    std::vector<std::vector<double>> rows;
    while (std::getline(input,line))
      {
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream stream(line);
        std::vector<double> row;
        double value;
        while (stream >> value) row.push_back(value);
        rows.push_back(std::move(row));
      }
    return rows;
  };
  std::map<std::pair<double,double>,double> phase;
  std::set<double> xs,ys;
  for (const auto &row : read_rows("phase"))
    {
      REQUIRE(row.size() == 4);
      xs.insert(row[1]); ys.insert(row[2]);
      phase[{row[1],row[2]}] = row[3];
    }
  const auto surface = read_rows("surface");
  std::vector<Point<2>> vertices;
  for (const auto &row : surface)
    {
      REQUIRE(row.size() == 8);
      vertices.emplace_back(row[2],row[3]);
    }
  parallel::distributed::Triangulation<2> triangulation(MPI_COMM_WORLD);
  const unsigned int nx = xs.size()-1;
  REQUIRE(ys.size()-1 == 4*nx);
  REQUIRE((nx & (nx-1)) == 0);
  // Preserve the production mesh hierarchy, not just its final cell boxes:
  // making 65536 coarse trees is not the saved four-tree AMR configuration.
  GridGenerator::subdivided_hyper_rectangle(
    triangulation, std::vector<unsigned int>{1,4},
    Point<2>(*xs.begin(),*ys.begin()), Point<2>(*xs.rbegin(),*ys.rbegin()));
  unsigned int refinements = 0;
  while ((1u << refinements) < nx) ++refinements;
  triangulation.refine_global(refinements);
  std::cout << "Saved I_h benchmark: reconstructed " << triangulation.n_active_cells()
            << " fine cells" << std::endl;
  FE_Q<2> fe(1);
  DoFHandler<2> dofs(triangulation);
  dofs.distribute_dofs(fe);
  Vector<double> values(dofs.n_dofs());
  for (const auto &cell : dofs.active_cell_iterators())
    for (unsigned int v=0; v<4; ++v)
      values[cell->vertex_dof_index(v,0)] = phase.at({cell->vertex(v)[0],cell->vertex(v)[1]});
  const MappingQ1<2> q1;
  const MappingCartesian<2> cartesian;
  const Mapping<2> &mapping = std::getenv("ASPECT_IH_CARTESIAN")
                             ? static_cast<const Mapping<2>&>(cartesian)
                             : static_cast<const Mapping<2>&>(q1);
  GridTools::Cache<2> grid(triangulation, mapping);
  const QGauss<1> surface_gauss(3);
  std::vector<Point<2>> origins;
  std::vector<Tensor<1,2>> normals;
  std::vector<double> saved_at_profiles;
  // Keep all production Gauss profiles on both end segments and one interior
  // segment. Their complete transverse paths and tolerances are unchanged.
  for (const unsigned int s : {0u, static_cast<unsigned int>((vertices.size()-1)/2),
                                static_cast<unsigned int>(vertices.size()-2)})
    {
      const auto tangent = (vertices[s+1]-vertices[s])/vertices[s].distance(vertices[s+1]);
      for (unsigned int q=0; q<surface_gauss.size(); ++q)
        {
          origins.push_back(vertices[s]+surface_gauss.point(q)[0]*(vertices[s+1]-vertices[s]));
          normals.push_back(Tensor<1,2>({-tangent[1],tangent[0]}));
          const double xi=surface_gauss.point(q)[0];
          saved_at_profiles.push_back((1-xi)*surface[s][7]+xi*surface[s+1][7]);
        }
    }
  Access::NormalizationPointLookupCache cache;
  const bool baseline = std::getenv("ASPECT_IH_LOOKUP_BASELINE") != nullptr;
  std::vector<Access::NormalizationPointLookupCache::Batch> baseline_batches;
  unsigned long long found = 0, missing = 0;
  std::uint64_t status_digest = 14695981039346656037ull;
  const auto evaluate = [&](const std::vector<Point<2>> &points)
  {
    // Comparison-only path: retain the original full-request lookup, without
    // changing the production cache or the adaptive integration algorithm.
    Access::NormalizationPointLookupCache::Batch original;
    if (baseline)
      {
        original.points = points;
        original.lookup = std::make_unique<Utilities::MPI::RemotePointEvaluation<2>>();
        original.lookup->reinit(grid,points);
        original.request_indices.resize(points.size());
        std::iota(original.request_indices.begin(),original.request_indices.end(),0);
      }
    const auto &batch = baseline ? original : cache.get(grid,points);
    const auto &lookup = *batch.lookup;
    const auto phi = VectorTools::point_values<1>(lookup,dofs,values,VectorTools::EvaluationFlags::avg);
    const auto diameters = lookup.evaluate_and_process<double>(
      [](const ArrayView<double> &data, const Utilities::MPI::RemotePointEvaluation<2>::CellData &cells)
      {
        for (const auto c : cells.cell_indices())
          {
            const auto view = cells.get_data_view(c,data);
            std::fill(view.begin(),view.end(),cells.get_active_cell_iterator(c)->diameter());
          }
      });
    std::vector<Access::PointSample> samples(points.size());
    for (unsigned int q=0; q<batch.request_indices.size(); ++q)
      if (lookup.point_found(q))
        {
          const unsigned int p = batch.request_indices[q];
          samples[p].found = true;
          samples[p].phase_field = phi[q];
          samples[p].cell_diameter = std::numeric_limits<double>::max();
          for (unsigned int j=lookup.get_point_ptrs()[q]; j<lookup.get_point_ptrs()[q+1]; ++j)
            samples[p].cell_diameter = std::min(samples[p].cell_diameter,diameters[j]);
        }
    for (const auto &sample : samples)
      {
        if (sample.found) ++found; else ++missing;
        status_digest = (status_digest ^ static_cast<unsigned int>(sample.found))*1099511628211ull;
      }
    if (baseline) baseline_batches.push_back(std::move(original));
    return samples;
  };
  // The saved K2 fixture has p=1, m=128, ell=0.15625 and these I_h tolerances.
  // Use the production degradation and adaptive kernel, not a fitted profile.
  const aspect::PhaseField::DegradationFunction degradation(1,128);
  const auto integrate = [&]()
  {
    return Access::integrate_normalization_profiles(
      origins,normals,.15625,1e-10,1e-10,MPI_COMM_WORLD,evaluate,
      [&](const double phi) { return degradation.value(phi); });
  };
  Timer timer;
  std::cout << "Saved I_h benchmark: cold profile integration" << std::endl;
  const auto cold = integrate();
  const double cold_seconds = timer.wall_time();
  std::cout << std::setprecision(17) << "Cold result: found=" << found << ", missing=" << missing
            << ", status digest=" << status_digest;
  for (const double value : cold) std::cout << ", I_h=" << value;
  std::cout << std::endl;
  std::cout << "Saved I_h benchmark: cold complete in " << cold_seconds
            << " s; starting lookup reuse" << std::endl;
  cache.next_batch = 0;
  timer.restart();
  const auto warm = baseline ? cold : integrate();
  const double warm_seconds = timer.wall_time();
  REQUIRE(cold == warm);
  REQUIRE(cache.hits == cache.next_batch);
  unsigned long long points = 0;
  for (const auto &batch : cache.batches) points += batch.points.size();

  // This is a diagnostic comparison with the saved Q1 projection, not an
  // assertion that individual profile integrals equal their mass projection.
  double maximum_saved_error = 0;
  for (unsigned int p=0; p<cold.size(); ++p)
    maximum_saved_error = std::max(maximum_saved_error,std::abs(cold[p]-saved_at_profiles[p]));
  std::cout << "Saved I_h mapping=" << typeid(mapping).name()
            << ", rejection eligible=" << cache.rejection_supported << std::endl;
  std::cout << std::setprecision(17) << "Saved I_h benchmark: cold seconds=" << cold_seconds
            << ", warm seconds=" << warm_seconds << ", batches=" << cache.next_batch
            << ", points=" << points << ", profiles=" << origins.size()
            << ", saved Q1 I_h max difference=" << maximum_saved_error << std::endl;
}

TEST_CASE("Phase-field Newton trials stay on the nonsingular degradation branch",
          "[phase_field_domain]")
{
  const aspect::PhaseField::DegradationFunction degradation(1.0, 480000.0);
  REQUIRE(degradation.is_in_domain(0.0));
  REQUIRE(degradation.is_in_domain(1.0));
  REQUIRE(degradation.is_in_domain(-1.e-7));
  REQUIRE_FALSE(degradation.is_in_domain(-3.e-6));
  REQUIRE_FALSE(degradation.is_in_domain(-2.0));
  REQUIRE_FALSE(degradation.is_in_domain(1.01));
  REQUIRE_FALSE(degradation.is_in_domain(std::numeric_limits<double>::infinity()));
  REQUIRE_FALSE(degradation.is_in_domain(std::numeric_limits<double>::quiet_NaN()));
  // No real denominator roots: the lower mathematical branch is unbounded.
  const aspect::PhaseField::DegradationFunction no_pole(1.0, 4.0);
  REQUIRE(no_pole.is_in_domain(-0.5));
}

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
