/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

namespace
{
  using namespace dealii;

  FullMatrix<double>
  inverse(const FullMatrix<double> &matrix)
  {
    FullMatrix<double> result(matrix);
    result.gauss_jordan();
    return result;
  }


  Vector<double>
  product(const FullMatrix<double> &matrix, const Vector<double> &vector)
  {
    Vector<double> result(matrix.m());
    matrix.vmult(result, vector);
    return result;
  }
}


TEST_CASE("reconstructed fault exact condensation matches a full block solve",
          "[reconstructed_fault_condensation]")
{
  FullMatrix<double> A(2,2);
  A(0,0) = 4.0;
  A(0,1) = 0.7;
  A(1,0) = -0.4;
  A(1,1) = 3.0;

  FullMatrix<double> B(2,2);
  B(0,0) = 1.0;
  B(0,1) = 0.2;
  B(1,0) = -0.3;
  B(1,1) = 0.8;

  FullMatrix<double> G(2,2);
  G(0,0) = 0.4;
  G(0,1) = -0.7;
  G(1,0) = 1.2;
  G(1,1) = 0.1;

  // This surface block is nonsingular and indefinite.
  FullMatrix<double> K(2,2);
  K(0,0) = 2.0;
  K(0,1) = 1.0;
  K(1,0) = 1.0;
  K(1,1) = -1.0;

  Vector<double> R_bulk(2);
  R_bulk[0] = 0.9;
  R_bulk[1] = -1.1;
  Vector<double> R_surface(2);
  R_surface[0] = 0.3;
  R_surface[1] = 0.6;

  FullMatrix<double> full_system(4,4);
  for (unsigned int i = 0; i < 2; ++i)
    for (unsigned int j = 0; j < 2; ++j)
      {
        full_system(i,j) = A(i,j);
        full_system(i,j+2) = -B(i,j);
        full_system(i+2,j) = G(i,j);
        full_system(i+2,j+2) = -K(i,j);
      }
  Vector<double> full_rhs(4);
  for (unsigned int i = 0; i < 2; ++i)
    {
      full_rhs[i] = -R_bulk[i];
      full_rhs[i+2] = -R_surface[i];
    }
  const FullMatrix<double> full_inverse = inverse(full_system);
  const Vector<double> full_solution = product(full_inverse, full_rhs);

  const FullMatrix<double> K_inverse = inverse(K);
  FullMatrix<double> K_inverse_G(2,2);
  K_inverse.mmult(K_inverse_G, G);
  FullMatrix<double> B_K_inverse_G(2,2);
  B.mmult(B_K_inverse_G, K_inverse_G);
  FullMatrix<double> condensed(A);
  condensed.add(-1.0, B_K_inverse_G);

  REQUIRE(std::abs(condensed(0,1)-condensed(1,0)) > 1.e-2);

  const Vector<double> K_inverse_R_surface = product(K_inverse, R_surface);
  const Vector<double> B_K_inverse_R_surface = product(B, K_inverse_R_surface);
  Vector<double> condensed_rhs(R_bulk);
  condensed_rhs *= -1.0;
  condensed_rhs += B_K_inverse_R_surface;

  Vector<double> bulk_increment(2);
  SolverControl control(20, 1.e-13);
  GrowingVectorMemory<Vector<double>> memory;
  SolverFGMRES<Vector<double>> solver(control, memory);
  solver.solve(condensed, bulk_increment, condensed_rhs, PreconditionIdentity());

  Vector<double> recovery_rhs = product(G, bulk_increment);
  recovery_rhs += R_surface;
  const Vector<double> slip_increment = product(K_inverse, recovery_rhs);

  for (unsigned int i = 0; i < 2; ++i)
    {
      REQUIRE(bulk_increment[i] == Approx(full_solution[i]).margin(1.e-12));
      REQUIRE(slip_increment[i] == Approx(full_solution[i+2]).margin(1.e-12));
    }

  Vector<double> full_residual = product(full_system, full_solution);
  full_residual -= full_rhs;
  REQUIRE(full_residual.linfty_norm() < 1.e-12);
}


TEST_CASE("reconstructed fault condensation preserves multiple fault blocks",
          "[reconstructed_fault_condensation]")
{
  constexpr unsigned int n_bulk = 2;
  constexpr unsigned int n_surface = 3;
  FullMatrix<double> A(n_bulk,n_bulk);
  A(0,0) = 5.0;
  A(0,1) = -0.5;
  A(1,0) = 0.25;
  A(1,1) = 4.0;

  FullMatrix<double> B(n_bulk,n_surface);
  B(0,0) = 0.8;
  B(0,1) = -0.2;
  B(0,2) = 0.5;
  B(1,0) = 0.1;
  B(1,1) = 0.6;
  B(1,2) = -0.4;
  FullMatrix<double> G(n_surface,n_bulk);
  G(0,0) = 0.3;
  G(0,1) = 0.9;
  G(1,0) = -0.7;
  G(1,1) = 0.2;
  G(2,0) = 1.1;
  G(2,1) = -0.1;

  // Vertices 0--1 are one fault; vertex 2 is a separate one-vertex block.
  FullMatrix<double> K(n_surface,n_surface);
  K(0,0) = 2.0;
  K(0,1) = 0.4;
  K(1,0) = 0.4;
  K(1,1) = -1.5;
  K(2,2) = 0.75;
  REQUIRE(K(0,2) == 0.0);
  REQUIRE(K(1,2) == 0.0);

  Vector<double> R_bulk(n_bulk);
  R_bulk[0] = -0.2;
  R_bulk[1] = 0.7;
  Vector<double> R_surface(n_surface);
  R_surface[0] = 0.1;
  R_surface[1] = -0.3;
  R_surface[2] = 0.5;

  FullMatrix<double> full_system(n_bulk+n_surface, n_bulk+n_surface);
  for (unsigned int i = 0; i < n_bulk; ++i)
    for (unsigned int j = 0; j < n_bulk; ++j)
      full_system(i,j) = A(i,j);
  for (unsigned int i = 0; i < n_bulk; ++i)
    for (unsigned int j = 0; j < n_surface; ++j)
      full_system(i,j+n_bulk) = -B(i,j);
  for (unsigned int i = 0; i < n_surface; ++i)
    for (unsigned int j = 0; j < n_bulk; ++j)
      full_system(i+n_bulk,j) = G(i,j);
  for (unsigned int i = 0; i < n_surface; ++i)
    for (unsigned int j = 0; j < n_surface; ++j)
      full_system(i+n_bulk,j+n_bulk) = -K(i,j);

  Vector<double> full_rhs(n_bulk+n_surface);
  for (unsigned int i = 0; i < n_bulk; ++i)
    full_rhs[i] = -R_bulk[i];
  for (unsigned int i = 0; i < n_surface; ++i)
    full_rhs[i+n_bulk] = -R_surface[i];
  const Vector<double> full_solution = product(inverse(full_system), full_rhs);

  const FullMatrix<double> K_inverse = inverse(K);
  FullMatrix<double> K_inverse_G(n_surface,n_bulk);
  K_inverse.mmult(K_inverse_G, G);
  FullMatrix<double> correction(n_bulk,n_bulk);
  B.mmult(correction, K_inverse_G);
  FullMatrix<double> condensed(A);
  condensed.add(-1.0, correction);

  Vector<double> condensed_rhs(R_bulk);
  condensed_rhs *= -1.0;
  condensed_rhs += product(B, product(K_inverse, R_surface));
  Vector<double> bulk_increment(n_bulk);
  SolverControl control(20, 1.e-13);
  GrowingVectorMemory<Vector<double>> memory;
  SolverFGMRES<Vector<double>> solver(control, memory);
  solver.solve(condensed, bulk_increment, condensed_rhs, PreconditionIdentity());

  Vector<double> recovery_rhs = product(G, bulk_increment);
  recovery_rhs += R_surface;
  const Vector<double> slip_increment = product(K_inverse, recovery_rhs);
  for (unsigned int i = 0; i < n_bulk; ++i)
    REQUIRE(bulk_increment[i] == Approx(full_solution[i]).margin(1.e-12));
  for (unsigned int i = 0; i < n_surface; ++i)
    REQUIRE(slip_increment[i]
            == Approx(full_solution[i+n_bulk]).margin(1.e-12));
}
