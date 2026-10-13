/* Copyright (C) 2026 by the authors of ASPECT.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include "common.h"
#include <aspect/simulator/solver/reconstructed_fault_surface_schur.h>
#include <deal.II/lac/block_sparsity_pattern.h>

TEST_CASE("Fault direct bulk inverse preserves pressure quotient and constraints",
          "[fault_direct_schur]")
{
  using namespace dealii;
  using BV=aspect::LinearAlgebra::BlockVector;
  const MPI_Comm comm=MPI_COMM_WORLD;
  if(dealii::Utilities::MPI::n_mpi_processes(comm)!=1)return;
  IndexSet owned(2);owned.add_range(0,2);
  const std::vector<IndexSet> partition{owned,owned};
  BlockDynamicSparsityPattern sparsity(2,2);
  for(unsigned int b=0;b<2;++b)for(unsigned int c=0;c<2;++c)
    {
      sparsity.block(b,c).reinit(2,2);
      for(unsigned int i=0;i<2;++i)for(unsigned int j=0;j<2;++j)sparsity.block(b,c).add(i,j);
    }
  sparsity.collect_sizes();
  aspect::LinearAlgebra::BlockSparseMatrix matrix;
  matrix.reinit(partition,sparsity,comm);
  const double entries[4][4]={{4.,1.,1.,-1.},{1.,3.,2.,-2.},{1.,2.,0.,0.},{-1.,-2.,0.,0.}};
  for(unsigned int b=0;b<2;++b)for(unsigned int c=0;c<2;++c)
    for(unsigned int i=0;i<2;++i)for(unsigned int j=0;j<2;++j)matrix.block(b,c).set(i,j,entries[2*b+i][2*c+j]);
  matrix.compress(VectorOperation::insert);
  BV state(partition,comm),rhs(state),result(state),q(state),error(state);
  state.block(0)[0]=.1;state.block(0)[1]=-.3;
  state.block(1)[0]=.4;state.block(1)[1]=-.4;
  state.compress(VectorOperation::insert);
  q.block(1)=1./std::sqrt(2.);
  matrix.vmult(rhs,state);
  AffineConstraints<double> constraints;constraints.close();
  aspect::internal::EndpointBulkDirect inverse(matrix,state,constraints,q,comm);
  inverse.vmult(result,rhs);
  error=result;error-=state;
  CHECK(error.linfty_norm()<2e-15);
  CHECK(std::abs(q*result)<2e-16);
  matrix.vmult(error,result);error-=rhs;
  CHECK(error.linfty_norm()<2e-15);
  // Reusing the factor is a genuine second RHS, not reuse of a solution.
  rhs*=3.;inverse.vmult(result,rhs);
  matrix.vmult(error,result);error-=rhs;
  CHECK(error.linfty_norm()<5e-15);
  CHECK(inverse.solves==2);

  // An actual fixed pressure coefficient receives no additional gauge pin.
  constraints.reinit();constraints.add_line(3);constraints.close();q=0.;
  state.block(1)[0]=.8;state.block(1)[1]=0.;state.compress(VectorOperation::insert);
  matrix.vmult(rhs,state);rhs.block(1)[1]=0.;rhs.compress(VectorOperation::insert);
  aspect::internal::EndpointBulkDirect fixed(matrix,state,constraints,q,comm);
  fixed.vmult(result,rhs);error=result;error-=state;
  CHECK(error.linfty_norm()<3e-15);
  CHECK(result.block(1)[1]==0.);
}
