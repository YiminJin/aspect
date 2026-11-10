/* Copyright (C) 2026 by the authors of ASPECT.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_test_endpoint_bulk_reference_h
#define _aspect_test_endpoint_bulk_reference_h

#include <aspect/global.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <chrono>

namespace aspect { namespace internal {
  /** Exact sparse inverse of the assembled velocity/pressure block. The
   * initial bounded implementation is serial; it does not alter the MPI path.
   * Homogeneous constrained entries and a verified pressure gauge are algebraic
   * eliminations, not changes to the physical boundary conditions. */
  class EndpointBulkDirect
  {
    public:
      EndpointBulkDirect(const LinearAlgebra::BlockSparseMatrix &blocks,
                         const LinearAlgebra::BlockVector &layout,
                         const AffineConstraints<double> &constraints,
                         const LinearAlgebra::BlockVector &nullspace,
                         const MPI_Comm communicator)
        : nu(layout.block(0).size()), np(layout.block(1).size()), q(nullspace)
      {
        AssertThrow(Utilities::MPI::n_mpi_processes(communicator)==1,
                    ExcMessage("Endpoint direct surface Schur is currently a serial verification option."));
        const auto start=std::chrono::steady_clock::now();
        IndexSet owned(nu+np);owned.add_range(0,nu+np);
        fixed.assign(nu+np,false);
        for(const auto &line:constraints.get_lines())
          if(line.index<fixed.size())fixed[line.index]=true;
        // Pin one independent pressure coefficient only when BOTH null tests
        // of the actual constrained coupled operator passed. Project the result
        // back to its original Euclidean pressure complement after inversion.
        if(q.l2_norm()>0.)
          for(unsigned int i=0;i<np;++i)if(q.block(1)[i]!=0.)
            {gauge=nu+i;fixed[gauge]=true;break;}
        DynamicSparsityPattern sparsity(nu+np);
        for(unsigned int i=0;i<fixed.size();++i)sparsity.add(i,i);
        for(unsigned int bi=0;bi<2;++bi)for(unsigned int bj=0;bj<2;++bj)
          {
            const auto &block=blocks.block(bi,bj);
            const unsigned int ro=bi?nu:0,co=bj?nu:0;
            for(unsigned int r=0;r<block.m();++r)
              if(!fixed[ro+r])for(auto e=block.begin(r);e!=block.end(r);++e)
                if(!fixed[co+e->column()])sparsity.add(ro+r,co+e->column());
          }
        matrix.reinit(owned,owned,sparsity,communicator);
        for(unsigned int i=0;i<fixed.size();++i)if(fixed[i])matrix.set(i,i,1.);
        for(unsigned int bi=0;bi<2;++bi)for(unsigned int bj=0;bj<2;++bj)
          {
            const auto &block=blocks.block(bi,bj);
            const unsigned int ro=bi?nu:0,co=bj?nu:0;
            for(unsigned int r=0;r<block.m();++r)
              if(!fixed[ro+r])for(auto e=block.begin(r);e!=block.end(r);++e)
                if(!fixed[co+e->column()])matrix.set(ro+r,co+e->column(),e->value());
          }
        matrix.compress(VectorOperation::insert);
        input.reinit(owned,communicator);output.reinit(input);
        inverse.initialize(matrix);
        setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      }
      void vmult(LinearAlgebra::BlockVector &dst,const LinearAlgebra::BlockVector &src) const
      {
        for(unsigned int i=0;i<nu;++i)input[i]=src.block(0)[i];
        for(unsigned int i=0;i<np;++i)input[nu+i]=src.block(1)[i];
        for(unsigned int i=0;i<fixed.size();++i)if(fixed[i])input[i]=0.;
        input.compress(VectorOperation::insert);
        inverse.vmult(output,input);
        for(unsigned int i=0;i<nu;++i)dst.block(0)[i]=output[i];
        for(unsigned int i=0;i<np;++i)dst.block(1)[i]=output[nu+i];
        dst.compress(VectorOperation::insert);
        if(q.l2_norm()>0.)dst.add(-(q*dst),q);
        ++solves;
      }
      double setup_seconds=0.;
      mutable unsigned int solves=0;
    private:
      unsigned int nu,np,gauge=numbers::invalid_unsigned_int;
      LinearAlgebra::BlockVector q;
      std::vector<bool> fixed;
      LinearAlgebra::SparseMatrix matrix;
      mutable LinearAlgebra::Vector input,output;
      mutable TrilinosWrappers::SolverDirect inverse;
  };

}
}
#endif
