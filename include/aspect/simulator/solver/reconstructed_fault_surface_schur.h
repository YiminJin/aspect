/* Copyright (C) 2026 by the authors of ASPECT.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_reconstructed_fault_surface_schur_h
#define _aspect_reconstructed_fault_surface_schur_h

#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
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

  /** Exact Woodbury/surface Schur solve, one current linearization/free set.
   * S=K_FF-G_F A^{-1}B_F. Never substitutes B^T for the native G action.
   * All velocity, pressure and tangential degrees of freedom remain present. */
  template<int dim> class EndpointSurfaceSchur
  {
    using BV=LinearAlgebra::BlockVector;
    using FV=ReconstructedFaultVector;
    using Lin=typename StokesSolver::ReconstructedFaultCondensedSystem<dim>::Linearization;
    public:
      EndpointSurfaceSchur(const Lin &linearization,
                           const ReconstructedFaultSurfaceSystem<dim> &surface,
                           const ReconstructedFaultActiveSet &active,
                           const BV &layout,std::shared_ptr<EndpointBulkDirect> bulk)
        : linearization(linearization),bulk(std::move(bulk)),shape(surface.get_linearization_residual().values),active(active)
      {
        const auto start=std::chrono::steady_clock::now();
        for(unsigned int f=0;f<active.size();++f)for(unsigned int i=0;i<active[f].size();++i)
          if(!active[f][i])free.emplace_back(f,i);
        for(auto &values:shape)std::fill(values.begin(),values.end(),0.);
        schur.reinit(free.size(),free.size());
        for(unsigned int j=0;j<free.size();++j)
          {
            FV mode=shape,K,G;mode[free[j].first][free[j].second]=1.;
            BV load(layout),response(layout);load=0.;response=0.;
            linearization.apply_B(mode,load);this->bulk->vmult(response,load);
            linearization.apply_G(response,G);surface.apply_surface_jacobian(mode,K);
            for(unsigned int i=0;i<free.size();++i)
              schur(i,j)=K[free[i].first][free[i].second]-G[free[i].first][free[i].second];
            responses.push_back(std::move(response));
          }
        if(!free.empty())
          {
            LAPACKFullMatrix<double> spectrum(schur);spectrum.compute_svd();
            condition=spectrum.singular_value(0)/spectrum.singular_value(free.size()-1);
            schur.compute_lu_factorization();
          }
        setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      }
      void vmult(BV &dst,const BV &src) const
      {
        bulk->vmult(dst,src);
        if(free.empty())return;
        FV G;linearization.apply_G(dst,G);
        Vector<double> v(free.size());
        for(unsigned int i=0;i<free.size();++i)v[i]=G[free[i].first][free[i].second];
        schur.solve(v);
        for(unsigned int i=0;i<free.size();++i)dst.add(v[i],responses[i]);
      }
      unsigned int size() const {return free.size();}
      bool matches(const ReconstructedFaultActiveSet &requested)const{return active==requested;}
      double setup_seconds=0.;
      double condition=1.;
    private:
      // Copy the immutable view rather than retaining a reference to a
      // potentially moved view. Its native generation checks remain active.
      const Lin linearization;
      std::shared_ptr<EndpointBulkDirect> bulk;
      FV shape;
      ReconstructedFaultActiveSet active;
      std::vector<std::pair<unsigned int,unsigned int>> free;
      std::vector<BV> responses;
      LAPACKFullMatrix<double> schur;
  };
}}
#endif
