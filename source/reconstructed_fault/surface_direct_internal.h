/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_surface_direct_internal_h
#define _aspect_surface_direct_internal_h

#include <aspect/global.h>
#ifdef DEAL_II_WITH_TRILINOS
#include <Teuchos_LAPACK.hpp>
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace aspect
{
  namespace internal
  {
    /** One replicated fault's principal free matrix. No history or MPI state.
     * LAPACK's adjacent row pivoting allows zero/negative diagonal entries and
     * creates a second superdiagonal; no definiteness assumption is made. */
    class FaultSurfaceDirect
    {
      public:
        FaultSurfaceDirect(const std::vector<double> &diagonal,
                           const std::vector<double> &off_diagonal,
                           const std::vector<bool> &active,
                           const unsigned int fault,
                           const std::vector<double> &lower_diagonal={})
          : diagonal(diagonal), off_diagonal(off_diagonal),
            lower_diagonal(lower_diagonal.empty() ? off_diagonal : lower_diagonal), active(active),
            fault(fault)
        {
          AssertDimension(active.size(),diagonal.size());
          AssertDimension(off_diagonal.size(),diagonal.empty() ? 0 : diagonal.size()-1);
          AssertDimension(this->lower_diagonal.size(),off_diagonal.size());
          // The specialized solve factors solely contiguous free principal blocks.
          for (unsigned int i=0;i<diagonal.size();++i)
            {
              AssertThrow(std::isfinite(diagonal[i]),dealii::ExcMessage(context(i)+": nonfinite diagonal."));
              if (i<off_diagonal.size())
                AssertThrow(std::isfinite(off_diagonal[i]) && std::isfinite(this->lower_diagonal[i]),
                            dealii::ExcMessage(context(i)+": nonfinite edge."));
            }
          for (unsigned int i=0;i<active.size();)
            {
              if (active[i]) { ++i; continue; }
              const unsigned int first=i;
              while (i<active.size() && !active[i]) ++i;
              Block block;
              block.first=first; block.size=i-first;
              block.d.assign(diagonal.begin()+first,diagonal.begin()+i);
              block.dl.assign(this->lower_diagonal.begin()+first,this->lower_diagonal.begin()+i-1);
              block.du.assign(off_diagonal.begin()+first,off_diagonal.begin()+i-1);
              block.du2.resize(block.size>2 ? block.size-2 : 1);
              block.pivots.resize(block.size);
              int info=0;
#ifdef DEAL_II_WITH_TRILINOS
              Teuchos::LAPACK<int,double>().GTTRF(block.size,block.dl.data(),block.d.data(),
                block.du.data(),block.du2.data(),block.pivots.data(),&info);
#else
              AssertThrow(false,dealii::ExcMessage("Pivoted surface solve requires the Trilinos LAPACK wrapper."));
#endif
              AssertThrow(info==0,dealii::ExcMessage(context(first)+": GTTRF failed, info="
                +std::to_string(info)+(info>0 ? ", singular pivot vertex="+std::to_string(first+info-1) : "")));
              for (unsigned int j=0;j<block.d.size();++j)
                AssertThrow(std::isfinite(block.d[j]) && block.d[j]!=0.,
                            dealii::ExcMessage(context(first+j)+": nonfinite/zero factored pivot."));
              for (const auto *values : {&block.dl,&block.du,&block.du2})
                for (const double value : *values)
                  AssertThrow(std::isfinite(value),dealii::ExcMessage(context(first)+": nonfinite LU factor."));
              blocks.push_back(std::move(block));
            }
        }

        void solve(const std::vector<double> &rhs,std::vector<double> &solution) const
        {
          AssertDimension(rhs.size(),diagonal.size());
          solution.assign(rhs.size(),0.);
          for (unsigned int i=0;i<rhs.size();++i)
            if (!active[i])
              AssertThrow(std::isfinite(rhs[i]),dealii::ExcMessage(context(i)+": nonfinite free RHS."));
          for (const auto &block : blocks)
            {
              std::copy_n(rhs.begin()+block.first,block.size,solution.begin()+block.first);
              int info=0;
#ifdef DEAL_II_WITH_TRILINOS
              Teuchos::LAPACK<int,double>().GTTRS('N',block.size,1,block.dl.data(),block.d.data(),
                block.du.data(),block.du2.data(),block.pivots.data(),solution.data()+block.first,block.size,&info);
#endif
              AssertThrow(info==0,dealii::ExcMessage(context(block.first)+": GTTRS failed, info="+std::to_string(info)));
            }
          // Check each free block against the original coefficients, not LU.
          // Keep the existing 100*epsilon*n_fault allowance, now with block-local
          // scales so a large neighboring block cannot hide a bad small solve.
          for (const auto &block : blocks)
            {
              double residual=0.,matrix_norm=0.,solution_norm=0.,rhs_norm=0.;
              unsigned int worst=block.first;
              for (unsigned int i=block.first;i<block.first+block.size;++i)
                {
                  AssertThrow(std::isfinite(solution[i]),dealii::ExcMessage(context(i)+": nonfinite solution."));
                  double value=diagonal[i]*solution[i],row=std::abs(diagonal[i]);
                  if (i>block.first)
                    { value+=lower_diagonal[i-1]*solution[i-1]; row+=std::abs(lower_diagonal[i-1]); }
                  if (i+1<block.first+block.size)
                    { value+=off_diagonal[i]*solution[i+1]; row+=std::abs(off_diagonal[i]); }
                  AssertThrow(std::isfinite(value) && std::isfinite(row),
                              dealii::ExcMessage(context(i)+": nonfinite backward residual."));
                  if (std::abs(value-rhs[i])>residual) { residual=std::abs(value-rhs[i]); worst=i; }
                  matrix_norm=std::max(matrix_norm,row);
                  solution_norm=std::max(solution_norm,std::abs(solution[i]));
                  rhs_norm=std::max(rhs_norm,std::abs(rhs[i]));
                }
              const double scale=matrix_norm*solution_norm+rhs_norm;
              const double error=residual/std::max(scale,std::numeric_limits<double>::min());
              AssertThrow(std::isfinite(scale) && std::isfinite(error)
                          && error<=100.*std::numeric_limits<double>::epsilon()*std::max<size_t>(1,diagonal.size()),
                          dealii::ExcMessage(context(worst)+", free block=["+std::to_string(block.first)+","+
                            std::to_string(block.first+block.size-1)+"] has excessive scaled backward residual="+
                            dealii::Utilities::to_string(error)));
            }
        }

      private:
        struct Block
        {
          unsigned int first;
          int size;
          std::vector<double> dl,d,du,du2;
          std::vector<int> pivots;
        };
        std::string context(const unsigned int vertex) const
        { return "K_V fault="+std::to_string(fault)+", vertex="+std::to_string(vertex); }
        const std::vector<double> diagonal,off_diagonal,lower_diagonal;
        const std::vector<bool> active;
        const unsigned int fault;
        std::vector<Block> blocks;
    };
  }
}
#endif
