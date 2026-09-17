/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#ifndef _aspect_fault_interface_preconditioner_internal_h
#define _aspect_fault_interface_preconditioner_internal_h

#include <aspect/simulator/solver/reconstructed_fault_condensed_system.h>
#include <aspect/reconstructed_fault/linear_performance.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace aspect
{
  namespace internal
  {
    /** Few-mode Woodbury approximation. All data die with this active/free
     * solve; neither a history cache nor an approximation to the operator. */
    template <int dim, class BasePreconditioner>
    class FaultInterfacePreconditioner
    {
      using BulkVector=LinearAlgebra::BlockVector;
      using FaultVector=ReconstructedFaultVector;
      using Linearization=typename StokesSolver::ReconstructedFaultCondensedSystem<dim>::Linearization;
      public:
        FaultInterfacePreconditioner(const BasePreconditioner &base,
                                     const Linearization &linearization,
                                     const ReconstructedFaultSurfaceSystem<dim> &surface,
                                     const ReconstructedFaultActiveSet &active,
                                     const BulkVector &layout,
                                     ConditionalOStream &output)
          : base(base), linearization(linearization)
        {
          const char *setting=std::getenv("ASPECT_FAULT_INTERFACE_MODES");
          if (!setting) return;
          const auto start=FaultLinearTiming::Clock::now();
          FaultLinearSection timing(FaultLinearTiming::interface_setup);
          const unsigned int requested=std::atoi(setting);
          AssertThrow(requested>=1 && requested<=4,
                      dealii::ExcMessage("ASPECT_FAULT_INTERFACE_MODES must be 1--4."));
          // Orthogonal cell-centered cosines in vertex-index coordinates.
          // Q is Euclidean-normalized; the same Q appears on both sides of
          // Q^T(K-G P_A B)Q. No joining of active regions or open fault tips.
          for (unsigned int f=0; f<active.size(); ++f)
            for (unsigned int first=0; first<active[f].size();)
              {
                if (active[f][first]) { ++first; continue; }
                unsigned int end=first;
                while (end<active[f].size() && !active[f][end]) ++end;
                for (unsigned int k=0; k<std::min(requested,end-first) && modes.size()<8; ++k)
                  {
                    FaultVector mode=surface.get_linearization_residual().values;
                    for (auto &fault : mode) std::fill(fault.begin(),fault.end(),0.);
                    double norm=0.;
                    for (unsigned int i=first; i<end; ++i)
                      {
                        mode[f][i]=std::cos(numbers::PI*k*(i-first+.5)/(end-first));
                        norm+=mode[f][i]*mode[f][i];
                      }
                    for (double &value : mode[f]) value/=std::sqrt(norm);
                    modes.push_back(std::move(mode));
                  }
                first=end;
              }
          coarse.reinit(modes.size(),modes.size());
          for (unsigned int j=0; j<modes.size(); ++j)
            {
              BulkVector load(layout),response(layout);
              linearization.apply_B(modes[j],load);
              // The existing Schur inverse skips a zero pressure RHS without
              // writing its destination. BQ has exactly that RHS: never seed
              // this response with the nonlinear load copied for its layout.
              response=0.;
              base.vmult(response,load);
              if (std::getenv("ASPECT_FAULT_VERIFY_INTERFACE"))
                AssertThrow(response.block(1).l2_norm()==0.,
                            dealii::ExcMessage("A B-generated interface response retained stale pressure."));
              FaultVector KQ,GY;
              surface.apply_surface_jacobian(modes[j],KQ);
              linearization.apply_G(response,GY);
              for (unsigned int i=0; i<modes.size(); ++i)
                coarse(i,j)=dot(modes[i],KQ)-dot(modes[i],GY);
              responses.push_back(std::move(response));
            }
          // This coarse approximation can be indefinite/nonsymmetric. A
          // singular coarse model disables only this preconditioner correction;
          // physical K_V and every fresh residual test remain untouched.
          try
            { if (!modes.empty()) coarse.compute_lu_factorization(); }
          catch (const dealii::ExceptionBase &error)
            {
              output << "Fault interface correction disabled: " << error.what() << std::endl;
              modes.clear(); responses.clear();
            }
          output << "Fault interface: modes=" << modes.size()
                 << ", owned response bytes=" << modes.size()*layout.locally_owned_size()*sizeof(double)
                 << ", setup inclusive s="
                 << std::chrono::duration<double>(FaultLinearTiming::Clock::now()-start).count()
                 << std::endl;
        }

        void vmult(BulkVector &result, const BulkVector &source) const
        {
          FaultLinearSection timing(FaultLinearTiming::interface_apply,!modes.empty());
          // Satisfy the same zero-RHS contract on reused FGMRES destinations.
          if (!modes.empty()) result=0.;
          base.vmult(result,source);
          if (modes.empty()) return;
          FaultVector Gz;
          linearization.apply_G(result,Gz);
          Vector<double> coefficients(modes.size());
          for (unsigned int i=0; i<modes.size(); ++i) coefficients[i]=dot(modes[i],Gz);
          coarse.solve(coefficients);
          for (unsigned int i=0; i<modes.size(); ++i)
            {
              AssertThrow(std::isfinite(coefficients[i]),dealii::ExcMessage("Nonfinite interface correction."));
              result.add(coefficients[i],responses[i]);
            }
        }

      private:
        static double dot(const FaultVector &left, const FaultVector &right)
        {
          double value=0.;
          for (unsigned int f=0; f<left.size(); ++f)
            for (unsigned int i=0; i<left[f].size(); ++i) value+=left[f][i]*right[f][i];
          return value;
        }
        const BasePreconditioner &base;
        const Linearization &linearization;
        std::vector<FaultVector> modes;
        std::vector<BulkVector> responses;
        LAPACKFullMatrix<double> coarse;
    };
  }
}
#endif
