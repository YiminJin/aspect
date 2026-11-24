/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_reconstructed_fault_stokes_coupling_h
#define _aspect_reconstructed_fault_stokes_coupling_h

#include <aspect/reconstructed_fault/phase_profile.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim> class PhaseFieldFault;
  }

  /** FE and MPI operations coupling a stationary section to Stokes. All
   * scratch is rebuilt from the accepted bulk fields and labels at attempt entry. */
  template <int dim>
  class FaultStokesCoupling : public SimulatorAccess<dim>
  {
    public:
      explicit FaultStokesCoupling(MaterialModel::PhaseFieldFault<dim> &material);
      /** Register plane-strain Stokes assembly and accepted-state publication. */
      void initialize();
      /** Couple trial rates and convergence to the fixed-age RSF section solve. */
      void connect_section_balance();
      /** Transport (when selected), rotate and project the accepted physical stress on owned cells. */
      void prepare_parent();
      /** Spatial stress (Pa) added to the implicit plane-strain viscous stress. */
      SymmetricTensor<2,dim> stress_offset(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           const Point<dim> &position, double phase) const;

      /** One bulk quadrature contribution to section balance. The tangent
       * is minus the fixed-age derivative with respect to label rates. */
      struct BulkPoint
      {
        typename FaultPhaseProfile<dim>::Sample transfer;
        double volume, driving_shear, compression, stiffness;
        void add_balance(const std::vector<double> &rates, double age,
                         const MaterialModel::Rheology::RateAndStateFriction &law,
                         std::vector<double> &rhs, std::vector<double> &diagonal,
                         std::vector<double> &upper, std::vector<double> &scale) const;
      };

      /** Collectively sample the supplied iterate against the current attempt's
       * relaxed parent. Every simulator rank participates in the closed-fault
       * check. The returned nonzero-support points are locally owned. */
      std::vector<BulkPoint> sample_bulk(const LinearAlgebra::BlockVector &iterate) const;
    private:
      friend class MaterialModel::PhaseFieldFault<dim>;
      using Stress = SymmetricTensor<2,dim>;
      const FiniteElement<dim> &stress_element() const;
      std::vector<double> solve_rates(const LinearAlgebra::BlockVector &iterate) const;
      void prepare_iteration(const LinearAlgebra::BlockVector &iterate, bool &rebuild);
      void check_convergence(const LinearAlgebra::BlockVector &solution, bool &converged);
      void accept_step(LinearAlgebra::BlockVector &solution);
      // The material owns both the physical parameters and the accepted curve.
      MaterialModel::PhaseFieldFault<dim> &material;
      std::vector<double> trial_rates;
      // Moving mechanics uses Q8 cell moments of the complete physical response.
      // Freeze its projection defect while Stokes and the Q1 section are linearized.
      struct Reaction
      {
        std::vector<Stress> stress, strain;
      };
      std::map<CellId,Reaction> reaction;
      std::vector<double> reaction_rates;
      double reaction_change=0.;
      void project_reaction(const LinearAlgebra::BlockVector &iterate);
      /** Owned DGQ2 rotated spatial parent in Pa. Apply the local Maxwell beta
       * after interpolation to each physical constitutive point. */
      std::map<CellId,std::vector<Stress>> objective_parent;
  };
}
#endif
