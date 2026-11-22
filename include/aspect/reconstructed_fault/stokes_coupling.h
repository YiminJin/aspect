/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_reconstructed_fault_stokes_coupling_h
#define _aspect_reconstructed_fault_stokes_coupling_h

#include <aspect/reconstructed_fault/stationary_fault.h>
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
      /** Register the stress load, plane-strain correction and timestep hooks. */
      void initialize();
      /** Rotate, relax and project the accepted physical stress on owned cells. */
      void prepare_parent(double relaxation);
      /** Spatial stress (Pa) added to the implicit plane-strain viscous stress. */
      SymmetricTensor<2,dim> stress_offset(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                           const Point<dim> &position) const;

    private:
      using Stress = SymmetricTensor<2,dim>;
      struct BulkPoint
      {
        typename StationaryFault<dim>::Projection foot;
        double weight, driving_shear, compression;
      };

      const FiniteElement<dim> &stress_element() const;
      std::vector<BulkPoint> sample_bulk(const LinearAlgebra::BlockVector &iterate) const;
      std::vector<double> solve_rates(const LinearAlgebra::BlockVector &iterate) const;
      void prepare_iteration(const LinearAlgebra::BlockVector &iterate, bool &rebuild);
      void check_convergence(const LinearAlgebra::BlockVector &solution, bool &converged);
      void accept_step(LinearAlgebra::BlockVector &solution);
      // The material owns both the physical parameters and the accepted curve.
      MaterialModel::PhaseFieldFault<dim> &material;
      std::vector<double> trial_rates;
      /** DGQ2 coefficients of the attempted rotated/relaxed parent, in Pa. */
      std::map<CellId,std::vector<Stress>> relaxed_parent;
  };
}
#endif
