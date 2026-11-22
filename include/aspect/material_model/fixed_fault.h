/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_material_model_fixed_fault_h
#define _aspect_material_model_fixed_fault_h

#include <aspect/material_model/simple.h>
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/reconstructed_fault/stationary_fault.h>

namespace aspect
{
  namespace MaterialModel
  {
    /** Mature stationary fault in plane strain. Physical spatial stress lives
     * in the simulator's three static DGQ2 fields. The curve owns the
     * material labels and their accepted history. */
    template <int dim>
    class FixedFault : public Simple<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;
        void update() override;
        void evaluate(const MaterialModelInputs<dim> &in, MaterialModelOutputs<dim> &out) const override;
        void create_additional_named_outputs(MaterialModelOutputs<dim> &out) const override;

        /** Solve the section balance at the given bulk Picard iterate.
         * Only the rate used in the next Stokes assembly is replaced. */
        void prepare_stokes(const LinearAlgebra::BlockVector &iterate);

        /** Publish the converged section and DG stress after the repeat
         * decision. The zero-duration initialization publishes only a rate. */
        void accept_step(LinearAlgebra::BlockVector &solution);

        /** Ordered geometry and last published age, slip and rate at each label. */
        const std::vector<typename StationaryFault<dim>::Label> &labels() const;
        void save(std::map<std::string,std::string> &status) const override;
        void load(const std::map<std::string,std::string> &status) override;

      private:
        using Stress = SymmetricTensor<2,dim>;
        using BulkPoint = typename StationaryFault<dim>::BulkPoint;
        std::vector<BulkPoint> sample_bulk(const LinearAlgebra::BlockVector &iterate) const;
        Stress stress_offset(const typename DoFHandler<dim>::active_cell_iterator &cell,
                             const Point<dim> &position, const std::vector<double> &rates) const;
        const FiniteElement<dim> &stress_element() const;

        Rheology::RateAndStateFriction friction;
        StationaryFault<dim> fault;
        double viscosity, modulus, background_shear, background_compression;
        double initial_interval, minimum_speed, kappa;
        std::vector<double> trial_rates;
        /** Owned cell-local DGQ2 coefficients of the rotated and relaxed
         * physical parent (Pa). Rebuilt from old_solution on each attempt. */
        std::map<CellId,std::vector<Stress>> relaxed_parent;
    };
  }
}
#endif
