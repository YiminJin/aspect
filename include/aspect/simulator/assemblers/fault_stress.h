/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#ifndef _aspect_simulator_assemblers_fault_stress_h
#define _aspect_simulator_assemblers_fault_stress_h

#include <aspect/simulator/assemblers/interface.h>

namespace aspect
{
  namespace MaterialModel
  {
    /** Spatial stress offset in Pa at Stokes quadrature points. The implicit
     * viscous stress is already supplied by the ordinary velocity matrix. */
    template <int dim>
    struct FaultStressOffset : AdditionalMaterialOutputs<dim>
    {
      explicit FaultStressOffset(unsigned int size) : stress(size) {}
      std::vector<SymmetricTensor<2,dim>> stress;
    };
  }

  namespace Assemblers
  {
    /** Add the negative internal virtual work of the material stress offset
     * to the velocity RHS on both full and RHS-only assembly. */
    template <int dim>
    class FaultStressLoad : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const override;
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch,
                     internal::Assembly::CopyData::CopyDataBase<dim> &data) const override;
    };
  }
}
#endif
