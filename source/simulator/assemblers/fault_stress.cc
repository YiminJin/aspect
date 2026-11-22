/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/simulator/assemblers/fault_stress.h>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    void FaultStressLoad<dim>::create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      if (!out.template has_additional_output_object<MaterialModel::FaultStressOffset<dim>>())
        out.additional_outputs.push_back(std::make_unique<MaterialModel::FaultStressOffset<dim>>(out.n_evaluation_points()));
    }

    template <int dim>
    void FaultStressLoad<dim>::execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                                       internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const
    {
      const auto &scratch = dynamic_cast<const internal::Assembly::Scratch::StokesSystem<dim>&>(scratch_base);
      auto &data = dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(data_base);
      const auto offset = scratch.material_model_outputs.template get_additional_output_object<MaterialModel::FaultStressOffset<dim>>();
      Assert(offset != nullptr, ExcInternalError());
      const auto &fe = scratch.finite_element_values;
      const auto &intro = this->introspection();
      for (unsigned int q=0; q<fe.n_quadrature_points; ++q)
        for (unsigned int i=0; i<data.local_dof_indices.size(); ++i)
          {
            const auto index = intro.stokes_dof_info[i].local_dof_index;
            data.local_rhs(i) -= fe.JxW(q) * (offset->stress[q] * fe[intro.extractors.velocities].symmetric_gradient(index,q));
          }
    }

#define INSTANTIATE(dim) template class FaultStressLoad<dim>;
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
