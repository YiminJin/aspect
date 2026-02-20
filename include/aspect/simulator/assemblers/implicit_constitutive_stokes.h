/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_simulator_assemblers_implicit_constitutive_stokes_h
#define _aspect_simulator_assemblers_implicit_constitutive_stokes_h

#include <aspect/simulator/assemblers/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Assemblers
  {
    template <int dim>
    class ImplicitConstitutiveStokesPreconditioner : public Assemblers::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void
        execute(internal::Assembly::Scratch::ScratchBase<dim>   &scratch_base,
                internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override;

        void create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const override;
    };

    template <int dim>
    class ImplicitConstitutiveStokesSystem : public Assemblers::Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        void
        execute(internal::Assembly::Scratch::ScratchBase<dim>   &scratch,
                internal::Assembly::CopyData::CopyDataBase<dim> &data) const override;

        void
        create_additional_material_model_outputs(MaterialModel::MaterialModelOutputs<dim> &outputs) const override;
    };
  }
}

#endif
