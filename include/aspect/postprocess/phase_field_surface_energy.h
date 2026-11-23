/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#ifndef _aspect_postprocess_phase_field_surface_energy_h
#define _aspect_postprocess_phase_field_surface_energy_h

#include <aspect/phase_field/crack_geometry.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Postprocess
  {
    /**
     * Integrate the local and gradient contributions to crack-surface energy
     * from the current compositional FE field.
     * @ingroup Postprocessing
     */
    template <int dim>
    class PhaseFieldSurfaceEnergy : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &statistics) override;
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

      private:
        const PhaseField::CrackGeometry *geometry = nullptr;
        unsigned int phase_index;
        double fracture_energy, length_scale;
    };
  }
}
#endif
