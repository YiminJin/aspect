/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

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
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

/* Internal implementation data for the restricted accepted FE-history path. */
#ifndef _aspect_material_model_phase_field_fault_fe_strain_internal_h
#define _aspect_material_model_phase_field_fault_fe_strain_internal_h

#include <aspect/material_model/phase_field_fault.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      struct FENormalIntegral
      {
        double v=0.;
        std::array<double,2> d {};
        std::array<std::array<double,2>,2> h {};
      };
    }
    template<int dim> struct PhaseFieldFault<dim>::FENormalization
    {
      Vector<double> phase_values;
      std::vector<double> values,action;
      FullMatrix<double> derivative,derivative_scale;
      std::vector<std::vector<internal::FENormalIntegral>> integrals;
      bool admissible=true;
    };

    template<int dim> struct PhaseFieldFault<dim>::FEStrainState
    {
      struct Ray
      {
        std::array<types::global_dof_index,GeometryInfo<dim>::vertices_per_cell> dofs;
        std::array<std::array<double,GeometryInfo<dim>::vertices_per_cell>,2> shape;
        double length,lower;
        template<class Archive> void serialize(Archive &archive,const unsigned int)
        {
          archive &dofs &shape &length &lower;
        }
      };
      struct Profile
      {
        unsigned int segment;
        double xi,weight;
        std::vector<Ray> rays;
      };
      struct BulkPoint
      {
        CellId cell;
        Point<dim> x;
        double weight,xi;
        unsigned int segment;
        std::array<types::global_dof_index,GeometryInfo<dim>::vertices_per_cell> dofs;
        std::array<double,GeometryInfo<dim>::vertices_per_cell> shape;
      };
      // The FE option's velocity block is authoritative d_R. In the fixed-profile
      // DG option that block stays zero: only the phase/geometry/Z/I data are
      // shared, and the independent DG tensor below owns bulk history.
      LinearAlgebra::BlockVector displacement_history;
      std::shared_ptr<DGHistory> dg_history;
      // Z is slip memory in mature mode, independently of zero cohesive
      // traction. In the cohesive P2 option Z = I_h * tau_coh.
      std::vector<double> Z,I;
      double shear_sense=1.;
      std::vector<Profile> profiles;
      std::vector<BulkPoint> bulk;
      FullMatrix<double> projection;
      std::map<CellId,typename DoFHandler<dim>::active_cell_iterator> cells,evaluation_cells;
      unsigned long long n_cells_global=0,n_particles_global=0;
      std::map<types::particle_index,Point<dim>> particles;
      std::vector<Point<dim>> vertices;
      std::vector<Point<dim>> mesh_vertices;
      // Derived point samples of the immutable FE anchor. This cache never
      // transfers samples back into d_R and is invalidated at publication.
      mutable std::map<std::pair<CellId,std::array<double,dim>>,SymmetricTensor<2,dim>> stress_samples;
      std::uint64_t geometry_version;
      types::global_dof_index n_dofs;
      double m=0.,G=0.,eta=0.,initial_fracture=0.;
      unsigned int committed_step=0;
      std::uint64_t accepted_publications=0;
    };

    template<int dim> struct PhaseFieldFault<dim>::FEStrainUpdate
    {
      LinearAlgebra::BlockVector displacement_history;
      std::shared_ptr<DGHistory> dg_history;
      std::vector<double>Z,I;
      // Only PF5 candidates use these publication preconditions.
      std::shared_ptr<const DGHistory> dg_parent;
      std::vector<double> dg_cohesive,dg_rates;
      unsigned int dg_step=0;
      double dg_time=0.,dg_dt=0.;
    };

  }
}
#endif
