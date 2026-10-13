/* Internal implementation data for the restricted accepted FE-history path. */
#ifndef _aspect_material_model_phase_field_fault_fe_strain_internal_h
#define _aspect_material_model_phase_field_fault_fe_strain_internal_h

#include <aspect/material_model/phase_field_fault.h>

namespace aspect
{
  namespace MaterialModel
  {
    template<int dim> struct PhaseFieldFault<dim>::FEStrainState
    {
      struct Ray
      {
        std::array<types::global_dof_index,GeometryInfo<dim>::vertices_per_cell> dofs;
        std::array<std::array<double,GeometryInfo<dim>::vertices_per_cell>,2> shape;
        double length,lower;
        template<class Archive> void serialize(Archive &archive,const unsigned int)
        {archive & dofs & shape & length & lower;}
      };
      struct Profile
      {
        unsigned int segment;double xi,weight;
        std::vector<Ray> rays;
      };
      struct BulkPoint
      {
        CellId cell;Point<dim> x;double weight,xi;
        unsigned int segment;
        std::array<types::global_dof_index,GeometryInfo<dim>::vertices_per_cell> dofs;
        std::array<double,GeometryInfo<dim>::vertices_per_cell> shape;
      };
      LinearAlgebra::BlockVector accepted;
      std::vector<double> Z,I;
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
      double m,G,eta,initial_fracture=0.;
      unsigned int committed_step=0;
    };

    template<int dim> struct PhaseFieldFault<dim>::FEStrainUpdate
    {
      LinearAlgebra::BlockVector accepted;
      std::vector<double>Z,I;
    };

  }
}
#endif
