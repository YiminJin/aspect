/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/phase_field/midpoint_fracture.h>
#include <deal.II/fe/fe_values.h>

namespace aspect::PhaseField
{
  FractureRows::FractureRows(const SimulatorAccess<2> &sim,const unsigned int field)
  {
    const auto &model=sim.get_parameters().surface_energy_model;
    AssertThrow(&model.geometry==&CrackGeometry::select("AT1"),ExcMessage("Growing fault requires the AT1 surface energy."));
    const double k=2.*model.length_scale*model.length_scale;
    const auto &system=sim.get_fe();
    const auto c=sim.introspection().component_indices.compositional_fields[field];
    const auto b=sim.introspection().block_indices.compositional_fields[field];
    const auto offset=sim.get_solution().get_block_indices().block_start(b);
    const auto &element=system.base_element(system.component_to_base_index(c).first);
    const QGauss<2> volume(5);
    const QGauss<1> face(5);
    const auto flags=update_values|update_gradients;
    FEValues<2> fe(sim.get_mapping(),element,volume,flags|update_JxW_values);
    FEFaceValues<2> side(sim.get_mapping(),element,face,flags|update_JxW_values|update_normal_vectors|update_quadrature_points);
    const auto indices=[&](const auto &cell)
    {
      std::vector<Index> global(system.n_dofs_per_cell()), result(element.n_dofs_per_cell());
      cell->get_dof_indices(global);
      for (unsigned int i=0; i<result.size(); ++i) result[i]=global[system.component_to_system_index(c,i)]-offset;
      return result;
    };
    for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto rows=indices(cell);
          fe.reinit(Triangulation<2>::cell_iterator(cell));
          for (unsigned int q=0; q<volume.size(); ++q)
            for (unsigned int i=0; i<rows.size(); ++i)
              {
                mass[rows[i]]+=fe.JxW(q)*fe.shape_value(i,q);
                for (unsigned int j=0; j<rows.size(); ++j)
                  stiffness[ {rows[i],rows[j]}]+=fe.JxW(q)*k*(fe.shape_grad(i,q)*fe.shape_grad(j,q));
              }
          for (const auto f:cell->face_indices())
            if (!cell->at_boundary(f))
              {
                const auto neighbor=cell->neighbor(f);
                AssertThrow(neighbor->is_active() && !cell->neighbor_is_coarser(f),ExcMessage("Growing fault requires conforming faces."));
                side.reinit(Triangulation<2>::cell_iterator(cell),f);
                std::vector<Point<2>> unit;
                for (const auto &p:side.get_quadrature_points()) unit.push_back(sim.get_mapping().transform_real_to_unit_cell(neighbor,p));
                FEValues<2> other(sim.get_mapping(),element,Quadrature<2>(unit),flags);
                other.reinit(Triangulation<2>::cell_iterator(neighbor));
                const auto columns=indices(neighbor);
                const double sigma=sim.get_parameters().phase_discontinuous_penalty*std::pow(element.degree+1,2)*k
                                   *cell->face(f)->measure()/std::min(cell->measure(),neighbor->measure());
                for (unsigned int q=0; q<face.size(); ++q)
                  for (unsigned int i=0; i<rows.size(); ++i)
                    {
                      const double vi=side.shape_value(i,q), di=.5*k*(side.normal_vector(q)*side.shape_grad(i,q));
                      for (unsigned int j=0; j<rows.size(); ++j)
                        {
                          const double jump=side.shape_value(j,q), outside=-other.shape_value(j,q);
                          stiffness[ {rows[i],rows[j]}]+=side.JxW(q)*((sigma*vi-di)*jump-.5*k*vi*(side.normal_vector(q)*side.shape_grad(j,q)));
                          stiffness[ {rows[i],columns[j]}]+=side.JxW(q)*((sigma*vi-di)*outside-.5*k*vi*(side.normal_vector(q)*other.shape_grad(j,q)));
                        }
                    }
              }
        }
  }
}
