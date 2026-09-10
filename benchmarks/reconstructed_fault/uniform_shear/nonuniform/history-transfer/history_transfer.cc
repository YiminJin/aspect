/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/particle/manager.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <fstream>
#include <iomanip>
#include <set>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class FrozenHistoryTransfer : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim == 2, ExcNotImplemented());
          const auto comm = this->get_mpi_communicator();
          const auto &intro = this->introspection();
          const auto &fe = this->get_fe();
          const auto &dofs = this->get_dof_handler();
          const auto &particles = this->get_particle_manager(0);
          const auto &handler = particles.get_particle_handler();
          const auto &info = particles.get_property_manager().get_data_info();
          const unsigned int stress_property = info.get_position_by_field_name("maxwell stress")+2;
          const unsigned int constant_property = info.get_position_by_field_name("phase field fault state");
          const unsigned int component = intro.component_indices.compositional_fields[2];
          const unsigned int block = intro.block_indices.compositional_fields[2];
          const auto &points = fe.base_element(intro.base_elements.compositional_fields[2]).get_unit_support_points();
          FEValues<dim> supports(this->get_mapping(),fe,points,update_quadrature_points);
          const auto &quadrature = intro.quadratures.velocities;
          FEValues<dim> reference(this->get_mapping(),fe,QGauss<dim>(4),
                                 update_values | update_gradients | update_JxW_values);
          auto &working = const_cast<LinearAlgebra::BlockVector &>(this->get_current_linearization_point());
          const LinearAlgebra::BlockVector saved(working), published(this->get_solution());
          LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,comm);
          LinearAlgebra::BlockVector transfer(intro.index_sets.system_partitioning,comm);
          LinearAlgebra::BlockVector contributions(intro.index_sets.system_partitioning,comm);
          AffineConstraints<double> homogeneous(this->get_current_constraints());
          for (const auto &line : homogeneous.get_lines())
            homogeneous.set_inhomogeneity(line.index,0.);
          homogeneous.close();
          std::vector<typename DoFHandler<dim>::active_cell_iterator> cells;
          for (const auto &cell : dofs.active_cell_iterators())
            if (cell->is_locally_owned()) cells.push_back(cell);
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          const unsigned int nc = dim*fe.base_element(intro.base_elements.velocities).dofs_per_cell
                                  +fe.base_element(intro.base_elements.pressure).dofs_per_cell;
          internal::Assembly::Scratch::StokesSystem<dim> scratch(
            fe,this->get_mapping(),quadrature,intro.face_quadratures.velocities,
            update_values | update_gradients | update_quadrature_points | update_JxW_values,
            update_default,intro.n_compositional_fields,nc,false,false,false,false,false,false);
          internal::Assembly::CopyData::StokesSystem<dim> data(nc,false);
          auto &assembler = this->get_reconstructed_fault_stokes_coupling();
          this->get_reconstructed_fault_manager().prepare_stokes_qp_projection_cache();
          auto csv = [&](const std::string &name, const std::string &header)
          {
            std::ofstream out(this->get_output_directory()+name+"_rank"+
                              std::to_string(Utilities::MPI::this_mpi_process(comm))+".csv");
            out.exceptions(std::ios::failbit | std::ios::badbit);
            out << std::setprecision(17) << header << '\n';
            return out;
          };
          auto particle_output = csv("particles","id,x,y,stress,constant");
          for (const auto &p : handler)
            particle_output << p.get_id() << ',' << p.get_location()[0] << ',' << p.get_location()[1]
                            << ',' << p.get_properties()[stress_property] << ',' << p.get_properties()[constant_property] << '\n';
          std::map<CellId,Vector<double>> zero_load;

          // Only the traversal of the test replay changes. The interpolator,
          // shared-node ADD/count semantics and distributed vector match production.
          for (unsigned int mode=0; mode<5; ++mode)
            {
              const bool constant = mode==1 || mode==2;
              const bool reverse = mode==2 || mode==4;
              const unsigned int property = constant ? constant_property : stress_property;
              transfer=0.;
              contributions=0.;
              ComponentMask mask(handler.n_properties_per_particle(),false);
              mask.set(property,true);
              for (unsigned int c=0; c<cells.size(); ++c)
                {
                  const auto &cell=cells[reverse ? cells.size()-1-c : c];
                  supports.reinit(cell);
                  const auto values=particles.get_interpolator().properties_at_points(
                    handler,supports.get_quadrature_points(),mask,cell);
                  cell->get_dof_indices(indices);
                  for (unsigned int i=0; i<points.size(); ++i)
                    {
                      const auto index=indices[fe.component_to_system_index(component,i)];
                      transfer[index]+=mode==0 ? 0. : values[i][property];
                      contributions[index]+=1.;
                    }
                }
              transfer.compress(VectorOperation::add);
              contributions.compress(VectorOperation::add);
              for (const auto index : transfer.block(block).locally_owned_elements())
                transfer.block(block)[index]/=contributions.block(block)[index];
              if (mode==3)
                {
                  LinearAlgebra::BlockVector delta(intro.index_sets.system_partitioning,comm);
                  delta=published;
                  delta.block(block)-=transfer.block(block);
                  AssertThrow(delta.block(block).linfty_norm()<1e-10,
                              ExcMessage("Forward replay differs from production-published history."));
                }
              // The published field is not the field used by mechanics:
              // physical constraints act on a private copy, without publication.
              owned=saved;
              owned.block(block)=transfer.block(block);
              LinearAlgebra::BlockVector unconstrained(working);
              unconstrained=owned;
              this->get_current_constraints().distribute(owned);
              working=owned;
              LinearAlgebra::BlockVector load(intro.index_sets.system_partitioning,comm);
              LinearAlgebra::BlockVector exact(intro.index_sets.system_partitioning,comm);
              auto fields=csv("fields_"+std::to_string(mode),"x,y,weight,published,working");
              double reference_error=0., reference_size=0.;
              for (const auto &cell : cells)
                {
                  scratch.reinit(cell);
                  data.local_rhs=0.;
                  data.local_frozen_fault_rhs=0.;
                  assembler.execute(scratch,data);
                  if (mode==0)
                    {
                      zero_load.emplace(cell->id(),data.local_frozen_fault_rhs);
                      continue;
                    }
                  Vector<double> increment=data.local_frozen_fault_rhs;
                  increment-=zero_load.at(cell->id());
                  const auto &values=scratch.finite_element_values;
                  std::vector<double> raw(quadrature.size()), constrained(quadrature.size());
                  values[intro.extractors.compositional_fields[2]].get_function_values(unconstrained,raw);
                  values[intro.extractors.compositional_fields[2]].get_function_values(working,constrained);
                  for (unsigned int q=0; q<quadrature.size(); ++q)
                    {
                      if (constant)
                        AssertThrow(std::abs(raw[q]-200.)<1e-10 && std::abs(constrained[q]-200.)<1e-10,
                                    ExcMessage("Nonzero constant history is not reproduced."));
                      fields << values.quadrature_point(q)[0] << ',' << values.quadrature_point(q)[1]
                             << ',' << values.JxW(q) << ',' << raw[q] << ',' << constrained[q] << '\n';
                    }
                  // Independent quadrature of the realized FE field checks sign
                  // and weak assembly, not exact analytic cell-average reproduction.
                  reference.reinit(cell);
                  std::vector<double> stress(reference.n_quadrature_points);
                  reference[intro.extractors.compositional_fields[2]].get_function_values(working,stress);
                  Vector<double> full(fe.dofs_per_cell), expected(fe.dofs_per_cell);
                  for (unsigned int i=0,j=0; i<fe.dofs_per_cell; ++i)
                    if (intro.is_stokes_component(fe.system_to_component_index(i).first))
                      {
                        full[i]=increment[j++];
                        for (unsigned int q=0; q<stress.size(); ++q)
                          {
                            SymmetricTensor<2,dim> tensor;
                            tensor[0][1]=std::exp(-.02)*stress[q];
                            expected[i]-=tensor*reference[intro.extractors.velocities].symmetric_gradient(i,q)*reference.JxW(q);
                          }
                        reference_error=std::max(reference_error,std::abs(full[i]-expected[i]));
                        reference_size=std::max(reference_size,std::abs(expected[i]));
                      }
                  cell->get_dof_indices(indices);
                  homogeneous.distribute_local_to_global(full,indices,load);
                  homogeneous.distribute_local_to_global(expected,indices,exact);
                }
              if (mode==0) continue;
              load.compress(VectorOperation::add);
              exact.compress(VectorOperation::add);
              exact-=load;
              reference_error=Utilities::MPI::max(reference_error,comm);
              reference_size=Utilities::MPI::max(reference_size,comm);
              AssertThrow(reference_error<1e-11*reference_size && exact.l2_norm()<1e-9,
                          ExcMessage("Production frozen-history load differs from independent FE weak integration."));
              if (constant)
                AssertThrow(load.l2_norm()<1e-9,ExcMessage("Constant history weak cancellation failed."));
              auto loads=csv("load_"+std::to_string(mode),"x,y,component,load");
              std::set<types::global_dof_index> written;
              const auto &unit_points=fe.get_unit_support_points();
              for (const auto &cell : cells)
                {
                  cell->get_dof_indices(indices);
                  for (unsigned int i=0; i<indices.size(); ++i)
                    if (fe.system_to_component_index(i).first<dim && dofs.locally_owned_dofs().is_element(indices[i])
                        && written.insert(indices[i]).second)
                      {
                        const auto point=this->get_mapping().transform_unit_to_real_cell(cell,unit_points[i]);
                        loads << point[0] << ',' << point[1] << ',' << fe.system_to_component_index(i).first
                              << ',' << load[indices[i]] << '\n';
                      }
                }
              this->get_pcout() << "Frozen transfer mode " << mode << ": load norm=" << std::setprecision(17)
                               << load.l2_norm() << ", independent load error=" << exact.l2_norm() << std::endl;
            }
          working=saved;
          return {"Frozen transfer audit:","published forward match, constant and independent weak-load controls passed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(FrozenHistoryTransfer,"frozen history transfer",
                                  "Frozen particle-transfer traversal and MPI weak-load audit.")
  }
}
