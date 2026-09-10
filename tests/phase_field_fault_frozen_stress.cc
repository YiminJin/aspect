/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator/assemblers/reconstructed_fault_stokes.h>
#include <aspect/reconstructed_fault/manager.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyFrozenMaxwellLoad : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim == 2, ExcNotImplemented());
          const auto &intro = this->introspection();
          const auto &fe = this->get_fe();
          const auto &quadrature = intro.quadratures.velocities;
          auto &manager = this->get_reconstructed_fault_manager();
          auto &assembler = this->get_reconstructed_fault_stokes_coupling();
          manager.prepare_stokes_qp_projection_cache();
          auto &working = const_cast<LinearAlgebra::BlockVector &>(this->get_current_linearization_point());
          const LinearAlgebra::BlockVector saved(working);
          const unsigned int nc = dim*fe.base_element(intro.base_elements.velocities).dofs_per_cell
                                  + fe.base_element(intro.base_elements.pressure).dofs_per_cell;
          internal::Assembly::Scratch::StokesSystem<dim> scratch(
            fe, this->get_mapping(), quadrature, intro.face_quadratures.velocities,
            update_values | update_gradients | update_quadrature_points | update_JxW_values,
            update_default, intro.n_compositional_fields, nc,
            false, false, false, false, false, false);
          internal::Assembly::CopyData::StokesSystem<dim> data(nc,false);
          std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
          std::map<CellId,Vector<double>> zero_history_rhs;
          AffineConstraints<double> constraints(this->get_current_constraints());
          for (const auto &line : this->get_current_constraints().get_lines())
            constraints.set_inhomogeneity(line.index,0.);
          constraints.close();
          const double beta = std::exp(-.02), kappa = -1e8*std::expm1(-.02);
          // K1 uses cell-average particle interpolation, not exact linear
          // reproduction. Independently integrate the realized Q2 FE history
          // with a different rule; do not change that initialization policy.
          FEValues<dim> reference_values(this->get_mapping(),fe,QGauss<dim>(4),
            update_values | update_gradients | update_quadrature_points | update_JxW_values);

          // Run the same production execute with zero, constant, then actual
          // initialized nonuniform FE history. V and all other fields stay fixed.
          for (unsigned int mode=0; mode<3; ++mode)
            {
              LinearAlgebra::BlockVector owned(intro.index_sets.system_partitioning,
                                                this->get_mpi_communicator());
              owned = saved;
              if (mode < 2)
                for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                  if (cell->is_locally_owned())
                    {
                      cell->get_dof_indices(indices);
                      for (unsigned int i=0; i<indices.size(); ++i)
                        if (fe.system_to_component_index(i).first == intro.component_indices.compositional_fields[2]
                            && this->get_dof_handler().locally_owned_dofs().is_element(indices[i]))
                          owned[indices[i]] = mode == 0 ? 0. : 1500.;
                    }
              owned.compress(VectorOperation::insert);
              working = owned;
              LinearAlgebra::BlockVector assembled(intro.index_sets.system_partitioning,
                                                    this->get_mpi_communicator());
              double error=0., expected_size=0., history_work=0., reference_work=0., outside_work=0., full_work=0.;
              for (const auto &cell : this->get_dof_handler().active_cell_iterators())
                if (cell->is_locally_owned())
                  {
                    scratch.reinit(cell);
                    data.local_rhs=0.;
                    data.local_frozen_fault_rhs=0.;
                    assembler.execute(scratch,data);
                    if (mode == 0)
                      {
                        zero_history_rhs.emplace(cell->id(),data.local_frozen_fault_rhs);
                        continue;
                      }
                    const auto &values=scratch.finite_element_values;
                    const auto &associations=manager.get_stokes_qp_fault_associations(
                      cell->id(),quadrature,values.get_quadrature_points());
                    std::vector<double> old_stress(quadrature.size()), phi(quadrature.size());
                    std::vector<Tensor<2,dim>> gradients(quadrature.size());
                    values[intro.extractors.compositional_fields[2]].get_function_values(working,old_stress);
                    values[intro.extractors.velocities].get_function_gradients(working,gradients);
                    values[FEValuesExtractors::Scalar(intro.variable("phase_field").first_component_index)]
                      .get_function_values(working,phi);
                    Vector<double> expected(nc), increment=data.local_frozen_fault_rhs;
                    increment -= zero_history_rhs.at(cell->id());
                    reference_values.reinit(cell);
                    std::vector<double> reference_stress(reference_values.n_quadrature_points);
                    reference_values[intro.extractors.compositional_fields[2]].get_function_values(working,reference_stress);
                    for (unsigned int q=0; q<reference_values.n_quadrature_points; ++q)
                      {
                        SymmetricTensor<2,dim> stress;
                        stress[0][1]=beta*reference_stress[q];
                        for (unsigned int i=0,j=0; j<nc; ++i)
                          if (intro.is_stokes_component(fe.system_to_component_index(i).first))
                            expected[j++] -= stress*reference_values[intro.extractors.velocities].symmetric_gradient(i,q)*reference_values.JxW(q);
                        reference_work-=2.*reference_values.quadrature_point(q)[1]*beta*reference_stress[q]*reference_values.JxW(q);
                      }
                    for (unsigned int q=0; q<quadrature.size(); ++q)
                      {
                        const double y=values.quadrature_point(q)[1];
                        if (mode==1)
                          AssertThrow(std::abs(old_stress[q]-1500.)<1e-8,
                                      ExcMessage("Constant test history was not installed."));
                        const double work=-2.*y*beta*old_stress[q]*values.JxW(q);
                        history_work+=work;
                        if (!associations[q].active)
                          outside_work+=work;
                        if (mode==2)
                          {
                            double slip=0.;
                            if (associations[q].active)
                              {
                                const auto &a=associations[q];
                                const auto &fault=manager.get_fault(a.fault_index);
                                const unsigned int p=manager.get_property_information()[
                                  manager.get_property_index("phase field fault previous I h")].position;
                                const double Ih=(1-a.xi)*fault.get_properties(a.segment_index)[p]
                                                 +a.xi*fault.get_properties(a.segment_index+1)[p];
                                const double effective=std::max(0.,phi[q]);
                                const double h=128.*effective*(1+effective)/std::pow(1-effective,2);
                                slip=h/Ih*manager.interpolate_slip_rate(a.fault_index,a.segment_index,a.xi);
                              }
                            // w=(y^2-1/4,0) is an admissible Q2 test function.
                            // Check the complete accepted production equilibrium,
                            // so a duplicate term elsewhere cannot escape the test.
                            const double tau=kappa*(gradients[q][0][1]+gradients[q][1][0]-slip)+beta*old_stress[q];
                            full_work+=2.*y*tau*values.JxW(q);
                          }
                      }
                    for (unsigned int j=0; j<nc; ++j)
                      {
                        error=std::max(error,std::abs(increment[j]-expected[j]));
                        expected_size=std::max(expected_size,std::abs(expected[j]));
                      }
                    Vector<double> full_local(fe.dofs_per_cell);
                    for (unsigned int i=0,j=0; j<nc; ++i)
                      if (intro.is_stokes_component(fe.system_to_component_index(i).first))
                        full_local[i]=increment[j++];
                    cell->get_dof_indices(indices);
                    constraints.distribute_local_to_global(full_local,indices,assembled);
                  }
              if (mode==0)
                continue;
              assembled.compress(VectorOperation::add);
              error=Utilities::MPI::max(error,this->get_mpi_communicator());
              expected_size=Utilities::MPI::max(expected_size,this->get_mpi_communicator());
              history_work=Utilities::MPI::sum(history_work,this->get_mpi_communicator());
              reference_work=Utilities::MPI::sum(reference_work,this->get_mpi_communicator());
              outside_work=Utilities::MPI::sum(outside_work,this->get_mpi_communicator());
              full_work=Utilities::MPI::sum(full_work,this->get_mpi_communicator());
              AssertThrow(error < 1e-11*expected_size, ExcMessage("Frozen stress RHS has wrong sign, support or multiplicity."));
              if (mode==1)
                AssertThrow(assembled.l2_norm()<1e-9,
                            ExcMessage("Constant frozen stress failed homogeneous weak cancellation."));
              else
                {
                  AssertThrow(std::abs(history_work-reference_work)<1e-9
                              && reference_work < -30. && assembled.l2_norm()>1.,
                              ExcMessage("Nonuniform frozen stress has no correct nonzero weak contribution."));
                  AssertThrow(std::abs(outside_work)>.5*std::abs(history_work),
                              ExcMessage("Fixture does not exercise bulk history outside fault support."));
                  AssertThrow(std::abs(full_work)<1e-7,
                              ExcMessage("Accepted full bulk residual omits or double-counts frozen stress."));
                  this->get_pcout()<<"Frozen stress weak work / outside / equilibrium: "
                                   <<history_work<<" / "<<outside_work<<" / "<<full_work<<std::endl;
                }
            }
          working=saved;
          return {"Frozen Maxwell bulk load:","sign, full support, single inclusion and constant cancellation verified"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyFrozenMaxwellLoad,"uniform shear pilot",
                                  "Check the frozen-history weak form without changing committed histories.")
  }
}
