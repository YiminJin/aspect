/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive_internal.h>
#include <aspect/material_model/phase_field_fault/bulk_cohesive_mpi.h>
#include <aspect/phase_field.h>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <aspect/utilities.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aspect
{
  namespace MaterialModel
  {
    namespace
    {
      // A constant FE translation has zero gradient. Remove it before summing
      // the accumulated memory, whose coefficients can be much larger than
      // their cellwise differences. This is the same native FE gradient, with
      // reduced cancellation; it changes neither the source nor its projection.
      template <int dim>
      std::vector<SymmetricTensor<2, dim>> memory_strain(
        const FEValues<dim> &fe, const LinearAlgebra::BlockVector &memory,
        const std::array<unsigned int, dim> &components,
        const typename DoFHandler<dim>::active_cell_iterator &cell)
      {
        std::vector<types::global_dof_index> ids(fe.get_fe().n_dofs_per_cell());
        cell->get_dof_indices(ids);
        std::array<long double, dim> origin {};
        std::array<bool, dim> found {};
        for (unsigned int i = 0; i < ids.size(); ++i)
          for (unsigned int c = 0; c < dim; ++c)
            if (fe.get_fe().system_to_component_index(i).first == components[c] && !found[c])
              {
                origin[c] = memory[ids[i]];
                found[c] = true;
              }
        std::vector<SymmetricTensor<2, dim>> result(fe.n_quadrature_points);
        for (unsigned int q = 0; q < fe.n_quadrature_points; ++q)
          {
            std::array<std::array<long double, dim>, dim> gradient {};
            for (unsigned int i = 0; i < ids.size(); ++i)
              for (unsigned int c = 0; c < dim; ++c)
                if (fe.get_fe().system_to_component_index(i).first == components[c])
                  {
                    const long double value = static_cast<long double>(memory[ids[i]]) - origin[c];
                    const auto shape = fe.shape_grad_component(i, q, components[c]);
                    for (unsigned int d = 0; d < dim; ++d)
                      gradient[c][d] += value * shape[d];
                  }
            for (unsigned int c = 0; c < dim; ++c)
              for (unsigned int d = c; d < dim; ++d)
                result[q][c][d] = static_cast<double>((gradient[c][d] + gradient[d][c]) / 2);
          }
        return result;
      }
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::BulkCohesiveState::rebuild_native_projection(
      const PhaseFieldFault<dim> &model)
    {
      native_projection = std::make_shared<TensorField>();
      native_projection->space = tensor->space;
      for (auto &field : native_projection->coefficients)
        field.reinit(tensor->space->dofs.locally_owned_dofs(), model.get_mpi_communicator());
      const QGauss<dim> quadrature(3);
      FEValues<dim> fe(model.get_mapping(), model.get_fe(), quadrature, update_gradients);
      FEValues<dim> dg(model.get_mapping(), tensor->space->element, quadrature, update_values | update_JxW_values);
      std::vector<types::global_dof_index> ids(9);
      for (const auto &entry : native_cells)
        {
          fe.reinit(entry.second);
          const auto gradient = memory_strain<dim>(fe, native_history, model.introspection().component_indices.velocities, entry.second);
          const auto cell = tensor->space->cells.at(entry.first);
          dg.reinit(cell);
          cell->get_dof_indices(ids);
          FullMatrix<double> mass(9, 9);
          std::array<Vector<double>, 3> rhs;
          for (auto &v : rhs)
            v.reinit(9);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              const auto T = 2 * parameters.G * Utilities::Tensors::consistent_deviator(gradient[q]);
              for (unsigned int i = 0; i < 9; ++i)
                {
                  const double w = dg.JxW(q) * dg.shape_value(i, q);
                  for (unsigned int j = 0; j < 9; ++j)
                    mass(i, j) += w * dg.shape_value(j, q);
                  rhs[0][i] += w * T[0][0];
                  rhs[1][i] += w * T[1][1];
                  rhs[2][i] += w * T[0][1];
                }
            }
          mass.gauss_jordan();
          for (unsigned int c = 0; c < 3; ++c)
            {
              Vector<double> value(9);
              mass.vmult(value, rhs[c]);
              for (unsigned int i = 0; i < 9; ++i)
                native_projection->coefficients[c][ids[i]] = value[i];
            }
        }
      for (auto &field : native_projection->coefficients)
        field.compress(VectorOperation::insert);
    }

    template <int dim>
    std::vector<SymmetricTensor<2, dim>>
    PhaseFieldFault<dim>::BulkCohesiveState::evaluate_tensor(
      const PhaseFieldFault<dim> &model, const CellId &id, const Quadrature<dim> &quadrature)const
    {
      auto result = model.evaluate_history_tensor(*tensor, id, quadrature);
      const auto projected = model.evaluate_history_tensor(*native_projection, id, quadrature);
      FEValues<dim> fe(model.get_mapping(), model.get_fe(), quadrature, update_gradients);
      fe.reinit(native_cells.at(id));
      const auto gradient = memory_strain<dim>(fe, native_history, model.introspection().component_indices.velocities, native_cells.at(id));
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        result[q] += 2 * parameters.G * Utilities::Tensors::consistent_deviator(gradient[q]) - projected[q];
      return result;
    }

    template <int dim>
    double
    PhaseFieldFault<dim>::BulkCohesiveState::check_tensor(
      const PhaseFieldFault<dim> &model, const CellId &id, const Quadrature<dim> &quadrature,
      const std::vector<BulkCohesive::Tensor> &expected)const
    {
      const auto actual = evaluate_tensor(model, id, quadrature);
      AssertDimension(actual.size(), expected.size());
      double maximum = 0.;
      unsigned int worst = 0;
      for (unsigned int q = 0; q < actual.size(); ++q)
        {
          BulkCohesive::Tensor T;
          T[0][0] = actual[q][0][0];
          T[1][1] = actual[q][1][1];
          T[0][1] = actual[q][0][1];
          T[2][2] = -T[0][0] - T[1][1];
          const double error = (T - expected[q]).norm() / std::max(1000., T.norm());
          if (error > maximum)
            {
              maximum = error;
              worst = q;
            }
        }
      if (maximum > 1e-12)
        {
          const auto unit = quadrature.point(worst);
          const auto point = model.get_mapping().transform_unit_to_real_cell(native_cells.at(id), unit);
          const auto dg = model.evaluate_history_tensor(*tensor, id, quadrature);
          const auto projection = model.evaluate_history_tensor(*native_projection, id, quadrature);
          std::ofstream out(model.get_output_directory() + "bulk-source-consistency-failure.csv");
          out << std::setprecision(17) << "step,time,cell,q,u,v,x,y,error,bound,ratio,Txx,Tyy,Tzz,Txy,expected_xx,expected_yy,expected_zz,expected_xy,DGxx,DGyy,DGxy,projection_xx,projection_yy,projection_xy\n";
          out << step << ',' << time << ',' << id << ',' << worst << ',' << unit[0] << ',' << unit[1] << ',' << point[0] << ',' << point[1] << ',' << maximum << ",1e-12," << maximum / 1e-12 << ','
              << actual[worst][0][0] << ',' << actual[worst][1][1] << ',' << -actual[worst][0][0] - actual[worst][1][1] << ',' << actual[worst][0][1] << ','
              << expected[worst][0][0] << ',' << expected[worst][1][1] << ',' << expected[worst][2][2] << ',' << expected[worst][0][1] << ','
              << dg[worst][0][0] << ',' << dg[worst][1][1] << ',' << dg[worst][0][1] << ',' << projection[worst][0][0] << ',' << projection[worst][1][1] << ',' << projection[worst][0][1] << std::endl;
          std::ostringstream message;
          message << std::setprecision(17)
                  << "Bulk native DG source projection fails its independent QP check. error=" << maximum
                  << ", bound=1e-12, ratio=" << maximum / 1e-12 << ", cell=" << id << ", reference=" << unit << ", position=" << point;
          AssertThrow(false, ExcMessage(message.str()));
        }
      return maximum;
    }

    template <int dim>
    std::shared_ptr<typename PhaseFieldFault<dim>::BulkCohesiveState>
    PhaseFieldFault<dim>::prepare_bulk_tip_candidate(
      const std::shared_ptr<const BulkCohesiveState> &parent,
      const std::map<CellId,std::vector<BulkCohesive::Tip::Response>> &points,
      const std::vector<double> &rates, const std::vector<double> &forces,
      const LinearAlgebra::Vector &phase) const
    {
      AssertThrow(uses_integrated_bulk_tip() && parent==bulk_cohesive_state
                  && parent->step+1==this->get_timestep_number(),
                  ExcMessage("Tip candidate must use the current immutable accepted parent."));
      const double dt=this->get_timestep();
      auto next=std::make_shared<BulkCohesiveState>(*parent);
      next->tip_phase=phase;next->step=this->get_timestep_number();
      next->time=this->get_time();next->publications=parent->publications+1;
      for(const auto &cell:points)for(unsigned int q=0;q<cell.second.size();++q)
        {
          const auto &r=cell.second[q];auto &h=next->history.at(cell.first)[q];
          const auto &old=parent->history.at(cell.first)[q];
          h.stress=bulk_tip_midpoint ? 2.*r.state.stress-old.stress : r.state.stress;
          h.A=bulk_tip_midpoint ? 2.*r.state.cohesive_strain-old.A : r.state.cohesive_strain;
          // Rate and reaction are last-stage observations, not endpoint laws.
          h.q=r.Q[0][1];h.rate=r.rate;
          h.signed_shear+=dt*r.rate;h.absolute_shear+=dt*std::abs(r.rate);
        }
      AssertDimension(rates.size(),next->sections.size());AssertDimension(forces.size(),rates.size());
      for(unsigned int k=0;k<next->sections.size();++k)
        {
          auto &s=next->sections[k];s.rate=rates[k];s.slip+=dt*s.rate;s.force=forces[k];
          s.age=fault_friction.evaluate_age(std::abs(s.rate),parent->sections[k].age,dt,
                  bulk_tip_midpoint ? Rheology::FaultFriction<dim>::AgingMethod::constant_rate_exact
                                    : Rheology::FaultFriction<dim>::AgingMethod::backward_euler).age;
        }
      return next;
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::initialize_bulk_cohesive_history(
      const LinearAlgebra::BlockVector &initial)
    {
      AssertThrow(bulk_cohesive_phase && !bulk_cohesive_state && dim == 2,
                  ExcMessage("Bulk cohesive history initializes once on a fixed mesh."));
      auto s = std::make_shared<BulkCohesiveState>();
      s->flow_rule = bulk_flow_rule;
      const double G = elastic_shear_moduli[0], Gc = critical_energy_release_rates[0];
      const double Kf = Gc / ((8. / 3.) * this->get_phase_field_handler().get_length_scale());
      std::vector<double> fractions(elastic_shear_moduli.size(), 0.);
      fractions[0] = 1.;
      const double mu = s->direct_tip()?0.:fault_friction.friction_coefficient(fractions, fault_friction.get_minimum_slip_rate());
      s->parameters = {G, reference_viscosities[0], Kf / (cohesions[0]*cohesions[0] / (2 * G)),
                       mu *this->get_parameters().surface_pressure
                      };
      if (s->flow_rule == BulkFlowRule::coulomb_fixed_phase)
        {
          s->background_pressure = this->get_parameters().surface_pressure;
          s->coulomb_mu = mu;
          AssertThrow(s->background_pressure > 0. && !evolve_phase_field,
                      ExcMessage("Closed Coulomb requires positive background confinement and fixed phase."));
        }
      const auto b = this->introspection().phase_field().block_index;
      if (uses_integrated_bulk_tip())
        {
          const auto &sets = this->introspection().index_sets;
          s->tip_phase.reinit(sets.system_partitioning[b], sets.system_relevant_partitioning[b], this->get_mpi_communicator());
          s->tip_phase = initial.block(b);
        }
      else
        s->phase = aspect::internal::endpoint_values(initial.block(b), this->get_mpi_communicator());
      for (unsigned int i = 0; i < s->phase.size(); ++i)
        if(s->direct_tip()) AssertThrow(s->phase[i]>=0. && s->phase[i]<=1.,ExcMessage("Invalid initial tip phase."));
        else BulkCohesive::compliance(s->phase[i], s->parameters.m);
      s->mesh = this->get_triangulation().get_vertices();
      s->quadrature = fe_bulk_normal_quadrature_points;
      if(!s->direct_tip())
        {
      s->tensor = std::make_shared<TensorField>();
      s->tensor->space = std::make_shared<typename TensorField::Space>(this->get_triangulation());
      for (auto &component : s->tensor->coefficients)
        {
          component.reinit(s->tensor->space->dofs.locally_owned_dofs(), this->get_mpi_communicator());
          component = 0.;
        }
      s->native_history.reinit(initial);
      s->native_history = 0.;
        }
      const unsigned int nq = mechanical_quadrature().size();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            s->history[cell->id()].resize(nq);
            s->native_cells.emplace(cell->id(), cell);
          }
      if(s->direct_tip())
        {
          // Freeze the native quadrature-point material representation. No
          // projection of true stress or cohesive strain onto uncut DGQ2.
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),mechanical_quadrature(),
                           update_quadrature_points|update_JxW_values);
          const QGauss<1> line(s->quadrature);
          const auto section_x=[&](const auto &cell,const unsigned int q)
            {return cell->vertex(0)[0]+(cell->vertex(1)[0]-cell->vertex(0)[0])*line.point(q%s->quadrature)[0];};
          std::map<double,double> local;
          double minimum_y=std::numeric_limits<double>::infinity(),maximum_y=-minimum_y;
          for(const auto &entry:s->native_cells)
            {
              const auto cell=entry.second;
              minimum_y=std::min(minimum_y,cell->vertex(0)[1]);
              maximum_y=std::max(maximum_y,cell->vertex(GeometryInfo<dim>::vertices_per_cell-1)[1]);
              for(const auto v:cell->vertex_indices())for(unsigned int d=0;d<dim;++d)
                AssertThrow(cell->vertex(v)[d]==cell->vertex(0)[d]
                            || cell->vertex(v)[d]==cell->vertex(GeometryInfo<dim>::vertices_per_cell-1)[d],
                            ExcMessage("The first section control requires Cartesian aligned cells."));
              fe.reinit(cell);
              for(unsigned int q=0;q<fe.n_quadrature_points;++q)
                local[section_x(cell,q)]=(cell->vertex(1)[0]-cell->vertex(0)[0])*line.weight(q%s->quadrature);
            }
          std::map<double,double> global;
          for(const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))
            for(const auto &entry:part)
              {const auto inserted=global.emplace(entry);
               AssertThrow(inserted.first->second==entry.second,ExcMessage("Section tangential measures disagree across normal cells."));}
          std::map<double,unsigned int> index;
          for(const auto &entry:global)
            {index[entry.first]=s->sections.size();s->sections.push_back({entry.first,entry.second,bulk_tip_initial_age,0.,0.,0.});}
          for(const auto &entry:s->native_cells)
            {
              fe.reinit(entry.second);
              for(unsigned int q=0;q<fe.n_quadrature_points;++q)
                {
                  const auto k=index.at(section_x(entry.second,q));
                  s->section_ids[entry.first].push_back(k);
                  s->normal_weights[entry.first].push_back(fe.JxW(q)/s->sections[k].weight);
                }
            }

          Vector<double> normal_measure(s->sections.size());
          for(const auto &cell:s->section_ids)for(unsigned int q=0;q<cell.second.size();++q)
            normal_measure[cell.second[q]]+=s->normal_weights.at(cell.first)[q];
          BulkCohesive::sum(normal_measure,this->get_mpi_communicator());
          const double height=Utilities::MPI::max(maximum_y,this->get_mpi_communicator())
                              -Utilities::MPI::min(minimum_y,this->get_mpi_communicator());
          for(const auto value:normal_measure)
            AssertThrow(std::abs(value-height)<1e-12*height,
                        ExcMessage("Bulk tip sections must span the complete normal extent with matching tangential cells."));
        }
      else s->rebuild_native_projection(*this);
      bulk_cohesive_state = std::move(s);
    }

    template <int dim>
    template<class Archive>
    void
    PhaseFieldFault<dim>::serialize_bulk_cohesive_history(Archive &ar)
    {
      AssertThrow(bulk_cohesive_phase, ExcInternalError());
      auto record = std::make_shared<BulkCohesiveArchive>();
      if(bulk_flow_rule==BulkFlowRule::bulk_tip_sections)
        {
          record->signature="bulk tip physical tau A DGQ2 sections BE v1";
          if constexpr(Archive::is_saving::value)
            {
              const auto &s=*bulk_cohesive_state;
              AssertThrow(s.step+1==this->get_timestep_number(),ExcMessage("Tip checkpoint requires accepted state."));
              record->step=s.step;record->time=s.time;record->publications=s.publications;record->quadrature=s.quadrature;
              record->material=s.material_record();
              record->material.push_back(critical_energy_release_rates[0]);
              record->material.push_back(this->get_phase_field_handler().get_length_scale());
              record->material.push_back(bulk_tip_initial_age);
              record->material.push_back(this->get_parameters().surface_pressure);
              record->material.push_back(radiation_damping_coefficients[0]);
              // Include full configured friction law identity, not just an age.
              const auto law=fault_friction.signed_parameter_record();
              record->material.insert(record->material.end(),law.begin(),law.end());
              for(const auto &a:s.sections)
                for(double value:{a.coordinate,a.weight,a.age,a.slip,a.rate,a.force})record->phase.push_back(value);
              std::map<std::string,std::vector<double>> local;
              std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell());
              const auto pc=this->introspection().phase_field().first_component_index;
              const auto po=this->get_solution().get_block_indices().block_start(this->introspection().phase_field().block_index);
              for(const auto &entry:s.native_cells)
                {
                  auto &row=local[entry.first.to_string()];entry.second->get_dof_indices(ids);
                  for(const auto v:entry.second->vertex_indices())for(unsigned int d=0;d<dim;++d)row.push_back(entry.second->vertex(v)[d]);
                  for(unsigned int i=0;i<ids.size();++i)if(this->get_fe().system_to_component_index(i).first==pc)row.push_back(uses_integrated_bulk_tip() ? s.tip_phase[ids[i]-po] : s.phase[ids[i]-po]);
                  for(const auto &h:s.history.at(entry.first))
                    {for(const auto *t:{&h.stress,&h.A})for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)row.push_back((*t)[i][j]);
                     for(double v:{h.signed_shear,h.absolute_shear,h.rate,h.q})row.push_back(v);}
                }
              std::map<std::string,std::vector<double>> global;
              for(const auto &part:Utilities::MPI::all_gather(this->get_mpi_communicator(),local))for(const auto &entry:part)global.emplace(entry);
              for(const auto &entry:global){record->cells.push_back(entry.first);record->local.insert(record->local.end(),entry.second.begin(),entry.second.end());}
            }
          ar &record->signature;
          AssertThrow(record->signature=="bulk tip physical tau A DGQ2 sections BE v1",ExcMessage("Tip checkpoint law mismatch; no conversion."));
          ar &record->quadrature &record->step &record->publications &record->time &record->material &record->phase &record->cells &record->local;
          if constexpr(Archive::is_loading::value)bulk_cohesive_archive=std::move(record);
          return;
        }
      if constexpr(Archive::is_saving::value)
        {
          const auto &s = *bulk_cohesive_state;
          AssertThrow(s.step + 1 == this->get_timestep_number(),
                      ExcMessage("Bulk checkpoint requires one accepted physical endpoint, after advance_time."));
          record->quadrature = s.quadrature;
          record->step = s.step;
          record->publications = s.publications;
          record->time = s.time;
          record->material = s.material_record();
          // Physical cell records avoid native and DG partition-dependent global
          // DoF numbering. Only uniquely owned cells contribute, as in accepted-DG
          // checkpointing. Serialization is called collectively by Simulator.
          record->signature = s.checkpoint_signature();
          std::map<std::string, std::vector<double>> local, global;
          std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell()), dg_ids(9);
          const auto pc = this->introspection().phase_field().first_component_index;
          const auto po = this->get_solution().get_block_indices().block_start(this->introspection().phase_field().block_index);
          for (const auto &entry : s.native_cells)
            {
              const auto cell = entry.second;
              auto &values = local[entry.first.to_string()];
              for (const auto v : cell->vertex_indices())
                {
                  for (unsigned int d = 0; d < dim; ++d)
                    values.push_back(cell->vertex(v)[d]);
                  values.push_back(s.phase[cell->vertex_dof_index(v, pc) - po]);
                }
              s.tensor->space->cells.at(entry.first)->get_dof_indices(dg_ids);
              for (const auto &field : s.tensor->coefficients)
                for (const auto i : dg_ids)
                  values.push_back(field[i]);
              cell->get_dof_indices(ids);
              for (unsigned int i = 0; i < ids.size(); ++i)
                if (this->get_fe().system_to_component_index(i).first < dim)
                  values.push_back(s.native_history[ids[i]]);
              for (const auto &h : s.history.at(entry.first))
                if (s.tensor_flow())
                  {
                    for (const auto *t :
                    {
                      &h.A, &h.K, &h.Q, &h.W
                    })
                    for (unsigned int i = 0; i < 3; ++i)
                      for (unsigned int j = i; j < 3; ++j)
                        values.push_back((*t)[i][j]);
                    values.push_back(h.L);
                  }
                else for (const double value :
                  {
                    h.a, h.signed_shear, h.absolute_shear, h.q, h.rate
                  })
              values.push_back(value);
            }
          for (const auto &part : Utilities::MPI::all_gather(this->get_mpi_communicator(), local))
            for (const auto &entry : part)
              AssertThrow(global.emplace(entry).second, ExcMessage("Duplicate bulk checkpoint CellId owner."));
          AssertDimension(global.size(), this->get_triangulation().n_global_active_cells());
          for (const auto &entry : global)
            {
              record->cells.push_back(entry.first);
              record->local.insert(record->local.end(), entry.second.begin(), entry.second.end());
            }

        }
      ar &record->signature;
      const auto expected = bulk_flow_rule == BulkFlowRule::coulomb_fixed_phase ? "bulk cohesive closed Coulomb fixed phase cell history v1"
                            : (bulk_flow_rule == BulkFlowRule::scalar ? "bulk cohesive cell history v3"
                               : (bulk_flow_rule == BulkFlowRule::fixed_tensor ? "bulk cohesive fixed tensor cell history v1"
                                  : "bulk cohesive radial tensor cell history v1"));
      AssertThrow(record->signature == expected
                  || (bulk_flow_rule == BulkFlowRule::scalar && record->signature == "bulk cohesive BE native remainder v2"),
                  ExcMessage("Bulk cohesive checkpoint mode/version mismatch; conversion is unsupported."));
      ar &record->quadrature &record->step &record->publications &record->time
      &record->material &record->mesh &record->phase &record->tensor &record->cells &record->local &record->native;
      if constexpr(Archive::is_loading::value)
        {
          bulk_cohesive_archive = std::move(record);
        }
    }

    template <int dim>
    void
    PhaseFieldFault<dim>::restore_bulk_cohesive_history()
    {
      AssertThrow(bulk_cohesive_archive && !bulk_cohesive_state, ExcMessage("Missing bulk cohesive restart history."));
      initialize_bulk_cohesive_history(this->get_solution());
      auto &s = *bulk_cohesive_state;
      const auto &a = *bulk_cohesive_archive;
      if(s.direct_tip())
        {
          AssertThrow(a.signature==s.checkpoint_signature() && a.quadrature==s.quadrature,ExcMessage("Tip checkpoint representation/quadrature mismatch."));
          std::vector<double> material=s.material_record();
          material.push_back(critical_energy_release_rates[0]);
          material.push_back(this->get_phase_field_handler().get_length_scale());
          material.push_back(bulk_tip_initial_age);
          material.push_back(this->get_parameters().surface_pressure);material.push_back(radiation_damping_coefficients[0]);
          const auto law=fault_friction.signed_parameter_record();
          material.insert(material.end(),law.begin(),law.end());
          AssertThrow(material==a.material,ExcMessage("Tip checkpoint material/RSF law changed."));
          AssertDimension(a.phase.size(),6*s.sections.size());
          for(unsigned int i=0;i<s.sections.size();++i)
            {auto &v=s.sections[i];AssertThrow(v.coordinate==a.phase[6*i] && v.weight==a.phase[6*i+1],ExcMessage("Tip section geometry changed."));
             v.age=a.phase[6*i+2];v.slip=a.phase[6*i+3];v.rate=a.phase[6*i+4];v.force=a.phase[6*i+5];}
          const auto &fe=this->get_fe();const auto pc=this->introspection().phase_field().first_component_index;
          const auto po=this->get_solution().get_block_indices().block_start(this->introspection().phase_field().block_index);
          const auto nd=fe.base_element(fe.component_to_base_index(pc).first).n_dofs_per_cell();
          const unsigned int stride=GeometryInfo<dim>::vertices_per_cell*dim+nd+16*mechanical_quadrature().size();
          AssertDimension(a.local.size(),stride*a.cells.size());
          std::map<std::string,unsigned int> offsets;for(unsigned int i=0;i<a.cells.size();++i)offsets[a.cells[i]]=i*stride;
          std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
          for(const auto &entry:s.native_cells)
            {unsigned int k=offsets.at(entry.first.to_string());entry.second->get_dof_indices(ids);
             for(const auto v:entry.second->vertex_indices())for(unsigned int d=0;d<dim;++d)AssertThrow(a.local[k++]==entry.second->vertex(v)[d],ExcMessage("Tip mesh changed."));
             for(unsigned int i=0;i<ids.size();++i)if(fe.system_to_component_index(i).first==pc)AssertThrow(a.local[k++]==(uses_integrated_bulk_tip() ? s.tip_phase[ids[i]-po] : s.phase[ids[i]-po]),ExcMessage("Tip phase obstacle changed."));
             for(auto &h:s.history[entry.first])
               {for(auto *t:{&h.stress,&h.A})for(unsigned int i=0;i<3;++i)for(unsigned int j=i;j<3;++j)(*t)[i][j]=a.local[k++];
                h.signed_shear=a.local[k++];h.absolute_shear=a.local[k++];h.rate=a.local[k++];h.q=a.local[k++];}}
          s.step=a.step;s.time=a.time;s.publications=a.publications;
          AssertThrow(s.step+1==this->get_timestep_number(),ExcMessage("Tip restart endpoint clock mismatch."));
          bulk_cohesive_archive.reset();return;
        }
      if (a.signature == s.checkpoint_signature())
        {
          AssertThrow(a.quadrature == s.quadrature
                      && a.material == s.material_record(),
                      ExcMessage("Bulk restart changed quadrature or coefficients."));
          const auto pc = this->introspection().phase_field().first_component_index;
          const auto po = this->get_solution().get_block_indices().block_start(this->introspection().phase_field().block_index);
          const unsigned int dpc = this->get_fe().n_dofs_per_cell(), nq = mechanical_quadrature().size();
          unsigned int velocity_dpc = 0;
          for (unsigned int i = 0; i < dpc; ++i)
            if (this->get_fe().system_to_component_index(i).first < dim)
              ++velocity_dpc;
          const unsigned int stride = GeometryInfo<dim>::vertices_per_cell * (dim + 1) + 27 + velocity_dpc + (s.tensor_flow() ? 25 : 5) * nq;
          AssertDimension(a.local.size(), stride * a.cells.size());
          AssertDimension(a.cells.size(), this->get_triangulation().n_global_active_cells());
          std::map<std::string, unsigned int> offsets;
          for (unsigned int i = 0; i < a.cells.size(); ++i)
            AssertThrow(offsets.emplace(a.cells[i], i * stride).second, ExcInternalError());
          LinearAlgebra::BlockVector owned;
          owned.reinit(this->introspection().index_sets.system_partitioning, this->get_mpi_communicator());
          owned = 0.;
          std::vector<types::global_dof_index> ids(dpc), dg_ids(9);
          const auto nv = this->get_solution().block(this->introspection().block_indices.velocities).size();
          std::set<types::global_dof_index> covered;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (!cell->is_artificial())
              {
                unsigned int k = offsets.at(cell->id().to_string());
                for (const auto v : cell->vertex_indices())
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      AssertThrow(a.local[k++] == cell->vertex(v)[d], ExcMessage("Bulk restart changed physical mesh."));
                    AssertThrow(a.local[k++] == s.phase[cell->vertex_dof_index(v, pc) - po], ExcMessage("Bulk restart changed physical obstacle."));
                  }
                if (cell->is_locally_owned())
                  {
                    s.tensor->space->cells.at(cell->id())->get_dof_indices(dg_ids);
                    for (auto &field : s.tensor->coefficients)
                      for (const auto i : dg_ids)
                        field[i] = a.local[k++];
                  }
                else k += 27;
                cell->get_dof_indices(ids);
                for (unsigned int i = 0; i < dpc; ++i)
                  if (this->get_fe().system_to_component_index(i).first < dim)
                    {
                      const double value = a.local[k++];
                      if (owned.block(this->introspection().block_indices.velocities).locally_owned_elements().is_element(ids[i]))
                        {
                          if (!covered.insert(ids[i]).second)
                            AssertThrow(owned[ids[i]] == value, ExcMessage("Inconsistent shared bulk FE history."));
                          owned[ids[i]] = value;
                        }
                    }
                if (cell->is_locally_owned())
                  for (auto &h : s.history.at(cell->id()))
                    if (s.tensor_flow())
                      {
                        for (auto *t :
                             {
                               &h.A, &h.K, &h.Q, &h.W
                             })
                          for (unsigned int i = 0; i < 3; ++i)
                            for (unsigned int j = i; j < 3; ++j)
                              (*t)[i][j] = a.local[k++];
                        h.L = a.local[k++];
                      }
                    else
                      {
                        h.a = a.local[k++];
                        h.signed_shear = a.local[k++];
                        h.absolute_shear = a.local[k++];
                        h.q = a.local[k++];
                        h.rate = a.local[k++];
                      }
              }
          (void)nv;
          AssertDimension(covered.size(), owned.block(this->introspection().block_indices.velocities).locally_owned_size());
          for (auto &field : s.tensor->coefficients)
            field.compress(VectorOperation::insert);
          owned.compress(VectorOperation::insert);
          s.native_history = owned;
          s.rebuild_native_projection(*this);
          s.step = a.step;
          s.time = a.time;
          s.publications = a.publications;
          AssertThrow(s.step + 1 == this->get_timestep_number(), ExcMessage("Bulk restart endpoint clock mismatch."));
          bulk_cohesive_archive.reset();
          return;
        }
      AssertThrow(Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) == 1,
                  ExcMessage("Legacy v2 bulk checkpoint requires its original serial layout."));
      AssertThrow(a.mesh == s.mesh && a.quadrature == s.quadrature
                  && a.material == s.material_record()
                  && a.phase == std::vector<double>(s.phase.begin(), s.phase.end()),
                  ExcMessage("Bulk restart changed mesh, quadrature, coefficients or physical phase."));
      const unsigned int n = s.tensor->space->dofs.n_dofs();
      AssertDimension(a.tensor.size(), s.tensor->coefficients.size()*n);
      unsigned int offset = 0;
      for (auto &field : s.tensor->coefficients)
        {
          for (unsigned int i = 0; i < n; ++i)
            field[i] = a.tensor[offset++];
          field.compress(VectorOperation::insert);
        }
      auto &velocity = s.native_history.block(this->introspection().block_indices.velocities);
      AssertDimension(a.native.size(), velocity.size());
      for (unsigned int i = 0; i < velocity.size(); ++i)
        velocity[i] = a.native[i];
      s.native_history.compress(VectorOperation::insert);
      s.rebuild_native_projection(*this);
      unsigned int row = 0;
      offset = 0;
      for (auto &entry : s.history)
        {
          AssertThrow(a.cells.at(row++) == entry.first.to_string(), ExcMessage("Bulk restart changed cell identities."));
          for (auto &h : entry.second)
            {
              h.a = a.local.at(offset++);
              h.signed_shear = a.local.at(offset++);
              h.absolute_shear = a.local.at(offset++);
              h.q = a.local.at(offset++);
              h.rate = a.local.at(offset++);
            }
        }
      AssertDimension(offset, a.local.size());
      s.step = a.step;
      s.time = a.time;
      s.publications = a.publications;
      AssertThrow(s.step + 1 == this->get_timestep_number(), ExcMessage("Bulk restart endpoint clock mismatch."));
      bulk_cohesive_archive.reset();
    }

#define INSTANTIATE(dim) \
  template void PhaseFieldFault<dim>::BulkCohesiveState::rebuild_native_projection(const PhaseFieldFault<dim> &); \
  template std::vector<SymmetricTensor<2,dim>> PhaseFieldFault<dim>::BulkCohesiveState::evaluate_tensor(const PhaseFieldFault<dim> &,const CellId &,const Quadrature<dim> &) const; \
  template double PhaseFieldFault<dim>::BulkCohesiveState::check_tensor(const PhaseFieldFault<dim> &,const CellId &,const Quadrature<dim> &,const std::vector<BulkCohesive::Tensor> &) const; \
  template std::shared_ptr<PhaseFieldFault<dim>::BulkCohesiveState> PhaseFieldFault<dim>::prepare_bulk_tip_candidate(const std::shared_ptr<const BulkCohesiveState> &,const std::map<CellId,std::vector<BulkCohesive::Tip::Response>> &,const std::vector<double> &,const std::vector<double> &,const LinearAlgebra::Vector &) const; \
  template void PhaseFieldFault<dim>::initialize_bulk_cohesive_history(const LinearAlgebra::BlockVector &); \
  template void PhaseFieldFault<dim>::serialize_bulk_cohesive_history<aspect::oarchive>(aspect::oarchive &); \
  template void PhaseFieldFault<dim>::serialize_bulk_cohesive_history<aspect::iarchive>(aspect::iarchive &); \
  template void PhaseFieldFault<dim>::restore_bulk_cohesive_history();
    ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
  }
}
