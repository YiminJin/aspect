/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/

#include <aspect/material_model/fixed_fault.h>
#include <aspect/material_model/rheology/elasticity.h>
#include <aspect/geometry_model/box.h>
#include <aspect/utilities.h>
#include <deal.II/physics/transformations.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>

namespace aspect
{
  namespace MaterialModel
  {
    template <int dim>
    const FiniteElement<dim> &FixedFault<dim>::stress_element() const
    {
      return this->get_fe().base_element(this->introspection().base_elements.compositional_fields[0]);
    }

    template <int dim>
    void FixedFault<dim>::update()
    {
      const double physical_dt = this->get_timestep();
      const double dt = this->get_timestep_number() == 0 ? initial_interval : physical_dt;
      const double beta = std::exp(-modulus*dt/viscosity);
      kappa = -viscosity*std::expm1(-modulus*dt/viscosity);
      trial_rates.clear();
      for (const auto &label : fault.labels())
        trial_rates.push_back(label.rate);

      const QGauss<dim> rule(8);
      const auto &dg = stress_element();
      FullMatrix<double> projector(dg.n_dofs_per_cell(),rule.size());
      FETools::compute_projection_from_quadrature_points_matrix(dg,rule,rule,projector);
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_gradients | update_quadrature_points);
      const auto &fields = this->get_old_solution();
      const auto &extract = this->introspection().extractors;
      std::vector<Tensor<2,dim>> gradients(rule.size());
      std::array<std::vector<double>,3> components;
      for (auto &v : components)
        v.resize(rule.size());
      std::vector<Stress> rotated(rule.size());
      std::vector<double> phase(rule.size());
      double min_phase = 1., max_phase = 0.;
      bool aligned = true;
      relaxed_parent.clear();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // With fixed constant phase and resolved Q1 knots the localized
            // slip stress is already in DGQ2. This makes the physical stress
            // representation exact also on the stationary donor's split path.
            aligned &= fault.resolves_segments(cell->vertex(0),cell->vertex(GeometryInfo<dim>::vertices_per_cell-1));
            fe.reinit(cell);
            fe[extract.velocities].get_function_gradients(fields,gradients);
            fe[extract.compositional_fields[3]].get_function_values(fields,phase);
            for (unsigned int c=0; c<3; ++c)
              fe[extract.compositional_fields[c]].get_function_values(fields,components[c]);
            for (unsigned int q=0; q<rule.size(); ++q)
              {
                min_phase = std::min(min_phase,phase[q]);
                max_phase = std::max(max_phase,phase[q]);
                Stress parent;
                for (unsigned int c=0; c<3; ++c)
                  parent.access_raw_entry(c) = components[c][q];
                const auto foot = fault.project(fe.quadrature_point(q));
                const double rate = foot.interpolate(trial_rates[foot.left],trial_rates[foot.left+1]);
                const auto elastic_gradient = gradients[q] - fault.localization()*rate*fault.slip_dyad();
                if constexpr (dim == 2)
                  {
                    // Initialization has a Maxwell interval but no accepted spin interval.
                    const double angle = .5*physical_dt*(elastic_gradient[1][0]-elastic_gradient[0][1]);
                    const auto rotation = Physics::Transformations::Rotations::rotation_matrix_2d(angle);
                    rotated[q] = beta*symmetrize(rotation*static_cast<Tensor<2,dim>>(parent)*transpose(rotation));
                  }
              }
            auto &coefficients = relaxed_parent[cell->id()];
            coefficients.resize(dg.n_dofs_per_cell());
            FETools::compute_projection_from_quadrature_points(projector,rotated,coefficients);
          }
      const auto comm = this->get_mpi_communicator();
      min_phase = Utilities::MPI::min(min_phase,comm);
      max_phase = Utilities::MPI::max(max_phase,comm);
      AssertThrow(min_phase > 0. && max_phase < 1. && max_phase-min_phase < 1.e-12,
                  ExcMessage("The stationary section normalization requires a constant phase between zero and one."));
      AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(aligned),comm),
                  ExcMessage("Interior fault labels must lie on transverse bulk cell faces."));
    }

    template <int dim>
    std::vector<typename FixedFault<dim>::BulkPoint>
    FixedFault<dim>::sample_bulk(const LinearAlgebra::BlockVector &iterate) const
    {
      const QGauss<dim> rule(8);
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_gradients | update_quadrature_points | update_JxW_values);
      FullMatrix<double> basis(rule.size(),stress_element().n_dofs_per_cell());
      FETools::compute_interpolation_to_quadrature_points_matrix(stress_element(),rule,basis);
      std::vector<Stress> strain(rule.size());
      std::vector<double> pressure(rule.size());
      std::vector<BulkPoint> samples;
      const auto S = symmetrize(fault.slip_dyad());
      double min_compression = std::numeric_limits<double>::max();
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(iterate,strain);
            fe[this->introspection().extractors.pressure].get_function_values(iterate,pressure);
            const auto &parent = relaxed_parent.at(cell->id());
            for (unsigned int q=0; q<rule.size(); ++q)
              {
                Stress stress = 2.*kappa*Utilities::Tensors::consistent_deviator(strain[q]);
                for (unsigned int i=0; i<parent.size(); ++i)
                  stress += basis(q,i)*parent[i];
                const double compression = background_compression + pressure[q] - fault.normal_vector()*(stress*fault.normal_vector());
                min_compression = std::min(min_compression,compression);
                samples.push_back({fault.project(fe.quadrature_point(q)),fe.JxW(q)*fault.localization(),
                                   background_shear + stress*S,compression
                                  });
              }
          }
      AssertThrow(Utilities::MPI::min(min_compression,this->get_mpi_communicator()) > 0.,
                  ExcMessage("Mature friction requires a closed fault."));
      return samples;
    }

    template <int dim>
    typename FixedFault<dim>::Stress
    FixedFault<dim>::stress_offset(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                   const Point<dim> &position, const std::vector<double> &rates) const
    {
      const auto foot = fault.project(position);
      const double rate = foot.interpolate(rates[foot.left],rates[foot.left+1]);
      Stress result = -2.*kappa*fault.localization()*rate*symmetrize(fault.slip_dyad());
      const auto unit = this->get_mapping().transform_real_to_unit_cell(cell,position);
      const auto &parent = relaxed_parent.at(cell->id());
      for (unsigned int i=0; i<parent.size(); ++i)
        result += stress_element().shape_value(i,unit)*parent[i];
      return result;
    }

    template <int dim>
    void FixedFault<dim>::prepare_stokes(const LinearAlgebra::BlockVector &iterate)
    {
      trial_rates = fault.solve_rates(sample_bulk(iterate),friction,kappa,minimum_speed,this->get_mpi_communicator());
    }

    template <int dim>
    void FixedFault<dim>::evaluate(const MaterialModelInputs<dim> &in, MaterialModelOutputs<dim> &out) const
    {
      Simple<dim>::evaluate(in,out);
      const auto force = out.template get_additional_output_object<ElasticOutputs<dim>>();
      const auto elastic = out.template get_additional_output_object<ElasticAdditionalOutputs<dim>>();
      for (unsigned int q=0; q<in.n_evaluation_points(); ++q)
        {
          out.viscosities[q] = kappa;
          if (force)
            force->elastic_force[q] = -stress_offset(in.current_cell,in.position[q],trial_rates);
          if (elastic)
            {
              elastic->elastic_shear_moduli[q] = modulus;
              elastic->elastic_viscosity[q] = kappa;
              for (unsigned int c=0; c<3; ++c)
                elastic->deviatoric_stress[q].access_raw_entry(c) = in.composition[q][c];
            }
        }
    }

    template <int dim>
    void FixedFault<dim>::create_additional_named_outputs(MaterialModelOutputs<dim> &out) const
    {
      if (!out.template has_additional_output_object<ElasticAdditionalOutputs<dim>>())
        out.additional_outputs.push_back(std::make_unique<ElasticAdditionalOutputs<dim>>(out.n_evaluation_points()));
    }

    template <int dim>
    void FixedFault<dim>::accept_step(LinearAlgebra::BlockVector &solution)
    {
      const auto rates = fault.solve_rates(sample_bulk(solution),friction,kappa,minimum_speed,this->get_mpi_communicator());
      for (unsigned int j=0; j<rates.size(); ++j)
        AssertThrow(std::abs(rates[j]-trial_rates[j]) <= 1.e-8*std::max(rates[j],trial_rates[j]),
                    ExcMessage("The converged Stokes field and fault rate disagree."));

      const double dt = this->get_timestep();
      if (dt > 0.)
        {
          // With constant coefficients and cell-resolved Q1 rates the endpoint
          // is DGQ2 on each cell. Evaluation at DG nodes is its exact projection.
          const auto &system = this->get_fe();
          const auto &intro = this->introspection();
          const Quadrature<dim> nodes(stress_element().get_unit_support_points());
          FEValues<dim> fe(this->get_mapping(),system,nodes,update_gradients | update_quadrature_points);
          std::vector<Stress> strain(nodes.size());
          std::vector<types::global_dof_index> indices(system.n_dofs_per_cell());
          std::array<LinearAlgebra::Vector,3> owned;
          for (unsigned int c=0; c<3; ++c)
            owned[c].reinit(solution.block(intro.block_indices.compositional_fields[c]).locally_owned_elements(),this->get_mpi_communicator());
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                cell->get_dof_indices(indices);
                fe[intro.extractors.velocities].get_function_symmetric_gradients(solution,strain);
                for (unsigned int q=0; q<nodes.size(); ++q)
                  {
                    const Stress stress = stress_offset(cell,fe.quadrature_point(q),rates)
                                          + 2.*kappa*Utilities::Tensors::consistent_deviator(strain[q]);
                    for (unsigned int c=0; c<3; ++c)
                      {
                        const auto local = system.component_to_system_index(intro.component_indices.compositional_fields[c],q);
                        const auto block = intro.block_indices.compositional_fields[c];
                        owned[c][indices[local]-solution.get_block_indices().block_start(block)] = stress.access_raw_entry(c);
                      }
                  }
              }
          for (unsigned int c=0; c<3; ++c)
            {
              owned[c].compress(VectorOperation::insert);
              solution.block(intro.block_indices.compositional_fields[c]) = owned[c];
            }
        }
      fault.accept(rates,dt,friction);
    }

    template <int dim>
    const std::vector<typename StationaryFault<dim>::Label> &FixedFault<dim>::labels() const
    {
      return fault.labels();
    }

    template <int dim>
    void FixedFault<dim>::save(std::map<std::string,std::string> &status) const
    {
      std::ostringstream buffer;
      {
        oarchive archive(buffer);
        fault.save(archive);
      }
      status["stationary Q1 labels, DGQ2 stress v2"] = buffer.str();
    }

    template <int dim>
    void FixedFault<dim>::load(const std::map<std::string,std::string> &status)
    {
      const auto record = status.find("stationary Q1 labels, DGQ2 stress v2");
      if (record != status.end())
        {
          std::istringstream buffer(record->second);
          iarchive archive(buffer);
          fault.load(archive);
          return;
        }
      const auto legacy = status.find("fixed vertical fault, constant section, DGQ2 stress v1");
      AssertThrow(legacy != status.end(),ExcMessage("Incompatible fixed-fault representation in checkpoint."));
      std::istringstream buffer(legacy->second);
      iarchive archive(buffer);
      Point<dim> origin, extent;
      double age, slip, rate;
      archive >> origin >> extent >> age >> slip >> rate;
      const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      auto end = origin;
      end[1] += extent[1];
      AssertThrow(box.get_origin() == origin && box.get_extents() == extent && labels().size() == 2 &&
                  labels().front().position == origin && labels().back().position == end,
                  ExcMessage("Incompatible fixed-fault representation in checkpoint."));
      fault.initialize(origin,extent, {origin,end},age,rate,slip);
    }

    template <int dim>
    void FixedFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      prm.enter_subsection("Fixed fault");
      prm.declare_entry("Fault vertices","",Patterns::List(Patterns::List(Patterns::Double(),2,2),0,
                                                           Patterns::List::max_int_value,";"),"Ordered x,y label positions in metres separated by semicolons. "
                        "The default joins the ends of the left box boundary.");
      prm.declare_entry("Shear modulus","1e10",Patterns::Double(0.),"Elastic shear modulus in Pa.");
      prm.declare_entry("Initialization interval","1",Patterns::Double(0.),"Maxwell interval in seconds for the initial Stokes solve.");
      prm.declare_entry("Background shear traction","0",Patterns::Double(),"Initial surface shear loading in Pa, positive along the slip sense.");
      prm.declare_entry("Background compression","5e7",Patterns::Double(0.),"Compression in Pa added to the dynamic normal traction.");
      prm.declare_entry("Reference friction coefficient","0.6",Patterns::Double(0.),"Dimensionless reference friction coefficient.");
      prm.declare_entry("Direct effect","0.01",Patterns::Double(0.),"Dimensionless direct effect parameter.");
      prm.declare_entry("Evolution effect","0.015",Patterns::Double(0.),"Dimensionless evolution effect parameter.");
      prm.declare_entry("Reference speed","1e-6",Patterns::Double(0.),"Reference slip speed in m/s.");
      prm.declare_entry("Minimum speed","1e-20",Patterns::Double(0.),"Minimum admissible sliding speed in m/s.");
      prm.declare_entry("Slip distance","0.01",Patterns::Double(0.),"Characteristic slip distance in m.");
      prm.declare_entry("Radiation damping","0",Patterns::Double(0.),"Radiation damping coefficient in Pa s/m.");
      prm.declare_entry("Initial age","1",Patterns::Double(0.),"Initial section age in seconds.");
      prm.leave_subsection();
      prm.leave_subsection();
    }

    template <int dim>
    void FixedFault<dim>::parse_parameters(ParameterHandler &prm)
    {
      Simple<dim>::parse_parameters(prm);
      prm.enter_subsection("Material model");
      prm.enter_subsection("Simple model");
      viscosity = prm.get_double("Viscosity");
      AssertThrow(prm.get_double("Thermal viscosity exponent") == 0 && prm.get_double("Composition viscosity prefactor") == 1.,
                  ExcMessage("The stationary fault requires a constant bulk viscosity."));
      prm.leave_subsection();
      prm.enter_subsection("Fixed fault");
      modulus = prm.get_double("Shear modulus");
      initial_interval = prm.get_double("Initialization interval");
      background_shear = prm.get_double("Background shear traction");
      background_compression = prm.get_double("Background compression");
      friction = {prm.get_double("Reference friction coefficient"),prm.get_double("Direct effect"),
                  prm.get_double("Evolution effect"),prm.get_double("Reference speed"),
                  prm.get_double("Slip distance"),prm.get_double("Radiation damping")
                 };
      minimum_speed = prm.get_double("Minimum speed");
      const double age = prm.get_double("Initial age");
      const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      std::vector<Point<dim>> vertices;
      for (const auto &entry : Utilities::split_string_list(prm.get("Fault vertices"),';'))
        {
          const auto coordinates = Utilities::string_to_double(Utilities::split_string_list(entry));
          Point<dim> point;
          point[0] = coordinates.at(0);
          point[1] = coordinates.at(1);
          vertices.push_back(point);
        }
      if (vertices.empty())
        {
          vertices = {box.get_origin(),box.get_origin()};
          vertices.back()[1] += box.get_extents()[1];
        }
      fault.initialize(box.get_origin(),box.get_extents(),vertices,age,friction.reference_speed);
      prm.leave_subsection();
      prm.leave_subsection();
      AssertThrow(std::min({viscosity,modulus,initial_interval,friction.direct_effect,minimum_speed,
                            friction.reference_speed,friction.slip_distance,age
                           }) > 0.,
                  ExcMessage("The fixed fault requires positive material, time and speed scales."));
      AssertThrow(friction.reference_speed >= minimum_speed,
                  ExcMessage("The reference speed must be at least the minimum admissible speed."));
      const auto &p = this->get_parameters();
      AssertThrow(dim == 2 && p.enable_elasticity && p.nonlinear_solver == Parameters<dim>::NonlinearSolver::no_Advection_iterated_Stokes
                  && p.stokes_solver_type == Parameters<dim>::StokesSolverType::block_amg && p.stokes_velocity_degree == 2
                  && !p.run_postprocessors_on_nonlinear_iterations && !p.mesh_deformation_enabled && !p.include_melt_transport && p.pressure_normalization == "no"
                  && p.nonlinear_solver_failure_strategy != Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep,
                  ExcMessage("The fixed fault requires a fixed 2-D Q2 box with elasticity, block AMG, no pressure normalization, "
                             "no Advection, iterated Stokes, and convergence or retry."));
      const std::vector<std::string> names = {"ve_stress_xx","ve_stress_yy","ve_stress_xy","phase_field"};
      AssertThrow(p.names_of_compositional_fields == names &&
                  p.composition_degrees == std::vector<unsigned int>(4,2) &&
                  p.use_discontinuous_composition_discretization == std::vector<bool>(4,true) &&
                  std::all_of(p.compositional_field_methods.begin(),p.compositional_field_methods.end(),
                              [](const auto method)
      {
        return method == Parameters<dim>::AdvectionFieldMethod::static_field;
      }),
      ExcMessage("Expected static DGQ2 fields ve_stress_xx, ve_stress_yy, ve_stress_xy and phase_field in this order."));

      kappa = -viscosity*std::expm1(-modulus*initial_interval/viscosity);

    }

    ASPECT_REGISTER_MATERIAL_MODEL(FixedFault,"fixed fault",
                                   "Stationary mature fault with independent Q1 labels spanning a Cartesian box. "
                                   "The phase is a spatial constant strictly between zero and one. "
                                   "The Simple model supplies the bulk viscosity and thermal properties.")
  }
}
