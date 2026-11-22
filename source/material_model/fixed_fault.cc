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

namespace aspect
{
  namespace MaterialModel
  {
    std::pair<double,double>
    Rheology::MatureFaultFriction::resistance(const double speed, const double age, const double compression) const
    {
      // Use log(z) to avoid overflow for large reference friction / direct effect.
      AssertThrow(speed > 0. && age > 0. && compression > 0., ExcMessage("Regularized mature friction requires positive speed, age and compression."));
      const double log_z = std::log(speed) - std::log(2.*reference_speed)
                           + (reference_coefficient + evolution_effect * std::log(age*reference_speed/slip_distance))/direct_effect;
      const double small = std::exp(-std::abs(log_z));
      const double asinh_z = log_z > 0 ? log_z + std::log1p(std::hypot(1.,small)) : std::asinh(small);
      const double ratio = (log_z > 0 ? 1. : small)/std::hypot(1.,small);
      return {compression *direct_effect *asinh_z + damping*speed,
              compression *direct_effect *ratio/speed + damping
             };
    }

    double
    Rheology::MatureFaultFriction::advance_age(const double speed, const double age, const double dt) const
    {
      AssertThrow(speed >= 0. && age > 0. && dt >= 0., ExcMessage("Invalid aging inputs."));
      const double x = speed*dt/slip_distance;
      return age*std::exp(-x) + dt*(x == 0. ? 1. : -std::expm1(-x)/x);
    }

    template <int dim>
    typename FixedFault<dim>::SectionMean
    FixedFault<dim>::section_mean(const LinearAlgebra::BlockVector &fields, const bool check_symmetry) const
    {
      // For spatially constant 0 < phi < 1 the normalized profile h/I is
      // exactly 1/width. This is the constant mode of both Q1 surface rows.
      // Use the same native quadrature as the mature bulk-work residual.
      const QGauss<dim> rule(8);
      FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values | update_gradients | update_quadrature_points | update_JxW_values);
      std::vector<Tensor<2,dim>> gradients(rule.size());
      std::vector<double> pressures(rule.size());
      std::vector<Tensor<1,dim>> velocities(rule.size());
      std::array<std::vector<double>,4> compositions;
      for (auto &values : compositions)
        values.resize(rule.size());

      // Four gradient entries, p, xx, yy, xy and phase. Every rank takes
      // part even when it owns no cells.
      constexpr unsigned int n = 9;
      double integrals[n+1] = {};
      double low[n], high[n];
      std::fill_n(low,n,std::numeric_limits<double>::max());
      std::fill_n(high,n,std::numeric_limits<double>::lowest());
      const auto &intro = this->introspection();
      const double left = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model()).get_origin()[0];
      double shear_error = 0.;
      for (const auto &cell : this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe.reinit(cell);
            fe[intro.extractors.velocities].get_function_gradients(fields,gradients);
            fe[intro.extractors.pressure].get_function_values(fields,pressures);
            if (check_symmetry)
              {
                fe[intro.extractors.velocities].get_function_values(fields,velocities);
                for (unsigned int c=0; c<4; ++c)
                  fe[intro.extractors.compositional_fields[c]].get_function_values(fields,compositions[c]);
              }
            for (unsigned int q=0; q<rule.size(); ++q)
              {
                const double weight = fe.JxW(q)/width;
                const std::array<double,n> values = {{
                    gradients[q][0][0],gradients[q][0][1],gradients[q][1][0],gradients[q][1][1],
                    pressures[q],compositions[0][q],compositions[1][q],compositions[2][q],compositions[3][q]
                  }
                };
                integrals[n] += weight;
                for (unsigned int i=0; i<(check_symmetry ? n : 5); ++i)
                  integrals[i] += weight*values[i];
                if (check_symmetry)
                  {
                    // v=(0,gamma*(x-left)) fixes every label on the wall.
                    // A normal flow or tangential translation would require
                    // transporting the geometry and its incoming history.
                    const auto &gradient = gradients[q];
                    shear_error = std::max({shear_error, std::abs(gradient[0][0]),std::abs(gradient[0][1]),std::abs(gradient[1][1]),
                                            std::abs(velocities[q][0])/width,
                                            std::abs(velocities[q][1]-gradient[1][0]*(fe.quadrature_point(q)[0]-left))/width
                                           });
                    for (unsigned int i=0; i<n; ++i)
                      {
                        low[i] = std::min(low[i],values[i]);
                        high[i] = std::max(high[i],values[i]);
                      }
                  }
              }
          }
      const auto comm = this->get_mpi_communicator();
      Utilities::MPI::sum(integrals,comm,integrals);
      if (check_symmetry)
        {
          Utilities::MPI::min(low,comm,low);
          Utilities::MPI::max(high,comm,high);
          AssertThrow(Utilities::MPI::max(shear_error,comm) < 1.e-9*std::max(1.,std::abs(high[2])),
                      ExcMessage("The fixed geometry requires tangential shear with zero velocity on the left wall."));
          for (unsigned int i=0; i<n; ++i)
            AssertThrow(high[i]-low[i] < 1.e-9*std::max({1.,std::abs(high[i]),std::abs(low[i])}),
                        ExcMessage("The fixed fault requires homogeneous shear, pressure, stress and phase."));
          AssertThrow(low[8] > 0. && high[8] < 1., ExcMessage("The constant phase must be strictly between zero and one."));
        }
      SectionMean result;
      for (unsigned int i=0; i<2; ++i)
        for (unsigned int j=0; j<2; ++j)
          result.gradient[i][j] = integrals[2*i+j]/integrals[n];
      result.pressure = integrals[4]/integrals[n];
      for (unsigned int c=0; c<3; ++c)
        result.stress.access_raw_entry(c) = integrals[5+c]/integrals[n];
      return result;
    }

    template <int dim>
    void FixedFault<dim>::update()
    {
      const auto parent = section_mean(this->get_old_solution(),true);
      const double dt = this->get_timestep_number() == 0 ? initial_interval : this->get_timestep();
      kappa = -viscosity*std::expm1(-modulus*dt/viscosity);
      // Rotate by the accepted bulk spin. With upward tangent and left
      // normal the slip gradient has yx component -V/width.
      const double angle = dt*.5*(parent.gradient[1][0]+accepted.rate/width-parent.gradient[0][1]);
      if constexpr (dim == 2)
        {
          const auto rotation = Physics::Transformations::Rotations::rotation_matrix_2d(angle);
          relaxed_parent = std::exp(-modulus*dt/viscosity)*symmetrize(rotation*static_cast<Tensor<2,dim>>(parent.stress)*transpose(rotation));
        }
      trial_rate = accepted.rate;
    }

    template <int dim>
    SymmetricTensor<2,dim> FixedFault<dim>::stress_offset(const double rate) const
    {
      auto stress = relaxed_parent;
      stress[0][1] += kappa*rate/width;
      return stress;
    }

    template <int dim>
    double FixedFault<dim>::section_rate(const SectionMean &mean) const
    {
      const auto trial = relaxed_parent + 2.*kappa*Utilities::Tensors::consistent_deviator(symmetrize(mean.gradient));
      const double driving = background_shear - trial[0][1];
      const double compression = background_compression + mean.pressure - trial[0][0];
      AssertThrow(compression > 0., ExcMessage("Mature friction requires a closed fault."));
      const double stiffness = kappa/width;
      double lower = minimum_speed;
      double upper = driving/(stiffness + friction.damping);
      AssertThrow(upper >= lower && friction.resistance(lower,accepted.age,compression).first + stiffness*lower <= driving,
                  ExcMessage("The positive sliding branch has no admissible rate."));
      double speed = std::clamp(accepted.rate,lower,upper);
      for (unsigned int iteration=0; iteration<100; ++iteration)
        {
          const auto response = friction.resistance(speed,accepted.age,compression);
          const double residual = response.first + stiffness*speed - driving;
          if (std::abs(residual) <= 32.*std::numeric_limits<double>::epsilon()*driving)
            return speed;
          if (residual > 0.)
            upper = speed;
          else
            lower = speed;
          const double next = speed-residual/(response.second+stiffness);
          speed = (lower < next && next < upper) ? next : .5*(lower+upper);
        }
      AssertThrow(false, ExcMessage("The fixed-fault section solve did not converge."));
      return speed;
    }

    template <int dim>
    void FixedFault<dim>::prepare_stokes(const LinearAlgebra::BlockVector &iterate)
    {
      trial_rate = section_rate(section_mean(iterate,false));
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
            force->elastic_force[q] = -stress_offset(trial_rate);
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
      const auto mean = section_mean(solution,true);
      const double rate = section_rate(mean);
      AssertThrow(std::abs(rate-trial_rate) <= 1.e-8*std::max(rate,trial_rate),
                  ExcMessage("The converged Stokes field and section rate disagree."));
      const double dt = this->get_timestep();
      if (dt > 0.)
        {
          const auto stress = stress_offset(rate) + 2.*kappa*Utilities::Tensors::consistent_deviator(symmetrize(mean.gradient));
          // The L2 projection of this spatial constant has the same value
          // at every DGQ2 support point on each owned and ghosted cell.
          for (unsigned int c=0; c<3; ++c)
            solution.block(this->introspection().block_indices.compositional_fields[c]) = stress.access_raw_entry(c);
        }
      accepted = {friction.advance_age(rate,accepted.age,dt), accepted.slip+dt*rate, rate};
    }

    template <int dim>
    const typename FixedFault<dim>::Section &FixedFault<dim>::accepted_section() const
    {
      return accepted;
    }

    template <int dim>
    void FixedFault<dim>::save(std::map<std::string,std::string> &status) const
    {
      const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      std::ostringstream buffer;
      {
        oarchive archive(buffer);
        archive << box.get_origin() << box.get_extents() << accepted.age << accepted.slip << accepted.rate;
      }
      status["fixed vertical fault, constant section, DGQ2 stress v1"] = buffer.str();
    }

    template <int dim>
    void FixedFault<dim>::load(const std::map<std::string,std::string> &status)
    {
      const auto record = status.find("fixed vertical fault, constant section, DGQ2 stress v1");
      AssertThrow(record != status.end(),ExcMessage("Incompatible fixed-fault representation in checkpoint."));
      std::istringstream buffer(record->second);
      iarchive archive(buffer);
      const auto &box = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      Point<dim> origin, extents;
      archive >> origin >> extents;
      AssertThrow(origin == box.get_origin() && extents == box.get_extents(),
                  ExcMessage("The checkpoint has a different fault geometry."));
      archive >> accepted.age >> accepted.slip >> accepted.rate;
    }

    template <int dim>
    void FixedFault<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      prm.enter_subsection("Fixed fault");
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
                  ExcMessage("The homogeneous fault requires a constant bulk viscosity."));
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
      accepted = {prm.get_double("Initial age"),0.,friction.reference_speed};
      prm.leave_subsection();
      prm.leave_subsection();
      AssertThrow(std::min({viscosity,modulus,initial_interval,friction.direct_effect,minimum_speed,
                            friction.reference_speed,friction.slip_distance,accepted.age
                           }) > 0.,
                  ExcMessage("The fixed fault requires positive material, time and speed scales."));
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
      width = Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model()).get_extents()[0];
      kappa = -viscosity*std::expm1(-modulus*initial_interval/viscosity);
      trial_rate = accepted.rate;
    }

    ASPECT_REGISTER_MATERIAL_MODEL(FixedFault,"fixed fault",
                                   "Homogeneous plane shear across a vertical mature fault on the left side of a box. "
                                   "The phase is a spatial constant strictly between zero and one. "
                                   "The Simple model supplies the bulk viscosity and thermal properties.")
  }
}
