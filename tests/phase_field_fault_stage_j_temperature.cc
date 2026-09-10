#include "phase_field_fault_stage_i.cc"

#include <deal.II/numerics/vector_tools.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyFaultSurfaceTemperature : public Interface<dim>,
      public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(coupled_solve_converged,
                      ExcMessage("The surface-temperature test requires both "
                                 "final nonlinear convergence criteria."));
          auto &model = const_cast<MaterialModel::PhaseFieldFault<dim> &>(
            Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
              this->get_material_model()));
          model.prepare_reconstructed_fault_mechanical_solve();
          auto &manager = this->get_reconstructed_fault_manager();
          const auto &fault = manager.get_fault(0);
          const unsigned int segment = fault.n_cells()/2;
          const Tensor<1,dim> tangent =
            (fault.vertex(segment+1)-fault.vertex(segment))
            / fault.vertex(segment+1).distance(fault.vertex(segment));
          Tensor<1,dim> normal;
          normal[0] = -tangent[1];
          normal[1] = tangent[0];
          const Point<dim> center =
            0.5*(fault.vertex(segment)+fault.vertex(segment+1));
          const double offset = std::min(
            0.1*this->get_phase_field_handler().get_length_scale(),
            0.25*fault.vertex(segment+1).distance(fault.vertex(segment)));
          const std::vector<Point<dim>> points =
            {center-offset*normal, center+offset*normal};

          // The actual reconstructed normal, not a vertical particle lattice,
          // defines the transverse pair. Verify the production association first.
          for (const auto &point : points)
            {
              const auto projection = manager.project_to_normal_profiles(point);
              AssertThrow(projection.active && projection.fault_index == 0
                          && projection.segment_index == segment
                          && std::abs(projection.xi-0.5) < 1.e-12,
                          ExcMessage("Controlled transverse points do not share "
                                     "the intended production fault coordinate."));
            }
          Utilities::MPI::RemotePointEvaluation<dim> cache;
          cache.reinit(this->get_phase_field_handler().get_grid_cache(), points);
          const auto temperatures = VectorTools::point_values<1>(
            cache, this->get_dof_handler(), this->get_solution(),
            VectorTools::EvaluationFlags::avg,
            this->introspection().component_indices.temperature);
          AssertThrow(std::abs(temperatures[1]-temperatures[0]) > 1.0,
                      ExcMessage("The transverse bulk-temperature contrast is not observable."));

          using Model = MaterialModel::PhaseFieldFault<dim>;
          typename Model::ReconstructedFaultPointInputs in;
          in.fault_index = 0;
          in.segment_index = segment;
          in.xi = 0.5;
          // Keep the slope times V comparable to the history intercept, so
          // extracting the latter does not subtract two enormous slip terms.
          in.slip_rate = model.minimum_fault_slip_rate();
          in.phase_field = in.previous_phase_field = 1.e-8;
          in.bulk_material_fractions = {0.5, 0.5};
          in.dynamic_pressure = 0.0;
          in.slip_tensor = symmetrize(outer_product(tangent, normal));
          in.normal_tensor = symmetrize(outer_product(normal, normal));
          std::vector<typename Model::ReconstructedFaultPointResponse> responses;
          for (unsigned int i=0; i<points.size(); ++i)
            {
              in.xi = manager.project_to_normal_profiles(points[i]).xi;
              in.position = points[i];
              in.temperature = temperatures[i];
              responses.push_back(model.evaluate_reconstructed_fault_point(in));
            }
          AssertThrow(std::abs(responses[0].kappa-responses[1].kappa)
                      > 1.e-3*responses[0].kappa,
                      ExcMessage("Bulk Maxwell temperature dependence was not exercised."));
          const auto surface_state =
            MaterialModel::internal::PhaseFieldFaultTestAccess<dim>
            ::surface_material_state_at_projection(model, 0, segment, 0.5);
          AssertThrow(std::abs(surface_state.second-(293.0+100.0*center[1])) < 1.e-10,
                      ExcMessage("Surface temperature is not the FE temperature on the fault."));

          // With zero friction/damping and unchanged phi/I_h, remove only the
          // bulk stress term from the production response. What remains is the
          // surface slope kappa_Gamma/I_h and history intercept beta_Gamma*T_old.
          std::vector<double> slopes, intercepts;
          for (const auto &response : responses)
            {
              const double bulk_slope = 2.0*response.kappa
                *response.localization_factor*(in.slip_tensor*in.slip_tensor);
              slopes.push_back(response.minus_derivative_wrt_slip_rate-bulk_slope);
              intercepts.push_back(-response.residual_density
                                    -response.minus_derivative_wrt_slip_rate*in.slip_rate);
            }
          AssertThrow(std::abs(slopes[0]-slopes[1]) < 1.e-11*std::abs(slopes[0])
                      && std::abs(intercepts[0]-intercepts[1])
                         < 1.e-11*std::abs(intercepts[0]),
                      ExcMessage("Surface cohesive coefficients depend on transverse bulk temperature."));
          return {"Stage-J controlled transverse-temperature evaluation:", "verified"};
        }
    };

    ASPECT_REGISTER_POSTPROCESSOR(VerifyFaultSurfaceTemperature,
      "verify stage j surface temperature",
      "Evaluate the production constitutive response at controlled transverse points.")
  }
}
