#include <aspect/initial_composition/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/phase_field.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>

namespace aspect
{
  namespace BP3
  {
    class AiryStressFunction
    {
      public:
        AiryStressFunction()
          : A(numbers::signaling_nan<double>())
          , B(numbers::signaling_nan<double>())
          , C(numbers::signaling_nan<double>())
          , D(numbers::signaling_nan<double>())
        {}

        void initialize(const double wedge_angle,
                        const double normal_stress,
                        const double shear_stress);

        SymmetricTensor<2, 2> 
        compute_stress(const double radius,
                       const double angle,
                       const double uniform_horizontal_stress,
                       const double regularization_radius,
                       const int    x_direction) const;

      private:
        double A;
        double B;
        double C;
        double D;
    };



    void 
    AiryStressFunction::initialize(const double alpha,
                                   const double normal_stress,
                                   const double shear_stress)
    {
      // Airy stress function
      //   Phi = r^2 f(theta),
      //   f = A + B theta + C cos(2 theta) + D sin(2 theta).
      // Zero top traction gives A=-C and B=-2D. C and D are determined by
      // sigma_theta_theta(alpha) = normal_stress and 
      // sigma_r_theta(alpha) = shear_stress.
      const double sin_2a = std::sin(2. * alpha);
      const double cos_2a = std::cos(2. * alpha);

      const double a11 = 2. * (cos_2a - 1.);
      const double a12 = 2. * (sin_2a - 2. * alpha);
      const double a21 = 2. * sin_2a;
      const double a22 = 2. * (1. - cos_2a);
      const double determinant = a11 * a22 - a12 * a21;

      AssertThrow(std::abs(determinant) > 1e-12,
                  ExcMessage("The Airy initial stress is singular for this dip angle."));

      C = (normal_stress * a22 - a12 * shear_stress) / determinant;
      D = (a11 * shear_stress - normal_stress * a21) / determinant;
      A = -C;
      B = -2. * D;
    }



    SymmetricTensor<2, 2>
    AiryStressFunction::compute_stress(const double r,
                                       const double theta,
                                       const double sigma_h,
                                       const double R,
                                       const int    x_sign) const
    {
      const double sin_2t = std::sin(2. * theta);
      const double cos_2t = std::cos(2. * theta);

      const double sin_t = std::sin(theta);
      const double cos_t = std::cos(theta);

      SymmetricTensor<2, 2> sigma_polar;

      if (R == 0)
        {
          // Airy stress function
          //   Phi = r^2 f(theta),
          //   f = A + B theta + C cos(2 theta) + D sin(2 theta).
          const double f = A + B * theta + C * cos_2t + D * sin_2t;

          sigma_polar[0][0] = 2. * f - 4. * C * cos_2t - 4. * D * sin_2t;
          sigma_polar[1][1] = 2. * f;
          sigma_polar[0][1] = -B + 2. * C * sin_2t - 2. * D * cos_2t;
        }
      else
        {
          // Regularize the Airy stress function near the fault/free-surface
          // intersection:
          //
          //   Phi = r^2 [f_U(theta) + q(r) (f_A(theta) - f_U(theta))].
          //
          // f_A is the original Airy solution and f_U corresponds to the
          // uniform Cartesian stress diag(sigma_h,0). Since Phi is still an
          // Airy stress function, div(sigma)=0 is preserved exactly. The radial
          // blending function
          //
          //   q = 1 - R/sqrt(r^2+R^2)
          //
          // satisfies q=0(r^2) at the origin, so the stress approaches the
          // uniform value smoothly, and q -> 1 away from the intersection.
          const double f_A = A + B * theta + C * cos_2t + D * sin_2t;
          const double df_A = B - 2. * C * sin_2t + 2. * D * cos_2t;
          const double d2f_A = -4. * C * cos_2t - 4. * D * sin_2t;

          // Phi_U = 1/2 sigma_h y^2 = r^2 f_U(theta).
          const double f_U = 0.5 * sigma_h * sin_t * sin_t;
          const double df_U = 0.5 * sigma_h * sin_2t;
          const double d2f_U = sigma_h * cos_2t;

          const double delta_f = f_A - f_U;
          const double delta_df = df_A - df_U;
          const double delta_d2f = d2f_A - d2f_U;

          const double s = R / std::sqrt(r * r + R * R);
          const double q = 1. - s;

          // These are r q'(r) and r^2 q''(r), written without divisions by r.
          const double r_dq = s * (1. - s * s);
          const double r2_d2q = s * (1. - s * s) * (3. * s * s - 2.);

          // Compute the stress in the local polar basis
          sigma_polar[0][0] = (2. * f_U + d2f_U) + q * (2. * delta_f + delta_d2f) + r_dq * delta_f;
          sigma_polar[1][1] = 2. * f_U + (2. * q + 4. * r_dq + r2_d2q) * delta_f;
          sigma_polar[0][1] = -df_U - (q + r_dq) * delta_df;
        }

      if (x_sign == 0)
        return sigma_polar;

      // Transform from the local polar basis to global (x,y). The local
      // positive x direction follows the top surface away from the fault
      // intersection, and local positive y points downward.
      if (std::abs(x_sign) == 1)
        {
          Tensor<2, 2> T;
          T[0][0] = x_sign * cos_t;
          T[0][1] = -sin_t;
          T[1][0] = -x_sign * sin_t;
          T[1][1] = -cos_t;

          return symmetrize(transpose(T) * sigma_polar * T);
        }

      return numbers::signaling_nan<SymmetricTensor<2, 2>>();
    }



    class BP3Model
    {
      public:
        BP3Model();

        double initial_crack_driving_force(const Point<2> &position,
                                           const PhaseField::PhaseFieldProfile &profile,
                                           const PhaseFieldHandler<2> &phase_field_handler,
                                           const unsigned int j) const;

        double initial_slip_rate() const;

        double initial_slip_state(const Point<2> &position,
                                  const PhaseField::PhaseFieldProfile &profile,
                                  const PhaseFieldHandler<2> &phase_field_handler,
                                  const unsigned int j) const;

        Tensor<1, 2> fault_normal() const;

        Tensor<1, 2> fault_slip_direction() const;

        SymmetricTensor<2, 2>
        initial_stress(const Point<2> &position) const;

        double slip_strengthing_region_fraction(const Point<2> &position) const;

        double peak_phase_field() const;

        static void declare_parameters(ParameterHandler &prm);
        
        void parse_parameters(ParameterHandler &prm);

      private:
        const double a_max;
        const double a0;
        const double b0;
        const double f0;
        const double L;
        const double sigma0_n;
        const double Vinit;
        const double V0;
        const double eta;
        const double G;
        const double D;
        const double d;
        const double tau0;

        double psi;
        double k;
        double w;
        double h;
        double R;
        double phi_max;

        BP3::AiryStressFunction airy_stress_function_left;
        BP3::AiryStressFunction airy_stress_function_right;
    };



    BP3Model::BP3Model()
      : a_max(0.025)
      , a0(0.010)
      , b0(0.015)
      , f0(0.6)
      , L(0.008)
      , sigma0_n(50e6)
      , Vinit(1e-9)
      , V0(1e-6)
      , eta(4.624e6)
      , G(3.204e10)
      , D(15e3)
      , d(3e3)
      , tau0(sigma0_n * a_max * std::asinh(Vinit / (2. * V0) * std::exp((f0 + b0 * std::log(V0 / Vinit)) / a_max)) + eta * Vinit)
    {}



    void BP3Model::declare_parameters(ParameterHandler &prm)
    {
      prm.declare_entry("Dip angle", "60",
                        Patterns::Double(0,90),
                        "");
      prm.declare_entry("Model width", "60e3",
                        Patterns::Double(0),
                        "");
      prm.declare_entry("Model height", "40e3",
                        Patterns::Double(0),
                        "");
      prm.declare_entry("Stress regularization radius", "0",
                        Patterns::Double(0),
                        "Length scale used to smooth the Airy prestress near the fault/free-surface "
                        "intersection. A value of zero disables regularization.");
      prm.declare_entry("Peak phase field", "0.9",
                        Patterns::Double(0, 1),
                        "");
    }



    void BP3Model::parse_parameters(ParameterHandler &prm)
    {
      const double dip_angle = prm.get_double("Dip angle");
      psi = dip_angle * numbers::PI / 180.;
      AssertThrow(psi > 0. && psi <= 0.5 * numbers::PI,
                  ExcMessage("Dip angle must be in the interval (0, 90] degrees."));

      k = std::tan(psi);

      w = prm.get_double("Model width");
      h = prm.get_double("Model height");
      R = prm.get_double("Stress regularization radius");
      phi_max = prm.get_double("Peak phase field");

      /*         Left                             Right
       *     ______________                 ________________
       *        theta=0    \                \    theta=0
       *                    \                \
       *                     \                \
       *                      \                \
       *                 theta=pi-psi       theta=psi
       *
       * (The sign of shear stress differs because the local angular basis
       * has opposite oritation on the two fault faces)
       */
      airy_stress_function_left.initialize(numbers::PI - psi, -sigma0_n, -tau0);
      airy_stress_function_right.initialize(psi, -sigma0_n, tau0);
    }



    double
    BP3Model::
    initial_crack_driving_force(const Point<2> &position,
                                const PhaseField::PhaseFieldProfile &profile,
                                const PhaseFieldHandler<2> &phase_field_handler,
                                const unsigned int j) const
    {
      AssertIndexRange(j, 2);
      std::vector<double> volume_fractions(2);
      volume_fractions[j] = 1;

      const double x = position[0];
      const double y = position[1];
      const double zeta = std::abs(k * x + y - (k * w + h) * 0.5) / std::sqrt(1. + k * k);
      const double phi = profile.value(zeta);

      return phase_field_handler.stationary_crack_driving_force(volume_fractions, phi, phi_max);
    }



    double BP3Model::initial_slip_rate() const
    {
      return Vinit;
    }



    double 
    BP3Model::initial_slip_state(const Point<2> &position,
                                 const PhaseField::PhaseFieldProfile &profile,
                                 const PhaseFieldHandler<2> &phase_field_handler,
                                 const unsigned int j) const
    {
      const double x = position[0];
      const double y = position[1];

      const SymmetricTensor<2, 2> sigma = initial_stress(position);

      const Tensor<1, 2> n = fault_normal();
      const Tensor<1, 2> s = fault_slip_direction();

      const Tensor<1, 2> traction = sigma * n;
      const double sigma_n = -traction * n;
      const double tau     =  traction * s;

      AssertThrow(sigma_n > 0,
                  ExcMessage("Initial effective normal stress is non-compressive."));

      const double fraction = slip_strengthing_region_fraction(position);
      const double a = a_max * fraction + a0 * (1. - fraction);

      // Compute the initial cohesive force
      AssertIndexRange(j, 2);
      std::vector<double> volume_fractions(2, 0);
      volume_fractions[j] = 1;

      const double zeta = std::abs(k * x + y - (k * w + h) * 0.5) / std::sqrt(1. + k * k);
      const double phi = profile.value(zeta);

      const double H = phase_field_handler.stationary_crack_driving_force(volume_fractions, phi, phi_max);
      const double g = phase_field_handler.energetic_degradation(volume_fractions, phi);

      const double tau_coh = g * std::sqrt(2. * G * H);

      const double mu_init = (tau - tau_coh - eta * Vinit) / sigma_n;

      AssertThrow(mu_init > 0,
                  ExcMessage("Initial stress is incompatible with positive slip in the "
                             "prescribed slip direction."));

      return L / V0 * std::exp(a / b0 * std::log(2. * V0 / Vinit * std::sinh(mu_init / a)) - f0 / b0);
    }



    Tensor<1, 2> BP3Model::fault_normal() const
    {
      const double denom = std::sqrt(1. + k*k);

      Tensor<1, 2> n;
      n[0] = k  / denom;
      n[1] = 1. / denom;

      return n;
    }



    Tensor<1, 2> BP3Model::fault_slip_direction() const
    {
      const double denom = std::sqrt(1. + k*k);

      Tensor<1, 2> s;
      s[0] = -1. / denom;
      s[1] =  k  / denom;

      return s;
    }



    SymmetricTensor<2, 2>
    BP3Model::initial_stress(const Point<2> &position) const
    {
      const double x = position[0];
      const double y = position[1];

      // Intersection of fault surface and top surface.
      const double x_surface = 0.5 * (w - h / k);

      const bool right_side = (k * x + y - 0.5 * (k * w + h) >= 0.);

      // Angular coodinate in the local polar basis
      const double local_x = right_side ? x - x_surface : x_surface - x;
      const double local_y = h - y;
      const double r     = std::hypot(local_x, local_y);
      const double theta = std::atan2(local_y, local_x);

      // The uniform stress used at r=0 is chosen to preserve the prescribed
      // compressive fault-normal stress sigma0_n. It also satisfies the
      // traction-free horizontal surface exactly.
      const double sigma_h = -sigma0_n / std::pow(std::sin(psi), 2);

      return (right_side ?
              airy_stress_function_right.compute_stress(r, theta, sigma_h, R, 1) :
              airy_stress_function_left.compute_stress(r, theta, sigma_h, R, -1));
    }



    double 
    BP3Model::
    slip_strengthing_region_fraction(const Point<2> &position) const
    {
      const double y = position[1];
      return std::min(1., std::max(0., std::sqrt(1. + k*k) * (h - y) / (d * k) - D / d));
    }



    double BP3Model::peak_phase_field() const
    {
      return phi_max;
    }
  }



  namespace InitialComposition
  {
    template <int dim>
    class BP3_QD : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void initialize() override;

        double
        initial_composition(const Point<dim>   &position,
                            const unsigned int  n_comp) const override;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm) override;

      private:
        BP3::BP3Model bp3_model;

        std::vector<std::unique_ptr<PhaseField::PhaseFieldProfile>> phase_field_profiles;
    };



    template <int dim>
    void BP3_QD<dim>::initialize()
    {
      this->get_signals().post_simulator_initialization.connect(
        [this](const SimulatorAccess<dim> &)
      {
        phase_field_profiles = this->get_phase_field_handler()
          .get_phase_field_profiles(bp3_model.peak_phase_field());
      });
    }



    template <>
    double
    BP3_QD<2>::initial_composition(const Point<2> &position,
                                   const unsigned int n_comp) const
    {
      switch (n_comp)
        {
          case 0: // H
            // We know that the two compositional fields share the same degradation
            // function, so just pick one of them for efficiency
            return bp3_model.initial_crack_driving_force(position,
                                                         *phase_field_profiles[0],
                                                         this->get_phase_field_handler(),
                                                         0);

          case 1: // V
            return bp3_model.initial_slip_rate();

          case 2: // Theta
            return bp3_model.initial_slip_state(position,
                                                *phase_field_profiles[0],
                                                this->get_phase_field_handler(),
                                                0);

          case 3: // n_x
            return bp3_model.fault_normal()[0];

          case 4: // n_y
            return bp3_model.fault_normal()[1];

          case 5: // tau_xx
            return deviator(bp3_model.initial_stress(position)).access_raw_entry(0);

          case 6: // tau_yy
            return deviator(bp3_model.initial_stress(position)).access_raw_entry(1);

          case 7: // tau_xy
            return deviator(bp3_model.initial_stress(position)).access_raw_entry(2);

          case 8: // inactive region
            return bp3_model.slip_strengthing_region_fraction(position);

          default:
            Assert(false, ExcInternalError());
        }

      return numbers::signaling_nan<double>();
    }



    template <>
    double
    BP3_QD<3>::initial_composition(const Point<3> &/*position*/,
                                   const unsigned int /*n_comp*/) const
    {
      AssertThrow(false, ExcNotImplemented());
    }



    template <int dim>
    void BP3_QD<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("BP3");
        {
          BP3::BP3Model::declare_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void BP3_QD<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("BP3");
        {
          bp3_model.parse_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }



  namespace BoundaryTraction
  {
    template <int dim>
    class BP3_QD : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        Tensor<1, dim>
        boundary_traction(const types::boundary_id  boundary_indicator,
                          const Point<dim>         &position,
                          const Tensor<1, dim>     &normal_vector) const override;

        static void declare_parameters(ParameterHandler &prm);

        void parse_parameters(ParameterHandler &prm) override;

      private:
        BP3::BP3Model bp3_model;
    };



    template <>
    Tensor<1, 2>
    BP3_QD<2>::boundary_traction(const types::boundary_id,
                                 const Point<2>     &position,
                                 const Tensor<1, 2> &normal_vector) const
    {
      const SymmetricTensor<2, 2> sigma = bp3_model.initial_stress(position);

      return sigma * normal_vector;
    }



    template <>
    Tensor<1, 3>
    BP3_QD<3>::boundary_traction(const types::boundary_id,
                                 const Point<3>     &,
                                 const Tensor<1, 3> &) const
    {
      AssertThrow(false, ExcNotImplemented());
      return numbers::signaling_nan<Tensor<1, 3>>();
    }

   
    
    template <int dim>
    void BP3_QD<dim>::declare_parameters(ParameterHandler &)
    {
      // Let the initial composition model declare the parameters
    }



    template <int dim>
    void BP3_QD<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Initial composition model");
      {
        prm.enter_subsection("BP3");
        {
          bp3_model.parse_parameters(prm);
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    } 
  }
}

// explicit instantiations
namespace aspect
{
  namespace InitialComposition
  {
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(BP3_QD,
                                              "bp3",
                                              "")
  }

  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(BP3_QD,
                                            "bp3",
                                            "")
  }
}
