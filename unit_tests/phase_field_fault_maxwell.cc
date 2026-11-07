/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.
*/

#include "common.h"
#include <aspect/time_stepping/reconstructed_fault.h>

#include <aspect/material_model/rheology/fault_friction.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/property/maxwell_stress.h>
#include <aspect/phase_field.h>
#include <aspect/reconstructed_fault/objective_stress.h>

#include "../tests/phase_field_fault_test_access.h"

#include <sstream>
#include <iomanip>

namespace
{
  using PlaneStress = dealii::SymmetricTensor<2,2>;
  using PlaneGradient = dealii::Tensor<2,2>;
  using MaxwellAccess = aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;

  double plane_norm_square(const PlaneStress &a)
  {
    return a*a + dealii::trace(a)*dealii::trace(a); // include zz = -xx-yy
  }

  PlaneGradient rigid_gradient(double angular_velocity)
  {
    PlaneGradient l;
    l[0][1]=-angular_velocity;
    l[1][0]=angular_velocity;
    return l;
  }

  PlaneStress rotation_reference(const PlaneStress &a, double angle)
  {
    // Independent component/eigenbasis solution, not a matrix congruence.
    PlaneStress b;
    const double mean=.5*(a[0][0]+a[1][1]);
    const double difference=.5*(a[0][0]-a[1][1]);
    b[0][0]=mean+difference*std::cos(2*angle)-a[0][1]*std::sin(2*angle);
    b[1][1]=2*mean-b[0][0];
    b[0][1]=difference*std::sin(2*angle)+a[0][1]*std::cos(2*angle);
    return b;
  }

  PlaneStress maxwell_lie(const PlaneStress &a,const PlaneStress &d,
                         double w,double eta,double g,double dt)
  {
    return MaxwellAccess::maxwell_stress(eta,g,dt,d,
      aspect::ReconstructedFaultUtilities::rotate_plane_strain_stress(a,rigid_gradient(w),dt));
  }

  PlaneStress ode_reference(PlaneStress a,double end,unsigned int steps,
                           const std::function<double(double)> &w,
                           const std::function<PlaneStress(double)> &d,double eta,double g)
  {
    // RK4 integrates the selected tensor ODE directly. No rotation helper or
    // exponential Maxwell coefficients occur in this independent reference.
    const auto rhs=[&](const PlaneStress &s,double t)
    {
      auto result=2*g*d(t)-(g/eta)*s;
      result[0][0]-=2*w(t)*s[0][1];
      result[1][1]+=2*w(t)*s[0][1];
      result[0][1]+=w(t)*(s[0][0]-s[1][1]);
      return result;
    };
    const double dt=end/steps;
    for (unsigned int i=0;i<steps;++i)
      {
        const double t=i*dt;
        const auto k1=rhs(a,t), k2=rhs(a+.5*dt*k1,t+.5*dt),
                   k3=rhs(a+.5*dt*k2,t+.5*dt), k4=rhs(a+dt*k3,t+dt);
        a+=dt/6*(k1+2*k2+2*k3+k4);
      }
    return a;
  }
}

TEST_CASE("Finite Jaumann rotation and Maxwell loss preserve plane strain", "[objective_stress]")
{
  PlaneStress a;
  a[0][0]=7.; a[1][1]=-2.; a[0][1]=1.3;
  const auto rotated=aspect::ReconstructedFaultUtilities::rotate_plane_strain_stress(
    a,rigid_gradient(.7),2.1);
  CHECK(std::sqrt(plane_norm_square(rotated-rotation_reference(a,1.47)))<2e-14);
  CHECK(std::abs(plane_norm_square(rotated)-plane_norm_square(a))<8e-14);
  CHECK(std::abs(dealii::trace(rotated)-dealii::trace(a))<4e-15);
  CHECK(std::abs(dealii::determinant(rotated)-dealii::determinant(a))<4e-14);
  PlaneStress rate;
  rate[0][0]=.2; rate[1][1]=.1; rate[0][1]=-.15;
  CHECK(maxwell_lie(a,rate,0.,23.,3.,.17)==MaxwellAccess::maxwell_stress(23.,3.,.17,rate,a));
  auto relaxed=a;
  for (unsigned int i=0;i<100;++i)
    relaxed=maxwell_lie(relaxed,{},.7,23.,3.,.021);
  const auto exact=std::exp(-2.1*3/23)*rotation_reference(a,1.47);
  const double error=std::sqrt(plane_norm_square(relaxed-exact));
  CHECK(error<2e-13);
  const double initial_energy=plane_norm_square(a)/12;
  const double final_energy=plane_norm_square(relaxed)/12;
  const double loss=initial_energy*(-std::expm1(-2*2.1*3/23));
  CHECK(std::abs(final_energy+loss-initial_energy)<2e-13);
  std::cout<<"Objective rotation/relaxation error="<<error<<", energy closure="
           <<final_energy+loss-initial_energy<<std::endl;
}

TEST_CASE("Lie spin source and time dependent observer converge", "[objective_stress]")
{
  PlaneStress initial;
  initial[0][0]=2.; initial[1][1]=-.3; initial[0][1]=.4;
  const auto w=[](double t){return .6+.2*std::sin(t);};
  const auto forcing=[](double t)
  {
    PlaneStress d;
    d[0][0]=.1*std::cos(t); d[1][1]=-.03; d[0][1]=.2+.05*t;
    return d;
  };
  const double g=3.,eta=23.,end=1.7;
  const auto exact=ode_reference(initial,end,20000,w,forcing,eta,g);
  CHECK(std::sqrt(plane_norm_square(exact-ode_reference(initial,end,10000,w,forcing,eta,g)))<1e-12);
  double previous_error=0.,previous_covariance=0.;
  for (const unsigned int steps:{40u,80u,160u,320u})
    {
      auto a=initial,observer=initial;
      const double dt=end/steps;
      for (unsigned int i=0;i<steps;++i)
        {
          const double t=i*dt;
          a=maxwell_lie(a,forcing(t+dt),w(t),eta,g,dt);
          // Superposed Q(t), angle=.2*t^2: W*=W+.4*t and D*=Q D Q^T.
          observer=maxwell_lie(observer,rotation_reference(forcing(t+dt),.2*(t+dt)*(t+dt)),
                               w(t)+.4*t,eta,g,dt);
        }
      const double error=std::sqrt(plane_norm_square(a-exact));
      const double covariance=std::sqrt(plane_norm_square(observer-rotation_reference(a,.2*end*end)));
      if (previous_error>0.)
        {
          CHECK(previous_error/error>1.9); CHECK(previous_error/error<2.1);
          CHECK(previous_covariance/covariance>1.9); CHECK(previous_covariance/covariance<2.1);
        }
      previous_error=error; previous_covariance=covariance;
      std::cout<<"Objective timestep convergence: N="<<steps<<", ODE error="<<error
               <<", superposed-motion error="<<covariance<<std::endl;
    }
}

TEST_CASE("Jaumann simple shear resolves its hypoelastic oscillation", "[objective_stress]")
{
  // Elastic simple shear L01=1: w=-1/2. Analytic tau01=G sin(gamma),
  // tau00=-tau11=G(1-cos(gamma)). This is a constitutive limit, not neo-Hookean elasticity.
  PlaneStress d;
  d[0][1]=.5;
  const double g=3.,eta=1e20,end=8.;
  PlaneStress exact;
  exact[0][0]=g*(1-std::cos(end)); exact[1][1]=-exact[0][0]; exact[0][1]=g*std::sin(end);
  double previous=0.;
  for (unsigned int n:{400u,800u,1600u})
    {
      PlaneStress a;
      for (unsigned int i=0;i<n;++i) a=maxwell_lie(a,d,-.5,eta,g,end/n);
      const double error=std::sqrt(plane_norm_square(a-exact));
      if (previous>0.) {CHECK(previous/error>1.99);CHECK(previous/error<2.01);}
      previous=error;
      std::cout<<"Objective elastic simple shear: N="<<n<<", error="<<error<<std::endl;
    }
}

TEST_CASE("Objective fault frame and width sensitivity", "[objective_stress]")
{
  PlaneStress stress;
  stress[0][0]=7.;stress[1][1]=-2.;stress[0][1]=1.3;
  dealii::Tensor<1,2> tangent,normal;
  tangent[0]=1.;normal[1]=1.;
  const double angle=.9,pressure=20.;
  const auto rotated=rotation_reference(stress,angle);
  const auto rotate_vector=[&](dealii::Tensor<1,2> x)
  {
    dealii::Tensor<1,2> y;
    y[0]=std::cos(angle)*x[0]-std::sin(angle)*x[1];
    y[1]=std::sin(angle)*x[0]+std::cos(angle)*x[1];return y;
  };
  const auto t=rotate_vector(tangent),n=rotate_vector(normal);
  CHECK(std::abs(t*(rotated*n)-tangent*(stress*normal))<4e-15);
  CHECK(std::abs((pressure-n*(rotated*n))-(pressure-normal*(stress*normal)))<4e-15);
  // Fixed physical slip delta=.5, simple diffuse L01=V/width. Total W gives
  // angle=-delta/(2*width); no automatic removal of this width dependence.
  for (double width:{1.,.5,.25})
    {
      PlaneGradient l;l[0][1]=.5/width;
      const auto actual=aspect::ReconstructedFaultUtilities::rotate_plane_strain_stress(stress,l,1.);
      CHECK(std::sqrt(plane_norm_square(actual-rotation_reference(stress,-.25/width)))<1e-14);
      std::cout<<"Objective diffuse-slip width="<<width<<", angle="<<-.25/width<<std::endl;
    }
}

TEST_CASE("Rotation plus forced Maxwell has a separate exact substep energy ledger", "[objective_stress]")
{
  PlaneStress old,rate;
  old[0][0]=2.;old[1][1]=-.3;old[0][1]=.4;
  rate[0][0]=.1;rate[1][1]=-.03;rate[0][1]=.2;
  const double g=3.,eta=23.,dt=.17,r=g/eta;
  const auto rotated=rotation_reference(old,.6*dt);
  const auto after=maxwell_lie(old,rate,.6,eta,g,dt);
  const auto equilibrium=2*eta*rate, departure=rotated-equilibrium;
  const auto dot=[](const PlaneStress &a,const PlaneStress &b)
  {return a*b+dealii::trace(a)*dealii::trace(b);};
  const double a=-std::expm1(-r*dt)/r,b=-std::expm1(-2*r*dt)/(2*r);
  const double work=dot(rate,equilibrium)*dt+dot(rate,departure)*a;
  const double viscous=(dot(equilibrium,equilibrium)*dt+2*dot(equilibrium,departure)*a
                        +dot(departure,departure)*b)/(2*eta);
  const double change=(plane_norm_square(after)-plane_norm_square(old))/(4*g);
  CHECK(std::abs(work-viscous-change)<2e-14);
  CHECK(std::abs(plane_norm_square(rotated)-plane_norm_square(old))<2e-14);
  std::cout<<"Objective forced-Maxwell substep: work="<<work<<", viscous="<<viscous
           <<", storage change="<<change<<", closure="<<work-viscous-change<<std::endl;
}

TEST_CASE("Homogeneous Maxwell coefficients are independent of chemical labels",
          "[homogeneous_maxwell]")
{
  using Access=aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;
  namespace Material=aspect::MaterialModel::MaterialUtilities;
  const double G=32038120320.,eta=1e26;
  double old_min=std::numeric_limits<double>::max(),old_max=0.;
  for (const auto averaging : {Material::arithmetic,Material::harmonic,
                               Material::geometric,Material::maximum_composition})
    for (unsigned int i=0; i<=1000; ++i)
      {
        const double f=i/1000.;
        const std::vector<double> fractions{1.-f,f};
        const double modulus=Access::material_shear_modulus(true,{G,G},fractions,averaging);
        REQUIRE(modulus==G);
        for (const double dt : {200000.,4e6,4e7})
          REQUIRE(Access::maxwell_coefficients(eta,modulus,dt)
                  ==Access::maxwell_coefficients(eta,G,dt));
        if (averaging==Material::harmonic)
          {
            const double mixed=Access::material_shear_modulus(false,{G,G},fractions,averaging);
            const auto c=Access::maxwell_coefficients(eta,mixed,4e6);
            old_min=std::min(old_min,c.second);old_max=std::max(old_max,c.second);
          }
      }
  // Independent heterogeneous-mixture values retain the legacy rules. The
  // homogeneous shortcut is selected by its validated configuration only.
  CHECK(Access::material_shear_modulus(false,{2.,8.},{.25,.75},Material::arithmetic)==6.5);
  CHECK(Access::material_shear_modulus(false,{2.,8.},{.25,.75},Material::harmonic)==Approx(32./7.));
  CHECK(Access::material_shear_modulus(false,{2.,8.},{.25,.75},Material::geometric)==Approx(4.*std::sqrt(2.)));
  CHECK(Access::material_shear_modulus(false,{2.,8.},{.25,.75},Material::maximum_composition)==8.);
  std::cout<<std::setprecision(17)<<"Homogeneous coefficient regression: harmonic old kappa range="
           <<old_min<<":"<<old_max<<", configured kappa="
           <<Access::maxwell_coefficients(eta,G,4e6).second<<std::endl;
}

namespace
{
  class TestPhaseFieldModel : public aspect::MaterialModel::PhaseFieldModel<2>
  {
    public:
      std::vector<double> get_critical_crack_driving_forces() const override
      {
        return {1.0};
      }

      std::vector<double> get_critical_energy_release_rates() const override
      {
        return {1.0};
      }
  };
}



TEST_CASE("Phase-field physical and activation ranges are distinct")
{
  const TestPhaseFieldModel model;
  REQUIRE(model.get_phase_field_range() == std::make_pair(0.0, 1.0));
  REQUIRE(model.get_phase_field_activation_threshold() == 0.01);
  REQUIRE(model.get_phase_field_upper_admissibility_threshold() == 1.0);

  aspect::MaterialModel::PhaseFieldFault<2> fault_model;
  REQUIRE(fault_model.get_phase_field_upper_admissibility_threshold() == 0.99);
}



TEST_CASE("I_h distinguishes physical phase-field range from singular degradation")
{
  using TestAccess =
    aspect::MaterialModel::internal::PhaseFieldFaultTestAccess<2>;

  const double tolerance =
    TestAccess::normalization_phase_field_undershoot_tolerance();
  REQUIRE(TestAccess::normalization_effective_phase_field(-0.5*tolerance) == 0.0);
  REQUIRE(TestAccess::normalization_effective_phase_field(0.25) == 0.25);
  REQUIRE(TestAccess::normalization_effective_phase_field(1.0) == 1.0);
  REQUIRE_NOTHROW(TestAccess::validate_normalization_phase_field_minimum(-tolerance));
  REQUIRE_THROWS_WITH(
    TestAccess::validate_normalization_phase_field_minimum(-2.0*tolerance),
    Catch::Matchers::Contains("error-detection"));
  REQUIRE(TestAccess::normalization_integrand(0.0, 1.0) == 0.0);
  REQUIRE_THROWS_WITH(TestAccess::normalization_integrand(1.0, 0.0),
                      Catch::Matchers::Contains("I_h singularity"));
  REQUIRE_THROWS_WITH(TestAccess::normalization_effective_phase_field(1.0+1.e-6),
                      Catch::Matchers::Contains("phase-field invariant"));
}



TEST_CASE("Legacy slip-rate normalization requires an interior phase-field interval")
{
  const aspect::PhaseField::GeometricFunction geometric_function(0.1, 1.0, 8.0/3.0);
  const aspect::PhaseField::DegradationFunction degradation_function(0.0, 1.0);
  REQUIRE_THROWS_WITH(
    aspect::PhaseField::SlipRateNormalizer(geometric_function,
                                           degradation_function,
                                           0.01,
                                           1.0),
    Catch::Matchers::Contains("strict interior"));
}



TEST_CASE("MaxwellStress particle property has one symmetric tensor")
{
  aspect::Particle::Property::MaxwellStress<2> property_2d;
  const auto information_2d = property_2d.get_property_information();
  REQUIRE(information_2d.size() == 1);
  REQUIRE(information_2d[0].first == "maxwell stress");
  REQUIRE(information_2d[0].second
          == dealii::SymmetricTensor<2,2>::n_independent_components);

  aspect::Particle::Property::MaxwellStress<3> property_3d;
  const auto information_3d = property_3d.get_property_information();
  REQUIRE(information_3d.size() == 1);
  REQUIRE(information_3d[0].first == "maxwell stress");
  REQUIRE(information_3d[0].second
          == dealii::SymmetricTensor<2,3>::n_independent_components);
}



TEST_CASE("FaultFriction declares the Stage E friction-law defaults")
{
  aspect::MaterialModel::Rheology::FaultFriction<2> friction;
  REQUIRE(friction.has_state_variable());

  dealii::ParameterHandler parameters;
  aspect::MaterialModel::Rheology::FaultFriction<2>::declare_parameters(parameters);
  REQUIRE(parameters.get("Friction law") == "rate state");
  REQUIRE(parameters.get("Dynamic friction coefficients") == "0.4");
  REQUIRE(parameters.get("Characteristic weakening slip rates") == "1.e-6");
  std::ostringstream parameter_text;
  parameters.print_parameters(parameter_text,
                              dealii::ParameterHandler::PRM);
  REQUIRE(parameter_text.str().find("Maximum slip rate") == std::string::npos);
}

TEST_CASE("Reconstructed fault state limiter parameter is opt in", "[fault_state_limiter]")
{
  using Limiter = aspect::TimeStepping::ReconstructedFault<2>;
  const std::string disabled = dealii::Utilities::to_string(std::numeric_limits<double>::max());
  for (const std::string &value : std::vector<std::string>{disabled, "0.1", "1e-12", "2"})
    {
      dealii::ParameterHandler parameters;
      Limiter::declare_parameters(parameters);
      parameters.enter_subsection("Time stepping");
      parameters.enter_subsection("Reconstructed fault time step");
      REQUIRE(parameters.get_double("Maximum logarithmic state change") == std::numeric_limits<double>::max());
      parameters.set("Maximum logarithmic state change", value);
      parameters.leave_subsection();
      parameters.leave_subsection();
      Limiter limiter;
      REQUIRE_NOTHROW(limiter.parse_parameters(parameters));
    }
  for (const std::string value : {"-0.1", "nan", "infinity", "-infinity", "invalid", "0.1 junk"})
    {
      dealii::ParameterHandler parameters;
      Limiter::declare_parameters(parameters);
      parameters.enter_subsection("Time stepping");
      parameters.enter_subsection("Reconstructed fault time step");
      REQUIRE_THROWS(parameters.set("Maximum logarithmic state change", value));
    }

  // The numeric pattern admits its lower endpoint; the plugin rejects zero.
  dealii::ParameterHandler parameters;
  Limiter::declare_parameters(parameters);
  parameters.enter_subsection("Time stepping");
  parameters.enter_subsection("Reconstructed fault time step");
  parameters.set("Maximum logarithmic state change", "0");
  parameters.leave_subsection();
  parameters.leave_subsection();
  Limiter limiter;
  REQUIRE_THROWS_WITH(limiter.parse_parameters(parameters),
                      Catch::Contains("Maximum logarithmic state change must be strictly positive."));
}



TEST_CASE("PhaseFieldFault declares dynamic fault pressure by default")
{
  dealii::ParameterHandler parameters;
  aspect::MaterialModel::PhaseFieldFault<2>::declare_parameters(parameters);
  parameters.enter_subsection("Material model");
  parameters.enter_subsection("Phase field fault");
  REQUIRE_FALSE(parameters.get_bool("Use adiabatic pressure in fault friction"));
}

TEST_CASE("Objective DG residual finite rotation keeps tiny-spin memory increments", "[phase_field_fault_maxwell][objective_stress]")
{
  PlaneStress b;
  b[0][0]=8e17;b[1][1]=-2e17;b[0][1]=3e17;
  const double angle=1e-12;
  const auto increment=aspect::ReconstructedFaultUtilities::rotate_plane_strain_stress_increment(
    b,rigid_gradient(angle),1.);
  // Independent small-angle component expansion; omitted cubic term is <1e-17Pa.
  PlaneStress reference;
  reference[0][0]=-2*angle*b[0][1]-angle*angle*(b[0][0]-b[1][1]);
  reference[1][1]=-reference[0][0];
  reference[0][1]=angle*(b[0][0]-b[1][1])-2*angle*angle*b[0][1];
  CHECK(std::sqrt(plane_norm_square(increment-reference))<1e-8);
  CHECK(plane_norm_square(aspect::ReconstructedFaultUtilities::rotate_plane_strain_stress_increment(
    b,rigid_gradient(0.),1.))==0.);
  CHECK(increment[0][0]+increment[1][1]==0.);
  const double finite_angle=.73;
  const auto finite=aspect::ReconstructedFaultUtilities::rotate_plane_strain_stress_increment(b,rigid_gradient(finite_angle),1.);
  CHECK(std::sqrt(plane_norm_square(finite-(rotation_reference(b,finite_angle)-b)))<1024.);
}

TEST_CASE("Bulk spin uses the signed slip dyad and ratio of interpolants", "[bulk_spin]")
{
  using namespace aspect::ReconstructedFaultUtilities;
  for (double angle : {0.,.37,1.9}) for (double sense : {-1.,1.})
    {
      MaxwellAccess::MatureLocalizationSample sample;
      sample.active=true;sample.segment=0;sample.xi=.31;sample.h=2.7;
      sample.tangent[0]=std::cos(angle);sample.tangent[1]=std::sin(angle);
      sample.normal[0]=-std::sin(angle);sample.normal[1]=std::cos(angle);sample.shear_sense=sense;
      const std::vector<double> rates={.13,.71},I={.2,1.3};
      sample.normalization=(1-sample.xi)*I[0]+sample.xi*I[1];
      const double q=sample.h*((1-sample.xi)*rates[0]+sample.xi*rates[1])/sample.normalization;
      const auto J=sample.slip_gradient(rates);
      for(unsigned i=0;i<2;++i)for(unsigned j=0;j<2;++j)
        CHECK(J[i][j]==Approx(q*sense*sample.tangent[i]*sample.normal[j]).margin(2e-15));
      CHECK((2*symmetrize(J)-sample.twice_symmetric(rates)).norm()<3e-15);
      CHECK(std::abs(q-sample.h*((1-sample.xi)*rates[0]/I[0]+sample.xi*rates[1]/I[1]))>.01);
      CHECK(std::abs(trace(J))<1e-15);
      CHECK(plane_strain_angular_rate(J)==Approx(-sense*q/2).margin(1e-15));
      sample.active=false;
      CHECK(sample.slip_gradient({}).norm()==0.);
      CHECK(sample.twice_symmetric({}).norm()==0.);
    }
}

TEST_CASE("Bulk-corotational pure slip removes width-dependent stress rotation", "[bulk_spin]")
{
  using namespace aspect::ReconstructedFaultUtilities;
  PlaneStress initial;initial[0][0]=7.;initial[1][1]=-2.;initial[0][1]=1.3;
  const double end=.4,slip=.3,G=2.,eta=5.;
  for(double width : {.05,.1,.4}) for(double sense : {-1.,1.}) for(double orientation : {0.,.71})
    {
      dealii::Tensor<1,2> t,n;t[0]=std::cos(orientation);t[1]=std::sin(orientation);n[0]=-t[1];n[1]=t[0];
      const auto J=(sense*slip/(width*end))*outer_product(t,n);
      for(double genuine_spin : {0.,.63})
        {
          const auto L=J+rigid_gradient(genuine_spin);
          const auto bulk=L-J;
          CHECK(symmetrize(bulk).norm()<1e-14);
          const auto actual=std::exp(-G*end/eta)*rotate_plane_strain_stress_with_spin(initial,plane_strain_angular_rate(bulk),end);
          const auto reference=std::exp(-G*end/eta)*rotation_reference(initial,genuine_spin*end);
          CHECK(std::sqrt(plane_norm_square(actual-reference))<1e-13);
          CHECK(std::abs(trace(actual)-std::exp(-G*end/eta)*trace(initial))<1e-14);
          const auto W=rigid_gradient(plane_strain_angular_rate(bulk));
          CHECK(std::abs(scalar_product(initial,symmetrize(W*dealii::Tensor<2,2>(initial)-dealii::Tensor<2,2>(initial)*W)))<1e-13);
        }
      const auto legacy=rotate_plane_strain_stress(initial,J,end);
      CHECK(std::sqrt(plane_norm_square(legacy-rotation_reference(initial,-sense*slip/(2*width))))<1e-13);
      CHECK(rotate_plane_strain_stress_with_spin(initial,plane_strain_angular_rate(J-J),end)==initial);
      // Zero total spin with nonzero fault spin must rotate the bulk tensor.
      CHECK(std::sqrt(plane_norm_square(rotate_plane_strain_stress_with_spin(initial,plane_strain_angular_rate(-J),end)
                                       -rotation_reference(initial,sense*slip/(2*width))))<1e-13);
    }
  for(double omega : {0.,.3,-1.7})
    CHECK((rotate_plane_strain_stress_with_spin(initial,omega,.1)-rotate_plane_strain_stress(initial,rigid_gradient(omega),.1)).norm()<1e-14);
}

TEST_CASE("Bulk spin retains regular curved varying-slip gradients and observer spin", "[bulk_spin]")
{
  using namespace aspect::ReconstructedFaultUtilities;
  const double width=.23;
  auto velocity=[&](dealii::Point<2> x)
  {
    double r=x.norm(),V=.4+.1*x[0]-.07*x[1],H=.5*(1+std::tanh((r-1)/width));
    dealii::Tensor<1,2> v;v[0]=-H*V*x[1]/r;v[1]=H*V*x[0]/r;return v;
  };
  for(auto x : {dealii::Point<2>(.8,.7),dealii::Point<2>(-.6,.9)})
    {
      const double r=x.norm(),V=.4+.1*x[0]-.07*x[1],H=.5*(1+std::tanh((r-1)/width));
      const double chi=.5/width/std::pow(std::cosh((r-1)/width),2);
      dealii::Tensor<1,2> n,t;n[0]=x[0]/r;n[1]=x[1]/r;t[0]=-n[1];t[1]=n[0];
      PlaneGradient grad_j;
      for(unsigned j=0;j<2;++j)
        {
          double dv=j==0 ? .1 : -.07;
          grad_j[0][j]=dv*t[0]+V*(-(j==1 ? 1./r : 0.)+x[1]*x[j]/(r*r*r));
          grad_j[1][j]=dv*t[1]+V*((j==0 ? 1./r : 0.)-x[0]*x[j]/(r*r*r));
        }
      PlaneGradient L;
      for(unsigned j=0;j<2;++j){auto plus=x,minus=x;plus[j]+=1e-5;minus[j]-=1e-5;const auto d=(velocity(plus)-velocity(minus))/(2e-5);for(unsigned i=0;i<2;++i)L[i][j]=d[i];}
      const auto J=chi*V*outer_product(t,n);
      CHECK((L-J-H*grad_j).norm()<2e-9);
      CHECK(std::abs(plane_strain_angular_rate(H*grad_j))>.01);
      PlaneGradient Q;const double angle=.67;Q[0][0]=Q[1][1]=std::cos(angle);Q[1][0]=std::sin(angle);Q[0][1]=-Q[1][0];
      const auto omega=rigid_gradient(.83),star_L=Q*L*transpose(Q)+omega,star_J=Q*J*transpose(Q);
      CHECK(std::abs(plane_strain_angular_rate(star_L-star_J)-plane_strain_angular_rate(L-J)-.83)<1e-14);
    }
}

TEST_CASE("Accepted bulk-spin Lie scheme converges under time-dependent observer rotation", "[bulk_spin]")
{
  using namespace aspect::ReconstructedFaultUtilities;
  PlaneStress initial;initial[0][0]=3.;initial[1][1]=-1.;initial[0][1]=.4;
  double previous=0.;
  for(unsigned steps : {10u,20u,40u,80u})
    {
      const double end=.8,dt=end/steps;auto stress=initial;
      for(unsigned k=0;k<steps;++k)
        {
          const double time=k*dt,angle=.3*std::sin(time)+.2*time*time,observer=.3*std::cos(time)+.4*time;
          PlaneGradient Q;Q[0][0]=Q[1][1]=std::cos(angle);Q[1][0]=std::sin(angle);Q[0][1]=-Q[1][0];
          PlaneGradient J;J[0][1]=4.;const auto L=rigid_gradient(.4)+J;
          const auto starL=Q*L*transpose(Q)+rigid_gradient(observer),starJ=Q*J*transpose(Q);
          stress=std::exp(-.7*dt)*rotate_plane_strain_stress_with_spin(stress,plane_strain_angular_rate(starL-starJ),dt);
        }
      const auto reference=std::exp(-.7*end)*rotation_reference(initial,.4*end+.3*std::sin(end)+.2*end*end);
      const double error=std::sqrt(plane_norm_square(stress-reference));
      if(previous>0.)CHECK(error/previous<.53);
      CHECK(error>0.);previous=error;
      std::cout<<"Bulk spin observer Lie: steps="<<steps<<", error="<<error<<'\n';
    }
}

TEST_CASE("Resolved Q2 slip gradients converge to smooth jump spin", "[bulk_spin]")
{
  // Interpolate v(y)=V/2*tanh(y/w) with Q2 nodal values; differentiate
  // that interpolant, independently of J01=V/(2w)*sech(y/w)^2.
  for(const double width:{.1,.2,.4})
    {
      const double V=.3;
      double previous=1e30;
      for(const unsigned per_width:{4u,8u,16u,32u})
        {
          const double h=width/per_width;
          double error2=0.,maximum=0.;
          const dealii::QGauss<1> quadrature(8);
          for(int cell=-int(4*per_width);cell<int(4*per_width);++cell)
            {
              const double y0=cell*h;
              const double v0=.5*V*std::tanh(y0/width);
              const double vm=.5*V*std::tanh((y0+.5*h)/width);
              const double v1=.5*V*std::tanh((y0+h)/width);
              for(unsigned q=0;q<quadrature.size();++q)
                {
                  const double u=quadrature.point(q)[0],y=y0+u*h;
                  const double derivative=((4*u-3)*v0+(4-8*u)*vm+(4*u-1)*v1)/h;
                  const double exact=.5*V/width/std::pow(std::cosh(y/width),2);
                  const double omega=.5*(exact-derivative);
                  error2+=h*quadrature.weight(q)*omega*omega;
                  maximum=std::max(maximum,std::abs(omega));
                }
            }
          const double error=std::sqrt(error2);
          REQUIRE(error<previous*.27);
          std::cout<<"Bulk spin Q2 profile: width="<<width<<", h/width="<<1./per_width
                   <<", residual_L2="<<error<<", residual_Linf="<<maximum<<'\n';
          previous=error;
        }
    }
}
