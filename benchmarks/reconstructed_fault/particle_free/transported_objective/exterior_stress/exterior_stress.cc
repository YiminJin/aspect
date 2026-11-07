/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later.
 * Constant-coefficient homogeneous shear background for this benchmark only. */
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <complex>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace ShearExterior
  {
    constexpr double G=10., eta=10., shear=.5, initial_interval=.1;
    std::string identity(const double dt)
    {
      std::ostringstream s;
      s<<std::setprecision(17)<<"homogeneous-shear-Maxwell-v1:G=10:eta=10:shear=.5:init=.1:p=0:dt="<<dt;
      return s.str();
    }
    // z=(tau_xx-tau_yy)/2+i*tau_xy. Zero stored initial history;
    // the step-zero diagnostic solve does not publish its .1-s reaction.
    std::complex<double> background(const unsigned n,const double dt)
    {
      if(n==0)return {};
      const auto r=std::exp(std::complex<double>(-G/eta*dt,-shear*dt));
      return std::complex<double>(0.,eta*shear*(-std::expm1(-G/eta*dt)))*(1.-std::pow(r,n))/(1.-r);
    }
    template<int dim> SymmetricTensor<2,dim> tensor(const std::complex<double> z)
    {
      SymmetricTensor<2,dim> a;a[0][0]=z.real();a[1][1]=-z.real();a[0][1]=z.imag();return a;
    }
    void validate(ParameterHandler &prm)
    {
      AssertThrow(prm.get("Pressure normalization")=="no" && prm.get_double("Surface pressure")==0.,
                  ExcMessage("Shear exterior benchmark requires the natural p=0 traction reference."));
      prm.enter_subsection("Material model");prm.enter_subsection("Phase field fault");
      AssertThrow(prm.get_double("Reference viscosities")==eta && prm.get_double("Elastic shear moduli")==G
                  && prm.get_double("Initial time step")==initial_interval,
                  ExcMessage("Shear exterior benchmark requires G=eta=10 and initialization interval .1."));
      prm.enter_subsection("Material viscosity multiplier");
      AssertThrow(prm.get("Function expression")=="1",ExcMessage("Homogeneous shear exterior requires constant viscosity."));
      prm.leave_subsection();prm.leave_subsection();prm.leave_subsection();
      prm.enter_subsection("Boundary velocity model");prm.enter_subsection("Function");
      AssertThrow(prm.get("Function expression")=="0.4+(0.5)*y; 0",ExcMessage("Shear exterior benchmark requires its retained affine boundary velocity."));
      prm.leave_subsection();prm.leave_subsection();
    }
  }
  namespace BoundaryTraction
  {
    template<int dim> class ExteriorShearTraction : public Interface<dim>,public SimulatorAccess<dim>
    {
    public:
      void parse_parameters(ParameterHandler &prm) override {ShearExterior::validate(prm);}
      Tensor<1,dim> boundary_traction(const types::boundary_id,const Point<dim> &,const Tensor<1,dim> &) const override
      {
        const auto n=this->get_timestep_number();
        // Same formula and initialization as the retained corner-shear plugin.
        const auto z=n ? ShearExterior::background(n,this->get_time()/n)
                     : std::complex<double>(0.,ShearExterior::eta*ShearExterior::shear*(-std::expm1(-ShearExterior::initial_interval)));
        Tensor<1,dim> t;t[0]=z.imag();t[1]=-z.real();return t;
      }
    };
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(ExteriorShearTraction,"prescribed shear background",
      "Compatible discrete homogeneous Maxwell/Jaumann background top traction, p=0; fixed benchmark inputs.")
  }
  namespace Postprocess
  {
    template<int dim> class ShearExteriorProvider : public Interface<dim>,public SimulatorAccess<dim>
    {
      std::array<unsigned long long,4> calls{};
      double nominal_dt=0.;
    public:
      void parse_parameters(ParameterHandler &prm) override {ShearExterior::validate(prm);}
      void initialize() override
      {
        nominal_dt=this->get_parameters().maximum_time_step;
        auto &model=const_cast<MaterialModel::PhaseFieldFault<dim>&>(
          Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()));
        model.set_material_exterior_stress(ShearExterior::identity(nominal_dt),
          [this](const Point<dim> &x,const double parent_time,const double stage_time,const double dt)
          {
            const auto n=this->get_timestep_number();
            AssertThrow(n>0 && std::abs(parent_time-(n-1)*nominal_dt)<1e-12
                        && std::abs(dt-nominal_dt)<1e-12,
                        ExcMessage("Homogeneous benchmark provider supports its declared uniform accepted schedule only."));
            AssertThrow(stage_time>=parent_time-1e-12 && stage_time<=parent_time+dt+1e-12,
                        ExcMessage("Exterior callback received an RK time outside the attempted interval."));
            for(unsigned axis=0;axis<2;++axis)for(unsigned side=0;side<2;++side)
              if(std::abs(x[axis]-(side ? 2. : -2.))<1e-12)++calls[2*axis+side];
            const auto start=ShearExterior::background(n-1,nominal_dt);
            // Spatially homogeneous: advection alone leaves this value fixed
            // at every SSPRK stage. Native rotation/reaction happens once later.
            const auto value=ShearExterior::tensor<dim>(start);
            return value;
          });
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto comm=this->get_mpi_communicator();
        std::array<unsigned long long,4> totals;
        for(unsigned i=0;i<4;++i)totals[i]=Utilities::MPI::sum(calls[i],comm);
        const auto n=this->get_timestep_number();
        const auto start=ShearExterior::background(n ? n-1 : 0,nominal_dt);
        const auto endpoint=ShearExterior::background(n,nominal_dt);
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {
            const std::string path=this->get_output_directory()+"exterior-stress.csv";
            std::ifstream previous(path);
            const bool header=n==0 || previous.peek()==std::ifstream::traits_type::eof();
            std::ofstream out(path,n==0 ? std::ios::out : std::ios::app);
            if(header)out<<"step,time,parent_xx,parent_xy,endpoint_xx,endpoint_xy,left_calls,right_calls,bottom_calls,top_calls\n";
            out<<std::setprecision(17)<<n<<','<<this->get_time()<<','<<start.real()<<','<<start.imag()<<','<<endpoint.real()<<','<<endpoint.imag();
            for(auto c:totals)out<<','<<c;
            out<<'\n';
          }
        calls.fill(0);
        return {"Prescribed exterior stress","transported parent; all incoming faces"};
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(ShearExteriorProvider,"homogeneous shear exterior",
      "Attach split-consistent physical exterior stress to native material advection; uniform benchmark schedule only.")
  }
}
