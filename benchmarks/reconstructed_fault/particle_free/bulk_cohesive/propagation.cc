/* Bounded fracture-scale specimen: reuse qualified MPI/lifecycle diagnostics. */
#include "parallel.cc"

namespace aspect { namespace Postprocess
{
  template<int dim> class BulkPropagationChecks:public MomentCycle<dim>
  {
    double G=0.,Gc=0.,strength_drop=0.;
    std::array<double,3> seed_parameters()const override
    {
      const double ell=this->get_phase_field_handler().get_length_scale();
      const double m=2.*G*Gc/((8./3.)*ell*strength_drop*strength_drop);
      const PhaseField::GeometricFunction geometric(ell,1.,8./3.);
      const PhaseField::DegradationFunction degradation(1.,m);
      const PhaseField::PhaseFieldProfile profile(geometric,degradation,.6);
      if(Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
        {
          std::ofstream out(this->get_output_directory()+"seed-prescription.csv");
          out<<std::setprecision(17)<<"radius_m,phi\n";
          const auto &r=profile.get_coordinate_values(),&p=profile.get_phase_field_values();
          for(unsigned int i=0;i<r.size();++i)out<<r[i]<<','<<p[i]<<'\n';
          std::ofstream parameters(this->get_output_directory()+"fracture-parameters.csv");
          parameters<<std::setprecision(17)<<"G_Pa,Gc_J_m2,strength_drop_Pa,ell_m,m,seed_peak,tip_x_m,tip_y_m\n"
                    <<G<<','<<Gc<<','<<strength_drop<<','<<ell<<','<<m<<",0.6,1.0,0.5\n";
        }
      return {{ell,m,1.}};
    }
    public:
      void initialize()override
      {
        MomentCycle<dim>::initialize();
        // The checkpoint clock describes the saved pending interval. Reduce
        // that interval through the native restart signal, preserving the
        // accepted history time and old_time_step.
        this->get_signals().post_resume_time_step.connect(
          [this](const SimulatorAccess<dim>&,double &dt)
          {dt=std::min(dt,this->get_parameters().maximum_time_step);});
      }
      void parse_parameters(ParameterHandler &prm)override
      {
        prm.enter_subsection("Material model");prm.enter_subsection("Phase field fault");
        const auto first=[&](const std::string &name)
          {return Utilities::string_to_double(Utilities::split_string_list(prm.get(name)).front());};
        G=first("Elastic shear moduli");Gc=first("Critical energy release rates");strength_drop=first("Cohesions");
        prm.leave_subsection();prm.leave_subsection();
      }
  };
  ASPECT_REGISTER_POSTPROCESSOR(BulkPropagationChecks,"bulk propagation checks",
    "Fracture-scale specimen with a fresh continuum seed matched to configured material parameters; qualified bulk MPI diagnostics.")
}}
