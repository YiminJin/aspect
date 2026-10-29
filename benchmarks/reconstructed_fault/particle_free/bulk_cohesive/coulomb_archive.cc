/* Read-only accepted-checkpoint pressure observer; never advances a specimen. */
#include "p5c_campaign.cc"
#include <deal.II/fe/fe_values.h>
namespace aspect { namespace Postprocess
{
  template<int dim> class CoulombArchivePressure:public Interface<dim>,public SimulatorAccess<dim>
  {
    public:
      void initialize()override
      {
        this->get_signals().post_resume_time_step.connect([this](const SimulatorAccess<dim>&,double &)
          {
            const auto rule=Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(this->get_material_model()).mechanical_quadrature();
            FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_JxW_values);
            const unsigned int rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
            std::ofstream out(this->get_output_directory()+"pressure-rank-"+std::to_string(rank)+".csv");
            out<<std::setprecision(17)<<"cell,x_m,y_m,area_m2,p_mech_mean_Pa,p_mech_min_Pa,p_mech_max_Pa\n";
            std::vector<double> p(rule.size());
            for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned()
                && cell->center()[0]>.85 && cell->center()[0]<1.3 && cell->center()[1]>.35 && cell->center()[1]<.65)
              {
                fe.reinit(cell);fe[this->introspection().extractors.pressure].get_function_values(this->get_solution(),p);
                double integral=0.,area=0.;for(unsigned int q=0;q<p.size();++q){integral+=fe.JxW(q)*p[q];area+=fe.JxW(q);}
                out<<cell->id()<<','<<cell->center()[0]<<','<<cell->center()[1]<<','<<area<<','<<integral/area<<','<<*std::min_element(p.begin(),p.end())<<','<<*std::max_element(p.begin(),p.end())<<'\n';
              }
          });
      }
      std::pair<std::string,std::string> execute(TableHandler&)override
      {AssertThrow(false,ExcMessage("Coulomb archive observer must terminate on the restored endpoint, before a physical step."));return {};}
  };
  ASPECT_REGISTER_POSTPROCESSOR(CoulombArchivePressure,"coulomb archive pressure","Read native physical pressure after restore and before termination; no advancement or publication.")
}}
