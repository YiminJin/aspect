/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/particle/manager.h>
#include <aspect/phase_field.h>
#include <aspect/plugins.h>
#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/time_stepping/interface.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/fe/fe_values.h>
#include <cstring>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace TimeStepping
  {
    template<int dim> class ObjectiveRejection : public Interface<dim>, public SimulatorAccess<dim>
    {
      bool rejected=false;
    public:
      double execute() override {return std::numeric_limits<double>::max();}
      std::pair<Reaction,double> determine_reaction(const TimeStepInfo &) override
      {
        if (this->get_timestep_number()==2 && !rejected)
          {rejected=true;return {Reaction::repeat_step,.5*this->get_timestep()};}
        return {Reaction::advance,std::numeric_limits<double>::max()};
      }
    };
    ASPECT_REGISTER_TIME_STEPPING_MODEL(ObjectiveRejection,"objective rejection",
      "Bounded verification: reject one completed trial before publication at step two.")
  }
  namespace Postprocess
  {
    template<int dim> class ObjectiveChecks : public Interface<dim>, public SimulatorAccess<dim>
    {
      using Model=MaterialModel::PhaseFieldFault<dim>;
      std::map<types::particle_index,std::vector<double>> incoming;
      std::vector<std::vector<double>> surface;
      std::vector<Tensor<2,dim>> old_gradients;
      unsigned int captured=numbers::invalid_unsigned_int,publications=0;
      unsigned int retries=0;

      void measure_branch_energy(const Model &model) const
      {
        if (this->get_timestep_number()==0) return;
        const auto &intro=this->introspection();
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto rule=fault_mechanical_quadrature(*this);
        FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,
                         update_values|update_gradients|update_quadrature_points|update_JxW_values);
        const unsigned int nq=rule.size();
        std::vector<Tensor<2,dim>> current(nq),lagged(nq);
        std::vector<double> phi(nq),pressure(nq);
        std::array<std::vector<double>,3> old;
        for (auto &c:old) c.resize(nq);
        const unsigned int fields[]={intro.compositional_index_for_name("tau_xx"),
          intro.compositional_index_for_name("tau_yy"),intro.compositional_index_for_name("tau_xy")};
        const double dt=this->get_timestep(),g=32038120320.,eta=1e26;
        const auto dot=[](const SymmetricTensor<2,dim> &a,const SymmetricTensor<2,dim> &b)
        {return a*b+trace(a)*trace(b);};
        long double energy_old=0.,energy_rotated=0.,energy_new=0.,work=0.,viscous=0.,endpoint=0.,slip_work=0.;
        const QGauss<1> time_rule(6);
        for (const auto &cell:this->get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe.reinit(cell);
              fe[intro.extractors.velocities].get_function_gradients(this->get_solution(),current);
              fe[intro.extractors.velocities].get_function_gradients(this->get_old_solution(),lagged);
              fe[intro.extractors.pressure].get_function_values(this->get_solution(),pressure);
              fe[FEValuesExtractors::Scalar(intro.phase_field().first_component_index)]
                .get_function_values(this->get_solution(),phi);
              for (unsigned int c=0;c<3;++c)
                fe[intro.extractors.compositional_fields[fields[c]]]
                  .get_function_values(this->get_current_linearization_point(),old[c]);
              const auto &associations=manager.get_stokes_qp_fault_associations(cell->id(),rule,fe.get_quadrature_points());
              for (unsigned int q=0;q<nq;++q)
                {
                  SymmetricTensor<2,dim> before;
                  before[0][0]=old[0][q];before[1][1]=old[1][q];before[0][1]=old[2][q];
                  const double angle=model.uses_jaumann_particle_stress() ? .5*dt*(lagged[q][1][0]-lagged[q][0][1]):0.;
                  const double mean=.5*(before[0][0]+before[1][1]),difference=.5*(before[0][0]-before[1][1]);
                  SymmetricTensor<2,dim> rotated,slip_rate;
                  rotated[0][0]=mean+difference*std::cos(2*angle)-before[0][1]*std::sin(2*angle);
                  rotated[1][1]=2*mean-rotated[0][0];
                  rotated[0][1]=difference*std::sin(2*angle)+before[0][1]*std::cos(2*angle);
                  auto total=symmetrize(current[q]);
                  const double tr=trace(total)/3.;total[0][0]-=tr;total[1][1]-=tr;
                  const auto &a=associations[q];
                  if (a.active)
                    {
                      typename Model::ReconstructedFaultBulkPointInputs input;
                      input.fault_index=a.fault_index;input.segment_index=a.segment_index;input.xi=a.xi;
                      input.phase_field=input.previous_phase_field=phi[q];input.temperature=293.;input.bulk_material_fractions={1.,0.};
                      const auto response=model.evaluate_reconstructed_fault_bulk_point(input);
                      slip_rate=response.localization_factor*manager.interpolate_slip_rate(a.fault_index,a.segment_index,a.xi)
                        *manager.get_shear_sense(a.fault_index)*symmetrize(outer_product(a.tangent,a.normal));
                    }
                  const auto rate=total-slip_rate;
                  const auto after=std::exp(-dt*g/eta)*rotated-2*eta*std::expm1(-dt*g/eta)*rate;
                  const double weight=fe.JxW(q);
                  energy_old+=weight*dot(before,before)/(4*g);
                  energy_rotated+=weight*dot(rotated,rotated)/(4*g);
                  energy_new+=weight*dot(after,after)/(4*g);
                  endpoint+=weight*dt*(dot(after,total)-pressure[q]*trace(current[q]));
                  slip_work+=weight*dt*dot(after,slip_rate);
                  for (unsigned int k=0;k<time_rule.size();++k)
                    {
                      const double s=dt*time_rule.point(k)[0];
                      const auto at=std::exp(-s*g/eta)*rotated-2*eta*std::expm1(-s*g/eta)*rate;
                      work+=weight*dt*time_rule.weight(k)*dot(at,rate);
                      viscous+=weight*dt*time_rule.weight(k)*dot(at,at)/(2*eta);
                    }
                }
            }
        std::array<double,7> values={double(energy_old),double(energy_rotated),double(energy_new),double(work),double(viscous),double(endpoint),double(slip_work)};
        for (auto &v:values) v=Utilities::MPI::sum(v,this->get_mpi_communicator());
        const double closure=values[3]-values[4]-(values[2]-values[0]);
        const double scale=std::max(1.,values[0]+values[2]+std::abs(values[3])+values[4]);
        const double bound=512*std::numeric_limits<double>::epsilon()*scale;
        AssertThrow(std::abs(closure)<bound,ExcMessage("Independent FE branch substep ledger failed."));
        if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
          {
            std::ofstream out(this->get_output_directory()+"objective-energy-"+std::to_string(this->get_timestep_number())+".csv");
            out<<std::setprecision(17)<<"old_J_m,rotated_J_m,new_J_m,exact_source_work_J_m,viscous_J_m,endpoint_total_work_J_m,endpoint_slip_work_J_m,closure_J_m,bound_J_m\n";
            for (unsigned int i=0;i<values.size();++i) {if (i) out<<',';out<<values[i];}
            out<<','<<closure<<','<<bound<<'\n';
          }
      }

      void capture(bool temperature)
      {
        if (!temperature) return;
        const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto &fault=manager.get_fault(0);
        const unsigned int step=this->get_timestep_number();
        if (captured==step)
          {
            bool unchanged=publications==0;
            for (const auto &particle:pm.get_particle_handler())
              {
                const auto p=particle.get_properties();const auto &old=incoming.at(particle.get_id());
                unchanged &= p.size()==old.size() && std::memcmp(p.data(),old.data(),p.size()*sizeof(double))==0;
              }
            for (unsigned int j=0;j<fault.n_vertices();++j)
              {
                const auto p=fault.get_properties(j);
                unchanged &= p.size()==surface[j].size() && std::memcmp(p.data(),surface[j].data(),p.size()*sizeof(double))==0;
              }
            AssertThrow(Utilities::MPI::min(static_cast<unsigned int>(unchanged),this->get_mpi_communicator()),
                        ExcMessage("Rejected objective trial changed accepted history or published."));
            ++retries;
          }
        captured=step;publications=0;incoming.clear();surface.clear();
        std::vector<Point<dim>> points;
        for (const auto &particle:pm.get_particle_handler())
          {
            const auto p=particle.get_properties();
            incoming[particle.get_id()]={p.begin(),p.end()};points.push_back(particle.get_location());
          }
        for (unsigned int j=0;j<fault.n_vertices();++j)
          {const auto p=fault.get_properties(j);surface.emplace_back(p.begin(),p.end());}
        Utilities::MPI::RemotePointEvaluation<dim> cache;
        cache.reinit(this->get_phase_field_handler().get_grid_cache(),points);
        old_gradients=VectorTools::point_gradients<dim>(cache,this->get_dof_handler(),this->get_old_solution(),
          VectorTools::EvaluationFlags::avg,this->introspection().component_indices.velocities[0]);
      }
    public:
      void initialize() override
      {
        this->get_signals().post_advection_solver.connect(
          [this](const auto &,bool temperature,unsigned int,const auto &){capture(temperature);});
        this->get_signals().post_reconstructed_fault_solver.connect(
          [this](unsigned int,unsigned int,double,const auto &){++publications;});
      }

      std::pair<std::string,std::string> execute(TableHandler &) override
      {
        const auto &model=Plugins::get_plugin_as_type<const Model>(this->get_material_model());
        const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
        const auto &info=pm.get_property_manager().get_data_info();
        const auto stress_position=info.get_position_by_field_name("maxwell stress");
        const auto &manager=this->get_reconstructed_fault_manager();
        const auto &fault=manager.get_fault(0);
        const auto &intro=this->introspection();
        const auto comm=this->get_mpi_communicator();
        const auto step=this->get_timestep_number();
        AssertThrow(publications==1,ExcMessage("Objective fixture requires one physical publication per accepted step."));
        const std::string prefix=this->get_output_directory()+"objective-"+std::to_string(step)
          +"-rank"+std::to_string(Utilities::MPI::this_mpi_process(comm));
        std::ofstream out(prefix+"-particles.csv");out<<std::setprecision(17)<<"id,x,y,xx,yy,xy,zz\n";
        double error=0.,relative=0.,angle_max=0.,energy=0.;
        std::vector<Point<dim>> points;
        for (const auto &p:pm.get_particle_handler()) points.push_back(p.get_location());
        Utilities::MPI::RemotePointEvaluation<dim> cache;
        cache.reinit(this->get_phase_field_handler().get_grid_cache(),points);
        const auto gradient=VectorTools::point_gradients<dim>(cache,this->get_dof_handler(),this->get_solution(),
          VectorTools::EvaluationFlags::avg,intro.component_indices.velocities[0]);
        const auto phi=VectorTools::point_values<1>(cache,this->get_dof_handler(),this->get_solution(),
          VectorTools::EvaluationFlags::avg,intro.phase_field().first_component_index);
        unsigned int k=0;
        for (const auto &p:pm.get_particle_handler())
          {
            const auto values=p.get_properties();
            SymmetricTensor<2,dim> actual;
            actual[0][0]=values[stress_position];actual[1][1]=values[stress_position+1];actual[0][1]=values[stress_position+2];
            out<<p.get_id()<<','<<p.get_location()[0]<<','<<p.get_location()[1]<<','<<actual[0][0]<<','<<actual[1][1]
               <<','<<actual[0][1]<<','<<-trace(actual)<<'\n';
            energy+=actual*actual+trace(actual)*trace(actual);
            if (step>0)
              {
                const auto &old=incoming.at(p.get_id());
                const double angle=model.uses_jaumann_particle_stress()
                  ? .5*this->get_timestep()*(old_gradients[k][1][0]-old_gradients[k][0][1]):0.;
                angle_max=std::max(angle_max,std::abs(angle));
                const double mean=.5*(old[stress_position]+old[stress_position+1]);
                const double difference=.5*(old[stress_position]-old[stress_position+1]);
                SymmetricTensor<2,dim> rotated;
                rotated[0][0]=mean+difference*std::cos(2*angle)-old[stress_position+2]*std::sin(2*angle);
                rotated[1][1]=2*mean-rotated[0][0];
                rotated[0][1]=difference*std::sin(2*angle)+old[stress_position+2]*std::cos(2*angle);
                auto strain=symmetrize(gradient[k]);
                const double tr=trace(strain)/3.;
                strain[0][0]-=tr;strain[1][1]-=tr;
                const auto a=manager.project_to_bulk_source(p.get_location());
                if (a.active)
                  {
                    auto tangent=fault.vertex(a.segment_index+1)-fault.vertex(a.segment_index);tangent/=tangent.norm();
                    Tensor<1,dim> normal;normal[0]=-tangent[1];normal[1]=tangent[0];
                    typename Model::ReconstructedFaultBulkPointInputs input;
                    input.fault_index=a.fault_index;input.segment_index=a.segment_index;input.xi=a.xi;
                    input.phase_field=input.previous_phase_field=phi[k];input.temperature=293.;
                    input.bulk_material_fractions={1.,0.};
                    const auto response=model.evaluate_reconstructed_fault_bulk_point(input);
                    strain-=response.localization_factor*manager.interpolate_slip_rate(a.fault_index,a.segment_index,a.xi)
                      *manager.get_shear_sense(a.fault_index)*symmetrize(outer_product(tangent,normal));
                  }
                // Fixed fixture's explicit G and eta. Independent component
                // rotation and endpoint source, including zz in every norm.
                const double dt=this->get_timestep(),g=32038120320.,eta=1e26;
                const auto expected=std::exp(-dt*g/eta)*rotated-2*eta*std::expm1(-dt*g/eta)*strain;
                const auto difference_tensor=actual-expected;
                const double e=std::sqrt(difference_tensor*difference_tensor+trace(difference_tensor)*trace(difference_tensor));
                const double scale=std::sqrt(expected*expected+trace(expected)*trace(expected));
                error=std::max(error,e);relative=std::max(relative,e/std::max(1000.,scale));
              }
            ++k;
          }
        error=Utilities::MPI::max(error,comm);relative=Utilities::MPI::max(relative,comm);
        angle_max=Utilities::MPI::max(angle_max,comm);energy=Utilities::MPI::sum(energy,comm);
        AssertThrow(relative<2e-12,ExcMessage("Independent objective particle recurrence failed."));
        AssertThrow(out,ExcMessage("Cannot write objective particle evidence."));
        measure_branch_energy(model);
        if (Utilities::MPI::this_mpi_process(comm)==0)
          {
            std::ofstream summary(this->get_output_directory()+"objective-summary.csv",step?std::ios::app:std::ios::out);
            if (!step) summary<<"step,time_s,dt_s,error_Pa,relative_error,max_angle_rad,tensor_square_sum_Pa2,publications,retries\n";
            summary<<std::setprecision(17)<<step<<','<<this->get_time()<<','<<this->get_timestep()<<','<<error<<','<<relative
                   <<','<<angle_max<<','<<energy<<','<<publications<<','<<retries<<'\n';
            std::ofstream surface_out(this->get_output_directory()+"objective-"+std::to_string(step)+"-surface.csv");
            surface_out<<std::setprecision(17)<<"node,V,theta\n";
            const auto position=manager.get_property_information()[manager.get_property_index("phase field fault state")].position;
            const auto &rate=manager.get_timestep_committed_slip_rate(0);
            for (unsigned int j=0;j<rate.size();++j)
              surface_out<<j<<','<<rate[j]<<','<<fault.get_properties(j)[position]<<'\n';
          }
        return {"Objective stress native checks", "independent recurrence and single publication passed"};
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR(ObjectiveChecks,"objective stress checks",
      "Bounded independent fixed-fixture recurrence, accepted-state rollback and publication checks.")
  }
}
