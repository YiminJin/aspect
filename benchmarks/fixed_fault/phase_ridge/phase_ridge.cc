#include <aspect/material_model/rheology/rate_and_state_friction.h>
/*
  This file is part of ASPECT, distributed under the GNU General Public
  License version 2 or later. See the file LICENSE for details.
*/
#include <aspect/initial_composition/interface.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator/assemblers/fault_stress.h>
#include <aspect/simulator_signals.h>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace aspect
{
  namespace InitialComposition
  {
    // The initializer alone knows the analytic line. The table is the mature
    // AT1 distance profile, supplied by run.py at the donor's 5000 abscissae.
    template <int dim>
    class StationaryRidge : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Initial composition model");
          prm.enter_subsection("Stationary ridge");
          prm.declare_entry("Angle", "60", Patterns::Double(),
                            "Angle from +x, in degrees.");
          prm.declare_entry("Center x", "0", Patterns::Double(),
                            "Horizontal translation in metres.");
          prm.declare_entry("Grid cells", "32", Patterns::Integer(1),
                            "Cells in each side of the centered square.");
          prm.declare_entry("Bilinear donor phase", "false", Patterns::Bool(),
                            "Embed the historical donor Q1 profile in DGQ2.");
          prm.declare_entry("Profile table", "", Patterns::Anything(),
                            "Distance in metres and phase.");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Initial composition model");
          prm.enter_subsection("Stationary ridge");
          const double angle=prm.get_double("Angle")*numbers::PI/180;
          normal[0]=-std::sin(angle);
          normal[1]=std::cos(angle);
          center=prm.get_double("Center x");
          cells=prm.get_integer("Grid cells");
          bilinear=prm.get_bool("Bilinear donor phase");
          std::ifstream input(prm.get("Profile table"));
          double s,d;
          while (input>>s>>d)
            {
              distance.push_back(s);
              phase.push_back(d);
            }
          AssertThrow(distance.size()==5000 && distance.front()==0. && std::is_sorted(distance.begin(),distance.end()),ExcMessage("Invalid mature profile table."));
          prm.leave_subsection();
          prm.leave_subsection();
        }
        double initial_composition(const Point<dim> &x, const unsigned int field) const override
        {
          if (field!=this->introspection().compositional_index_for_name(this->get_parameters().surface_energy_model.composition)) return 0.;
          std::array<unsigned int,2> index;
          std::array<double,2> unit;
          for (unsigned int d=0; d<2; ++d)
            {
              const double coord=std::clamp((x[d]+.5)*cells,0.,double(cells));
              index[d]=std::min(static_cast<unsigned int>(coord),cells-1);
              unit[d]=coord-index[d];
            }
          // Use analytic values only to initialize these local shape functions.
          // The regular quadratic interpolant undershoots at the compact edge.
          // Bound its Bernstein control values before writing ASPECT's DG nodes.
          double coeff[3][3];
          const unsigned int degree=bilinear ? 1 : 2;
          for (unsigned int j=0; j<=degree; ++j)
            for (unsigned int i=0; i<=degree; ++i)
              coeff[j][i]=ridge_value((index[0]+double(i)/degree)/cells-.5,(index[1]+double(j)/degree)/cells-.5);
          if (!bilinear)
            {
              for (unsigned int j=0; j<3; ++j) coeff[j][1]=2.*coeff[j][1]-.5*(coeff[j][0]+coeff[j][2]);
              for (unsigned int i=0; i<3; ++i) coeff[1][i]=2.*coeff[1][i]-.5*(coeff[0][i]+coeff[2][i]);
            }
          auto shapes=[&](const double u) -> std::array<double,3>
          {
            if (bilinear) return {{1.-u,u,0.}};
            return {{(1.-u) *(1.-u),2.*u *(1.-u),u*u}};
          };
          const auto bx=shapes(unit[0]), by=shapes(unit[1]);
          double value=0.;
          for (unsigned int j=0; j<=degree; ++j)
            for (unsigned int i=0; i<=degree; ++i)
              value+=bx[i]*by[j]*std::clamp(coeff[j][i],0.,phase.front());
          return value;
        }
      private:
        double ridge_value(const double x,const double y) const
        {
          const double s=std::abs(normal[0]*(x-center)+normal[1]*y);
          const unsigned int i=std::lower_bound(distance.begin(),distance.end(),s)-distance.begin();
          if (i==0) return phase.front();
          if (i==distance.size()) return 0.;
          const double fraction=(s-distance[i-1])/(distance[i]-distance[i-1]);
          return (1.-fraction)*phase[i-1]+fraction*phase[i];
        }
        unsigned int cells;
        bool bilinear;
        Tensor<1,dim> normal;
        double center;
        std::vector<double> distance,phase;
    };
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(StationaryRidge,"stationary ridge","Mature distance-profile initializer for the stationary ridge benchmark.")
  }
  namespace MaterialModel
  {
    template <int dim>
    class ObservedRidge : public PhaseFieldFault<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &) {}

        using PhaseFieldFault<dim>::normal_compression;
        // All ranks in RidgeProbe enter after acceptance and before the next
        // attempt replaces the relaxed parent. No collective on material reads.
        auto collect_attempt_samples(const LinearAlgebra::BlockVector &fields) const
        {
          return this->coupling->sample_bulk(fields);
        }
    };
    ASPECT_REGISTER_MATERIAL_MODEL(ObservedRidge,"observed ridge","Unmodified free RSF solve with a collective fixture probe.")

    // Test the ordinary Stokes RHS on the cell containing an interior inclined
    // knot. The initial parent stress is zero and the imposed rate is constant.
    template <int dim>
    class MeasuredStressLoad : public Assemblers::FaultStressLoad<dim>
    {
      public:
        void execute(internal::Assembly::Scratch::ScratchBase<dim> &scratch_base,
                     internal::Assembly::CopyData::CopyDataBase<dim> &data_base) const override
        {
          auto &scratch=dynamic_cast<internal::Assembly::Scratch::StokesSystem<dim>&>(scratch_base);
          auto &data=dynamic_cast<internal::Assembly::CopyData::StokesSystem<dim>&>(data_base);
          const auto &m=Plugins::get_plugin_as_type<const PhaseFieldFault<dim>>(this->get_material_model());
          const auto &profile=m.phase_profile();
          const auto &knot=m.labels()[m.labels().size()/2].position;
          const auto lo=scratch.cell->vertex(0), hi=scratch.cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
          if (this->get_timestep_number()!=0 || std::abs(profile.geometry.segment(0).tangent[0])<1.e-8 ||
              knot[0]<lo[0] || knot[1]<lo[1] || knot[0]>=hi[0] || knot[1]>=hi[1])
            {
              Assemblers::FaultStressLoad<dim>::execute(scratch_base,data_base);
              return;
            }
          const unsigned int n=data.local_dof_indices.size();
          internal::Assembly::CopyData::StokesSystem<dim> isolated(n,false);
          Assemblers::FaultStressLoad<dim>::execute(scratch_base,isolated);
          data.local_rhs+=isolated.local_rhs;
          const auto &intro=this->introspection();
          const auto field=intro.compositional_index_for_name(this->get_parameters().surface_energy_model.composition);
          // Pair the assembled Q3 vector with a Q2 velocity. Integrate the
          // unprojected physical slip independently on the section's Q8 rule.
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),QGauss<dim>(8),
                           update_values|update_quadrature_points|update_JxW_values);
          fe.reinit(scratch.cell);
          std::vector<double> phase(fe.n_quadrature_points);
          fe[intro.extractors.compositional_fields[field]].get_function_values(this->get_old_solution(),phase);
          AssertThrow(profile.geometry.cut_quadrature(lo,hi).size()>0,ExcInternalError());
          AssertThrow(*std::max_element(phase.begin(),phase.end())>*std::min_element(phase.begin(),phase.end()),ExcInternalError());
          // v=(y+xy,0.4x) belongs to Q2. Apply its nodal coefficients to
          // the assembled vector and its analytic gradient to the raw FE load.
          double actual=0.,reference=0.,scale=0.;
          for (unsigned int i=0; i<n; ++i)
            {
              const auto idx=intro.stokes_dof_info[i].local_dof_index;
              const auto component=this->get_fe().system_to_component_index(idx).first;
              if (component!=intro.component_indices.velocities[0] && component!=intro.component_indices.velocities[1]) continue;
              const auto x=this->get_mapping().transform_unit_to_real_cell(scratch.cell,this->get_fe().unit_support_point(idx));
              actual+=isolated.local_rhs(i)*(component==intro.component_indices.velocities[0] ? x[1]*(1.+x[0]) : .4*x[0]);
            }
          // This prescribed fixture uses one uniform Maxwell coefficient.
          const auto &viscosities=scratch.material_model_outputs.viscosities;
          const double kappa=viscosities.front();
          const auto has_reference_viscosity = [kappa](const double value)
          {
            return value == kappa;
          };
          AssertThrow(std::all_of(viscosities.begin(),viscosities.end(),has_reference_viscosity),
                      ExcInternalError());
          for (unsigned int q=0; q<phase.size(); ++q)
            {
              const auto x=fe.quadrature_point(q);
              const auto foot=profile.geometry.project(x);
              if (!foot.active) continue;
              const auto &f=profile.geometry.segment(foot.first_vertex);
              const auto &I=profile.normal_integrals();
              const double chi=profile.weight(phase[q])/foot.interpolate(I[foot.first_vertex],I[foot.first_vertex+1]);
              SymmetricTensor<2,dim> strain;
              strain[0][0]=x[1];
              strain[0][1]=.5*(1.4+x[0]);
              const double work=2.*kappa*.005*chi*fe.JxW(q)*profile.slip_sense*(f.tangent*(strain*f.normal));
              reference+=work;
              scale+=std::abs(work);
            }
          std::ofstream out(this->get_output_directory()+"rhs-action.txt");
          out<<std::setprecision(17)<<actual<<' '<<reference<<' '<<scale<<std::endl;
          AssertThrow(scale>0. && std::abs(actual-reference)<512.*std::numeric_limits<double>::epsilon()*scale,
                      ExcMessage("Projected Stokes RHS failed its independent Q8 slip-work check."));
        }
    };

    template <int dim>
    class PrescribedRidge : public ObservedRidge<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &) {}
        void initialize() override
        {
          // Retain the prescribed initial rate on every label. Only the free
          // material registers the RSF preparation and convergence callbacks.
          this->coupling->initialize_simulator(this->get_simulator());
          this->coupling->initialize();
          this->get_signals().set_assemblers.connect([](const SimulatorAccess<dim> &, Assemblers::Manager<dim> &assemblers)
          {
            for (auto &assembler:assemblers.stokes_system)
              if (dynamic_cast<Assemblers::FaultStressLoad<dim> *>(assembler.get()))
                {
                  assembler=std::make_unique<MeasuredStressLoad<dim>>();
                  return;
                }
            AssertThrow(false,ExcMessage("Missing fault stress assembler in prescribed fixture."));
          });
        }
    };
    ASPECT_REGISTER_MATERIAL_MODEL(PrescribedRidge,"prescribed ridge diagnostic","Fixed 0.005 m/s diagnostic with the ordinary Maxwell and Stokes operators.")
  }
  namespace Postprocess
  {
    /** Pair the actual section row samples with the FE quadrature identities. */
    template <int dim>
    class PressureProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &m = Plugins::get_plugin_as_type<const MaterialModel::ObservedRidge<dim>>(this->get_material_model());
          const auto points = m.collect_attempt_samples(this->get_solution());
          const QGauss<dim> rule(8);
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_quadrature_points);
          std::vector<double> phase(rule.size()), rates;
          for (const auto &label : m.labels()) rates.push_back(label.rate);
          const auto field = this->introspection().compositional_index_for_name(this->get_parameters().surface_energy_model.composition);
          const auto rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          std::ofstream out(this->get_output_directory()+"trace-pressure-"+std::to_string(this->get_timestep_number())
                            +"-r"+std::to_string(rank)+".csv");
          out<<std::setprecision(17)<<"cell,q,x,y,left,xi,foot_x,foot_y,ax,ay,bx,by,N,V,volume,chi,shear,stiffness,delta_pressure,delta_normal\n";
          unsigned int index=0;
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                fe[this->introspection().extractors.compositional_fields[field]].get_function_values(this->get_solution(),phase);
                for (unsigned int q=0; q<rule.size(); ++q)
                  {
                    const auto x=fe.quadrature_point(q);
                    if (m.phase_profile().sample(x,phase[q]).chi==0.) continue;
                    const auto &p=points.at(index++);
                    const auto &foot=p.transfer.foot;
                    const auto &a=m.labels()[foot.first_vertex].position;
                    const auto &b=m.labels()[foot.first_vertex+1].position;
                    const Point<dim> position=a+foot.xi*(b-a);
                    const double base=m.normal_compression(position,3.,2.);
                    out<<cell->id()<<','<<q<<','<<x[0]<<','<<x[1]<<','<<foot.first_vertex<<','<<foot.xi
                       <<','<<position[0]<<','<<position[1]<<','<<a[0]<<','<<a[1]<<','<<b[0]<<','<<b[1]<<','<<p.compression<<','<<p.transfer.rate(rates)
                       <<','<<p.volume<<','<<p.transfer.chi<<','<<p.driving_shear<<','<<p.stiffness
                       <<','<<m.normal_compression(position,4.,2.)-base<<','<<m.normal_compression(position,3.,3.)-base<<'\n';
                  }
              }
          AssertThrow(index==points.size(),ExcInternalError());
          return {"Fault friction pressure","sampled section rows"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(PressureProbe,"fault friction pressure samples","Export Q8 fault pressure, material projection and slip rate.")

    template <int dim>
    class RidgeProbe : public Interface<dim>, public SimulatorAccess<dim>
    {
        using Material = MaterialModel::ObservedRidge<dim>;
        using Stress = SymmetricTensor<2,dim>;
        using PointLoad = typename FaultStokesCoupling<dim>::BulkPoint;
        double initial_age,compression,shear,eta,G,initial_dt;
        bool compare_donor_load;
        MaterialModel::Rheology::RateAndStateFriction law;
        std::vector<double> parent_age;

      public:
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Initial composition model");
          prm.enter_subsection("Stationary ridge");
          compare_donor_load=prm.get_bool("Bilinear donor phase");
          prm.leave_subsection();
          prm.leave_subsection();
          prm.enter_subsection("Material model");
          prm.enter_subsection("Simple model");
          eta=prm.get_double("Viscosity");
          prm.leave_subsection();
          prm.enter_subsection("Phase field fault");
          G=prm.get_double("Shear modulus");
          initial_dt=prm.get_double("Initialization interval");
          initial_age=prm.get_double("Initial age");
          compression=prm.get("Background compression").empty() ? 5e7 : prm.get_double("Background compression");
          shear=prm.get_double("Background shear traction");
          law= {prm.get_double("Reference friction coefficient"),prm.get_double("Direct effect"),prm.get_double("Evolution effect"),
                prm.get_double("Reference speed"),prm.get_double("Slip distance"),prm.get_double("Radiation damping")
               };
          prm.leave_subsection();
          prm.leave_subsection();
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          const auto &m=Plugins::get_plugin_as_type<const Material>(this->get_material_model());
          const auto &profile=m.phase_profile();
          const auto &labels=m.labels();
          const unsigned int n=labels.size(),step=this->get_timestep_number();
          const double dt=step ? this->get_timestep() : initial_dt;
          const double kappa=-eta*std::expm1(-G*dt/eta);
          const auto comm=this->get_mpi_communicator();
          const auto rank=Utilities::MPI::this_mpi_process(comm);
          const auto prefix=this->get_output_directory();
          const auto suffix=std::to_string(step)+"-"+std::to_string(rank)+".txt";
          std::vector<double> rates(n);
          std::ofstream geometry(prefix+"labels-"+suffix);
          geometry<<std::setprecision(17)<<"id x y I age slip rate time\n";
          for (unsigned int j=0; j<n; ++j)
            {
              const auto &l=labels[j];
              rates[j]=l.rate;
              geometry<<j<<' '<<l.position<<' '<<profile.normal_integrals()[j]<<' '<<m.label_ages().at(j)<<' '<<l.slip<<' '<<l.rate<<' '<<this->get_time()<<'\n';
            }
          if (parent_age.empty()) parent_age.assign(n,initial_age);
          const auto points=m.collect_attempt_samples(this->get_solution());
          std::ofstream metrics(prefix+"checks-"+suffix);
          metrics<<std::setprecision(17);
          metrics<<"imposed_balance "<<(dynamic_cast<const MaterialModel::PrescribedRidge<dim>*>(&m)!=nullptr)<<'\n';
          section_checks(m,points,rates,kappa,metrics);
          // Three interior samples per coordinate determine every DGQ2 history
          // coefficient on its own cell, even where the adjacent DG trace differs.
          sample_fields(m,rates,kappa,points,metrics,prefix+"bulk-"+suffix);
          if (step==0)
            central_transect(m,rates,prefix+"transect-"+suffix);
          for (unsigned int j=0; j<n; ++j) parent_age[j]=m.label_ages().at(j);
          return {"Stationary ridge","exported accepted state and transfer checks"};
        }
      private:
        void central_transect(const Material &m,const std::vector<double> &rates,const std::string &path) const
        {
          // Select the upper DG trace on y=0. Report the engineering slip rate
          // t . L_slip . n and the physical quadrature weight along the cut.
          const QGauss<1> line(24);
          std::vector<Point<dim>> unit(line.size());
          for (unsigned int q=0; q<line.size(); ++q) unit[q][0]=line.point(q)[0];
          FEValues<dim> fe(this->get_mapping(),this->get_fe(),Quadrature<dim>(unit),update_values|update_quadrature_points);
          std::vector<double> phi(line.size());
          const auto field=this->introspection().compositional_index_for_name(this->get_parameters().surface_energy_model.composition);
          const auto &profile=m.phase_profile();
          std::ofstream out(path);
          out<<std::setprecision(17)<<"x y w phase h slip_gradient\n";
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned() && cell->vertex(0)[1]==0.)
              {
                fe.reinit(cell);
                fe[this->introspection().extractors.compositional_fields[field]].get_function_values(this->get_solution(),phi);
                for (unsigned int q=0; q<line.size(); ++q)
                  {
                    const auto x=fe.quadrature_point(q);
                    const auto sample=profile.sample(x,phi[q]);
                    const auto &frame=profile.geometry.segment(sample.foot.first_vertex);
                    const auto grad=profile.slip_gradient(sample,sample.rate(rates));
                    out<<x<<' '<<line.weight(q)*(cell->vertex(1)[0]-cell->vertex(0)[0])<<' '<<phi[q]<<' '<<profile.weight(phi[q])<<' '<<frame.tangent*(grad*frame.normal)<<'\n';
                  }
              }
        }

        void section_checks(const Material &m,
                            const std::vector<PointLoad> &points,
                            const std::vector<double> &rates,
                            double kappa,
                            std::ostream &out) const
        {
          const auto &profile=m.phase_profile();
          const unsigned int n=rates.size();
          std::vector<double> test(n),direction(n),plus(n),minus(n),load(n);
          const double eps=1.e-5;
          for (unsigned int j=0; j<n; ++j)
            {
              const double s=double(j)/(n-1);
              test[j]=.005*(1.+.3*s+.2*s*s);
              direction[j]=.003*(.2+std::cos(4.*s));
              plus[j]=test[j]+eps*direction[j];
              minus[j]=test[j]-eps*direction[j];
            }
          std::vector<double> residual(n),scale(n);
          // Assemble the production tridiagonal tangent at the test speeds,
          // then evaluate only residuals at the two perturbed speed vectors.
          std::vector<double> diagonal(n), upper(n-1), test_residual(n);
          std::vector<double> residual_plus(n), residual_minus(n);
          std::vector<double> scratch_scale(n), scratch_diagonal(n), scratch_upper(n-1);
          double work=0.;
          for (const auto &p:points)
            {
              const auto foot=p.transfer.foot;
              const auto &frame=profile.geometry.segment(foot.first_vertex);
              const double virtual_traction=.7+std::sin(3.*(frame.station+frame.length*foot.xi));
              p.transfer.add_load(p.volume*profile.slip_sense*virtual_traction,load);
              const auto grad=profile.slip_gradient(p.transfer,p.transfer.rate(test));
              work+=p.volume*virtual_traction*(frame.tangent*(grad*frame.normal));
              const double age=foot.interpolate(parent_age[foot.first_vertex],parent_age[foot.first_vertex+1]);
              p.add_balance(p.transfer.rate(test),
                            law.resistance(p.transfer.rate(test),age,p.compression),
                            test_residual,diagonal,upper,scratch_scale);
              p.add_balance(p.transfer.rate(plus),
                            law.resistance(p.transfer.rate(plus),age,p.compression),
                            residual_plus,scratch_diagonal,scratch_upper,scratch_scale);
              p.add_balance(p.transfer.rate(minus),
                            law.resistance(p.transfer.rate(minus),age,p.compression),
                            residual_minus,scratch_diagonal,scratch_upper,scratch_scale);
              p.add_balance(p.transfer.rate(rates),
                            law.resistance(p.transfer.rate(rates),age,p.compression),
                            residual,scratch_diagonal,scratch_upper,scale);
            }
          const auto comm=this->get_mpi_communicator();
          for (auto *v:
               {
                 &load,&diagonal,&upper,&residual_plus,&residual_minus,&residual,&scale
               }) Utilities::MPI::sum(*v,comm,*v);
          const double section_work=std::inner_product(load.begin(),load.end(),test.begin(),0.);
          out<<"work "<<Utilities::MPI::sum(work,comm)<<' '<<section_work<<'\n';
          double error=0.,largest=0.,balance=0.;
          for (unsigned int j=0; j<n; ++j)
            {
              const double action=diagonal[j]*direction[j]
                                  + (j ? upper[j-1]*direction[j-1] : 0.)
                                  + (j+1<n ? upper[j]*direction[j+1] : 0.);
              error=std::max(error,std::abs(action+(residual_plus[j]-residual_minus[j])/(2.*eps)));
              largest=std::max(largest,std::abs(action));
              balance=std::max(balance,std::abs(residual[j])/scale[j]);
            }
          out<<"tangent "<<error<<' '<<largest<<'\n'<<"residual "<<balance<<'\n';
        }

        void sample_fields(const Material &m,
                           const std::vector<double> &rates,
                           double kappa,
                           const std::vector<PointLoad> &points,
                           std::ostream &metrics,
                           const std::string &path) const
        {
          const auto &ex=this->introspection().extractors;
          const auto &solution=this->get_solution();
          const auto &profile=m.phase_profile();
          const auto phase=this->introspection().compositional_index_for_name(this->get_parameters().surface_energy_model.composition);
          const auto xx=this->introspection().compositional_index_for_name("fault_stress_xx");
          const auto yy=this->introspection().compositional_index_for_name("fault_stress_yy");
          const auto xy=this->introspection().compositional_index_for_name("fault_stress_xy");
          std::ofstream bulk(path);
          bulk<<std::setprecision(17)<<"cell q x y phi h chi seg xi rate p xx yy xy exx eyy exy tau compression gradxx gradxy gradyx gradyy associated";
          if (compare_donor_load) bulk<<" slip_xx slip_yy slip_xy";
          bulk<<'\n';
          std::vector<double> balance(rates.size()),forces(rates.size());
          double normal_error=0.,shear_error=0.,moment=0.,moment_scale=0.,total_moment=0.;
          unsigned int count=0,cut_cells=0;
          for (const unsigned int order:
          {
            3u,8u,16u
          })
          {
            const QGauss<dim> rule(order);
            FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_values|update_gradients|update_quadrature_points|update_JxW_values);
            std::vector<double> phi(rule.size()),p(rule.size());
            std::array<std::vector<double>,3> dg;
            for (auto &v:dg) v.resize(rule.size());
            std::vector<Tensor<2,dim>> grad(rule.size());
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  if (order==16)
                    {
                      if (!this->get_timestep_number()) continue;
                      const auto left=profile.geometry.project(cell->vertex(0)).first_vertex;
                      bool cut=false;
                      for (unsigned int corner=1; corner<GeometryInfo<dim>::vertices_per_cell; ++corner)
                        cut |= profile.geometry.project(cell->vertex(corner)).first_vertex!=left;
                      if (!cut) continue;
                    }
                  fe.reinit(cell);
                  fe[ex.compositional_fields[phase]].get_function_values(solution,phi);
                  if (order==16 && *std::max_element(phi.begin(),phi.end())<.1) continue;
                  fe[ex.velocities].get_function_gradients(solution,grad);
                  if (order!=16) fe[ex.pressure].get_function_values(solution,p);
                  if (order!=8 || cell->vertex(0).norm_square()==0.)
                    {
                      fe[ex.compositional_fields[xx]].get_function_values(solution,dg[0]);
                      fe[ex.compositional_fields[yy]].get_function_values(solution,dg[1]);
                      fe[ex.compositional_fields[xy]].get_function_values(solution,dg[2]);
                    }
                  std::array<std::array<double,9>,3> moments {};
                  for (unsigned int q=0; q<rule.size(); ++q)
                    {
                      const auto x=fe.quadrature_point(q);
                      const auto s=profile.sample(x,phi[q]);
                      if (order==8 && s.chi==0.) continue;
                      const auto &frame=profile.geometry.segment(s.foot.first_vertex);
                      const auto stress=m.endpoint_stress(cell,x,phi[q],symmetrize(grad[q]));
                      const double tau=shear+profile.slip_sense*(frame.tangent*(stress*frame.normal));
                      const double sigma=compression+p[q]-frame.normal*(stress*frame.normal);
                      if (order==3)
                        {
                          bulk<<cell->id()<<' '<<q<<' '<<x<<' '<<phi[q]<<' '<<profile.weight(phi[q])<<' '<<s.chi<<' '<<s.foot.first_vertex<<' '<<s.foot.xi<<' '<<s.rate(rates)<<' '<<p[q];
                          for (const auto &v:dg) bulk<<' '<<v[q];
                          for (unsigned int c=0; c<3; ++c) bulk<<' '<<stress.access_raw_entry(c);
                          bulk<<' '<<tau<<' '<<sigma<<' '<<grad[q][0][0]<<' '<<grad[q][0][1]<<' '<<grad[q][1][0]<<' '<<grad[q][1][1]<<' '<<s.foot.active;
                          if (compare_donor_load)
                            {
                              // Isolate the pointwise constitutive slip stress in Pa
                              // for the donor comparison. The initial
                              // parent is identically zero. On a physical step cancel
                              // the same parent with a zero-phase material evaluation.
                              Stress slip=m.endpoint_stress(cell,x,phi[q],Stress());
                              if (this->get_timestep_number())
                                slip-=m.endpoint_stress(cell,x,0.,Stress());
                              for (unsigned int c=0; c<3; ++c) bulk<<' '<<slip.access_raw_entry(c);
                            }
                          bulk<<'\n';
                        }
                      else if (order==8)
                        {
                          const auto &point=points.at(count++);
                          // Build the two hat-function residuals directly from the
                          // sampled tensor, mechanical pressure and incoming age.
                          const unsigned int left=s.foot.first_vertex;
                          const double xi=s.foot.xi;
                          const double age=(1.-xi)*parent_age[left]+xi*parent_age[left+1];
                          const double speed=(1.-xi)*rates[left]+xi*rates[left+1];
                          const double psi=law.reference_coefficient+law.evolution_effect*std::log(age*law.reference_speed/law.slip_distance);
                          const double resistance=sigma*law.direct_effect*std::asinh(speed/(2.*law.reference_speed)*std::exp(psi/law.direct_effect))+law.damping*speed;
                          for (unsigned int i=0; i<2; ++i)
                            {
                              const double weight=fe.JxW(q)*s.chi*(i ? xi : 1.-xi);
                              balance[left+i]+=weight*(tau-resistance);
                              forces[left+i]+=weight*std::max(std::abs(tau),std::abs(resistance));
                            }
                          if (cell->vertex(0).norm_square()==0. && q==0)
                            {
                              metrics<<"central_section "<<x;
                              for (const auto &value:dg) metrics<<' '<<value[q];
                              metrics<<' '<<p[q]<<' '<<point.compression<<' '<<tau<<'\n';
                            }
                          normal_error=std::max(normal_error,std::abs(point.compression-sigma));
                          shear_error=std::max(shear_error,std::abs(point.driving_shear-kappa*s.chi*s.rate(rates)-tau));
                        }
                      else
                        for (unsigned int c=0; c<3; ++c)
                          for (unsigned int j=0; j<9; ++j)
                            {
                              const double basis=std::pow(rule.point(q)[0],j%3)*std::pow(rule.point(q)[1],j/3);
                              moments[c][j]+=fe.JxW(q)*basis*(dg[c][q]-stress.access_raw_entry(c));
                              moment_scale=std::max(moment_scale,cell->measure()*std::abs(stress.access_raw_entry(c)));
                            }
                    }
                  if (order==16)
                    {
                      ++cut_cells;
                      for (const auto &v:moments)
                        for (const double value:v)
                          {
                            moment=std::max(moment,std::abs(value));
                            total_moment+=std::abs(value);
                          }
                    }
                }
          }
          AssertThrow(count==points.size(),ExcInternalError());
          const auto comm=this->get_mpi_communicator();
          Utilities::MPI::sum(balance,comm,balance);
          Utilities::MPI::sum(forces,comm,forces);
          double relative=0.;
          for (unsigned int j=0; j<rates.size(); ++j)
            relative=std::max(relative,std::abs(balance[j])/forces[j]);
          metrics<<"independent_balance "<<relative<<'\n';
          metrics<<"traction "<<normal_error<<' '<<shear_error<<'\n'<<"samples "<<count<<'\n'
                 <<"moments "<<cut_cells<<' '<<moment<<' '<<moment_scale<<' '<<total_moment<<'\n';
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(RidgeProbe,"ridge probe","Sample the stationary DG field, section work, Newton tangent and accepted stress moments.")
  }
}
