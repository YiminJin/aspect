/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */

#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/particle/property/crack_driving_force.h>
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/plugins.h>
#include "../../../../tests/phase_field_fault_test_access.h"
#include "geometry_reference.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>

#include <fstream>
#include <iomanip>
#include <map>

namespace aspect
{
  namespace TransportReference
  {
    constexpr double velocity = .4, viscosity = 4., modulus = 2., pulse_width = .0125;

    template <int dim>
    double pulse(const Point<dim> &point)
    {
      const double distance = std::remainder(point[0]-.213, 1.);
      return std::exp(-.5*distance*distance/(pulse_width*pulse_width));
    }

    template <int dim>
    SymmetricTensor<2,dim> initial_stress(const Point<dim> &point)
    {
      SymmetricTensor<2,dim> result;
      const double x = 2*numbers::PI*point[0], y = 2*numbers::PI*point[1];
      result[0][0] = 1.+.2*std::sin(x)*std::cos(y);
      result[1][1] = -result[0][0];
      result[0][1] = .3+.15*std::cos(x)*std::sin(y);
      return result;
    }

    template <int dim>
    Point<dim> departure(Point<dim> point, const double time, const double ramp)
    {
      point[0] -= velocity*time+.5*ramp*time*time*(point[1]-.5);
      return point;
    }

    template <int dim>
    SymmetricTensor<2,dim> stress(const Point<dim> &label, const double time, const double ramp)
    {
      const double lambda = viscosity/modulus;
      auto result = std::exp(-time/lambda)*initial_stress(label);
      // Exact convolution for the shear-rate ramp gamma(t)=ramp*t.
      result[0][1] += viscosity*ramp*(time+lambda*std::expm1(-time/lambda));
      return result;
    }
  }

  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      class TransportPulse : public CrackDrivingForce<dim>
      {
        public:
          void initialize_one_particle_property(const Point<dim> &position,
                                                std::vector<double> &data) const override
          {
            data.push_back(TransportReference::pulse(position));
          }
      };
      ASPECT_REGISTER_PARTICLE_PROPERTY(TransportPulse, "transport passive pulse",
                                        "Prescribed-flow diagnostic: an advected passive crack_driving_force pulse.")
    }
  }

  namespace Postprocess
  {
    template <int dim>
    class PrescribedTransport : public Interface<dim>, public SimulatorAccess<dim>
    {
        using F = MaterialModel::internal::PhaseFieldFaultTestAccess<dim>;
        struct Entry
        {
          Point<dim> label;
          std::string cell;
          unsigned int crossings = 0;
        };
        std::map<types::particle_index, Entry> particles;
        unsigned int last_update = numbers::invalid_unsigned_int;
        double ramp = 0.;
        bool audit_domains = false;
        double fe_error = 0., lls_error = 0., transfer_difference = 0.;
        double particle_error = 0., pulse_error = 0., position_error = 0., one_step_error = 0.;
        unsigned int minimum_crossings = 0;

        void update_history()
        {
          using namespace TransportReference;
          const unsigned int step = this->get_timestep_number();
          AssertThrow(step == 0 ? last_update == numbers::invalid_unsigned_int : step == last_update+1,
                      ExcMessage("Prescribed history must be constructed once per accepted step."));
          auto &pm = const_cast<Particle::Manager<dim> &>(this->get_particle_manager(0));
          const auto &info = pm.get_property_manager().get_data_info();
          const unsigned int stress_position = info.get_position_by_field_name("maxwell stress");
          const unsigned int H = info.get_position_by_field_name("crack_driving_force");
          const double time = this->get_time(), dt = step ? this->get_timestep() : 0.;
          const double expected_dt = this->get_parameters().maximum_time_step;
          AssertThrow(step == 0 || std::abs(dt-expected_dt)<1.e-12,
                      ExcMessage("Prescribed transport schedule changed."));
          AssertThrow(!Plugins::get_plugin_as_type<const MaterialModel::PhaseFieldFault<dim>>(
                        this->get_material_model()).uses_fe_strain_history(), ExcInternalError());
          const QGauss<dim> rule(3);
          FEValues<dim> fe(this->get_mapping(), this->get_fe(), rule,
                           update_values|update_gradients|update_quadrature_points|update_JxW_values);
          std::vector<SymmetricTensor<2,dim>> rates(rule.size());
          std::vector<std::vector<double>> fields(3, std::vector<double>(rule.size()));
          fe_error = lls_error = transfer_difference = 0.;
          ComponentMask mask(info.n_components(), false);
          for (unsigned int c=0; c<3; ++c)
            mask.set(stress_position+c, true);

          // These are the actual pre-publication mapped fields used by bulk assembly.
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(
                  this->get_solution(), rates);
                for (unsigned int c=0; c<3; ++c)
                  fe[this->introspection().extractors.compositional_fields[c]].get_function_values(
                    this->get_solution(), fields[c]);
                const auto values = pm.get_interpolator().properties_at_points(
                                      pm.get_particle_handler(), fe.get_quadrature_points(), mask, cell);
                for (unsigned int q=0; q<rule.size(); ++q)
                  {
                    SymmetricTensor<2,dim> old_fe, old_lls;
                    for (unsigned int c=0; c<3; ++c)
                      {
                        const auto indices = SymmetricTensor<2,dim>::unrolled_to_component_indices(c);
                        old_fe[indices] = fields[c][q];
                        old_lls[indices] = values[q][stress_position+c];
                      }
                    AssertThrow(std::abs(trace(rates[q]))<1.e-12,
                                ExcMessage("This control must remain trace-free."));
                    const auto reference = stress(departure(fe.quadrature_point(q), time, ramp), time, ramp);
                    const auto fe_current = F::maxwell_stress(viscosity, modulus, dt, rates[q], old_fe);
                    const auto lls_current = F::maxwell_stress(viscosity, modulus, dt, rates[q], old_lls);
                    fe_error += std::pow((fe_current-reference).norm(),2)*fe.JxW(q);
                    lls_error += std::pow((lls_current-reference).norm(),2)*fe.JxW(q);
                    transfer_difference += std::pow((fe_current-lls_current).norm(),2)*fe.JxW(q);
                  }
              }
          fe_error = std::sqrt(fe_error);
          lls_error = std::sqrt(lls_error);
          transfer_difference = std::sqrt(transfer_difference);

          std::map<types::particle_index, SymmetricTensor<2,dim>> candidates;
          particle_error = pulse_error = position_error = one_step_error = 0.;
          minimum_crossings = std::numeric_limits<unsigned int>::max();
          if (step)
            AssertThrow(particles.size() == pm.get_particle_handler().n_locally_owned_particles(),
                        ExcMessage("This isolated transport control has no births or losses."));
          for (const auto &cell : this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                std::vector<Point<dim>> points;
                for (const auto &p : pm.get_particle_handler().particles_in_cell(cell))
                  points.push_back(p.get_reference_location());
                if (points.empty()) continue;
                FEValues<dim> evaluation(this->get_mapping(), this->get_fe(), Quadrature<dim>(points), update_gradients);
                evaluation.reinit(cell);
                std::vector<SymmetricTensor<2,dim>> strains(points.size());
                evaluation[this->introspection().extractors.velocities].get_function_symmetric_gradients(
                  this->get_solution(), strains);
                unsigned int i = 0;
                for (const auto &p : pm.get_particle_handler().particles_in_cell(cell))
                  {
                    if (step == 0)
                      particles.emplace(p.get_id(), Entry {p.get_location(), cell->id().to_string(), 0});
                    auto &entry = particles.at(p.get_id());
                    if (entry.cell != cell->id().to_string()) ++entry.crossings;
                    entry.cell = cell->id().to_string();
                    minimum_crossings = std::min(minimum_crossings, entry.crossings);
                    SymmetricTensor<2,dim> old;
                    for (unsigned int c=0; c<3; ++c)
                      old[SymmetricTensor<2,dim>::unrolled_to_component_indices(c)] = p.get_properties()[stress_position+c];
                    const auto candidate = F::maxwell_stress(viscosity, modulus, dt, strains[i++], old);
                    candidates.emplace(p.get_id(), candidate);
                    particle_error = std::max(particle_error, (candidate-stress(entry.label, time, ramp)).norm());
                    pulse_error = std::max(pulse_error, std::abs(p.get_properties()[H]-pulse(entry.label)));
                    auto expected = entry.label;
                    expected[0] += velocity*time+.5*ramp*time*time*(expected[1]-.5);
                    const double dx = std::remainder(p.get_location()[0]-expected[0], 1.);
                    position_error = std::max(position_error, std::hypot(dx, p.get_location()[1]-expected[1]));
                    const double beta = std::exp(-dt*modulus/viscosity);
                    const double kappa = -viscosity*std::expm1(-dt*modulus/viscosity);
                    one_step_error = std::max(one_step_error, (candidate-beta*old-2*kappa*strains[i-1]).norm());
                  }
              }
          AssertThrow(candidates.size() == particles.size(), ExcInternalError());
          // Test-only driver for prescribed flow: production has no constitutive
          // terminal call in the no-Stokes scheme. It calls the native kernel once.
          for (auto &p : pm.get_particle_handler())
            for (unsigned int c=0; c<3; ++c)
              p.get_properties()[stress_position+c] = candidates.at(p.get_id())[
                                                        SymmetricTensor<2,dim>::unrolled_to_component_indices(c)];
          last_update = step;
        }

        void inspect_domains()
        {
          using namespace TransportGeometry;
          using Point = TransportGeometry::Point;
          const auto &pm=this->get_particle_manager(0);
          const unsigned int H=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          const unsigned int side=std::lround(std::sqrt(particles.size()));
          const unsigned int n=std::lround(std::sqrt(this->get_triangulation().n_global_active_cells()));
          const double h=1./side;
          std::vector<Polygon> bins(side*side);
          for (const auto &p : pm.get_particle_handler())
            bins[std::min(side-1,unsigned(p.get_location()[1]*side))*side+
            std::min(side-1,unsigned(p.get_location()[0]*side))].push_back({{p.get_location()[0],p.get_location()[1]}});
          std::array<double,14> error = {{}};
          double native_area=0, independent_area=0, absolute_area=0, symmetric_area=0;
          double mass=0, weak_bound=0, gradient_bound=0;
          double linear_error=0, minimum_y_derivative=1, maximum_y_derivative=1;
          unsigned int negative_y_derivatives=0;
          Operator weak_difference;
          std::ofstream domains;
          if (this->get_timestep_number()==2)
            {
              domains.open(this->get_output_directory()+"domains-step2.csv");
              domains<<std::setprecision(17)<<"id,kind,x,y,a,b,c\n";
            }
          for (const auto &p : pm.get_particle_handler())
            {
              const Point parent= {{p.get_location()[0],p.get_location()[1]}};
              const int ix=std::min(side-1,unsigned(parent[0]*side));
              const int iy=std::min(side-1,unsigned(parent[1]*side));
              Polygon reference= {{{-2*h,std::max(-parent[1],-2*h)}},{{2*h,std::max(-parent[1],-2*h)}},
                {{2*h,std::min(1-parent[1],2*h)}},{{-2*h,std::min(1-parent[1],2*h)}}
              };
              for (int j=std::max(0,iy-3); j<=std::min(int(side)-1,iy+3); ++j)
                for (int i=ix-3; i<=ix+3; ++i)
                  for (const auto &q : bins[j*side+(i+side)%side])
                    {
                      const Point d= {{std::remainder(q[0]-parent[0],1.),q[1]-parent[1]}};
                      if (std::hypot(d[0],d[1])>h/10)
                        reference=clip(reference,d,.5*(d[0]*d[0]+d[1]*d[1]));
                    }
              double radius=0;
              for (const auto &v : reference) radius=std::max(radius,std::hypot(v[0],v[1]));
              // Every omitted spatial bin is at least 3h away. Thus its sites
              // cannot cut a polygon whose circumscribed radius is below 1.5h.
              AssertThrow(2*radius<3*h,ExcMessage("Independent neighbor stencil is not certified complete."));
              const auto domain=pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
              Polygon native;
              bool seam=false;
              for (const auto &v : domain.vertices())
                {
                  native.push_back({{v[0]-parent[0],v[1]-parent[1]}});
                  seam=seam || v[0]<0 || v[0]>1;
                }
              const auto a=moments(native), b=moments(reference);
              const double area_error=std::abs(domain.volume()-b[0])/b[0];
              error[0]=std::max(error[0],area_error);
              error[1]=std::max(error[1],std::hypot(a[1]/a[0]-b[1]/b[0],a[2]/a[0]-b[2]/b[0])/h);
              for (unsigned int k=3; k<6; ++k)
                error[2]=std::max(error[2],std::abs(a[k]/a[0]-b[k]/b[0])/(h*h));
              const double difference=symmetric_difference(native,reference);
              error[3]=std::max(error[3],difference/b[0]);
              native_area+=domain.volume();
              independent_area+=b[0];
              absolute_area+=std::abs(domain.volume()-b[0]);
              symmetric_area+=difference;
              const auto exact=cpdi(reference,parent,n), same_polygon=cpdi(native,parent,n);
              Operator actual;
              const auto &vertices=this->get_triangulation().get_vertices();
              for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                {
                  const auto &v=vertices[domain.relevant_vertex_index(i)];
                  const unsigned int node=std::lround(n*v[1])*n+std::lround(n*v[0])%n;
                  const auto gradient=domain.weighting_function_gradient(i);
                  auto &entry=actual[node];
                  entry[0]+=domain.volume()*domain.weighting_function_value(i);
                  entry[1]+=domain.volume()*gradient[0];
                  entry[2]+=domain.volume()*gradient[1];
                }
              std::array<double,4> linear = {{}};
              for (const auto &entry : actual)
                {
                  const double x=std::remainder(double(entry.first%n)/n-parent[0],1.);
                  const double y=double(entry.first/n)/n-parent[1];
                  linear[0]+=x*entry.second[1]/domain.volume();
                  linear[1]+=x*entry.second[2]/domain.volume();
                  linear[2]+=y*entry.second[1]/domain.volume();
                  linear[3]+=y*entry.second[2]/domain.volume();
                }
              minimum_y_derivative=std::min(minimum_y_derivative,linear[3]);
              maximum_y_derivative=std::max(maximum_y_derivative,linear[3]);
              if (linear[3]<0) ++negative_y_derivatives;
              linear[0]-=1;
              linear[3]-=1;
              for (const auto value : linear) linear_error=std::max(linear_error,std::abs(value));
              auto delta=actual, implementation=actual, normalized=actual;
              for (const auto &entry : exact)
                for (unsigned int k=0; k<3; ++k) delta[entry.first][k]-=entry.second[k];
              for (const auto &entry : same_polygon)
                for (unsigned int k=0; k<3; ++k) implementation[entry.first][k]-=entry.second[k];
              for (auto &entry : normalized)
                for (auto &value : entry.second) value/=domain.volume();
              for (const auto &entry : exact)
                for (unsigned int k=0; k<3; ++k) normalized[entry.first][k]-=entry.second[k]/b[0];
              double weights=0, gradients=0, local_weak=0, local_gradient=0;
              for (const auto &entry : normalized)
                {
                  weights+=std::abs(entry.second[0]);
                  gradients+=std::hypot(entry.second[1],entry.second[2])/n;
                }
              error[4]=std::max(error[4],weights);
              error[5]=std::max(error[5],gradients);
              for (const auto &entry : implementation)
                {
                  error[6]=std::max(error[6],std::abs(entry.second[0])/a[0]);
                  error[7]=std::max(error[7],std::hypot(entry.second[1],entry.second[2])/(n*a[0]));
                }
              for (const auto &entry : delta)
                {
                  local_weak+=std::abs(entry.second[0]);
                  local_gradient+=std::hypot(entry.second[1],entry.second[2])/n;
                  for (unsigned int k=0; k<3; ++k) weak_difference[entry.first][k]+=p.get_properties()[H]*entry.second[k];
                }
              error[8]=std::max(error[8],local_weak/b[0]);
              error[9]=std::max(error[9],local_gradient/b[0]);
              if (seam) error[10]=std::max(error[10],local_weak/b[0]);
              if (parent[1]<h || parent[1]>1-h) error[11]=std::max(error[11],local_weak/b[0]);
              weak_bound+=p.get_properties()[H]*local_weak;
              gradient_bound+=p.get_properties()[H]*local_gradient;
              mass+=p.get_properties()[H]*b[0];
              if (domains)
                {
                  domains<<p.get_id()<<",parent,"<<parent[0]<<','<<parent[1]<<','<<domain.volume()<<','<<p.get_properties()[H]<<",0\n";
                  for (const auto &v : domain.vertices()) domains<<p.get_id()<<",vertex,"<<v[0]<<','<<v[1]<<",0,0,0\n";
                  for (const auto &entry : actual) domains<<p.get_id()<<",operator,"<<entry.first<<",0,"<<entry.second[0]<<','<<entry.second[1]<<','<<entry.second[2]<<'\n';
                }
            }
          for (const auto &entry : weak_difference)
            {
              error[12]+=std::abs(entry.second[0])/mass;
              error[13]+=std::hypot(entry.second[1],entry.second[2])/(n*mass);
            }
          std::ofstream out(this->get_output_directory()+"geometry.csv",this->get_timestep_number() ? std::ios::app : std::ios::out);
          if (this->get_timestep_number()==0)
            out<<"step,time,native_area,independent_area,absolute_area_error,symmetric_difference_area,max_area_relative,max_centroid_over_h,max_second_over_h2,max_symmetric_difference_relative,max_weight_l1,max_gradient_l1_over_n,same_polygon_weight_error,same_polygon_gradient_error,max_local_weak_relative,max_local_gradient_relative,seam_weak_relative,wall_weak_relative,weak_relative,weak_gradient_relative,weak_absolute_bound,gradient_absolute_bound,max_linear_gradient_error,min_y_derivative,max_y_derivative,negative_y_derivatives\n";
          out<<std::setprecision(17)<<this->get_timestep_number()<<','<<this->get_time()<<','<<native_area<<','<<independent_area<<','<<absolute_area<<','<<symmetric_area;
          for (const auto value : error) out<<','<<value;
          out<<','<<weak_bound/mass<<','<<gradient_bound/mass<<','<<linear_error<<','<<minimum_y_derivative<<','<<maximum_y_derivative<<','<<negative_y_derivatives<<'\n';
          out.flush();
          AssertThrow(linear_error<1.e-10 && error[0]<1.e-10 && error[3]<1.e-10 && error[4]<1.e-10 && error[5]<1.e-10,
                      ExcMessage("Domain geometry or CPDI operator exceeds the independent 1e-10 accuracy checks."));
        }

        std::array<double,4> weak_pulse(const bool final)
        {
          using namespace TransportReference;
          const auto &pm = this->get_particle_manager(0);
          const auto H = pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          const auto &vertices = this->get_triangulation().get_vertices();
          const unsigned int n = std::lround(std::sqrt(this->get_triangulation().n_global_active_cells()));
          const auto index = [&](const Point<dim> &x)
          {
            return std::lround(n*x[1])*n+std::lround(n*x[0])%n;
          };
          std::vector<double> assembled(n*(n+1),0.), reference(assembled.size(),0.);
          double maximum = 0., volume = 0., partition_error = 0.;
          types::particle_index worst_particle = 0;
          Point<dim> worst_position;
          long double extended_volume = 0.;
          double maximum_area_difference = 0.;
          types::particle_index worst_area_id = 0;
          const double lattice_area = 1./particles.size();
          for (const auto &p : pm.get_particle_handler())
            {
              const auto domain = pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
              const double value = p.get_properties()[H];
              maximum = std::max(maximum, value);
              volume += domain.volume();
              extended_volume += domain.volume();
              if (std::abs(domain.volume()-lattice_area)>maximum_area_difference)
                {
                  maximum_area_difference = std::abs(domain.volume()-lattice_area);
                  worst_area_id = p.get_id();
                }
              double partition = 0.;
              for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                {
                  partition += domain.weighting_function_value(i);
                  assembled[index(vertices[domain.relevant_vertex_index(i)])] +=
                    value*domain.volume()*domain.weighting_function_value(i);
                }
              if (std::abs(partition-1.)>partition_error)
                {
                  partition_error = std::abs(partition-1.);
                  worst_particle = p.get_id();
                  worst_position = p.get_location();
                }
            }
          if (audit_domains) inspect_domains();
          if (!(std::abs(volume-1.)<1.e-10 && partition_error<1.e-10))
            {
              this->get_pcout() << std::setprecision(17) << "CPDI failed coverage: volume=" << volume
                                << " extended_volume=" << extended_volume
                                << " partition_error=" << partition_error << " particle=" << worst_particle
                                << " position=" << worst_position << " maximum_lattice_area_difference="
                                << maximum_area_difference << std::endl;
              std::ofstream geometry(this->get_output_directory()+"failed_domain.csv");
              geometry << std::setprecision(17) << "kind,x,y,area\n";
              for (const auto &p : pm.get_particle_handler())
                if (p.get_id()==worst_area_id)
                  {
                    const auto domain = pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
                    geometry << "parent," << p.get_location()[0] << ',' << p.get_location()[1] << ',' << domain.volume() << '\n';
                    for (const auto &x : domain.vertices())
                      geometry << "vertex," << x[0] << ',' << x[1] << ",0\n";
                    for (const auto &neighbor : pm.get_particle_handler())
                      {
                        const double dx=std::remainder(neighbor.get_location()[0]-p.get_location()[0],1.);
                        const double dy=neighbor.get_location()[1]-p.get_location()[1];
                        if (std::hypot(dx,dy)<3*std::sqrt(lattice_area))
                          geometry << "neighbor," << p.get_location()[0]+dx << ',' << neighbor.get_location()[1] << ",0\n";
                      }
                  }
            }
          AssertThrow(std::abs(volume-1.)<1.e-10 && partition_error<1.e-10,
                      ExcMessage("Moved CPDI domains fail volume coverage or partition of unity."));
          const FE_Q<dim> q1(1);
          const QGauss<dim> rule(32);
          FEValues<dim> fe(this->get_mapping(), q1, rule, update_values|update_quadrature_points|update_JxW_values);
          for (const auto &cell : this->get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                fe.reinit(cell);
                for (unsigned int q=0; q<rule.size(); ++q)
                  for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                    reference[index(cell->vertex(v))] += pulse(departure(fe.quadrature_point(q), this->get_time(), ramp))
                                                         *fe.shape_value(v,q)*fe.JxW(q);
              }
          double mass = 0., exact_mass = 0., difference = 0.;
          for (unsigned int i=0; i<assembled.size(); ++i)
            {
              mass += assembled[i];
              exact_mass += reference[i];
              difference += std::abs(assembled[i]-reference[i]);
            }
          if (final)
            {
              std::ofstream out(this->get_output_directory()+"weak_pulse.csv");
              out << std::setprecision(17) << "vertex,assembled,reference\n";
              for (unsigned int i=0; i<assembled.size(); ++i)
                out << i << ',' << assembled[i] << ',' << reference[i] << '\n';
              std::ofstream samples(this->get_output_directory()+"pulse_particles.csv");
              samples << std::setprecision(17) << "id,x,y,label_x,pulse,reference\n";
              for (const auto &p : pm.get_particle_handler())
                if (p.get_properties()[H]>1.e-4 && std::abs(p.get_location()[1]-.5)<.08)
                  samples << p.get_id() << ',' << p.get_location()[0] << ',' << p.get_location()[1] << ','
                          << particles.at(p.get_id()).label[0] << ',' << p.get_properties()[H] << ','
                          << pulse(particles.at(p.get_id()).label) << '\n';
            }
          return {{difference/exact_mass, std::abs(mass-exact_mass)/exact_mass, maximum, mass}};
        }

      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Prescribed transport");
          prm.declare_entry("Shear rate ramp", "0", Patterns::Double(0.));
          prm.declare_entry("Audit particle domains", "false", Patterns::Bool());
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("Prescribed transport");
          ramp = prm.get_double("Shear rate ramp");
          audit_domains = prm.get_bool("Audit particle domains");
          prm.leave_subsection();
          prm.leave_subsection();
        }

        void initialize() override
        {
          AssertThrow(dim == 2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator()) == 1,
                      ExcMessage("This bounded diagnostic is serial 2D."));
          this->get_signals().post_nonlinear_solver.connect([this](const SolverControl &control)
          {
            AssertThrow(control.last_check() == SolverControl::success, ExcInternalError());
            update_history();
          });
        }

        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(last_update == this->get_timestep_number(), ExcInternalError());
          const bool final = this->get_time() >= 1.-1.e-12;
          const auto weak = weak_pulse(final);
          const auto path = this->get_output_directory()+"transport.csv";
          std::ofstream out(path, this->get_timestep_number() ? std::ios::app : std::ios::out);
          if (this->get_timestep_number() == 0)
            out << "step,time,dt,particles,min_crossings,position_error,particle_stress_error,pulse_particle_error,one_step_defect,FE_total_rms,LLS_total_rms,FE_LLS_difference,weak_relative_l1,mass_relative_error,pulse_peak,pulse_mass\n";
          out << std::setprecision(17) << this->get_timestep_number() << ',' << this->get_time() << ','
              << this->get_timestep() << ',' << particles.size() << ',' << minimum_crossings << ','
              << position_error << ',' << particle_error << ',' << pulse_error << ',' << one_step_error << ','
              << fe_error << ',' << lls_error << ',' << transfer_difference << ',' << weak[0] << ',' << weak[1]
              << ',' << weak[2] << ',' << weak[3] << '\n';
          AssertThrow(pulse_error == 0. && one_step_error < 1.e-12 && particle_error < .2,
                      ExcMessage("Transport or once-only Maxwell update failed."));
          if (final)
            AssertThrow(minimum_crossings >= 2, ExcMessage("Particles did not cross enough cells."));
          return {"Prescribed particle transport:", "measured"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(PrescribedTransport, "prescribed transport checks",
                                  "Separate particle, FE reconstruction and native CPDI weak errors in prescribed flow.")
  }
}
