/* Copyright (C) 2026 by the authors of the ASPECT code.
 * SPDX-License-Identifier: GPL-2.0-or-later */
// Include the existing benchmark registration and its verified fingerprint in
// one translation unit; this diagnostic stops before any mechanical solve.
#include "../../uniform_shear.cc"
#include <cstdlib>
#include <sstream>

namespace aspect
{
  namespace K3SeamAudit
  {
    using Rows = std::vector<std::vector<double>>;

    Rows read_csv(const std::string &path)
    {
      std::ifstream in(path);
      AssertThrow(in.good(), ExcMessage("Cannot read " + path));
      std::string line;
      std::getline(in,line);
      Rows rows;
      while (std::getline(in,line))
        {
          std::replace(line.begin(),line.end(),',',' ');
          std::istringstream values(line);
          std::vector<double> row;
          double value;
          while (values >> value) row.push_back(value);
          if (!row.empty()) rows.push_back(std::move(row));
        }
      return rows;
    }

    std::map<double,double> average_profile(const Rows &rows,
                                          const unsigned int y_index,
                                          const unsigned int value_index)
    {
      std::map<double,std::pair<long double,unsigned int>> sums;
      for (const auto &row : rows)
        {
          auto &sum = sums[std::round(row[y_index]*1e12)/1e12];
          sum.first += row[value_index];
          ++sum.second;
        }
      std::map<double,double> profile;
      for (const auto &sum : sums) profile[sum.first] = sum.second.first/sum.second.second;
      return profile;
    }

    double evaluate(const std::map<double,double> &profile, const double y)
    {
      const auto upper = profile.lower_bound(y);
      if (upper == profile.begin()) return upper->second;
      if (upper == profile.end()) return profile.rbegin()->second;
      const auto lower = std::prev(upper);
      const double fraction = (y-lower->first)/(upper->first-lower->first);
      return (1-fraction)*lower->second + fraction*upper->second;
    }

    template <int dim>
    void probe(const SimulatorAccess<dim> &sim, const std::string &tag,
               const std::map<double,double> &phi_bar,
               const std::map<double,double> &H_bar, const bool force_failure)
    {
      const auto before = K3::fingerprint(sim);
      auto &handler = const_cast<PhaseFieldHandler<dim>&>(sim.get_phase_field_handler());
      auto &pm = handler.get_associated_particle_manager();
      const auto H_index = pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
      const auto block = sim.introspection().variable("phase_field").block_index;
      const auto &dofs = internal::PhaseFieldTestAccess<dim>::vertex_dofs(handler);
      const auto &vertices = sim.get_triangulation().get_vertices();
      const auto &constraints = sim.get_current_constraints();
      LinearAlgebra::BlockSparseMatrix matrix;
      matrix.reinit(sim.get_system_matrix().n_block_rows(),sim.get_system_matrix().n_block_cols());
      matrix.block(block,block).copy_from(sim.get_system_matrix().block(block,block));
      LinearAlgebra::BlockVector state(sim.get_solution());
      LinearAlgebra::BlockVector rhs(sim.introspection().index_sets.system_partitioning,sim.get_mpi_communicator());
      LinearAlgebra::BlockVector reaction(rhs), gradient(rhs), mass(rhs);
      for (unsigned int v=0; v<dofs.size(); ++v)
        if (dofs[v] != numbers::invalid_dof_index) state[dofs[v]] = evaluate(phi_bar,vertices[v][1]);
      state.compress(VectorOperation::insert);
      constraints.distribute(state);

      // Substitution is limited to H, with restoration on success and exception.
      // The production residual receives private matrix/vector buffers only.
      struct Restore
      {
        std::vector<std::pair<double*,double>> values;
        ~Restore() { for (const auto &entry : values) *entry.first=entry.second; }
      };
      std::vector<std::pair<double*,double>> saved;
      for (auto &p : pm.get_particle_handler()) saved.emplace_back(&p.get_properties()[H_index],p.get_properties()[H_index]);
      std::exception_ptr failure;
      try
        {
          Restore restore{std::move(saved)};
          for (auto &p : pm.get_particle_handler()) p.get_properties()[H_index]=evaluate(H_bar,p.get_location()[1]);
          AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler,matrix,rhs,state,false),
                      ExcMessage("Homogeneous phase input was inadmissible."));
          if (force_failure) throw K3::ForcedProbeFailure();
          std::ofstream particles(sim.get_output_directory()+tag+"_particles.csv");
          particles << std::setprecision(17)
                    << "id,x,y,volume,polygon_area,cx,cy,sum_w,grad_constant,first_moment_error,linear_gradient_error,phi,gphi_x,gphi_y,H\n";

          // Geometric moments use unwrapped physical vertex coordinates, not
          // periodic master coordinates (the coordinate x is not periodic).
          for (const auto &p : pm.get_particle_handler())
            {
              const auto domain = pm.get_particle_domain_handler().get_particle_domain(p.get_local_index());
              const auto &polygon = domain.vertices();
              double twice_area=0;
              Point<dim> centroid;
              for (unsigned int j=0; j<polygon.size(); ++j)
                {
                  const auto &a=polygon[j], &b=polygon[(j+1)%polygon.size()];
                  const double cross=a[0]*b[1]-b[0]*a[1];
                  twice_area+=cross;
                  for (unsigned int d=0; d<2; ++d) centroid[d]+=(a[d]+b[d])*cross;
                }
              centroid/=3*twice_area;
              double sum_w=0, phi=0;
              Tensor<1,dim> constant_gradient, first, grad_phi;
              Tensor<2,dim> linear_gradient;
              const auto n=domain.n_relevant_vertices();
              std::vector<types::global_dof_index> indices(n);
              for (unsigned int i=0; i<n; ++i)
                {
                  const auto v=domain.relevant_vertex_index(i);
                  Point<dim> vertex=vertices[v];
                  if (std::getenv("K3_CORRECTED_PERIODIC_AUDIT"))
                    vertex[0]-=.25*std::round((vertex[0]-p.get_location()[0])/.25);
                  const double w=domain.weighting_function_value(i);
                  const auto g=domain.weighting_function_gradient(i);
                  indices[i]=dofs[v];
                  sum_w+=w;
                  constant_gradient+=g;
                  first+=w*Tensor<1,dim>(vertex);
                  const double nodal_phi=state[dofs[v]];
                  phi+=w*nodal_phi;
                  grad_phi+=g*nodal_phi;
                  for (unsigned int d=0; d<dim; ++d)
                    for (unsigned int e=0; e<dim; ++e) linear_gradient[d][e]+=vertex[d]*g[e];
                }
              for (unsigned int d=0; d<dim; ++d) linear_gradient[d][d]-=1;
              particles << p.get_id() << ',' << p.get_location()[0] << ',' << p.get_location()[1]
                        << ',' << domain.volume() << ',' << std::abs(twice_area)/2
                        << ',' << centroid[0] << ',' << centroid[1] << ',' << sum_w
                        << ',' << constant_gradient.norm() << ',' << (first-Tensor<1,dim>(centroid)).norm()
                        << ',' << linear_gradient.norm() << ',' << phi << ',' << grad_phi[0] << ',' << grad_phi[1]
                        << ',' << p.get_properties()[H_index] << '\n';

              // Independently split the weak load, using production constitutive
              // coefficients. Apply exactly the same physical FE constraints.
              const auto coefficients=internal::PhaseFieldTestAccess<dim>::single_material_coefficients(handler,phi,p.get_properties()[H_index]);
              Vector<double> local_r(n),local_g(n),local_m(n);
              for (unsigned int i=0; i<n; ++i)
                {
                  local_r[i]=-domain.volume()*domain.weighting_function_value(i)*coefficients.second;
                  local_g[i]=-domain.volume()*coefficients.first*(domain.weighting_function_gradient(i)*grad_phi);
                  local_m[i]=domain.volume()*domain.weighting_function_value(i);
                }
              constraints.distribute_local_to_global(local_r,indices,reaction);
              constraints.distribute_local_to_global(local_g,indices,gradient);
              constraints.distribute_local_to_global(local_m,indices,mass);
            }
          reaction.compress(VectorOperation::add);
          gradient.compress(VectorOperation::add);
          mass.compress(VectorOperation::add);
          LinearAlgebra::BlockVector difference(rhs);
          difference-=reaction;
          difference-=gradient;
          AssertThrow(difference.block(block).l2_norm()<1e-12,
                      ExcMessage("Split phase diagnostic disagrees with production assembly."));
          std::ofstream nodes(sim.get_output_directory()+tag+"_nodes.csv");
          nodes << std::setprecision(17) << "vertex,dof,x,y,constrained,phi,rhs,reaction,gradient,mass\n";
          for (unsigned int v=0; v<dofs.size(); ++v)
            if (dofs[v]!=numbers::invalid_dof_index)
              nodes << v << ',' << dofs[v] << ',' << vertices[v][0] << ',' << vertices[v][1]
                    << ',' << constraints.is_constrained(dofs[v]) << ',' << state[dofs[v]]
                    << ',' << rhs[dofs[v]] << ',' << reaction[dofs[v]] << ',' << gradient[dofs[v]] << ',' << mass[dofs[v]] << '\n';
          sim.get_pcout() << "K3_SEAM_PROBE " << tag << " production_rhs=" << rhs.block(block).l2_norm()
                          << " split_difference=" << difference.block(block).l2_norm() << std::endl;
        }
      catch (...) { failure=std::current_exception(); }
      AssertThrow(before==K3::fingerprint(sim),ExcMessage("Homogeneous residual probe did not restore live state."));
      if (failure) std::rethrow_exception(failure);
    }

    template <int dim>
    void run(const SimulatorAccess<dim> &sim, Assemblers::Manager<dim> &)
    {
      if (sim.get_reconstructed_fault_manager().get_faults().empty()) return;
      AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(sim.get_mpi_communicator())==1,
                  ExcMessage("K3 frozen seam audit is 2-D, one-rank only."));
      const std::string source=std::getenv("ASPECT_SOURCE_DIR");
      const std::string data=source+"/benchmarks/reconstructed_fault/uniform_shear/evolving/spatial0375_n128_f32/";
      const auto initial=read_csv(data+"particles_0.csv");
      const auto phi_bar=average_profile(read_csv(data+"phase_0.csv"),2,3);
      const auto H_bar=average_profile(initial,2,4);
      auto &pm=const_cast<Particle::Manager<dim>&>(sim.get_phase_field_handler().get_associated_particle_manager());
      auto &particles=pm.get_particle_handler();
      auto &domains=const_cast<Particle::ParticleDomainHandler<dim>&>(pm.get_particle_domain_handler());
      std::map<types::particle_index,Point<dim>> original;
      for (const auto &p : particles) original[p.get_id()]=p.get_location();
      std::ofstream volumes(sim.get_output_directory()+"domain_recovery.csv");
      volumes << std::setprecision(17) << "snapshot,max_saved_volume_relative_error,total_volume\n";

      // Rebuild only disposable diagnostic geometry, with the production
      // generator. Compare stable-ID volumes against each saved actual state.
      for (const unsigned int step : {0u,2u,3u})
        {
          const auto rows=read_csv(data+"particles_"+std::to_string(step)+".csv");
          std::map<types::particle_index,std::vector<double>> saved;
          for (const auto &row : rows) saved[static_cast<types::particle_index>(row[0])]=row;
          AssertThrow(saved.size()==original.size(),ExcMessage("Saved particle set differs."));
          for (auto &p : particles)
            {
              auto location=p.get_location();
              location[0]=saved.at(p.get_id())[1]; location[1]=saved.at(p.get_id())[2];
              p.set_location(location);
            }
          particles.sort_particles_into_subdomains_and_cells();
          domains.generate_particle_domains();
          double max_error=0,total=0;
          for (const auto &p : particles)
            {
              const double volume=domains.get_particle_domain(p.get_local_index()).volume();
              max_error=std::max(max_error,std::abs(volume/saved.at(p.get_id())[3]-1));
              total+=volume;
            }
          volumes << step << ',' << max_error << ',' << total << std::endl;
          // The baseline recovers old wall-bounded volumes exactly. The approved
          // periodic revision deliberately changes them; retain that difference.
          if (!std::getenv("K3_CORRECTED_PERIODIC_AUDIT"))
            AssertThrow(max_error<1e-10,ExcMessage("Regenerated domains do not reproduce saved volumes."));
          probe(sim,"state"+std::to_string(step),phi_bar,H_bar,false);
          bool restored=false;
          try { probe(sim,"failure",phi_bar,H_bar,true); }
          catch (const K3::ForcedProbeFailure &) { restored=true; }
          AssertThrow(restored,ExcMessage("Missing forced-failure restoration test."));
        }
      for (auto &p : particles) p.set_location(original.at(p.get_id()));
      particles.sort_particles_into_subdomains_and_cells();
      domains.generate_particle_domains();
      sim.get_pcout() << "K3_SEAM_AUDIT_COMPLETE: production residual, split assembly and normal/failure restoration verified; no mechanical solve." << std::endl;
      AssertThrow(false,ExcMessage("K3_SEAM_AUDIT_COMPLETE"));
    }
  }

  template <int dim>
  void connect_k3_seam_audit(SimulatorSignals<dim> &signals)
  { signals.set_assemblers.connect(&K3SeamAudit::run<dim>); }

  namespace k3_seam_audit_registration
  { ASPECT_REGISTER_SIGNALS_CONNECTOR(connect_k3_seam_audit<2>,connect_k3_seam_audit<3>) }
}
