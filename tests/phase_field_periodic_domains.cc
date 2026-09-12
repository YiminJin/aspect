// Frozen production-residual regression: images have no particle/history state.
#include "../benchmarks/reconstructed_fault/uniform_shear/uniform_shear.cc"

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyPeriodicPhaseDomains : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          AssertThrow(dim==2,ExcMessage("Periodic phase-domain regression is 2-D."));
          auto &handler=this->get_phase_field_handler();
          auto &pm=handler.get_associated_particle_manager();
          auto &particles=pm.get_particle_handler();
          auto &domains=const_cast<Particle::ParticleDomainHandler<dim>&>(pm.get_particle_domain_handler());
          const auto H_index=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          const auto &dofs=internal::PhaseFieldTestAccess<dim>::vertex_dofs(handler);
          const auto &vertices=this->get_triangulation().get_vertices();
          const auto block=this->introspection().variable("phase_field").block_index;
          const auto comm=this->get_mpi_communicator();
          constexpr double length=.25;
          unsigned int nx=0, ny=0, remote_faces=0;
          for (const auto &cell : this->get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                nx=std::lround(length/(cell->vertex(1)[0]-cell->vertex(0)[0]));
                ny=std::lround(1/(cell->vertex(2)[1]-cell->vertex(0)[1]));
                for (const auto face : cell->face_indices())
                  if (cell->has_periodic_neighbor(face) && cell->periodic_neighbor(face)->is_ghost()) ++remote_faces;
              }
          const double hx=length/nx, pitch=hx/3;
          remote_faces=Utilities::MPI::sum(remote_faces,comm);
          this->get_pcout() << "Periodic regression mesh " << nx << 'x' << ny
                           << ", remote periodic faces=" << remote_faces << std::endl;
          if (std::getenv("K3_REQUIRE_REMOTE_PERIODIC_IMAGES"))
            AssertThrow(remote_faces>0,ExcMessage("Regression did not place periodic images across MPI ranks."));

          // A nonlinear transverse input is decisive; constants alone cannot
          // distinguish a wall-bounded partition from a periodic one.
          LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
          owned=this->get_solution();
          for (unsigned int v=0; v<dofs.size(); ++v)
            if (dofs[v]!=numbers::invalid_dof_index && owned.locally_owned_elements().is_element(dofs[v]))
              owned[dofs[v]]=.2+.15*std::cos(2*numbers::PI*vertices[v][1]);
          owned.compress(VectorOperation::insert);
          this->get_current_constraints().distribute(owned);
          LinearAlgebra::BlockVector state(this->get_solution());
          state=owned;
          LinearAlgebra::BlockSparseMatrix matrix;
          matrix.reinit(this->get_system_matrix().n_block_rows(),this->get_system_matrix().n_block_cols());
          matrix.block(block,block).copy_from(this->get_system_matrix().block(block,block));
          LinearAlgebra::BlockVector rhs(owned);

          // Uniform reference positions are recoverable from each parent's y
          // and its current x by undoing the preceding prescribed displacement.
          double previous_rho=0;
          for (const double rho : {0.0,.12,.18,.8})
            {
              unsigned int local_wraps=0;
              for (auto &p : particles)
                {
                  auto point=p.get_location();
                  point[0]+=(rho-previous_rho)*pitch*(.7+.3*std::sin(2*numbers::PI*point[1]));
                  if (point[0]<0 || point[0]>=length) ++local_wraps;
                  point[0]-=length*std::floor(point[0]/length);
                  p.set_location(point);
                }
              particles.sort_particles_into_subdomains_and_cells();
              particles.exchange_ghost_particles();
              domains.generate_particle_domains();
              previous_rho=rho;

              const auto before=K3::fingerprint(*this);
              struct Restore
              {
                std::vector<std::pair<double*,double>> values;
                ~Restore() { for (const auto &value : values) *value.first=value.second; }
              };
              std::vector<std::pair<double*,double>> original;
              for (auto &p : particles) original.emplace_back(&p.get_properties()[H_index],p.get_properties()[H_index]);
              {
                Restore restore{std::move(original)};
                for (auto &p : particles)
                  p.get_properties()[H_index]=1+.2*std::cos(4*numbers::PI*p.get_location()[1]);
                AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler,matrix,rhs,state,false),
                            ExcMessage("Periodic nonlinear phase input is inadmissible."));
              }
              AssertThrow(before==K3::fingerprint(*this),ExcMessage("Periodic phase probe modified live state."));

              double local_measure=0, constant_error=0, gradient_error=0, first_error=0, fragment_error=0;
              unsigned int local_split=0;
              std::vector<Point<2>> fault_nodes;
              for (unsigned int i=0; i<=nx; ++i) fault_nodes.emplace_back(i*hx,0);
              const ReconstructedFault<2> open_fault(fault_nodes);
              std::vector<double> local_mass((nx+1)*(nx+1),0), surface_mass(local_mass.size());
              for (const auto &p : particles)
                {
                  const auto domain=domains.get_particle_domain(p.get_local_index());
                  local_measure+=domain.volume();
                  const auto &polygon=domain.vertices();
                  const auto area_centroid=[](const std::vector<Point<dim>> &points)
                  {
                    double twice_area=0;
                    Point<dim> centroid;
                    const Point<dim> origin=points.front();
                    for (unsigned int j=0; j<points.size(); ++j)
                      {
                        const auto a=points[j]-origin, b=points[(j+1)%points.size()]-origin;
                        const double cross=a[0]*b[1]-a[1]*b[0];
                        twice_area+=cross;
                        centroid+=Point<dim>((a+b)*cross);
                      }
                    centroid/=3*twice_area;
                    return std::make_pair(std::abs(twice_area)/2,centroid+origin);
                  };
                  const auto geometry=area_centroid(polygon);
                  double sum=0;
                  Tensor<1,dim> gradient, first;
                  for (unsigned int i=0; i<domain.n_relevant_vertices(); ++i)
                    {
                      const auto v=domain.relevant_vertex_index(i);
                      auto point=vertices[v];
                      point[0]-=length*std::round((point[0]-p.get_location()[0])/length);
                      sum+=domain.weighting_function_value(i);
                      gradient+=domain.weighting_function_gradient(i);
                      first+=domain.weighting_function_value(i)*Tensor<1,dim>(point);
                    }
                  constant_error=std::max(constant_error,std::abs(sum-1));
                  gradient_error=std::max(gradient_error,gradient.norm());
                  first_error=std::max(first_error,(first-Tensor<1,dim>(geometry.second)).norm());
                  double fragment_measure=0;
                  if (domain.periodic_fragments().empty()) fragment_measure=geometry.first;
                  else
                    {
                      ++local_split;
                      for (const auto &piece : domain.periodic_fragments())
                        {
                          for (const auto &point : piece)
                            AssertThrow(point[0]>=-1e-14 && point[0]<=length+1e-14,
                                        ExcMessage("Periodic fragment lies outside physical box."));
                          fragment_measure+=area_centroid(piece).first;
                        }
                    }
                  fragment_error=std::max(fragment_error,std::abs(fragment_measure/domain.volume()-1));
                  // Fragment integration must recover the physical-strip Q1
                  // mass without creating cyclic coupling of the open tips.
                  if constexpr (dim==2)
                    {
                      const auto integrate = [&](const std::vector<Point<2>> &piece)
                      {
                        for (const auto &q : ReconstructedFaultUtilities::domain_quadrature(piece,open_fault))
                          {
                            const double N[2]={1-q.xi,q.xi};
                            for (unsigned int a=0; a<2; ++a)
                              for (unsigned int b=0; b<2; ++b)
                                local_mass[(q.segment_index+a)*(nx+1)+q.segment_index+b]+=q.weight*N[a]*N[b];
                          }
                      };
                      if (domain.periodic_fragments().empty()) integrate(polygon);
                      else for (const auto &piece : domain.periodic_fragments()) integrate(piece);
                    }
                }
              Utilities::MPI::sum(local_mass,comm,surface_mass);
              double surface_mass_error=0;
              for (unsigned int a=0; a<=nx; ++a)
                for (unsigned int b=0; b<=nx; ++b)
                  {
                    const double exact=a==b ? hx*(a==0 || a==nx ? 1 : 2)/3
                                       : (a+1==b || b+1==a ? hx/6 : 0);
                    surface_mass_error=std::max(surface_mass_error,std::abs(surface_mass[a*(nx+1)+b]-exact));
                  }
              AssertThrow(surface_mass_error<1e-12 && surface_mass[nx]==0,
                          ExcMessage("Periodic fragments changed the physical-strip mass or joined the open tips."));

              // Compare canonical independent nodal loads, not zeroed periodic
              // slaves. MPI ADD gives one contribution for every owned DoF.
              std::vector<double> local(nx*(ny+1),0), loads(local.size()), counts(local.size()), global_counts(local.size());
              std::set<types::global_dof_index> seen;
              for (unsigned int v=0; v<dofs.size(); ++v)
                if (dofs[v]!=numbers::invalid_dof_index && rhs.locally_owned_elements().is_element(dofs[v])
                    && !this->get_current_constraints().is_constrained(dofs[v]) && seen.insert(dofs[v]).second)
                  {
                    const unsigned int ix=std::lround(vertices[v][0]/hx)%nx;
                    const unsigned int iy=std::lround((vertices[v][1]+.5)*ny);
                    local[iy*nx+ix]=rhs[dofs[v]];
                    counts[iy*nx+ix]=1;
                  }
              Utilities::MPI::sum(local,comm,loads);
              Utilities::MPI::sum(counts,comm,global_counts);
              double column_error=0;
              for (unsigned int j=0; j<=ny; ++j)
                for (unsigned int i=0; i<nx; ++i)
                  {
                    AssertThrow(global_counts[j*nx+i]==1,ExcMessage("Periodic nodal request was lost or duplicated."));
                    column_error=std::max(column_error,std::abs(loads[j*nx+i]-loads[j*nx]));
                  }
              const double measure=Utilities::MPI::sum(local_measure,comm);
              constant_error=Utilities::MPI::max(constant_error,comm);
              gradient_error=Utilities::MPI::max(gradient_error,comm);
              first_error=Utilities::MPI::max(first_error,comm);
              fragment_error=Utilities::MPI::max(fragment_error,comm);
              const auto splits=Utilities::MPI::sum(local_split,comm), wraps=Utilities::MPI::sum(local_wraps,comm);
              this->get_pcout() << std::setprecision(17) << "Periodic phase domains: rho=" << rho
                << " measure=" << measure << " constant=" << constant_error << " gradient=" << gradient_error
                << " first=" << first_error << " fragments=" << fragment_error << " column=" << column_error
                << " surface_mass=" << surface_mass_error
                << " split parents=" << splits << " wraps=" << wraps << std::endl;
              AssertThrow(std::abs(measure-.25)<1e-12 && constant_error<1e-12 && gradient_error<1e-10
                          && first_error<1e-12 && fragment_error<1e-11 && column_error<1e-10,
                          ExcMessage("Periodic domain/production phase-load invariance failed."));
              if (rho==.8) AssertThrow(wraps>0 && splits>0,ExcMessage("Wrapped regression state was not exercised."));
              if (Utilities::MPI::this_mpi_process(comm)==0)
                {
                  std::ofstream out(this->get_output_directory()+"periodic_phase_"+std::to_string(rho)+".csv");
                  out << std::setprecision(17) << "node,rhs\n";
                  for (unsigned int i=0; i<loads.size(); ++i) out << i << ',' << loads[i] << '\n';
                }
            }
          // Restore the disposable geometry as well; no physical history or
          // Maxwell update is evaluated during this frozen test.
          for (auto &p : particles)
            {
              auto point=p.get_location();
              point[0]-=previous_rho*pitch*(.7+.3*std::sin(2*numbers::PI*point[1]));
              point[0]-=length*std::floor(point[0]/length);
              p.set_location(point);
            }
          particles.sort_particles_into_subdomains_and_cells();
          particles.exchange_ghost_particles();
          domains.generate_particle_domains();
          return {"Periodic production phase weak load:","verified"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyPeriodicPhaseDomains,"verify periodic phase domains",
                                  "Frozen nonlinear periodic phase/domain invariance.")
  }
}
