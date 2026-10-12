/* Restricted native P2 total-energy model. Existing particle history is unchanged. */
#include <aspect/material_model/phase_field_fault.h>
#include <aspect/geometry_model/box.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/particle_domain.h>
#include <aspect/plugins.h>
#include <aspect/utilities.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/gravity_model/interface.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace aspect
{
  namespace
  {
    // Second derivatives of the two endpoint values of a linear normal ray.
    // This small jet differentiates the analytic integral, not quadrature samples.
    struct RayJet
    {
      double v;
      std::array<double,2> d{};
      std::array<std::array<double,2>,2> h{};
      RayJet(double value=0.) : v(value) {}
      static RayJet variable(double value,unsigned int i)
      { RayJet x(value);x.d[i]=1.;return x; }
    };
    RayJet operator+(const RayJet &a,const RayJet &b)
    {
      RayJet r(a.v+b.v);
      for(unsigned int i=0;i<2;++i) {r.d[i]=a.d[i]+b.d[i];
        for(unsigned int j=0;j<2;++j)r.h[i][j]=a.h[i][j]+b.h[i][j];}
      return r;
    }
    RayJet operator*(const RayJet &a,const RayJet &b)
    {
      RayJet r(a.v*b.v);
      for(unsigned int i=0;i<2;++i) {r.d[i]=a.d[i]*b.v+a.v*b.d[i];
        for(unsigned int j=0;j<2;++j)r.h[i][j]=a.h[i][j]*b.v+a.v*b.h[i][j]
          +a.d[i]*b.d[j]+b.d[i]*a.d[j];}
      return r;
    }
    RayJet compose(const RayJet &x,double v,double first,double second)
    {
      RayJet r(v);
      for(unsigned int i=0;i<2;++i) {r.d[i]=first*x.d[i];
        for(unsigned int j=0;j<2;++j)r.h[i][j]=first*x.h[i][j]+second*x.d[i]*x.d[j];}
      return r;
    }
    RayJet inverse(const RayJet &x)
    {return compose(x,1./x.v,-1./(x.v*x.v),2./(x.v*x.v*x.v));}
    RayJet operator-(const RayJet &a,const RayJet &b) {return a+RayJet(-1.)*b;}
    RayJet operator/(const RayJet &a,const RayJet &b) {return a*inverse(b);}
    RayJet ray_integral(double left,double right,double length,double m)
    {
      const auto a=RayJet(1.)-RayJet::variable(left,0);
      const auto b=RayJet(1.)-RayJet::variable(right,1);
      const auto z=(b-a)/a;
      RayJet ratio;
      if(std::abs(z.v)<1.e-3)
        {
          RayJet power(1.);
          for(unsigned int k=0;k<13;++k) {ratio=ratio+RayJet((k%2?-1.:1.)/(k+1))*power;power=power*z;}
        }
      else
        ratio=compose(z,std::log1p(z.v),1./(1.+z.v),-1./std::pow(1.+z.v,2))/z;
      return RayJet(m*length)*(RayJet(2.)/(a*b)-RayJet(3.)*ratio/a+RayJet(1.));
    }

  }

  namespace MaterialModel
  {
    template<int dim> bool PhaseFieldFault<dim>::initialize_fe_shared_phase(LinearAlgebra::BlockVector &trial) const
    {
      if(fe_initial_phase_profile.empty())return false;
      AssertThrow(consistent_endpoint_phase && this->get_timestep_number()==0
                  && !this->get_parameters().resume_computation && !fe_strain_state,
                  ExcMessage("Shared profile is fresh pre-existing damage; history conversion is prohibited."));
      std::ifstream input(fe_initial_phase_profile);
      AssertThrow(bool(input),ExcMessage("Cannot open FE initial phase profile."));
      std::vector<std::pair<double,double>> data;double y,value;
      while(input>>y>>value)
        {
          AssertThrow(std::isfinite(y) && value>=0. && value<1.
                      && (data.empty() || y>data.back().first),ExcMessage("Invalid shared phase data."));
          data.emplace_back(y,value);
        }
      AssertThrow(input.eof() && data.size()>=2,ExcMessage("Malformed shared phase data."));
      LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
      owned=trial;
      const auto component=this->introspection().variable("phase_field").first_component_index;
      std::vector<types::global_dof_index> indices(this->get_fe().n_dofs_per_cell());
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {
        cell->get_dof_indices(indices);
        for(unsigned int v=0;v<GeometryInfo<dim>::vertices_per_cell;++v)
          {
            const auto i=indices[this->get_fe().component_to_system_index(component,v)];const double yy=cell->vertex(v)[1];
            AssertThrow(yy>=data.front().first && yy<=data.back().first,
                        ExcMessage("Shared phase profile does not cover the domain."));
            auto right=std::upper_bound(data.begin(),data.end(),yy,[](double x,const auto &p){return x<p.first;});
            if(right==data.end())owned[i]=data.back().second;
            else if(right==data.begin())owned[i]=right->second;
            else {const auto &left=*(right-1);const double t=(yy-left.first)/(right->first-left.first);
                  owned[i]=(1-t)*left.second+t*right->second;}
          }
        }
      owned.compress(VectorOperation::insert);trial=owned;
      this->get_pcout()<<"   Accepted supplied continuum phase seed; native particle H retained, unloaded FE histories constructed afterward."<<std::endl;
      return true;
    }
    template<int dim> struct PhaseFieldFault<dim>::FEStrainState
    {
      struct Ray
      {
        std::array<types::global_dof_index,GeometryInfo<dim>::vertices_per_cell> dofs;
        std::array<std::array<double,GeometryInfo<dim>::vertices_per_cell>,2> shape;
        double length;
      };
      struct Profile
      {
        unsigned int segment;double xi,weight;
        std::vector<Ray> rays;
      };
      struct BulkPoint
      {
        CellId cell;Point<dim> x;double weight,xi;
        unsigned int segment;
        std::array<types::global_dof_index,GeometryInfo<dim>::vertices_per_cell> dofs;
        std::array<double,GeometryInfo<dim>::vertices_per_cell> shape;
      };
      LinearAlgebra::BlockVector accepted;
      std::vector<double> Z,I;
      std::vector<Profile> profiles;
      std::vector<BulkPoint> bulk;
      FullMatrix<double> projection;
      std::map<CellId,typename DoFHandler<dim>::active_cell_iterator> cells;
      std::map<types::particle_index,Point<dim>> particles;
      std::vector<Point<dim>> vertices;
      std::vector<Point<dim>> mesh_vertices;
      // Derived point samples of the immutable FE anchor. This cache never
      // transfers samples back into d_R and is invalidated at publication.
      mutable std::map<std::pair<CellId,std::array<double,dim>>,SymmetricTensor<2,dim>> stress_samples;
      std::uint64_t geometry_version;
      types::global_dof_index n_dofs;
      double m,G,eta,initial_fracture=0.;
      unsigned int committed_step=0;
    };

    template<int dim> struct PhaseFieldFault<dim>::FEStrainUpdate
    {
      LinearAlgebra::BlockVector accepted;
      std::vector<double>Z,I;
    };

    template<int dim> void PhaseFieldFault<dim>::set_fe_diagnostic_reference(
      const LinearAlgebra::BlockVector &reference,const std::vector<double> &Z,const std::vector<double> &I)
    {
      AssertThrow(consistent_endpoint_phase && fe_strain_state && this->get_timestep_number()>0
                  && Z.size()==fe_strain_state->Z.size()
                  && I.size()==Z.size(),ExcMessage("Invalid restricted fixed-entry reference."));
      for(unsigned int i=0;i<I.size();++i)
        AssertThrow(std::isfinite(Z[i]) && std::isfinite(I[i]) && I[i]>0.,ExcMessage("Undefined diagnostic surface history."));
      auto trial=std::make_shared<FEStrainState>(*fe_strain_state);
      trial->accepted=reference;trial->Z=Z;trial->I=I;trial->stress_samples.clear();
      trial->committed_step=this->get_timestep_number()-1;
      fe_strain_state=std::move(trial);
    }

    template<int dim> Quadrature<dim> PhaseFieldFault<dim>::mechanical_quadrature() const
    {
      if(!fe_strain_history)return this->introspection().quadratures.velocities;
      AssertThrow(dim==2,ExcNotImplemented());
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const auto extent=box.get_extents();
      const auto cell=this->get_dof_handler().begin_active();
      const double dx=cell->vertex(1)[0]-cell->vertex(0)[0];
      const double dy=cell->vertex(2)[1]-cell->vertex(0)[1];
      const unsigned int n=std::lround(extent[0]/dx), ny=std::lround(extent[1]/dy);
      const auto &faults=this->get_reconstructed_fault_manager().get_faults();
      // Before initial geometry construction no mechanical assembly is allowed.
      AssertThrow(n*ny==this->get_triangulation().n_global_active_cells() && faults.size()==1,
                  ExcMessage("FE strain quadrature requires a uniform Cartesian grid and one fault."));
      const unsigned int ns=faults[0].n_cells();
      const unsigned int panels=std::lcm(n,ns)/n;
      // One rule supplies mechanics, the slip map and its adjoint, and the
      // bulk part of the phase functional. Normalization/CPDI stay separate.
      // Resolve tangential polynomial products as well as sharp Q1 panels.
      const unsigned int tangential_points = std::max(3u,this->get_fe().degree+1);
      QIterated<1>x(QGauss<1>(tangential_points),panels);
      QGauss<1>y(fe_bulk_normal_quadrature_points);
      std::vector<Point<dim>>points;std::vector<double>weights;
      for(unsigned int i=0;i<x.size();++i)for(unsigned int j=0;j<y.size();++j)
        {Point<dim>p;p[0]=x.point(i)[0];p[1]=y.point(j)[0];points.push_back(p);weights.push_back(x.weight(i)*y.weight(j));}
      return Quadrature<dim>(points,weights);
    }

    template<int dim> const LinearAlgebra::BlockVector &
    PhaseFieldFault<dim>::accepted_fe_strain_state() const
    {
      AssertThrow(fe_strain_state,ExcMessage("FE strain history has not been initialized."));
      return fe_strain_state->accepted;
    }

    template<int dim> void PhaseFieldFault<dim>::validate_fe_strain_restrictions() const
    {
      AssertThrow(!benchmark_retained_stress && fe_strain_state,
                  ExcMessage("FE strain cannot consume an external stress override or missing accepted state."));
      const auto&s=*fe_strain_state;const auto &faults=this->get_reconstructed_fault_manager().get_faults();
      AssertThrow(this->get_dof_handler().n_dofs()==s.n_dofs && faults.size()==1
                  && faults[0].geometry_version()==s.geometry_version && faults[0].n_vertices()==s.vertices.size()
                  && this->get_triangulation().n_global_active_cells()==s.cells.size(),
                  ExcMessage("Accepted FE strain rejects mesh/geometry changes."));
      for(unsigned int i=0;i<s.vertices.size();++i)
        AssertThrow(faults[0].vertex(i)==s.vertices[i],ExcMessage("Accepted FE strain geometry changed."));
      AssertThrow(this->get_triangulation().get_vertices()==s.mesh_vertices,
                  ExcMessage("Accepted FE strain rejects mesh motion."));
      const auto &particles=this->get_phase_field_handler().get_associated_particle_manager().get_particle_handler();
      AssertThrow(particles.n_global_particles()==s.particles.size(),ExcMessage("Accepted FE strain rejects particle birth/deletion."));
      for(const auto&p:particles)
        AssertThrow(s.particles.count(p.get_id()) && s.particles.at(p.get_id())==p.get_location(),
                    ExcMessage("Accepted FE strain rejects history transport."));
    }

    template<int dim> void PhaseFieldFault<dim>::prepare_fe_strain_history()
    {
      if(!fe_strain_history)return;
      if(fe_strain_state) {validate_fe_strain_restrictions();return;}
      AssertThrow(this->get_timestep_number()==0 && !this->get_parameters().resume_computation
                  && Plugins::plugin_type_matches<GeometryModel::Box<dim>>(this->get_geometry_model())
                  && !benchmark_retained_stress && normalization_surface_subdivisions==1,
                  ExcMessage("FE strain initializes only on a fresh fixed Cartesian Box without stress overrides."));
      auto state=std::make_shared<FEStrainState>();
      state->accepted.reinit(this->get_solution());state->accepted=0.;
      const auto &intro=this->introspection();const unsigned int b=intro.variable("phase_field").block_index;
      state->accepted.block(b)=this->get_solution().block(b);
      const auto phase_offset=state->accepted.get_block_indices().block_start(b);
      for(auto i:state->accepted.block(b).locally_owned_elements())
        AssertThrow(!this->get_current_constraints().is_constrained(phase_offset+i),
                    ExcMessage("Accepted FE strain currently requires unconstrained Q1 phase degrees of freedom."));
      state->mesh_vertices=this->get_triangulation().get_vertices();
      state->G=elastic_shear_moduli[0];state->eta=reference_viscosities[0];
      state->n_dofs=this->get_dof_handler().n_dofs();
      const auto &fault=this->get_reconstructed_fault_manager().get_fault(0);
      state->geometry_version=fault.geometry_version();
      const unsigned int ns=fault.n_cells();
      for(unsigned int i=0;i<fault.n_vertices();++i)
        {
          AssertThrow(std::abs(fault.vertex(i)[0]-double(i)/ns)<1.e-13
                      && fault.vertex(i)[1]==.5,ExcMessage("FE strain requires a unit horizontal through-going fault."));
          state->vertices.push_back(fault.vertex(i));
        }
      const auto &pm=this->get_phase_field_handler().get_associated_particle_manager();
      const auto sp=pm.get_property_manager().get_data_info().get_position_by_field_name("maxwell stress");
      for(const auto&p:pm.get_particle_handler())
        {
          for(unsigned int c=0;c<SymmetricTensor<2,dim>::n_independent_components;++c)
            AssertThrow(p.get_properties()[sp+c]==0.,ExcMessage("FE strain requires unloaded initial particle stress; no conversion is performed."));
          state->particles[p.get_id()]=p.get_location();
        }
      const auto fractions=std::vector<double>(elastic_shear_moduli.size(),1./elastic_shear_moduli.size());
      state->m=(1./this->get_phase_field_handler().energetic_degradation(fractions,.5)-1.)/3.;
      const auto &box=Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(this->get_geometry_model());
      const auto extent=box.get_extents(),origin=box.get_origin();
      AssertThrow(extent[0]==1. && origin[0]==0. && extent[1]>0.
                  && std::abs(origin[1]+extent[1]/2-.5)<1.e-13,
                  ExcMessage("FE strain requires unit tangential width and a normal Box centered on y=0.5."));
      const auto profiles=build_owned_normalization_profiles(true);
      prepare_cell_normalization_geometry(profiles);
      AssertThrow(normalization_cell_cache.supported,ExcMessage("FE strain requires affine normal-ray cell geometry."));
      const auto offset=state->accepted.get_block_indices().block_start(b);
      for(unsigned int k=0;k<profiles.size();++k)
        {
          typename FEStrainState::Profile p;
          p.segment=profiles[k].segment_index;p.xi=profiles[k].xi;p.weight=profiles[k].surface_weight;
          double covered=0.;
          for(const auto &cell:normalization_cell_cache.profiles[k].intervals)
            {
              typename FEStrainState::Ray ray;ray.length=cell.upper-cell.lower;covered+=ray.length;
              for(unsigned int v=0;v<ray.dofs.size();++v)
                {
                  ray.dofs[v]=cell.phase_dofs[v]-offset;
                  for(unsigned int end=0;end<2;++end)
                    {
                      const auto unit=cell.reference_origin+(end?cell.upper:cell.lower)*cell.reference_direction;
                      double shape=1.;for(unsigned int d=0;d<dim;++d)shape*=v&(1u<<d)?unit[d]:1.-unit[d];
                      ray.shape[end][v]=shape;
                    }
                }
              p.rays.push_back(ray);
            }
          AssertThrow(std::abs(covered-extent[1])<1.e-12*std::max(1.,extent[1]),
                      ExcMessage("FE strain normal integration must cover the full Box cross-section."));
          state->profiles.push_back(p);
        }
      FullMatrix<double>mass(ns+1,ns+1),rhs(ns+1,profiles.size());
      for(unsigned int k=0;k<profiles.size();++k)
        {
          const auto&p=state->profiles[k];const double weights[2]={1-p.xi,p.xi};
          for(unsigned int i=0;i<2;++i)
            {rhs(p.segment+i,k)=p.weight*weights[i];
             for(unsigned int j=0;j<2;++j)mass(p.segment+i,p.segment+j)+=p.weight*weights[i]*weights[j];}
        }
      mass.gauss_jordan();state->projection.reinit(ns+1,profiles.size());mass.mmult(state->projection,rhs);
      const auto quad=mechanical_quadrature();
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_values|update_quadrature_points|update_JxW_values);
      std::vector<types::global_dof_index>ids(this->get_fe().dofs_per_cell);
      const auto pc=intro.variable("phase_field").first_component_index;
      const auto first=this->get_dof_handler().begin_active();
      std::array<double,dim> spacing{};
      spacing[0]=first->vertex(1)[0]-first->vertex(0)[0];
      spacing[1]=first->vertex(2)[1]-first->vertex(0)[1];
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {
          AssertThrow(cell->is_locally_owned(),ExcMessage("FE strain is serial."));
          fe.reinit(cell);cell->get_dof_indices(ids);state->cells.emplace(cell->id(),cell);
          for(const auto v:cell->vertex_indices())for(unsigned int d=0;d<dim;++d)
            {
              const double coordinate=(cell->vertex(v)[d]-origin[d])/spacing[d];
              AssertThrow(std::abs(coordinate-std::round(coordinate))<1.e-12
                          && cell->vertex(v)[d]>=origin[d]
                          && cell->vertex(v)[d]<=origin[d]+extent[d],
                          ExcMessage("FE strain requires a fixed uniform Cartesian Box mesh."));
            }
          AssertThrow(std::abs(cell->vertex(1)[0]-cell->vertex(0)[0]-spacing[0])<1.e-13
                      && std::abs(cell->vertex(2)[1]-cell->vertex(0)[1]-spacing[1])<1.e-13,
                      ExcMessage("FE strain cell sizes must be uniform."));
          for(unsigned int q=0;q<quad.size();++q)
            {
              typename FEStrainState::BulkPoint p;
              p.cell=cell->id();p.x=fe.quadrature_point(q);p.weight=fe.JxW(q);
              if(consistent_endpoint_phase)
                AssertThrow(this->get_gravity_model().gravity_vector(p.x).norm()==0.,
                            ExcMessage("Endpoint reduced potential requires the unloaded verification fixture without body forces."));
              p.segment=std::min(ns-1,static_cast<unsigned int>(p.x[0]*ns));p.xi=p.x[0]*ns-p.segment;
              for(unsigned int v=0;v<p.dofs.size();++v)
                {const auto j=this->get_fe().component_to_system_index(pc,v);p.dofs[v]=ids[j]-offset;
                 p.shape[v]=fe.shape_value_component(j,q,pc);}
              state->bulk.push_back(p);
            }
        }
      std::stable_sort(state->bulk.begin(),state->bulk.end(),
                       [](const auto&a,const auto&b){return a.cell<b.cell;});
      state->Z.assign(ns+1,0.);state->I=current_normalization_integrals.at(0);
      fe_strain_state=state;
      const auto initial=evaluate_fe_phase_energy(this->get_solution(),false);
      AssertThrow(initial.admissible,ExcMessage("Initial FE strain normalization is undefined."));
      current_normalization_integrals[0]=initial.normalization;state->I=initial.normalization;
      validate_cohesive_state_commit({state->Z});commit_cohesive_state({state->Z});
      this->get_pcout()<<std::setprecision(17)<<"   Accepted FE strain: G="<<state->G
                       <<" eta="<<state->eta<<" lambda="<<state->eta/state->G<<" m="<<state->m
                       <<" bulk QPs="<<state->bulk.size()<<" phase DoFs="<<state->accepted.block(b).size()<<'\n';
    }

    template<int dim> typename PhaseFieldFault<dim>::FEPhaseEnergy
    PhaseFieldFault<dim>::evaluate_fe_phase_energy(const LinearAlgebra::BlockVector &trial,
                                                  bool tangent,bool positive_search) const
    {
      validate_fe_strain_restrictions();
      return evaluate_fe_phase_energy_at(trial,tangent,positive_search,
                                         fe_strain_state->accepted,fe_strain_state->Z);
    }

    template<int dim> typename PhaseFieldFault<dim>::FEPhaseEnergy
    PhaseFieldFault<dim>::evaluate_fe_endpoint_energy(const LinearAlgebra::BlockVector &trial,
                                                     const FEEndpointTrial &endpoint,
                                                     const std::vector<double> *phase_direction) const
    {
      return evaluate_fe_phase_energy_at(trial,false,false,endpoint.strain,endpoint.Z,phase_direction);
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_endpoint_stress(
      const FEEndpointTrial &endpoint,const std::vector<double> &I,const CellId &cell,const Point<dim> &x) const
    {
      validate_fe_strain_restrictions();
      return evaluate_fe_strain_stress(endpoint.strain,endpoint.Z,I,cell,x);
    }

    template<int dim> typename PhaseFieldFault<dim>::FEEndpointTrial
    PhaseFieldFault<dim>::make_fe_endpoint_trial(const LinearAlgebra::BlockVector &bulk,
                                                const std::vector<std::vector<double>> &V) const
    {
      validate_fe_strain_restrictions();
      const auto &s=*fe_strain_state;
      const auto c=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
      FEEndpointTrial endpoint;
      endpoint.strain.reinit(s.accepted);endpoint.strain=s.accepted;
      const unsigned int vb=this->introspection().block_indices.velocities;
      endpoint.strain.block(vb)*=c.beta;
      endpoint.strain.block(vb).add(c.eta_ve/s.G,bulk.block(vb));
      const auto pb=this->introspection().variable("phase_field").block_index;
      endpoint.strain.block(pb)=bulk.block(pb);
      endpoint.Z=s.Z;
      long double friction_potential=0.;
      AssertDimension(V.size(),1);AssertDimension(V[0].size(),s.Z.size());
      for(unsigned int i=0;i<s.Z.size();++i)
        endpoint.Z[i]=c.beta*s.Z[i]+c.eta_ve*V[0][i];
      const std::vector<double> fractions(elastic_shear_moduli.size(),1./elastic_shear_moduli.size());
      for(const auto &p:s.profiles)
        {
          const double v=ReconstructedFaultUtilities::interpolate_slip_rate(
            V[0][p.segment],V[0][p.segment+1],p.xi);
          const double mu=fault_friction.friction_coefficient(fractions,v);
          AssertThrow(mu==fault_friction.friction_coefficient(fractions,minimum_fault_slip_rate())
                      && fault_friction.friction_coefficient_derivative_wrt_slip_rate(fractions,v)==0.,
                      ExcMessage("Endpoint variational solve requires fixed Coulomb friction."));
          const auto bg=reconstructed_fault_background_tractions(0,p.segment,p.xi);
          AssertThrow(bg.first==0.,ExcMessage("Endpoint verification prohibits an external shear-traction offset."));
          const Point<dim> x=(1-p.xi)*s.vertices[p.segment]+p.xi*s.vertices[p.segment+1];
          const double pressure=this->get_adiabatic_conditions().pressure(x)+bg.second;
          friction_potential+=static_cast<long double>(p.weight)*c.eta_ve/s.G*mu*pressure*v;
        }
      endpoint.friction_potential=friction_potential;
      return endpoint;
    }

    template<int dim> typename PhaseFieldFault<dim>::FEPhaseEnergy
    PhaseFieldFault<dim>::evaluate_fe_phase_energy_at(const LinearAlgebra::BlockVector &trial,
                                                     bool tangent,bool positive_search,
                                                     const LinearAlgebra::BlockVector &strain,
                                                     const std::vector<double> &Z_reference,
                                                     const std::vector<double> *phase_direction) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(), "Fault: FE phase energy");
      validate_fe_strain_restrictions();const auto&s=*fe_strain_state;
      FEPhaseEnergy result;
      long double bulk_energy=0.,cohesive_energy=0.;
      const unsigned int pb=this->introspection().variable("phase_field").block_index;
      const unsigned int n=trial.block(pb).size(),ns=Z_reference.size();
      AssertDimension(ns,s.Z.size());
      result.gradient.assign(n,0.);result.normalization.assign(ns,0.);result.roundoff_scale.assign(n,0.);
      if(tangent)result.hessian.reinit(n,n);
      for(unsigned int i=0;i<n;++i)
        if(!std::isfinite(trial.block(pb)[i]) || trial.block(pb)[i]<0 || trial.block(pb)[i]>=1.)
          {result.admissible=false;return result;}
      FullMatrix<double>A(s.profiles.size(),n),Iderivative(ns,n),positive_A(s.profiles.size(),n),positive_DI(ns,n);
      std::vector<std::vector<RayJet>>integrals(s.profiles.size());
      for(unsigned int k=0;k<s.profiles.size();++k)
        {
          double direct=0.;const auto&p=s.profiles[k];
          for(const auto&ray:p.rays)
            {
              double endpoint[2]={0.,0.};
              for(unsigned int v=0;v<ray.dofs.size();++v)for(unsigned int e=0;e<2;++e)
                endpoint[e]+=ray.shape[e][v]*trial.block(pb)[ray.dofs[v]];
              const auto f=ray_integral(endpoint[0],endpoint[1],ray.length,s.m);
              integrals[k].push_back(f);direct+=f.v;
              for(unsigned int v=0;v<ray.dofs.size();++v)
                {
                  A(k,ray.dofs[v])+=f.d[0]*ray.shape[0][v]+f.d[1]*ray.shape[1][v];
                  positive_A(k,ray.dofs[v])+=std::abs(f.d[0]*ray.shape[0][v])+std::abs(f.d[1]*ray.shape[1][v]);
                }
            }
          for(unsigned int i=0;i<ns;++i)result.normalization[i]+=s.projection(i,k)*direct;
        }
      for(const auto I:result.normalization)
        if(!(I>0) || !std::isfinite(I)) {result.admissible=false;return result;}
      s.projection.mmult(Iderivative,A);
      if(phase_direction)
        {
          AssertDimension(phase_direction->size(),n);result.normalization_action.assign(ns,0.);
          for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<n;++j)
            result.normalization_action[i]+=Iderivative(i,j)*(*phase_direction)[j];
        }
      // Assemble first in independent nodal (phi,I) coordinates, then apply
      // the normalization pullback once. This is the same full Hessian as
      // per-QP dense rank updates, without repeating nonlocal dense work.
      for(unsigned int i=0;i<ns;++i)for(unsigned int k=0;k<s.profiles.size();++k)
        for(unsigned int j=0;j<n;++j)positive_DI(i,j)+=std::abs(s.projection(i,k))*positive_A(k,j);
      std::vector<double>forceI(ns,0.),scaleI(ns,0.);
      FullMatrix<double>cross,II;
      if(tangent){cross.reinit(n,ns);II.reinit(ns,ns);}
      SymmetricTensor<2,dim>S;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
      const auto quad=mechanical_quadrature();
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),quad,update_gradients);
      std::vector<SymmetricTensor<2,dim>>R(quad.size());
      unsigned int point=0;
      for(const auto&entry:s.cells)
        {
          fe.reinit(entry.second);
          fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(strain,R);
          for(unsigned int q=0;q<quad.size();++q,++point)
            {
              const auto&p=s.bulk[point];AssertThrow(p.cell==entry.first,ExcInternalError());
              double phi=0.;
              for(unsigned int v=0;v<p.dofs.size();++v)phi+=p.shape[v]*trial.block(pb)[p.dofs[v]];
              const double w[2]={1-p.xi,p.xi};
              const double I=w[0]*result.normalization[p.segment]+w[1]*result.normalization[p.segment+1];
              const double Z=w[0]*Z_reference[p.segment]+w[1]*Z_reference[p.segment+1];
              const double h=s.m*phi*(1+phi)/std::pow(1-phi,2);
              const double hp=s.m*(4/std::pow(1-phi,3)-3/std::pow(1-phi,2));
              const double hpp=s.m*(12/std::pow(1-phi,4)-6/std::pow(1-phi,3));
              // R[q] is a raw zero-zz plane-strain gradient. Deviate once.
              const auto tau=2*s.G*Utilities::Tensors::consistent_deviator(R[q])-2*Z*h/I*S;
              result.bulk+=p.weight*((tau*tau)+std::pow(trace(tau),2))/(4*s.G);
              bulk_energy+=static_cast<long double>(p.weight)*((tau*tau)+std::pow(trace(tau),2))/(4*s.G);
              const double c=-Z/s.G*(tau*S);
              const double positive_c=std::abs(Z/s.G)*(2*s.G*std::abs(R[q]*S)+std::abs(2*Z*h/I)*(S*S));
              const double a=2*Z*Z/s.G*(S*S);
              for(unsigned int v=0;v<p.dofs.size();++v)
                {
                  result.gradient[p.dofs[v]]+=p.weight*c*hp/I*p.shape[v];
                  result.roundoff_scale[p.dofs[v]]+=p.weight*positive_c*std::abs(hp/I*p.shape[v]);
                }
              for(unsigned int i=0;i<2;++i)
                {
                  forceI[p.segment+i]-=p.weight*w[i]*c*h/(I*I);
                  scaleI[p.segment+i]+=p.weight*std::abs(w[i])*positive_c*std::abs(h/(I*I));
                }
              if(tangent)
                {
                  const double pp=a*hp*hp/(I*I)+(positive_search?0.:c*hpp/I);
                  const double pi=-a*hp*h/std::pow(I,3)-(positive_search?0.:c*hp/(I*I));
                  const double ii=a*h*h/std::pow(I,4)+(positive_search?0.:2*c*h/std::pow(I,3));
                  for(unsigned int v=0;v<p.dofs.size();++v)
                    {
                      for(unsigned int u=0;u<p.dofs.size();++u)
                        result.hessian(p.dofs[v],p.dofs[u])+=p.weight*pp*p.shape[v]*p.shape[u];
                      for(unsigned int i=0;i<2;++i)
                        cross(p.dofs[v],p.segment+i)+=p.weight*pi*p.shape[v]*w[i];
                    }
                  for(unsigned int i=0;i<2;++i)for(unsigned int j=0;j<2;++j)
                    II(p.segment+i,p.segment+j)+=p.weight*ii*w[i]*w[j];
                }
            }
        }
      for(const auto&p:s.profiles)
        {
          const double w[2]={1-p.xi,p.xi};
          const double I=w[0]*result.normalization[p.segment]+w[1]*result.normalization[p.segment+1];
          const double Z=w[0]*Z_reference[p.segment]+w[1]*Z_reference[p.segment+1];
          result.cohesive+=p.weight*Z*Z/(2*s.G*I);
          cohesive_energy+=static_cast<long double>(p.weight)*Z*Z/(2*s.G*I);
          const double c=-p.weight*Z*Z/(2*s.G*I*I);
          for(unsigned int i=0;i<2;++i)
            {
              forceI[p.segment+i]+=c*w[i];scaleI[p.segment+i]+=std::abs(c*w[i]);
              if(tangent)for(unsigned int j=0;j<2;++j)
                II(p.segment+i,p.segment+j)+=p.weight*Z*Z/(s.G*I*I*I)*w[i]*w[j];
            }
        }
      for(unsigned int j=0;j<n;++j)for(unsigned int i=0;i<ns;++i)
        {
          result.gradient[j]+=Iderivative(i,j)*forceI[i];
          result.roundoff_scale[j]+=positive_DI(i,j)*scaleI[i];
        }
      if(tangent)
        {
          FullMatrix<double>product(n,n),II_DI(ns,n),pullback(n,n);
          cross.mmult(product,Iderivative);II.mmult(II_DI,Iderivative);Iderivative.Tmmult(pullback,II_DI);
          for(unsigned int i=0;i<n;++i)for(unsigned int j=0;j<n;++j)
            result.hessian(i,j)+=product(i,j)+product(j,i)+pullback(i,j);
        }
      if(tangent && !positive_search)
        for(unsigned int k=0;k<s.profiles.size();++k)
          {
            double adjoint=0.;for(unsigned int i=0;i<ns;++i)adjoint+=s.projection(i,k)*forceI[i];
            unsigned int j=0;
            for(const auto&ray:s.profiles[k].rays)
              {
                const auto&f=integrals[k][j++];
                for(unsigned int v=0;v<ray.dofs.size();++v)for(unsigned int u=0;u<ray.dofs.size();++u)
                  for(unsigned int a=0;a<2;++a)for(unsigned int b=0;b<2;++b)
                    result.hessian(ray.dofs[v],ray.dofs[u])+=adjoint*f.h[a][b]*ray.shape[a][v]*ray.shape[b][u];
              }
          }
      this->get_pcout()<<"   FE energy summation: bulk old-new="<<static_cast<double>(result.bulk-bulk_energy)
                       <<", cohesive old-new="<<static_cast<double>(result.cohesive-cohesive_energy)<<std::endl;
      result.bulk=bulk_energy;result.cohesive=cohesive_energy;
      result.energy=result.bulk+result.cohesive;
      result.admissible=std::isfinite(result.energy);
      for(const auto g:result.gradient)result.admissible=result.admissible && std::isfinite(g);
      return result;
    }

    template<int dim> typename PhaseFieldFault<dim>::FEEndpointAction
    PhaseFieldFault<dim>::evaluate_fe_endpoint_action(
      const LinearAlgebra::BlockVector &trial, const FEEndpointTrial &endpoint,
      const std::vector<double> &direction,
      const LinearAlgebra::BlockVector *velocity_direction,
      const std::vector<std::vector<double>> *slip_direction, bool assemble_rhs) const
    {
      TimerOutput::Scope timer(this->get_computing_timer(),"Fault: FE tangent pullback");
      validate_fe_strain_restrictions();
      const auto &s=*fe_strain_state;
      const auto &intro=this->introspection();
      const unsigned int pb=intro.variable("phase_field").block_index;
      const unsigned int n=trial.block(pb).size(),ns=s.Z.size();
      AssertDimension(direction.size(),n);
      AssertThrow((velocity_direction==nullptr)==(slip_direction==nullptr),ExcInternalError());
      const auto coefficients=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
      FEEndpointAction result;result.gradient.assign(n,0.);
      std::vector<double>I(ns,0.),dI(ns,0.),force(ns,0.),dforce(ns,0.);
      std::vector<std::vector<RayJet>> jets(s.profiles.size());
      for(unsigned int k=0;k<s.profiles.size();++k)
        {
          double integral=0.,derivative=0.;
          for(const auto &ray:s.profiles[k].rays)
            {
              double phi[2]={},dphi[2]={};
              for(unsigned int e=0;e<2;++e)for(unsigned int v=0;v<ray.dofs.size();++v)
                {phi[e]+=ray.shape[e][v]*trial.block(pb)[ray.dofs[v]];
                 dphi[e]+=ray.shape[e][v]*direction[ray.dofs[v]];}
              const auto jet=ray_integral(phi[0],phi[1],ray.length,s.m);
              jets[k].push_back(jet);integral+=jet.v;
              derivative+=jet.d[0]*dphi[0]+jet.d[1]*dphi[1];
            }
          for(unsigned int i=0;i<ns;++i)
            {I[i]+=s.projection(i,k)*integral;dI[i]+=s.projection(i,k)*derivative;}
        }
      for(const double value:I)AssertThrow(value>0. && std::isfinite(value),ExcMessage("Undefined tangent normalization."));
      SymmetricTensor<2,dim>S;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
      const auto quadrature=mechanical_quadrature();
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),quadrature,update_gradients);
      std::vector<SymmetricTensor<2,dim>>R(quadrature.size()),du(quadrature.size());
      Vector<double>cell_rhs(this->get_fe().n_dofs_per_cell());
      std::vector<types::global_dof_index>indices(cell_rhs.size());
      AffineConstraints<double>constraints(this->get_current_constraints());
      for(const auto &line:constraints.get_lines())constraints.set_inhomogeneity(line.index,0.);
      if(assemble_rhs)
        {result.bulk_rhs.reinit(intro.index_sets.system_partitioning,this->get_mpi_communicator());
         result.surface_rhs={{}};result.surface_rhs[0].assign(ns,0.);}
      unsigned int point=0;
      for(const auto &entry:s.cells)
        {
          fe.reinit(entry.second);cell_rhs=0.;
          fe[intro.extractors.velocities].get_function_symmetric_gradients(endpoint.strain,R);
          if(velocity_direction)fe[intro.extractors.velocities].get_function_symmetric_gradients(*velocity_direction,du);
          for(unsigned int q=0;q<quadrature.size();++q,++point)
            {
              const auto &p=s.bulk[point];AssertThrow(p.cell==entry.first,ExcInternalError());
              double phi=0.,dp=0.;
              for(unsigned int v=0;v<p.dofs.size();++v)
                {phi+=p.shape[v]*trial.block(pb)[p.dofs[v]];dp+=p.shape[v]*direction[p.dofs[v]];}
              const double w[2]={1-p.xi,p.xi};
              const double ii=w[0]*I[p.segment]+w[1]*I[p.segment+1];
              const double di=w[0]*dI[p.segment]+w[1]*dI[p.segment+1];
              const double Z=w[0]*endpoint.Z[p.segment]+w[1]*endpoint.Z[p.segment+1];
              const double dZ=slip_direction?coefficients.eta_ve*(w[0]*(*slip_direction)[0][p.segment]+w[1]*(*slip_direction)[0][p.segment+1]):0.;
              const double h=s.m*phi*(1+phi)/std::pow(1-phi,2);
              const double hp=s.m*(4/std::pow(1-phi,3)-3/std::pow(1-phi,2));
              const double hpp=s.m*(12/std::pow(1-phi,4)-6/std::pow(1-phi,3));
              const double chi=h/ii,dchi=hp*dp/ii-h*di/(ii*ii);
              const auto tau=2*s.G*Utilities::Tensors::consistent_deviator(R[q])-2*Z*chi*S;
              const auto dtau=(velocity_direction?2*coefficients.eta_ve*Utilities::Tensors::consistent_deviator(du[q]):SymmetricTensor<2,dim>())
                             -2*(dZ*chi+Z*dchi)*S;
              const double c=-Z/s.G*(tau*S),dc=-(dZ*(tau*S)+Z*(dtau*S))/s.G;
              for(unsigned int v=0;v<p.dofs.size();++v)
                result.gradient[p.dofs[v]]+=p.weight*p.shape[v]*(dc*hp/ii+c*(hpp*dp/ii-hp*di/(ii*ii)));
              for(unsigned int i=0;i<2;++i)
                {
                  force[p.segment+i]-=p.weight*w[i]*c*h/(ii*ii);
                  dforce[p.segment+i]-=p.weight*w[i]*(dc*h/(ii*ii)+c*hp*dp/(ii*ii)-2*c*h*di/std::pow(ii,3));
                  if(assemble_rhs)
                    result.surface_rhs[0][p.segment+i]+=p.weight*w[i]*dchi*((tau*S)-2*Z*chi*(S*S));
                }
              if(assemble_rhs)
                for(unsigned int i=0;i<cell_rhs.size();++i)
                  cell_rhs[i]+=p.weight*2*Z*dchi*(S*fe[intro.extractors.velocities].symmetric_gradient(i,q));
            }
          if(assemble_rhs)
            {entry.second->get_dof_indices(indices);
             constraints.distribute_local_to_global(cell_rhs,indices,result.bulk_rhs);}
        }
      if(assemble_rhs)result.bulk_rhs.compress(VectorOperation::add);
      for(const auto &p:s.profiles)
        {
          const double w[2]={1-p.xi,p.xi};
          const double ii=w[0]*I[p.segment]+w[1]*I[p.segment+1];
          const double di=w[0]*dI[p.segment]+w[1]*dI[p.segment+1];
          const double Z=w[0]*endpoint.Z[p.segment]+w[1]*endpoint.Z[p.segment+1];
          const double dZ=slip_direction?coefficients.eta_ve*(w[0]*(*slip_direction)[0][p.segment]+w[1]*(*slip_direction)[0][p.segment+1]):0.;
          for(unsigned int i=0;i<2;++i)
            {
              force[p.segment+i]-=p.weight*w[i]*Z*Z/(2*s.G*ii*ii);
              dforce[p.segment+i]-=p.weight*w[i]*(Z*dZ/(s.G*ii*ii)-Z*Z*di/(s.G*std::pow(ii,3)));
              if(assemble_rhs)result.surface_rhs[0][p.segment+i]+=p.weight*w[i]*Z*di/(ii*ii);
            }
        }
      // Matrix-free pullback DI^T dforce + (D DI[d])^T force. In particular,
      // the second term differentiates the analytic rays through P^T.
      for(unsigned int k=0;k<s.profiles.size();++k)
        {
          double adjoint=0.,dadjoint=0.;
          for(unsigned int i=0;i<ns;++i)
            {adjoint+=s.projection(i,k)*force[i];dadjoint+=s.projection(i,k)*dforce[i];}
          unsigned int j=0;
          for(const auto &ray:s.profiles[k].rays)
            {
              const auto &jet=jets[k][j++];double dp[2]={};
              for(unsigned int a=0;a<2;++a)for(unsigned int v=0;v<ray.dofs.size();++v)
                dp[a]+=ray.shape[a][v]*direction[ray.dofs[v]];
              for(unsigned int v=0;v<ray.dofs.size();++v)for(unsigned int a=0;a<2;++a)
                {
                  double second=0.;for(unsigned int b=0;b<2;++b)second+=jet.h[a][b]*dp[b];
                  result.gradient[ray.dofs[v]]+=ray.shape[a][v]*(dadjoint*jet.d[a]+adjoint*second);
                }
            }
        }
      return result;
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::retained_history_stress(
      const CellId &id,const Point<dim> &x,const SymmetricTensor<2,dim> &legacy) const
    {
      if(!fe_strain_history)return benchmark_retained_stress?benchmark_retained_stress(id,x):legacy;
      AssertThrow(fe_strain_state && !benchmark_retained_stress,ExcMessage("FE strain needs its single accepted history."));
      const auto&s=*fe_strain_state;
      std::array<double,dim> coordinates;
      for(unsigned int d=0;d<dim;++d)coordinates[d]=x[d];
      const auto key=std::make_pair(id,coordinates);
      const auto cached=s.stress_samples.find(key);
      if(cached!=s.stress_samples.end())return cached->second;
      const auto stress=evaluate_fe_strain_stress(s.accepted,s.Z,s.I,id,x);
      s.stress_samples.emplace(key,stress);
      return stress;
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_strain_stress(
      const LinearAlgebra::BlockVector &state,const std::vector<double>&Znodes,
      const std::vector<double>&Inodes,const CellId&id,const Point<dim>&x,double *evaluation_scale) const
    {
      const auto&s=*fe_strain_state;const auto cell=s.cells.at(id);
      const auto unit=this->get_mapping().transform_real_to_unit_cell(cell,x);
      FEValues<dim>fe(this->get_mapping(),this->get_fe(),Quadrature<dim>({unit}),update_values|update_gradients);
      fe.reinit(cell);std::vector<SymmetricTensor<2,dim>>R(1);std::vector<double>phi(1);
      fe[this->introspection().extractors.velocities].get_function_symmetric_gradients(state,R);
      fe[FEValuesExtractors::Scalar(this->introspection().variable("phase_field").first_component_index)].get_function_values(state,phi);
      const auto ns=Znodes.size()-1;const auto j=std::min<unsigned int>(ns-1,x[0]*ns);const double t=x[0]*ns-j;
      const double Z=(1-t)*Znodes[j]+t*Znodes[j+1],I=(1-t)*Inodes[j]+t*Inodes[j+1];
      const double h=s.m*phi[0]*(1+phi[0])/std::pow(1-phi[0],2);
      SymmetricTensor<2,dim>S;S[0][1]=.5*this->get_reconstructed_fault_manager().get_shear_sense(0);
      if(evaluation_scale)
        {
          // Bound differentiation before cancellation of nearby FE values.
          // A constant d_R can have a tiny gradient with O(1) coefficients.
          Vector<double> values(this->get_fe().dofs_per_cell);
          cell->get_dof_values(state,values);double strain_scale=0.;
          for(unsigned int i=0;i<values.size();++i)
            if(this->get_fe().system_to_component_index(i).first<dim)
              strain_scale+=std::abs(values[i])
                            *fe[this->introspection().extractors.velocities].symmetric_gradient(i,0).norm();
          *evaluation_scale=2*s.G*strain_scale+std::abs(2*Z*h/I)*S.norm();
        }
      return 2*s.G*Utilities::Tensors::consistent_deviator(R[0])-2*Z*h/I*S;
    }

    template<int dim> std::shared_ptr<typename PhaseFieldFault<dim>::FEStrainUpdate>
    PhaseFieldFault<dim>::prepare_fe_strain_commit(const LinearAlgebra::BlockVector &bulk,
                                                  const std::vector<std::vector<double>>&tractions) const
    {
      validate_fe_strain_restrictions();const auto&s=*fe_strain_state;
      AssertThrow(s.committed_step!=this->get_timestep_number(),ExcMessage("FE strain cannot commit twice in a physical step."));
      const auto c=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
      auto update=std::make_shared<FEStrainUpdate>();update->accepted.reinit(s.accepted);update->accepted=s.accepted;
      const auto vb=this->introspection().block_indices.velocities;
      update->accepted.block(vb)*=c.beta;
      update->accepted.block(vb).add(c.eta_ve/s.G,bulk.block(vb));
      update->accepted.block(this->introspection().variable("phase_field").block_index)=
        bulk.block(this->introspection().variable("phase_field").block_index);
      update->I=current_normalization_integrals.at(0);update->Z.resize(s.Z.size());
      const auto&V=this->get_reconstructed_fault_manager().get_slip_rate(0);
      for(unsigned int j=0;j<s.Z.size();++j)
        {
          update->Z[j]=tractions.at(0).at(j)*update->I[j];
          const double expected=c.beta*s.Z[j]+c.eta_ve*V[j];
          AssertThrow(std::isfinite(update->Z[j]) && std::abs(update->Z[j]-expected)
                      <=64*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(expected),std::abs(update->Z[j])}),
                      ExcMessage("FE strain surface history and accepted Maxwell state disagree."));
        }
      return update;
    }

    template<int dim> SymmetricTensor<2,dim> PhaseFieldFault<dim>::candidate_fe_strain_stress(
      const std::shared_ptr<FEStrainUpdate>&update,const CellId&id,const Point<dim>&x,double *evaluation_scale) const
    {
      double current_scale=0.;
      const auto stress=evaluate_fe_strain_stress(update->accepted,update->Z,update->I,id,x,&current_scale);
      if(evaluation_scale)
        {
          double old_scale=0.;const auto&s=*fe_strain_state;
          evaluate_fe_strain_stress(s.accepted,s.Z,s.I,id,x,&old_scale);
          const auto c=compute_maxwell_coefficients(s.eta,s.G,this->get_timestep());
          // The triangle inequality also bounds kappa*epsilon by the old
          // and new R terms. Count both endpoint evaluations and forcing.
          *evaluation_scale=2*(current_scale+c.beta*old_scale);
        }
      return stress;
    }

    template<int dim> void PhaseFieldFault<dim>::publish_fe_strain_commit(
      const std::shared_ptr<FEStrainUpdate>&update) noexcept
    {
      fe_strain_state->accepted.swap(update->accepted);
      fe_strain_state->Z.swap(update->Z);fe_strain_state->I.swap(update->I);
      fe_strain_state->stress_samples.clear();
      fe_strain_state->committed_step=this->get_timestep_number();
    }
  }

  template<int dim> Quadrature<dim> fault_mechanical_quadrature(const SimulatorAccess<dim>&sim)
  {
    return fault_mechanical_quadrature(sim.get_material_model(),sim.introspection().quadratures.velocities);
  }
  template<int dim> Quadrature<dim> fault_mechanical_quadrature(
    const MaterialModel::Interface<dim>&material,const Quadrature<dim>&native)
  {
    if(const auto*m=dynamic_cast<const MaterialModel::PhaseFieldFault<dim>*>(&material))
      return m->mechanical_quadrature();
    return native;
  }

  namespace MaterialModel
  {
#define INSTANTIATE(dim) \
  template PhaseFieldFault<dim>::FEEndpointAction PhaseFieldFault<dim>::evaluate_fe_endpoint_action(const LinearAlgebra::BlockVector &,const FEEndpointTrial &,const std::vector<double> &,const LinearAlgebra::BlockVector *,const std::vector<std::vector<double>> *,bool) const; \
    template bool PhaseFieldFault<dim>::initialize_fe_shared_phase(LinearAlgebra::BlockVector &) const; \
    template Quadrature<dim> PhaseFieldFault<dim>::mechanical_quadrature() const; \
    template void PhaseFieldFault<dim>::prepare_fe_strain_history(); \
    template void PhaseFieldFault<dim>::validate_fe_strain_restrictions() const; \
    template const LinearAlgebra::BlockVector &PhaseFieldFault<dim>::accepted_fe_strain_state() const; \
    template PhaseFieldFault<dim>::FEPhaseEnergy PhaseFieldFault<dim>::evaluate_fe_phase_energy(const LinearAlgebra::BlockVector &,bool,bool) const; \
    template PhaseFieldFault<dim>::FEPhaseEnergy PhaseFieldFault<dim>::evaluate_fe_phase_energy_at(const LinearAlgebra::BlockVector &,bool,bool,const LinearAlgebra::BlockVector &,const std::vector<double>&,const std::vector<double>*) const; \
    template PhaseFieldFault<dim>::FEPhaseEnergy PhaseFieldFault<dim>::evaluate_fe_endpoint_energy(const LinearAlgebra::BlockVector &,const FEEndpointTrial &,const std::vector<double>*) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_endpoint_stress(const FEEndpointTrial &,const std::vector<double>&,const CellId&,const Point<dim>&) const; \
    template PhaseFieldFault<dim>::FEEndpointTrial PhaseFieldFault<dim>::make_fe_endpoint_trial(const LinearAlgebra::BlockVector &,const std::vector<std::vector<double>> &) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::retained_history_stress(const CellId &,const Point<dim>&,const SymmetricTensor<2,dim>&) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::evaluate_fe_strain_stress(const LinearAlgebra::BlockVector &,const std::vector<double>&,const std::vector<double>&,const CellId&,const Point<dim>&,double*) const; \
    template std::shared_ptr<PhaseFieldFault<dim>::FEStrainUpdate> PhaseFieldFault<dim>::prepare_fe_strain_commit(const LinearAlgebra::BlockVector &,const std::vector<std::vector<double>>&) const; \
    template SymmetricTensor<2,dim> PhaseFieldFault<dim>::candidate_fe_strain_stress(const std::shared_ptr<FEStrainUpdate>&,const CellId&,const Point<dim>&,double*) const; \
    template void PhaseFieldFault<dim>::publish_fe_strain_commit(const std::shared_ptr<FEStrainUpdate>&) noexcept;
    INSTANTIATE(2)
    INSTANTIATE(3)
#undef INSTANTIATE
  }
  template Quadrature<2> fault_mechanical_quadrature(const SimulatorAccess<2>&);
  template Quadrature<3> fault_mechanical_quadrature(const SimulatorAccess<3>&);
  template Quadrature<2> fault_mechanical_quadrature(const MaterialModel::Interface<2>&,const Quadrature<2>&);
  template Quadrature<3> fault_mechanical_quadrature(const MaterialModel::Interface<3>&,const Quadrature<3>&);
}
