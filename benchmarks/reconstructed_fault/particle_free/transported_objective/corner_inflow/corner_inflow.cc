/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later.
 * Benchmark-only boundary isolation; no material history is published. */
#include <aspect/postprocess/interface.h>
#include <aspect/boundary_traction/interface.h>
#include <aspect/material_model/history/tensor_advection.h>
#include <aspect/material_model/phase_field_fault/stress_law.h>
#include <aspect/reconstructed_fault/objective_stress.h>
#include <deal.II/numerics/vector_tools.h>
#include <complex>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace CornerReference
  {
    const double shear=.5, speed=.4, G=10.;
    // Independent component/complex recurrence, not the native rotation helper.
    template<int dim> SymmetricTensor<2,dim>
    stress(Point<dim> x, double time, double dt, unsigned steps, bool reaction, bool continuous)
    {
      x[0]-=(speed+shear*x[1])*time;
      const double f=1.+.15*x[0]+.1*x[1]+.08*x[0]*x[1];
      double mean=.35*f; std::complex<double> z(.65*f,.45*f);
      if(reaction)
        {
          const std::complex<double> lambda(-1.,-shear), factor=std::exp(lambda*time);
          mean*=std::exp(-time);z*=factor;
          if(continuous)z+=std::complex<double>(0.,G*shear)*(factor-1.)/lambda;
          else if(steps)z+=std::complex<double>(0.,G*shear*(-std::expm1(-dt)))
                               *(1.-std::pow(std::exp(lambda*dt),steps))/(1.-std::exp(lambda*dt));
        }
      SymmetricTensor<2,dim> a;a[0][0]=mean+z.real();a[1][1]=mean-z.real();a[0][1]=z.imag();return a;
    }
    std::complex<double> uniform_background(unsigned step, double time)
    {
      if(!step)return {0.,G*shear*(-std::expm1(-.1))};
      const double dt=time/step;
      const std::complex<double> r=std::exp(std::complex<double>(-dt,-shear*dt));
      return std::complex<double>(0.,G*shear*(-std::expm1(-dt)))*(1.-std::pow(r,step))/(1.-r);
    }
  }

  namespace BoundaryTraction
  {
    template<int dim> class CornerShear : public Interface<dim>, public SimulatorAccess<dim>
    {
      Tensor<1,dim> boundary_traction(const types::boundary_id, const Point<dim> &, const Tensor<1,dim> &) const override
      {
        const auto z=CornerReference::uniform_background(this->get_timestep_number(),this->get_time());
        Tensor<1,dim> t;t[0]=z.imag();t[1]=-z.real();return t;
      }
    };
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(CornerShear,"corner shear background",
      "Benchmark top traction of the uniform discrete Maxwell/Jaumann background, G=eta=10, shear=.5, initial interval=.1; p=0.")
  }

  namespace Postprocess
  {
    template<int dim> class CornerInflow : public Interface<dim>, public SimulatorAccess<dim>
    {
      using Field=MaterialModel::HistoryDiscretization::TensorField<dim>;
      using Op=MaterialModel::HistoryDiscretization::TensorAdvection<dim>;
      using Sym=SymmetricTensor<2,dim>;
      static double dot(const Sym &a,const Sym &b) {return scalar_product(a,b)+trace(a)*trace(b);}
      Sym sample(const Field &a,const typename DoFHandler<dim>::active_cell_iterator &cell,const Point<dim> &x) const
      {
        std::vector<types::global_dof_index> ids(9);cell->get_dof_indices(ids);
        return a.tensor_at(ids,this->get_mapping().transform_real_to_unit_cell(cell,x));
      }
      // E, dE/dt, supplied-boundary flux, upwind loss, interior-boundary flux.
      std::array<double,5> energy(const Op &op,const Field &a,const typename Op::Field &inflow,double time) const
      {
        Field rhs;rhs.space=op.space;const auto comm=this->get_mpi_communicator();
        for(unsigned c=0;c<3;++c)
          {
            LinearAlgebra::Vector owned(op.space->dofs.locally_owned_dofs(),comm);
            for(const auto &block:op.blocks)
              for(unsigned i=0;i<block.rows.size();++i)for(unsigned j=0;j<block.columns.size();++j)
                owned[block.rows[i]]+=block.values(i,j)*a.coefficients[c][block.columns[j]];
            for(const auto &load:op.inflow)
              {
                const auto g=inflow(load.point,time);
                for(unsigned i=0;i<load.indices.size();++i)
                  owned[load.indices[i]]+=load.weights[i]*g[Sym::unrolled_to_component_indices(c)];
              }
            owned.compress(VectorOperation::insert);
            rhs.coefficients[c].reinit(op.space->dofs.locally_owned_dofs(),op.space->relevant,comm);
            rhs.coefficients[c]=owned;
          }
        // Volume mass products are Q4. Face upwind signs change at y=-.8:
        // use the operator's q8 rule for its semidiscrete identity.
        const QGauss<dim> q(3);const QGauss<dim-1> face(8);
        FEValues<dim> fe(this->get_mapping(),op.space->element,q,update_quadrature_points|update_JxW_values);
        FEFaceValues<dim> ff(this->get_mapping(),op.space->element,face,update_quadrature_points|update_JxW_values|update_normal_vectors);
        std::array<double,5> e{};
        for(const auto &entry:op.cells)
          {
            const auto cell=op.space->cells.at(entry.id);fe.reinit(cell);
            for(unsigned k=0;k<q.size();++k)
              {
                const auto x=fe.quadrature_point(k),u=this->get_mapping().transform_real_to_unit_cell(cell,x);
                const auto value=a.tensor_at(entry.indices,u),rate=rhs.tensor_at(entry.indices,u);
                e[0]+=fe.JxW(k)*dot(value,value)/40.;e[1]+=fe.JxW(k)*dot(value,rate)/20.;
              }
            for(unsigned side=0;side<GeometryInfo<dim>::faces_per_cell;++side)
              {
                ff.reinit(cell,side);
                for(unsigned k=0;k<face.size();++k)
                  {
                    const auto x=ff.quadrature_point(k);const auto inside=sample(a,cell,x);
                    const double vn=(.4+.5*x[1])*ff.normal_vector(k)[0],w=ff.JxW(k);
                    if(cell->at_boundary(side))
                      {
                        // Evaluate the same parent polynomial for held trace;
                        // the callback never substitutes the current iterate.
                        const auto outside=inflow(x,time),flux_value=vn>=0 ? inside : outside;
                        e[2]+=w*vn*dot(flux_value,flux_value)/40.;
                        e[4]+=w*vn*dot(inside,inside)/40.;
                        if(vn<0)e[3]+=w*(-vn)*dot(inside-outside,inside-outside)/40.;
                      }
                    else
                      {
                        const auto jump=inside-sample(a,cell->neighbor(side),x);
                        e[3]+=.5*w*std::abs(vn)*dot(jump,jump)/40.;
                      }
                  }
              }
          }
        for(auto &v:e)v=Utilities::MPI::sum(v,comm);
        return e;
      }

      void run() const
      {
        AssertThrow(dim==2,ExcMessage("Corner experiment requires 2D."));
        const auto vertices=this->get_triangulation().get_vertices();
        const auto comm=this->get_mpi_communicator();Field prototype;
        LinearAlgebra::BlockVector saved_owned(this->introspection().index_sets.system_partitioning,comm);
        saved_owned=this->get_solution();
        prototype.space=std::make_shared<typename Field::Space>(this->get_triangulation());
        typename Field::Cells cells;
        for(const auto &c:this->get_dof_handler().active_cell_iterators())if(c->is_locally_owned())cells.emplace(c->id(),c);
        struct Flow : Function<dim>
        {
          Flow(unsigned n,unsigned first):Function<dim>(n),first(first){} unsigned first;
          double value(const Point<dim> &x,unsigned c) const override {return c==first ? .4+.5*x[1] : 0.;}
        };
        LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
        VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),Flow(this->get_fe().n_components(),this->introspection().component_indices.velocities[0]),owned);
        LinearAlgebra::BlockVector velocity(this->get_solution());velocity=owned;
        const auto op=Op::assemble(*this,prototype,cells,velocity,8,0.,true);
        const auto n=unsigned(std::sqrt(this->get_triangulation().n_global_active_cells()));const double h=4./n;
        std::map<std::pair<unsigned,unsigned>,CellId> spatial_cells;
        for(const auto &entry:op->space->cells)
          {const auto center=entry.second->center();spatial_cells.emplace(std::make_pair(unsigned((center[0]+2)/h),unsigned((center[1]+2)/h)),entry.first);}
        const auto at=[&](const Field &a,const Point<dim> &x)
          {
            const unsigned i=std::min(n-1,unsigned(std::max(0.,std::floor((x[0]+2)/h))));
            const unsigned j=std::min(n-1,unsigned(std::max(0.,std::floor((x[1]+2)/h))));
            return sample(a,op->space->cells.at(spatial_cells.at({i,j})),x);
          };
        const typename Op::Field zero=[](const Point<dim> &,double){return Sym();};
        const QGauss<dim> q(6);FEValues<dim> fe(this->get_mapping(),op->space->element,q,update_quadrature_points|update_JxW_values);
        std::ofstream out;
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {out.open(this->get_output_directory()+"corner-isolation.csv");out<<std::setprecision(17)<<"cells,reaction,held_trace,dt,step,time,substeps,relative_discrete,relative_continuous,corner_relative,interior_relative,energy,transport_change,reaction_change,flux_supplied_integral,flux_interior_integral,upwind_integral,transport_balance,max_identity,max_stress\n";}
        for(bool reaction:{false,true})for(bool held:{false,true})
          for(double dt:n==32 ? std::vector<double>{.1,.05,.025} : std::vector<double>{.025})
            {
              auto state=Op::project(*this,*op,[&](const Point<dim> &x,double t){return CornerReference::stress(x,t,dt,0,reaction,false);},0.,8);
              const unsigned steps=std::lround(.4/dt),substeps=std::max(1u,unsigned(std::ceil(dt*op->inverse_advection_time/.1)));
              for(unsigned step=0;step<steps;++step)
                {
                  const auto parent=state;const double time=step*dt;
                  const typename Op::Field boundary=[&](Point<dim> x,double t)
                    {
                      if(held)return at(*parent,x);
                      x[0]-=(.4+.5*x[1])*(t-time);
                      return CornerReference::stress(x,time,dt,step,reaction,false);
                    };
                  auto previous=energy(*op,*state,boundary,time);const double start=previous[0];
                  double flux=0.,inside_flux=0.,loss=0.,identity=0.;
                  for(unsigned k=0;k<substeps;++k)
                    {
                      identity=std::max(identity,std::abs(previous[1]+previous[2]+previous[3]));
                      state=Op::advance(*this,*op,*state,boundary,zero,time+k*dt/substeps,dt/substeps);
                      const auto next=energy(*op,*state,boundary,time+(k+1)*dt/substeps);
                      flux+=.5*dt/substeps*(previous[2]+next[2]);inside_flux+=.5*dt/substeps*(previous[4]+next[4]);
                      loss+=.5*dt/substeps*(previous[3]+next[3]);previous=next;
                    }
                  identity=std::max(identity,std::abs(previous[1]+previous[2]+previous[3]));
                  const double transported=previous[0];
                  if(reaction)
                    {
                      const auto incoming=state;
                      state=Field::project(*this,*incoming,cells,QGauss<dim>(8),
                        [&](const CellId &,const std::vector<types::global_dof_index> &ids,const FEValues<dim> &target,std::vector<Sym> &values)
                        {
                          Tensor<2,dim> gradient;gradient[0][1]=.5;Sym D;D[0][1]=.25;
                          for(unsigned k=0;k<values.size();++k)
                            {
                              const auto a=incoming->tensor_at(ids,target.get_quadrature().point(k));
                              Sym rotated;if constexpr(dim==2)rotated=ReconstructedFaultUtilities::rotate_plane_strain_stress(a,gradient,dt);
                              values[k]=MaterialModel::Rheology::mature_stress_endpoint(std::exp(-dt),-10*std::expm1(-dt),rotated,D,Sym());
                            }
                        });
                    }
                  std::array<double,8> sums{};double peak=0.;
                  for(const auto &cell:op->cells)
                    {
                      fe.reinit(op->space->cells.at(cell.id));
                      for(unsigned k=0;k<q.size();++k)
                        {
                          const auto x=fe.quadrature_point(k);const auto actual=state->tensor_at(cell.indices,q.point(k));
                          const auto exact=CornerReference::stress(x,(step+1)*dt,dt,step+1,reaction,false);
                          const auto continuous=CornerReference::stress(x,(step+1)*dt,dt,step+1,reaction,true);
                          const double w=fe.JxW(k),e=dot(actual-exact,actual-exact),b=dot(exact,exact);
                          sums[0]+=w*e;sums[1]+=w*b;sums[2]+=w*dot(actual-continuous,actual-continuous);sums[3]+=w*dot(continuous,continuous);
                          if(x[0]<-1.5 && x[1]>1.5){sums[4]+=w*e;sums[5]+=w*b;}
                          if(std::abs(x[0])<1.5 && std::abs(x[1])<1.5){sums[6]+=w*e;sums[7]+=w*b;}
                          peak=std::max(peak,std::sqrt(dot(actual,actual)));
                        }
                    }
                  for(auto &v:sums)v=Utilities::MPI::sum(v,comm);
                  peak=Utilities::MPI::max(peak,comm);
                  const double final=energy(*op,*state,boundary,(step+1)*dt)[0];
                  AssertThrow(identity<1e-10*std::max(1.,final),ExcMessage("Corner advection semidiscrete energy identity failed."));
                  if(!held)AssertThrow(std::sqrt(sums[0]/sums[1])<1e-9,ExcMessage("Exact inflow failed the discrete polynomial reference."));
                  if(out.is_open())out<<n*n<<','<<reaction<<','<<held<<','<<dt<<','<<step+1<<','<<(step+1)*dt<<','<<substeps<<','<<std::sqrt(sums[0]/sums[1])<<','<<std::sqrt(sums[2]/sums[3])<<','<<std::sqrt(sums[4]/sums[5])<<','<<std::sqrt(sums[6]/sums[7])<<','<<final<<','<<transported-start<<','<<final-transported<<','<<flux<<','<<inside_flux<<','<<loss<<','<<transported-start+flux+loss<<','<<identity<<','<<peak<<'\n';
                }
            }
        LinearAlgebra::BlockVector difference(this->introspection().index_sets.system_partitioning,comm);
        difference=this->get_solution();difference-=saved_owned;
        AssertThrow(difference.l2_norm()==0. && this->get_triangulation().get_vertices()==vertices,
                    ExcMessage("Scratch corner experiment changed accepted solution/mesh."));
      }
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {if(this->get_timestep_number()==0)run();return {"Corner inflow isolation","scratch controls complete"};}
    };
    ASPECT_REGISTER_POSTPROCESSOR(CornerInflow,"corner inflow isolation","Nonpublishing prescribed affine-shear inflow control using native DGQ2 and SSPRK3.")
  }
}
