/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later.
 * Saved-velocity replay: benchmark scratch fields, no material publication.
 * Energy assembly follows the retained corner_inflow benchmark, with explicit
 * manufactured source and pointwise divergence for the non-affine flow. */
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/history/tensor_advection.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace aspect
{
  template<int dim> class ReplayFlow : public Function<dim>
  {
    unsigned n,first;std::vector<std::array<double,18>> coefficients;
    std::pair<unsigned,Point<dim>> location(const Point<dim> &x) const
    {
      Point<dim> u;unsigned index=0;
      for(unsigned d=0;d<2;++d)
        {
          const int cell=std::max(0,std::min(int(n)-1,int(std::floor((x[d]+2)*n/4))));
          u[d]=2*((x[d]+2)*n/4-cell)-1;index=index*n+cell;
        }
      return {index,u};
    }
  public:
    ReplayFlow(unsigned components,unsigned first):Function<dim>(components),first(first)
    {
      const char *path=std::getenv("ASPECT_CORNER_REPLAY_VELOCITY");
      AssertThrow(path,ExcMessage("Missing replay velocity file."));
      std::ifstream in(path);in>>n;AssertThrow(in && n==32,ExcMessage("Replay requires saved 32x32 Q2 velocity."));
      coefficients.resize(n*n);for(auto &cell:coefficients)for(auto &c:cell)in>>c;
      AssertThrow(in,ExcMessage("Truncated replay velocity."));
    }
    Tensor<1,dim> velocity(const Point<dim> &x) const
    {
      const auto p=location(x);const auto &u=p.second;
      const double a[3]={1,u[0],.5*(3*u[0]*u[0]-1)},b[3]={1,u[1],.5*(3*u[1]*u[1]-1)};
      Tensor<1,dim> v;
      for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)for(unsigned c=0;c<2;++c)
        v[c]+=a[i]*b[j]*coefficients[p.first][2*(3*i+j)+c];
      return v;
    }
    Tensor<2,dim> gradient(const Point<dim> &x) const
    {
      const auto p=location(x);const auto &u=p.second;
      const double a[3]={1,u[0],.5*(3*u[0]*u[0]-1)},b[3]={1,u[1],.5*(3*u[1]*u[1]-1)};
      const double da[3]={0,1,3*u[0]},db[3]={0,1,3*u[1]};Tensor<2,dim> L;
      for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j)for(unsigned c=0;c<2;++c)
        {
          L[c][0]+=n/2.*da[i]*b[j]*coefficients[p.first][2*(3*i+j)+c];
          L[c][1]+=n/2.*a[i]*db[j]*coefficients[p.first][2*(3*i+j)+c];
        }
      return L;
    }
    double value(const Point<dim> &x,unsigned c) const override
    {return c>=first && c<first+dim ? velocity(x)[c-first] : 0.;}
  };
  namespace Postprocess
  {
    template<int dim> class CornerReplay : public Interface<dim>, public SimulatorAccess<dim>
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
      // E, dE/dt, supplied flux, upwind loss, interior flux, divergence work, source work.
      std::array<double,7> energy(const Op &op,const Field &a,const typename Op::Field &inflow,
                                  const typename Op::Field &source,const ReplayFlow<dim> &flow,double time) const
      {
        Field rhs;rhs.space=op.space;const auto comm=this->get_mpi_communicator();
        for(unsigned c=0;c<3;++c)
          {
            LinearAlgebra::Vector owned(op.space->dofs.locally_owned_dofs(),comm);
            for(const auto &block:op.blocks)
              for(unsigned i=0;i<block.rows.size();++i)for(unsigned j=0;j<block.columns.size();++j)
                owned[block.rows[i]]+=block.values(i,j)*a.coefficients[c][block.columns[j]];
            for(unsigned kind=0;kind<2;++kind)for(const auto &load:kind ? op.inflow : op.source)
              {
                const auto g=(kind ? inflow : source)(load.point,time);
                for(unsigned i=0;i<load.indices.size();++i)
                  owned[load.indices[i]]+=load.weights[i]*g[Sym::unrolled_to_component_indices(c)];
              }
            owned.compress(VectorOperation::insert);
            rhs.coefficients[c].reinit(op.space->dofs.locally_owned_dofs(),op.space->relevant,comm);
            rhs.coefficients[c]=owned;
          }
        // Use the operator quadrature for flux signs and variable-velocity volume terms.
        const QGauss<dim> q(8);const QGauss<dim-1> face(8);
        FEValues<dim> fe(this->get_mapping(),op.space->element,q,update_quadrature_points|update_JxW_values);
        FEFaceValues<dim> ff(this->get_mapping(),op.space->element,face,update_quadrature_points|update_JxW_values|update_normal_vectors);
        std::array<double,7> e{};
        for(const auto &entry:op.cells)
          {
            const auto cell=op.space->cells.at(entry.id);fe.reinit(cell);
            for(unsigned k=0;k<q.size();++k)
              {
                const auto x=fe.quadrature_point(k),u=this->get_mapping().transform_real_to_unit_cell(cell,x);
                const auto value=a.tensor_at(entry.indices,u),rate=rhs.tensor_at(entry.indices,u);
                e[0]+=fe.JxW(k)*dot(value,value)/40.;e[1]+=fe.JxW(k)*dot(value,rate)/20.;
                e[5]+=fe.JxW(k)*dot(value,value)/40.*trace(flow.gradient(x));
                e[6]+=fe.JxW(k)*dot(value,source(x,time))/20.;
              }
            for(unsigned side=0;side<GeometryInfo<dim>::faces_per_cell;++side)
              {
                ff.reinit(cell,side);
                for(unsigned k=0;k<face.size();++k)
                  {
                    const auto x=ff.quadrature_point(k);const auto inside=sample(a,cell,x);
                    const double vn=flow.velocity(x)*ff.normal_vector(k),w=ff.JxW(k);
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

      static Sym tensor(double f) {Sym a;a[0][0]=f;a[1][1]=-.3*f;a[0][1]=.45*f;return a;}
      static Sym exact(Point<dim> x,double t)
      {x[0]-=(.4+.5*x[1])*t;return tensor(1+.15*x[0]+.1*x[1]+.08*x[0]*x[1]);}
      void run() const
      {
        AssertThrow(dim==2,ExcMessage("Replay requires 2D."));
        const auto vertices=this->get_triangulation().get_vertices();
        const auto comm=this->get_mpi_communicator();Field prototype;
        LinearAlgebra::BlockVector saved(this->introspection().index_sets.system_partitioning,comm);saved=this->get_solution();
        prototype.space=std::make_shared<typename Field::Space>(this->get_triangulation());
        typename Field::Cells cells;
        for(const auto &c:this->get_dof_handler().active_cell_iterators())if(c->is_locally_owned())cells.emplace(c->id(),c);
        ReplayFlow<dim> flow(this->get_fe().n_components(),this->introspection().component_indices.velocities[0]);
        LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
        VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),flow,owned);
        LinearAlgebra::BlockVector velocity(this->get_solution());velocity=owned;
        // Verify replay in native FE space, independently of DG tensor coefficients.
        QGauss<dim> check_q(8);FEValues<dim> check(this->get_mapping(),this->get_fe(),check_q,update_values|update_gradients|update_quadrature_points);
        std::vector<Tensor<1,dim>> vv(check_q.size());std::vector<Tensor<2,dim>> gg(check_q.size());
        double ve=0,ge=0;
        for(const auto &cell:cells)
          {
            check.reinit(cell.second);check[this->introspection().extractors.velocities].get_function_values(velocity,vv);
            check[this->introspection().extractors.velocities].get_function_gradients(velocity,gg);
            for(unsigned k=0;k<check_q.size();++k)
              {ve=std::max(ve,(vv[k]-flow.velocity(check.quadrature_point(k))).norm());ge=std::max(ge,(gg[k]-flow.gradient(check.quadrature_point(k))).norm());}
          }
        ve=Utilities::MPI::max(ve,comm);ge=Utilities::MPI::max(ge,comm);
        AssertThrow(ve<1e-12 && ge<1e-10,ExcMessage("Saved velocity FE reconstruction failed."));
        const auto op=Op::assemble(*this,prototype,cells,velocity,8,0.,true);
        const unsigned n=32;const double h=4./n;
        AssertThrow(this->get_triangulation().n_global_active_cells()==n*n,ExcMessage("Replay mesh mismatch."));
        std::map<std::pair<unsigned,unsigned>,CellId> spatial;
        for(const auto &entry:op->space->cells)
          {const auto x=entry.second->center();spatial.emplace(std::make_pair(unsigned((x[0]+2)/h),unsigned((x[1]+2)/h)),entry.first);}
        const auto at=[&](const Field &a,const Point<dim> &x)
          {
            const unsigned i=std::min(n-1,unsigned(std::max(0.,std::floor((x[0]+2)/h))));
            const unsigned j=std::min(n-1,unsigned(std::max(0.,std::floor((x[1]+2)/h))));
            return sample(a,op->space->cells.at(spatial.at({i,j})),x);
          };
        const typename Op::Field source=[&](const Point<dim> &x,double t)
          {
            const auto v=flow.velocity(x);const double gx=.15+.08*x[1];
            const double gy=.1+.08*x[0]-t*(.5*gx+.08*(.4+.5*x[1]));
            return tensor((v[0]-(.4+.5*x[1]))*gx+v[1]*gy);
          };
        const unsigned base_substeps=4*std::max(1u,unsigned(std::ceil(op->inverse_advection_time/4.)));
        const double subdt=.1/base_substeps;
        const QGauss<dim> q(6);FEValues<dim> fe(this->get_mapping(),op->space->element,q,update_quadrature_points|update_JxW_values);
        std::ofstream out;
        if(Utilities::MPI::this_mpi_process(comm)==0)
          {
            std::ofstream audit(this->get_output_directory()+"replay-native-audit.txt");
            audit<<std::setprecision(17)<<"velocity_max_error "<<ve<<"\ngradient_max_error "<<ge<<"\ninverse_advection_time "<<op->inverse_advection_time<<"\nsubdt "<<subdt<<'\n';
            out.open(this->get_output_directory()+"corner-replay.csv");out<<std::setprecision(17)
              <<"mode,dt,step,time,substeps,relative,corner_relative,interior_relative,energy,energy_change,flux_supplied,flux_interior,upwind,divergence,source_work,balance,max_identity,max_stress,corner_error_share\n";
          }
        // mode 0: exact stage exterior; 1: exact macro-parent exterior;
        // mode 2: accepted numerical macro-parent trace (feedback).
        for(unsigned mode=0;mode<3;++mode)for(double dt:{.1,.05,.025})
          {
            auto state=Op::project(*this,*op,exact,0.,8);
            const unsigned steps=std::lround(.4/dt),substeps=std::lround(dt/subdt);
            for(unsigned step=0;step<steps;++step)
              {
                const auto parent=state;const double time=step*dt;
                const typename Op::Field boundary=[&](const Point<dim> &x,double t)
                  {return mode==2 ? at(*parent,x) : exact(x,mode==1 ? time : t);};
                auto previous=energy(*op,*state,boundary,source,flow,time);const double start=previous[0];
                std::array<double,7> integral{};double identity=0;
                const auto defect=[](const std::array<double,7> &e){return std::abs(e[1]+e[2]+e[3]-e[5]-e[6]);};
                for(unsigned k=0;k<substeps;++k)
                  {
                    identity=std::max(identity,defect(previous));
                    state=Op::advance(*this,*op,*state,boundary,source,time+k*subdt,subdt);
                    const auto next=energy(*op,*state,boundary,source,flow,time+(k+1)*subdt);
                    for(unsigned j=2;j<7;++j)integral[j]+=.5*subdt*(previous[j]+next[j]);
                    previous=next;
                  }
                identity=std::max(identity,defect(previous));
                std::array<double,6> sums{};double peak=0.;
                for(const auto &cell:op->cells)
                  {
                    fe.reinit(op->space->cells.at(cell.id));
                    for(unsigned k=0;k<q.size();++k)
                      {
                        const auto x=fe.quadrature_point(k);const auto a=state->tensor_at(cell.indices,q.point(k));
                        const auto ref=exact(x,(step+1)*dt);const double w=fe.JxW(k),e=dot(a-ref,a-ref),b=dot(ref,ref);
                        sums[0]+=w*e;sums[1]+=w*b;
                        if(x[0]<-1.5 && x[1]>1.5){sums[2]+=w*e;sums[3]+=w*b;}
                        if(std::abs(x[0])<1.5 && std::abs(x[1])<1.5){sums[4]+=w*e;sums[5]+=w*b;}
                        peak=std::max(peak,std::sqrt(dot(a,a)));
                      }
                  }
                for(auto &v:sums)v=Utilities::MPI::sum(v,comm);
                peak=Utilities::MPI::max(peak,comm);
                const double balance=previous[0]-start+integral[2]+integral[3]-integral[5]-integral[6];
                AssertThrow(identity<1e-10*std::max(1.,previous[0]),ExcMessage("Replay energy identity failed."));
                if(mode==0)AssertThrow(std::sqrt(sums[0]/sums[1])<1e-9,ExcMessage("Exact manufactured inflow reference failed."));
                if(out.is_open())out<<mode<<','<<dt<<','<<step+1<<','<<(step+1)*dt<<','<<substeps<<','<<std::sqrt(sums[0]/sums[1])<<','<<std::sqrt(sums[2]/sums[3])<<','<<std::sqrt(sums[4]/sums[5])<<','<<previous[0]<<','<<previous[0]-start<<','<<integral[2]<<','<<integral[4]<<','<<integral[3]<<','<<integral[5]<<','<<integral[6]<<','<<balance<<','<<identity<<','<<peak<<','<<sums[2]/std::max(1e-300,sums[0])<<'\n';
              }
          }
        LinearAlgebra::BlockVector difference(this->introspection().index_sets.system_partitioning,comm);difference=this->get_solution();difference-=saved;
        AssertThrow(difference.l2_norm()==0 && this->get_triangulation().get_vertices()==vertices,ExcMessage("Replay modified accepted solution or geometry."));
      }
    public:
      std::pair<std::string,std::string> execute(TableHandler &) override
      {if(this->get_timestep_number()==0)run();return {"Corner velocity replay","scratch controls complete; accepted solution/mesh unchanged"};}
    };
    ASPECT_REGISTER_POSTPROCESSOR(CornerReplay,"corner velocity replay","Nonpublishing saved non-affine velocity manufactured-inflow control, native DGQ2/SSPRK3.")
  }
}
