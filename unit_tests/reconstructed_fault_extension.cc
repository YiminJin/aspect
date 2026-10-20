/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// P3 is an imposed geometry-transfer experiment, NOT a finite-tip evolution
// backend. The fixture bundles native manager state and cell records only to
// test atomic publication without advancing the simulator's physical clock.
#include "common.h"
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/boundary_contact.h>
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/phase_field.h>
#include <aspect/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace
{
  using namespace dealii;
  using Manager=aspect::ReconstructedFaultManager<2>;
  using Coefficients=std::array<std::array<double,9>,3>;
  using Records=std::map<std::string,Coefficients>;
  constexpr double ell=.15625,G=1e6,m=128.;
  constexpr unsigned int nx=32,ny=64;

  struct ExtensionExceptions
  {
    ExtensionExceptions(){deal_II_exceptions::disable_abort_on_exception();}
    ~ExtensionExceptions(){deal_II_exceptions::enable_abort_on_exception();}
  };

  // Material fields are owned independently of geometry. No bulk material is
  // born in an append, including the region ahead of the old physical tip.
  double prescribed_phase(const Point<2> &x)
  {
    static const aspect::PhaseField::GeometricFunction geometry(ell,1.,8./3.);
    static const aspect::PhaseField::DegradationFunction degradation(1.,m);
    static const aspect::PhaseField::PhaseFieldProfile profile(geometry,degradation,.6);
    const double dx=std::max(0.,x[0]-.5),dy=x[1]-.5;
    return profile.value(std::hypot(dx,dy));
  }

  double physical_phase(const Point<2> &x)
  {
    const unsigned int i=std::min(nx-1,static_cast<unsigned int>(x[0]*nx));
    const unsigned int j=std::min(ny-1,static_cast<unsigned int>(x[1]*ny));
    const double u=x[0]*nx-i,v=x[1]*ny-j;
    double value=0.;
    for (unsigned int b=0;b<2;++b)
      for (unsigned int a=0;a<2;++a)
        value+=(a?u:1-u)*(b?v:1-v)*prescribed_phase(Point<2>(double(i+a)/nx,double(j+b)/ny));
    return value;
  }

  double h(const double phi)
  {
    // Same p=1 Lorentz law as the cohesive material. The independent Python
    // replay evaluates its rational form and integrates each physical Q1 cell.
    static const aspect::PhaseField::DegradationFunction degradation(1.,m);
    return 1./degradation.value(phi)-1.;
  }

  struct State
  {
    Manager manager;
    Records tensor;
    std::vector<double> I,obstacle;
    // Derived point layout belongs to the candidate, not the accepted fixture.
    // Rebuild it after restart; never persist geometry caches.
    std::vector<std::pair<std::string,Point<2>>> associations;
    unsigned int publications=0;
    std::uint64_t parent_version=0;

    template<class Archive> void serialize(Archive &ar,const unsigned int)
    {
      ar & manager & tensor & I & obstacle & publications & parent_version;
    }
  };

  std::string checkpoint(const State &state)
  {
    std::ostringstream out;
    {aspect::oarchive archive(out);archive<<state;}
    return out.str();
  }

  std::unique_ptr<State> restore(const std::string &bytes)
  {
    auto state=std::make_unique<State>();
    std::istringstream in(bytes);
    {aspect::iarchive archive(in);archive>>*state;}
    return state;
  }

  struct ExtensionFixture
  {
    const MPI_Comm comm=MPI_COMM_WORLD;
    parallel::distributed::Triangulation<2> mesh;
    MappingQ1<2> mapping;
    FE_DGQ<2> element{2};
    DoFHandler<2> dofs;
    IndexSet relevant;
    AffineConstraints<double> constraints;

    ExtensionFixture():mesh(comm),dofs(mesh)
    {
      GridGenerator::subdivided_hyper_rectangle(mesh,{nx,ny},Point<2>(),Point<2>(1,1));
      dofs.distribute_dofs(element);
      relevant=DoFTools::extract_locally_relevant_dofs(dofs);
      // Fixed-grid DG constraints are empty, but rebuilt and checked on every
      // geometry candidate; no artificial constrained rows enter a norm.
      constraints.reinit(dofs.locally_owned_dofs(),relevant);
      DoFTools::make_hanging_node_constraints(dofs,constraints);
      constraints.close();
    }

    static std::array<double,3> initial_tensor(const Point<2> &x)
    {
      const double a=x[0],b=x[1];
      // Nonproportional tensor components; all complete Q2 coefficients are
      // retained, including x^2 y^2. zz is always -(xx+yy).
      return {{400+71*a+33*b+19*a*b+11*a*a+13*b*b+7*a*a*b+5*a*b*b+3*a*a*b*b,
               -130+17*a-21*b+9*a*b+5*a*a-7*b*b-4*a*a*b+3*a*b*b+2*a*a*b*b,
               800-29*a+41*b-17*a*b+23*a*a+31*b*b+11*a*a*b-13*a*b*b+7*a*a*b*b}};
    }

    std::vector<double> normalization(const Manager &manager,const unsigned int order) const
    {
      const auto &fault=manager.get_fault(0);
      std::vector<double> diagonal(fault.n_vertices()),off(fault.n_cells()),rhs(fault.n_vertices());
      const QGauss<1> quad(order);
      for (unsigned int s=0;s<fault.n_cells();++s)
        {
          const double length=fault.vertex(s).distance(fault.vertex(s+1));
          for (unsigned int q=0;q<quad.size();++q)
            {
              const double t=quad.point(q)[0],w[2]={1-t,t};
              const double x=(1-t)*fault.vertex(s)[0]+t*fault.vertex(s+1)[0];
              double integral=0.;
              // Every ordinary-FE normal cell is integrated, without domain
              // reads, arbitrary support truncation, endpoint continuation or
              // a normalization floor at the physical interior tip.
              for (unsigned int j=0;j<ny;++j)
                for (unsigned int p=0;p<quad.size();++p)
                  integral+=quad.weight(p)/ny*h(physical_phase(Point<2>(x,(j+quad.point(p)[0])/ny)));
              const double weight=length*quad.weight(q);
              for (unsigned int i=0;i<2;++i)
                {
                  diagonal[s+i]+=weight*w[i]*w[i];
                  rhs[s+i]+=weight*w[i]*integral;
                }
              off[s]+=weight*w[0]*w[1];
            }
        }
      return aspect::ReconstructedFaultUtilities::solve_tridiagonal_system(diagonal,off,rhs);
    }

    void boundary_check(const Manager &manager) const
    {
      std::vector<aspect::FaultBoundaryFace<2>> faces(4);
      const Point<2> corners[4][2]={{{0,0},{0,1}},{{1,0},{1,1}},{{0,0},{1,0}},{{0,1},{1,1}}};
      for(unsigned int i=0;i<4;++i)
        {
          faces[i].boundary_id=i;faces[i].vertices={{corners[i][0],corners[i][1]}};
          faces[i].cell_upper=Point<2>(1,1);
          faces[i].inward_normal[i/2]=i%2 ? -1. : 1.;
        }
      const aspect::PrescribedInitialFault<2> seed={manager.get_fault(0).get_vertices(),
                                                  std::vector<double>(manager.get_fault(0).n_vertices(),.6)};
      auto contacts=aspect::ReconstructedFaultUtilities::detect_boundary_contacts<2>({seed},faces);
      aspect::ReconstructedFaultUtilities::qualify_boundary_contact_support<2>(contacts,{seed},faces,{.5,.5},{.015625});
      REQUIRE(contacts.size()==1);
      REQUIRE(contacts[0].endpoint==0);
      REQUIRE(contacts[0].position==Point<2>(0,.5));
      REQUIRE(contacts[0].influence_length==0.);
      REQUIRE(contacts[0].unsupported_reason.empty());
    }

    std::unique_ptr<State> initial()
    {
      auto state=std::make_unique<State>();
      state->manager.register_property("cohesive Z",1);
      state->manager.register_property("cumulative slip",1);
      std::vector<Point<2>> vertices;
      for (unsigned int j=0;j<=32;++j)vertices.emplace_back(double(j)/64,.5);
      state->manager.add_reconstructed_fault(vertices,std::vector<double>(vertices.size(),.5));
      state->manager.set_shear_sense(0,1);
      std::vector<double> rates(vertices.size());
      for (unsigned int j=0;j<vertices.size();++j)
        {
          // Exact zero at the old physical tip: continuous zero-history birth
          // is an imposed fixture datum, not a rule for a loaded nonzero tip.
          const double amplitude=j==32 ? 0. : std::pow(std::sin(2*numbers::PI*vertices[j][0]),2);
          state->manager.get_fault(0).get_properties(j)[0]=1000*amplitude;
          state->manager.get_fault(0).get_properties(j)[1]=.002*amplitude;
          rates[j]=1e-5*amplitude;
        }
      state->manager.initialize_slip_rate(0,rates);
      for (unsigned int j=0;j<=ny;++j)
        for (unsigned int i=0;i<=nx;++i)
          state->obstacle.push_back(prescribed_phase(Point<2>(double(i)/nx,double(j)/ny)));
      state->I=normalization(state->manager,8);
      Records local;
      for (const auto &cell:dofs.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            Coefficients data;
            for (unsigned int i=0;i<9;++i)
              {
                const auto point=mapping.transform_unit_to_real_cell(cell,element.get_unit_support_points()[i]);
                const auto value=initial_tensor(point);
                for(unsigned int c=0;c<3;++c)data[c][i]=value[c];
              }
            local.emplace(cell->id().to_string(),data);
          }
      for(const auto &part:dealii::Utilities::MPI::all_gather(comm,local))
        for(const auto &row:part)AssertThrow(state->tensor.emplace(row).second,ExcMessage("Duplicate DG cell owner."));
      return state;
    }

    static double profile_stress(const State &state,const Point<2> &point)
    {
      const auto source=state.manager.project_to_normal_profiles(point);
      if(!source.active)return 0.;
      const auto &fault=state.manager.get_fault(0);
      const unsigned int s=source.segment_index;
      const double t=source.xi;
      const double Z=(1-t)*fault.get_properties(s)[0]+t*fault.get_properties(s+1)[0];
      const double I=(1-t)*state.I[s]+t*state.I[s+1];
      AssertThrow(I>0. && std::isfinite(I),ExcMessage("Undefined finite-tip normalization."));
      // tau_xy=T_xy-Z*h/I because 2*S_xy=1.
      return Z*h(physical_phase(point))/I;
    }

    std::unique_ptr<State> trial(const State &old,const bool extend,const unsigned int order=8)
    {
      auto next=restore(checkpoint(old));
      if(!extend)return next; // identity: do not reproject an unchanged field
      next->parent_version=old.manager.get_fault(0).geometry_version();
      AssertThrow(old.manager.get_fault(0).get_properties(32)[0]==0.
                  && old.manager.get_fault(0).get_properties(32)[1]==0.
                  && old.manager.get_slip_rate(0)[32]==0.,ExcMessage("This fixture requires zero histories at the old tip."));
      std::vector<Point<2>> append;
      for(unsigned int j=33;j<=48;++j)append.emplace_back(double(j)/64,.5);
      next->manager.append_reconstructed_fault(0,append,std::vector<double>(append.size(),.5),
                                               std::vector<std::vector<double>>(append.size(),{0.,0.}),
                                               std::vector<double>(append.size(),0.));
      next->I=normalization(next->manager,order);
      for(const double I:next->I)
        AssertThrow(std::isfinite(I) && I>0.,ExcMessage("Append requires positive physical normalization; no birth floor is selected."));
      // Rebuild all exercised geometry-dependent data. Fixed mesh/DG basis and
      // physical obstacle are invariant. No mechanics/phase solver is active.
      next->associations.clear();
      constraints.clear();
      constraints.reinit(dofs.locally_owned_dofs(),relevant);
      DoFTools::make_hanging_node_constraints(dofs,constraints);
      constraints.close();
      const QAnisotropic<2> quad(QIterated<1>(QGauss<1>(order),2),QGauss<1>(order));
      FEValues<2> fe(mapping,element,quad,update_values|update_quadrature_points|update_JxW_values);
      Records local;
      for(const auto &cell:dofs.active_cell_iterators())
        if(cell->is_locally_owned())
          {
            fe.reinit(cell);
            FullMatrix<double> mass(9,9);
            Vector<double> rhs(9),correction(9);
            for(unsigned int q=0;q<quad.size();++q)
              {
                const auto x=fe.quadrature_point(q);
                const auto source=next->manager.project_to_normal_profiles(x);
                if(source.active)next->associations.emplace_back(cell->id().to_string(),x);
                // Exact change of stress coordinates at fixed physical state:
                // T+ = T- + 2 (Z+ chi+ - Z- chi-) S. It is not a Maxwell step.
                const double delta=profile_stress(*next,x)-profile_stress(old,x);
                for(unsigned int i=0;i<9;++i)
                  {
                    rhs[i]+=fe.JxW(q)*fe.shape_value(i,q)*delta;
                    for(unsigned int j=0;j<9;++j)
                      mass(i,j)+=fe.JxW(q)*fe.shape_value(i,q)*fe.shape_value(j,q);
                  }
              }
            mass.gauss_jordan();
            mass.vmult(correction,rhs);
            auto data=old.tensor.at(cell->id().to_string());
            for(unsigned int i=0;i<9;++i)data[2][i]+=correction[i];
            local.emplace(cell->id().to_string(),data);
          }
      next->tensor.clear();
      for(const auto &part:dealii::Utilities::MPI::all_gather(comm,local))
        for(const auto &row:part)AssertThrow(next->tensor.emplace(row).second,ExcMessage("Duplicate trial DG owner."));
      return next;
    }

    // Single geometry publication for this synthetic transition. Existing
    // simulator physical-step counters are deliberately not impersonated.
    static void publish(std::unique_ptr<State> &accepted,std::unique_ptr<State> &candidate)
    {
      AssertThrow(candidate && candidate->parent_version==accepted->manager.get_fault(0).geometry_version()
                  && candidate->manager.get_fault(0).geometry_version()==candidate->parent_version+1,
                  ExcMessage("Stale or duplicate geometry publication."));
      candidate->publications=accepted->publications+1;
      accepted.swap(candidate); // all geometry/history fields become visible together
      candidate.reset();
    }

    void ghost_check(const State &state) const
    {
      std::array<aspect::LinearAlgebra::Vector,3> owned,ghost;
      for(unsigned int c=0;c<3;++c)
        {
          owned[c].reinit(dofs.locally_owned_dofs(),comm);
          ghost[c].reinit(dofs.locally_owned_dofs(),relevant,comm);
        }
      std::vector<types::global_dof_index> ids(9);
      for(const auto &cell:dofs.active_cell_iterators())if(cell->is_locally_owned())
          {
            cell->get_dof_indices(ids);
            const auto &data=state.tensor.at(cell->id().to_string());
            for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)owned[c][ids[i]]=data[c][i];
          }
      for(unsigned int c=0;c<3;++c){owned[c].compress(VectorOperation::insert);ghost[c]=owned[c];}
      double error=0.;
      unsigned int ghost_cells=0;
      for(const auto &cell:dofs.active_cell_iterators())if(!cell->is_artificial())
          {
            ghost_cells+=cell->is_ghost();
            cell->get_dof_indices(ids);
            const auto &data=state.tensor.at(cell->id().to_string());
            for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)
                error=std::max(error,std::abs(ghost[c][ids[i]]-data[c][i]));
          }
      REQUIRE(dealii::Utilities::MPI::max(error,comm)==0.);
      if(dealii::Utilities::MPI::n_mpi_processes(comm)>1)REQUIRE(dealii::Utilities::MPI::sum(ghost_cells,comm)>0);
    }

    void measure(const State &old,const State &next) const
    {
      const QAnisotropic<2> quad(QIterated<1>(QGauss<1>(12),2),QGauss<1>(12));
      FEValues<2> fe(mapping,element,quad,update_values|update_quadrature_points|update_JxW_values);
      const auto norm=[](const std::array<double,3> &t)
      {return t[0]*t[0]+t[1]*t[1]+std::pow(t[0]+t[1],2)+2*t[2]*t[2];};
      double error=0.,scale=0.,naive=0.,old_energy=0.,new_energy=0.,mean_defect=0.;
      std::vector<double> old_load(old.I.size()),new_load(next.I.size()),ideal_load(next.I.size());
      for(const auto &cell:dofs.active_cell_iterators())if(cell->is_locally_owned())
          {
            fe.reinit(cell);
            const auto &a=old.tensor.at(cell->id().to_string());
            const auto &b=next.tensor.at(cell->id().to_string());
            double cell_error=0.;
            for(unsigned int q=0;q<quad.size();++q)
              {
                const auto x=fe.quadrature_point(q);
                std::array<double,3> u{{0.,0.,0.}},v{{0.,0.,0.}};
                for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)
                    {u[c]+=a[c][i]*fe.shape_value(i,q);v[c]+=b[c][i]*fe.shape_value(i,q);}
                const double before=profile_stress(old,x),after=profile_stress(next,x);
                u[2]-=before;v[2]-=after;
                const double difference=v[2]-u[2],weight=fe.JxW(q);
                error+=weight*2*difference*difference;scale+=weight*norm(u);
                naive+=weight*2*std::pow(after-before,2);
                old_energy+=weight*norm(u)/(4*G);new_energy+=weight*norm(v)/(4*G);
                cell_error+=weight*difference;
                const auto add_load=[&](const State &state,const double stress,std::vector<double> &load)
                {
                  const auto p=state.manager.project_to_normal_profiles(x);
                  if(!p.active)return;
                  const double I=(1-p.xi)*state.I[p.segment_index]+p.xi*state.I[p.segment_index+1];
                  const double value=weight*h(physical_phase(x))*stress/I;
                  load[p.segment_index]+=(1-p.xi)*value;
                  load[p.segment_index+1]+=p.xi*value;
                };
                add_load(old,u[2],old_load);add_load(next,v[2],new_load);add_load(next,u[2],ideal_load);
              }
            mean_defect=std::max(mean_defect,std::abs(cell_error));
          }
      for(auto *value:{&error,&scale,&naive,&old_energy,&new_energy})*value=dealii::Utilities::MPI::sum(*value,comm);
      mean_defect=dealii::Utilities::MPI::max(mean_defect,comm);
      for(auto *load:{&old_load,&new_load,&ideal_load})
        {const auto local=*load;dealii::Utilities::MPI::sum(local,comm,*load);}
      double load_error=0.,load_scale=1.;
      for(unsigned int i=0;i<new_load.size();++i)
        {load_error=std::max(load_error,std::abs(new_load[i]-ideal_load[i]));load_scale=std::max(load_scale,std::abs(ideal_load[i]));}
      const auto storage=[&](const State &state)
      {
        double result=0.;const QGauss<1> rule(12);const auto &fault=state.manager.get_fault(0);
        for(unsigned int s=0;s<fault.n_cells();++s)for(unsigned int q=0;q<rule.size();++q)
            {
              const double t=rule.point(q)[0];
              const double Z=(1-t)*fault.get_properties(s)[0]+t*fault.get_properties(s+1)[0];
              const double I=(1-t)*state.I[s]+t*state.I[s+1];
              result+=fault.vertex(s).distance(fault.vertex(s+1))*rule.weight(q)*Z*Z/(2*G*I);
            }
        return result;
      };
      REQUIRE(std::sqrt(error/scale)<.005);
      REQUIRE(std::abs(new_energy-old_energy)/old_energy<.005);
      REQUIRE(load_error/load_scale<.005);
      REQUIRE(mean_defect<1e-11); // quadrature-independent cell moment screen
      if(dealii::Utilities::MPI::this_mpi_process(comm)==0)
        {
          std::cout<<std::setprecision(17)<<"P3 stress relative L2="<<std::sqrt(error/scale)
                   <<", unchanged-T negative control="<<std::sqrt(naive/scale)
                   <<", bulk energy change="<<new_energy-old_energy
                   <<", mean defect="<<mean_defect<<", weak load defect="<<load_error/load_scale<<std::endl;
          if(const char *directory=std::getenv("ASPECT_P3_OUTPUT"))
            {
              std::ofstream f(std::string(directory)+"/native-metrics.csv");
              f<<std::setprecision(17)<<"stress_relative_L2,unchanged_T_relative_L2,bulk_old,bulk_new,cohesive_old,cohesive_new,mean_defect,weak_load_relative_defect\n"
               <<std::sqrt(error/scale)<<','<<std::sqrt(naive/scale)<<','<<old_energy<<','<<new_energy<<','<<storage(old)<<','<<storage(next)<<','<<mean_defect<<','<<load_error/load_scale<<'\n';
              std::ofstream loads(std::string(directory)+"/native-loads.csv");
              loads<<std::setprecision(17)<<"vertex,old,new,ideal\n";
              for(unsigned int i=0;i<new_load.size();++i)loads<<i<<','<<(i<old_load.size()?old_load[i]:0.)<<','<<new_load[i]<<','<<ideal_load[i]<<'\n';
            }
        }
    }

    void export_state(const State &state,const std::string &prefix) const
    {
      const char *directory=std::getenv("ASPECT_P3_OUTPUT");
      if(!directory)return;
      if(dealii::Utilities::MPI::this_mpi_process(comm)==0)
        {
          std::ofstream phase(std::string(directory)+"/"+prefix+"-phase.csv");
          phase<<std::setprecision(17)<<"x,y,phi,obstacle\n";
          for(unsigned int j=0;j<=ny;++j)for(unsigned int i=0;i<=nx;++i)
              phase<<double(i)/nx<<','<<double(j)/ny<<','<<state.obstacle[j*(nx+1)+i]<<','<<state.obstacle[j*(nx+1)+i]<<'\n';
          std::ofstream f(std::string(directory)+"/"+prefix+"-surface.csv");
          f<<std::setprecision(17)<<"vertex,x,y,Z,cumulative_slip,rate,I\n";
          for(unsigned int i=0;i<state.manager.get_fault(0).n_vertices();++i)
            f<<i<<','<<state.manager.get_fault(0).vertex(i)[0]<<",0.5,"
             <<state.manager.get_fault(0).get_properties(i)[0]<<','<<state.manager.get_fault(0).get_properties(i)[1]
             <<','<<state.manager.get_slip_rate(0)[i]<<','<<state.I[i]<<'\n';
        }
      const auto rank=dealii::Utilities::MPI::this_mpi_process(comm);
      std::ofstream out(std::string(directory)+"/"+prefix+"-cells-"+std::to_string(rank)+".csv");
      out<<std::setprecision(17)<<"cell,x0,y0,hx,hy";
      for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)out<<",c"<<c<<'_'<<i;
      out<<'\n';
      for(const auto &cell:dofs.active_cell_iterators())if(cell->is_locally_owned())
          {
            out<<cell->id().to_string()<<','<<cell->vertex(0)[0]<<','<<cell->vertex(0)[1]
               <<','<<cell->vertex(1)[0]-cell->vertex(0)[0]<<','<<cell->vertex(2)[1]-cell->vertex(0)[1];
            for(const auto &c:state.tensor.at(cell->id().to_string()))for(const auto value:c)out<<','<<value;
            out<<'\n';
          }
      std::ofstream shape(std::string(directory)+"/basis-"+std::to_string(rank)+".csv");
      shape<<std::setprecision(17)<<"i,u,v\n";
      for(unsigned int i=0;i<9;++i)shape<<i<<','<<element.get_unit_support_points()[i][0]<<','<<element.get_unit_support_points()[i][1]<<'\n';
    }
  };
}

TEST_CASE("P3 prescribed cohesive straight append", "[p3_extension]")
{
  ExtensionExceptions exceptions;
  ExtensionFixture fixture;
  auto accepted=fixture.initial();
  const auto initial=checkpoint(*accepted);
  REQUIRE(accepted->tensor.size()==nx*ny);
  REQUIRE(fixture.constraints.n_constraints()==0);
  REQUIRE(accepted->manager.get_fault(0).vertex(32)==Point<2>(.5,.5));
  fixture.boundary_check(accepted->manager);
  // Initialized rounded cap is already present beyond the physical tip. It is
  // not threshold-triggered birth, and the interior tip gets no continuation.
  REQUIRE(physical_phase(Point<2>(.625,.5))>0.);
  REQUIRE(prescribed_phase(Point<2>(.875,.5))==0.);
  REQUIRE_FALSE(accepted->manager.project_to_bulk_source(Point<2>(.625,.5)).active);
  REQUIRE_FALSE(accepted->manager.project_to_bulk_source(Point<2>(.8,.5)).active);
  REQUIRE(checkpoint(*fixture.trial(*accepted,false))==initial);
  auto first=fixture.trial(*accepted,true);
  fixture.boundary_check(first->manager);
  const auto proposed=checkpoint(*first);
  REQUIRE(checkpoint(*accepted)==initial);
  REQUIRE(accepted->associations.empty());
  fixture.export_state(*accepted,"old");
  fixture.export_state(*first,"new");
  fixture.ghost_check(*accepted);
  fixture.ghost_check(*first);
  // Reject the fully prepared geometry/history candidate. Retrying from the
  // same immutable accepted state must reproduce all coefficients, not merely
  // a scalar functional or an iteration count.
  first.reset();
  REQUIRE(checkpoint(*accepted)==initial);
  REQUIRE(accepted->associations.empty());
  auto retry=fixture.trial(*accepted,true);
  REQUIRE(checkpoint(*retry)==proposed);
  REQUIRE(retry->obstacle==accepted->obstacle);
  for(unsigned int i=0;i<=32;++i)
    {
      REQUIRE(retry->manager.get_fault(0).vertex(i)==accepted->manager.get_fault(0).vertex(i));
      for(unsigned int c=0;c<2;++c)
        REQUIRE(retry->manager.get_fault(0).get_properties(i)[c]==accepted->manager.get_fault(0).get_properties(i)[c]);
      REQUIRE(retry->manager.get_slip_rate(0)[i]==accepted->manager.get_slip_rate(0)[i]);
    }
  for(unsigned int i=33;i<=48;++i)
    {
      REQUIRE(retry->manager.get_fault(0).get_properties(i)[0]==0.);
      REQUIRE(retry->manager.get_fault(0).get_properties(i)[1]==0.);
      REQUIRE(retry->manager.get_slip_rate(0)[i]==0.);
    }
  for(const double I:retry->I)REQUIRE(I>0.);
  REQUIRE(retry->I!=accepted->I);
  fixture.measure(*accepted,*retry);
  REQUIRE(retry->manager.project_to_bulk_source(Point<2>(.625,.5)).active);
  REQUIRE_FALSE(retry->manager.project_to_bulk_source(Point<2>(.8,.5)).active);
  const auto high=fixture.trial(*accepted,true,12);
  fixture.export_state(*high,"gauss12");
  double coefficient_difference=0.,coefficient_scale=1.;
  for(const auto &cell:retry->tensor)for(unsigned int c=0;c<3;++c)for(unsigned int i=0;i<9;++i)
      {
        coefficient_difference=std::max(coefficient_difference,std::abs(cell.second[c][i]-high->tensor.at(cell.first)[c][i]));
        coefficient_scale=std::max(coefficient_scale,std::abs(cell.second[c][i]));
      }
  REQUIRE(coefficient_difference/coefficient_scale<1e-11);
  // Restart BEFORE the transition; independently rebuild normalization and
  // projection, then compare the entire candidate with uninterrupted staging.
  auto pre_restart=restore(initial);
  const auto restarted_trial=fixture.trial(*pre_restart,true);
  REQUIRE(checkpoint(*restarted_trial)==proposed);
  ExtensionFixture::publish(accepted,retry);
  REQUIRE(accepted->publications==1);
  REQUIRE_THROWS(ExtensionFixture::publish(accepted,retry));
  auto stale=restore(proposed);
  REQUIRE_THROWS(ExtensionFixture::publish(accepted,stale));
  REQUIRE(accepted->publications==1);
  const auto final=checkpoint(*accepted);
  if(dealii::Utilities::MPI::this_mpi_process(fixture.comm)==0)
    if(const char *directory=std::getenv("ASPECT_P3_OUTPUT"))
      {
        std::ofstream before(std::string(directory)+"/before.archive",std::ios::binary);
        before<<initial;
        std::ofstream after(std::string(directory)+"/after.archive",std::ios::binary);
        after<<final;
      }
  auto post_restart=restore(final);
  REQUIRE(checkpoint(*post_restart)==final);
  REQUIRE(checkpoint(*fixture.trial(*post_restart,false))==final);
  fixture.ghost_check(*post_restart);
  fixture.export_state(*post_restart,"restart");
  if(dealii::Utilities::MPI::this_mpi_process(fixture.comm)==0)
    std::cout<<std::setprecision(17)<<"P3 candidate Gauss8/12 coefficient relative difference="
             <<coefficient_difference/coefficient_scale<<", cells="<<nx*ny<<", DG scalar DoFs="
             <<fixture.dofs.n_dofs()<<", publications="<<accepted->publications<<std::endl;
}

TEST_CASE("Fault append validates birth data before mutation", "[p3_extension_storage]")
{
  ExtensionExceptions exceptions;
  State state;
  state.manager.register_property("generic history",1);
  state.manager.add_reconstructed_fault({{0.,.5},{.5,.5}},{.5,.5});
  state.manager.set_shear_sense(0,1);
  state.manager.get_fault(0).get_properties(0)[0]=3.;
  state.manager.get_fault(0).get_properties(1)[0]=0.;
  state.manager.initialize_slip_rate(0,{2.,0.});
  const auto saved=checkpoint(state);
  state.manager.append_reconstructed_fault(0,{},{},{},{});
  REQUIRE(checkpoint(state)==saved);
  REQUIRE_THROWS(state.manager.append_reconstructed_fault(0,{{.75,.5}},{.5},{{}}, {0.}));
  REQUIRE(checkpoint(state)==saved);
  REQUIRE_THROWS(state.manager.append_reconstructed_fault(0,{{.75,.5}},{.5},{{std::numeric_limits<double>::quiet_NaN()}}, {0.}));
  REQUIRE(checkpoint(state)==saved);
  REQUIRE_THROWS(state.manager.append_reconstructed_fault(0,{{.75,.5}},{.5},{{0.}}, {-1.}));
  REQUIRE(checkpoint(state)==saved);
  REQUIRE_THROWS(state.manager.append_reconstructed_fault(0,{{.5,.5}},{.5},{{0.}}, {0.}));
  REQUIRE(checkpoint(state)==saved);
  state.manager.begin_slip_rate_nonlinear_solve();
  REQUIRE_THROWS(state.manager.append_reconstructed_fault(0,{{.75,.5}},{.5},{{0.}}, {0.}));
  state.manager.rollback_slip_rate_nonlinear_solve();
  REQUIRE(checkpoint(state)==saved);
  state.manager.append_reconstructed_fault(0,{{.75,.5}},{.5},{{0.}}, {0.});
  REQUIRE(state.manager.get_fault(0).geometry_version()==1);
  REQUIRE(state.manager.get_fault(0).get_properties(0)[0]==3.);
  REQUIRE(state.manager.get_timestep_committed_slip_rate(0)==std::vector<double>{2.,0.,0.});
  REQUIRE(state.manager.get_slip_rate(0)==std::vector<double>{2.,0.,0.});
  REQUIRE(state.manager.prescribed_slip_rate_mask()[0].size()==3);
  // Knot ownership must retain the original lower segment; no extrapolated
  // endpoint source may turn an interior physical tip into box completion.
  REQUIRE(state.manager.project_to_normal_profiles({.5,.5}).segment_index==0);
  REQUIRE_FALSE(state.manager.project_to_bulk_source({.8,.5}).active);
  REQUIRE(checkpoint(*restore(checkpoint(state)))==checkpoint(state));
  Manager completed;
  completed.add_reconstructed_fault({{.5,0.},{.5,.5}},{.5,.5});
  completed.initialize_slip_rate(0,{0.,0.});
  completed.enable_bottom_source_continuation(0,{0.,0.},{1.,1.});
  REQUIRE_THROWS(completed.append_reconstructed_fault(0,{{.5,.75}},{.5},{{}}, {0.}));
  REQUIRE(completed.get_fault(0).n_vertices()==2);
}

TEST_CASE("P3 process restart of prescribed geometry transition", "[.][p3_extension_restart]")
{
  ExtensionExceptions exceptions;
  const char *directory=std::getenv("ASPECT_P3_RESTART");
  REQUIRE(directory!=nullptr);
  const auto read=[&](const std::string &name)
  {
    std::ifstream file(std::string(directory)+"/"+name+".archive",std::ios::binary);
    AssertThrow(file.good(),ExcMessage("Missing P3 process-restart archive."));
    return std::string(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
  };
  const auto before=read("before"),after=read("after");
  ExtensionFixture fixture;
  auto state=restore(before);
  REQUIRE(checkpoint(*state)==before);
  auto candidate=fixture.trial(*state,true);
  ExtensionFixture::publish(state,candidate);
  REQUIRE(checkpoint(*state)==after);
  auto terminal=restore(after);
  fixture.export_state(*restore(before),"old");
  fixture.export_state(*terminal,"new");
  REQUIRE(fixture.normalization(terminal->manager,8)==terminal->I);
  REQUIRE(checkpoint(*fixture.trial(*terminal,false))==after);
  fixture.boundary_check(terminal->manager);
  fixture.ghost_check(*terminal);
}
