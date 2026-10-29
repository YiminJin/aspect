/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
// Opt-in native FE/Gamma qualification. No production solver is redirected.
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/cohesive_interface.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <numeric>
#include "cohesive_interface_evolution.h"

namespace aspect { namespace Postprocess
{
  template<int dim> class CohesiveInterfaceNative:public Interface<dim>,public SimulatorAccess<dim>
  {
    using PointState=CohesiveInterface::PointState;
    using PointTrial=CohesiveInterface::PointTrial;
    using Row=std::vector<std::pair<unsigned int,double>>;
    CohesiveInterface::Parameters law;
    mutable ReconstructedFaultManager<dim> interface_manager;
    bool ready=false,resumed=false,exact_jump=false,saved_exact_jump=false;unsigned int ns=0,property=0,gauss_property=0,rate_property=0,publications=0,saved_ns=0;
    double saved_width=0.,load_end=.84,saved_load_end=.84;bool activation_checked=false;std::string saved_mode;
    double wall=0.,accepted_time=0.,width=.025,L=1.,H=.125,yfault=.05;
    std::string mode;
    std::vector<double> weights,qweights,qpositions;
    FullMatrix<double> Q,QP;
    FullMatrix<double> K,N,W,B,normal,normal_lower,normal_upper;
    Vector<double> mean_pressure;
    SparsityPattern sparsity;SparseMatrix<double> A;
    std::map<types::global_dof_index,Row> expansion;
    std::map<types::global_dof_index,types::global_dof_index> upper_pressure;
    double setup_seconds=0.;
    bool finite_surface=false;
    unsigned int first_segment=0,last_segment=0,geometry_publications=0;
    double virgin_age=1.,insertion_time=-1.;
    std::string saved_finite_configuration;
    std::string finite_configuration() const
    {
      std::ostringstream s;s<<std::setprecision(17);
      for(const auto key:{"CI_FINITE","CI_LENGTH","CI_HEIGHT","CI_FIRST","CI_LAST","CI_APPEND_FIRST","CI_APPEND_LAST","CI_INSERT_TIME","CI_SEED_FIRST","CI_SEED_LAST"})
        {const char *value=std::getenv(key);s<<key<<'='<<(value?value:"")<<';';}
      const char *signed_solver=std::getenv("CI_SIGNED_SOLVER");if(signed_solver)s<<"CI_SIGNED_SOLVER="<<signed_solver<<';';
      // Preserve old checkpoint strings when this diagnostic setting is absent.
      if(const char *fraction=std::getenv("CI_FAULT_FRACTION"))s<<"CI_FAULT_FRACTION="<<fraction<<';';
      return s.str();
    }
    bool movable_node(unsigned int i) const
    {return !finite_surface || (i>first_segment && i<last_segment);}
    bool installed(unsigned int q) const
    {return !finite_surface || (q/3>=first_segment && q/3<last_segment);}
    unsigned int first_vertex() const {return finite_surface?first_segment:0;}
    unsigned int last_vertex() const {return finite_surface?last_segment:ns;}
    auto properties(unsigned int i) const
    {return manager().get_fault(0).get_properties(i-first_vertex());}
    static double setting(const char *key,double fallback)
    {const char *v=std::getenv(key);return v?std::stod(v):fallback;}
    std::ofstream output(const std::string &name,bool append=false)const
    {std::ofstream f(this->get_output_directory()+name,append?std::ios::app:std::ios::out);f<<std::setprecision(17);return f;}
    auto &manager()const{return interface_manager;}
    static SymmetricTensor<2,3> dev(const SymmetricTensor<2,dim> &e)
    {SymmetricTensor<2,3>d;for(unsigned int i=0;i<dim;++i)for(unsigned int j=i;j<dim;++j)d[i][j]=e[i][j];const double tr=trace(d)/3.;for(unsigned int i=0;i<3;++i)d[i][i]-=tr;return d;}
    double chi(double y)const{return std::max(0.,1-std::abs(y-yfault)/width)/width;}
    // The enrichment vanishes at both outer walls and has unit tangential jump.
    double primitive(const double y) const
    {
      const double d=y-yfault;
      return d<=-width?0.:d>=width?1.:d<0?.5*std::pow(1+d/width,2):1-.5*std::pow(1-d/width,2);
    }
    double jump_lift(const double y,const bool upper=false) const
    {return (y>yfault || (y==yfault && upper)?1.:0.)-primitive(y);}
    void surface_basis(const Point<dim> &x,unsigned int (&ids)[2],double (&shape)[2],double (&gradient)[2]) const
    {
      const double coordinate=std::min(L,std::max(0.,x[0]))*ns/L;
      const unsigned int segment=std::min(ns-1,static_cast<unsigned int>(coordinate));
      const double xi=coordinate-segment;
      ids[0]=segment;ids[1]=(segment+1)%ns;
      shape[0]=1-xi;shape[1]=xi;gradient[0]=-static_cast<double>(ns)/L;gradient[1]=static_cast<double>(ns)/L;
    }
    SymmetricTensor<2,3> source_strain(const Point<dim> &x,const double shape,
                                      const double gradient,const bool upper=false) const
    {
      SymmetricTensor<2,dim> e;e[0][1]=-.5*chi(x[1])*shape;
      if(exact_jump)e[0][0]=jump_lift(x[1],upper)*gradient;
      return dev(e);
    }
    template<class Cell> void cell_indices(const Cell &cell,std::vector<types::global_dof_index> &dofs) const
    {
      cell->get_dof_indices(dofs);
      if(exact_jump && cell->center()[1]>yfault)
        for(auto &i:dofs){const auto it=upper_pressure.find(i);if(it!=upper_pressure.end())i=it->second;}
    }
    std::string archive()const
    {std::ostringstream out;{aspect::oarchive ar(out);ar<<manager();}return out.str();}
    std::vector<PointState> points()const
    {
      std::vector<PointState> r(ns);for(auto &v:r)v.theta=virgin_age;for(unsigned int i=first_vertex();i<std::min(ns,last_vertex()+1);++i)
        {auto p=properties(i);r[i]={p[property],p[property+1],p[property+2],p[property+3],p[property+4],p[property+5]};}
      return r;
    }
    std::vector<PointState> integration_points()const
    {
      std::vector<PointState> r(3*ns);for(auto &v:r)v.theta=virgin_age;for(unsigned int i=3*first_vertex();i<3*(finite_surface?last_segment:ns);++i)
        {auto p=properties(i/3);const auto offset=gauss_property+6*(i%3);r[i]={p[offset],p[offset+1],p[offset+2],p[offset+3],p[offset+4],p[offset+5]};}
      return r;
    }
    void build()
    {
      const auto started=std::chrono::steady_clock::now();
      AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                  ExcMessage("Native interface fixture is fixed-mesh, serial and 2D."));
      exact_jump=setting("CI_EXACT_JUMP",0)!=0;
      load_end=setting("CI_LOAD_END",.84);
      ns=setting("CI_NX",4);
      finite_surface=setting("CI_FINITE",0)!=0;
      L=setting("CI_LENGTH",1.);H=setting("CI_HEIGHT",.125);yfault=setting("CI_FAULT_FRACTION",.4)*H;
      insertion_time=setting("CI_INSERT_TIME",-1.);
      AssertThrow(saved_finite_configuration.empty() || saved_finite_configuration==finite_configuration(),ExcMessage("Finite-interface restart configuration changed."));
      if(!resumed){first_segment=std::lround(ns*setting("CI_FIRST",.25));last_segment=std::lround(ns*setting("CI_LAST",.75));}
      AssertThrow(!finite_surface || (first_segment<last_segment && last_segment<ns),ExcMessage("Invalid finite surface extent."));
      width=setting("CI_WIDTH",.025);mode=std::getenv("CI_MODE")?std::getenv("CI_MODE"):"formation";
      bool loading_plan_matches=mode!="front" || load_end==saved_load_end;
      if(resumed && !loading_plan_matches && std::getenv("CI_EXTEND_LOAD_END"))
        {
          // Only the not-yet-exercised future boundary plan may be extended.
          // The accepted state must predate the first changed boundary datum.
          AssertThrow(mode=="front" && std::getenv("CI_SIGNED_SOLVER") &&
                      load_end>saved_load_end && accepted_time<saved_load_end,
                      ExcMessage("Loading extension would reinterpret accepted boundary history"));
          auto log=output("loading-plan-extension.csv");log<<"accepted_s,old_load_end_s,new_load_end_s,accepted_histories_changed\n"<<accepted_time<<','<<saved_load_end<<','<<load_end<<",0\n";
          loading_plan_matches=true;
        }
      AssertThrow(!resumed || (ns==saved_ns && width==saved_width && mode==saved_mode && exact_jump==saved_exact_jump && loading_plan_matches),ExcMessage("Interface restart configuration changed."));
      AssertThrow(width>0 && width<yfault && width<H-yfault,ExcMessage("Closed interior support required."));
      Q.reinit(3*ns,ns);qweights.resize(3*ns);qpositions.resize(3*ns);
      QGauss<1> surface_rule(3);FullMatrix<double> mass(ns);
      for(unsigned int i=0;i<ns;++i)for(unsigned int q=0;q<3;++q)
        {const unsigned int k=3*i+q;const double xi=surface_rule.point(q)[0];Q(k,i)=1-xi;Q(k,(i+1)%ns)=xi;qweights[k]=L/ns*surface_rule.weight(q);qpositions[k]=L/ns*(i+xi);for(unsigned int a=0;a<ns;++a)for(unsigned int b=0;b<ns;++b)mass(a,b)+=qweights[k]*Q(k,a)*Q(k,b);}
      mass.gauss_jordan();QP.reinit(3*ns,ns);for(unsigned int q=0;q<3*ns;++q)for(unsigned int j=0;j<ns;++j)for(unsigned int i=0;i<ns;++i)QP(q,j)+=Q(q,i)*mass(i,j)*L/ns;
      auto &m=manager();
      if(m.get_faults().empty())
        {
          property=m.register_property("experimental interface jump damage age reference relaxed-jump absolute-slip",6);
          property=m.get_property_information()[property].position;
          gauss_property=m.register_property("experimental interface Gauss3 physical histories v1",18);
          gauss_property=m.get_property_information()[gauss_property].position;
          rate_property=m.register_property("experimental signed geometric interface velocity",1);rate_property=m.get_property_information()[rate_property].position;
          std::vector<Point<dim>> vertices(last_vertex()-first_vertex()+1);for(unsigned int i=first_vertex();i<=last_vertex();++i){vertices[i-first_vertex()][0]=L*i/ns;vertices[i-first_vertex()][1]=yfault;}
          m.add_reconstructed_fault(vertices,std::vector<double>(vertices.size(),width));
          for(unsigned int i=first_vertex();i<=last_vertex();++i)
            {auto p=properties(i);p[rate_property]=0.;for(unsigned int j=0;j<6;++j)p[property+j]=0.;p[property+2]=1.;p[property+1]=mode=="formation"?0.:1.;}
          for(unsigned int i=first_vertex();i<=last_vertex();++i)for(unsigned int q=0;q<3;++q)
            {auto p=properties(i);const unsigned int offset=gauss_property+6*q;for(unsigned int j=0;j<6;++j)p[offset+j]=0.;p[offset+2]=1.;p[offset+1]=mode=="front"?((qpositions[3*(i%ns)+q]>L*setting("CI_SEED_FIRST",.375) && qpositions[3*(i%ns)+q]<L*setting("CI_SEED_LAST",.625))?1.:0.):mode=="mature"?1.:mode=="nonuniform"?.6+.1*std::cos(2*numbers::PI*qpositions[3*(i%ns)+q]/L):0.;}
        }
      else property=m.get_property_information()[m.get_property_index("experimental interface jump damage age reference relaxed-jump absolute-slip")].position;
      gauss_property=m.get_property_information()[m.get_property_index("experimental interface Gauss3 physical histories v1")].position;
      rate_property=m.get_property_information()[m.get_property_index("experimental signed geometric interface velocity")].position;
      AssertDimension(m.get_fault(0).n_vertices(),last_vertex()-first_vertex()+1);
      const auto &intro=this->introspection();const auto &fe=this->get_fe();const auto &constraints=this->get_current_constraints();
      const auto velocity=intro.extractors.velocities;const auto pressure=intro.extractors.pressure;
      std::map<types::global_dof_index,unsigned int> components;
      std::map<types::global_dof_index,Point<dim>> locations;
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {cell->get_dof_indices(dofs);for(unsigned int j=0;j<dofs.size();++j)
          {const unsigned int c=fe.system_to_component_index(j).first;if(c<dim || c==intro.component_indices.pressure)components[dofs[j]]=c;locations[dofs[j]]=this->get_mapping().transform_unit_to_real_cell(cell,fe.get_unit_support_points()[j]);}}
      if(exact_jump)
        {
          for(const auto &cell:this->get_dof_handler().active_cell_iterators())
            for(const double knot:{yfault-width,yfault,yfault+width})
              AssertThrow(!(cell->vertex(0)[1]<knot-1e-12 && cell->vertex(GeometryInfo<dim>::vertices_per_cell-1)[1]>knot+1e-12),ExcMessage("Exact-jump fixture requires aligned interface and lift knots."));
          auto next=this->get_dof_handler().n_dofs();
          const auto original=components;
          for(const auto &entry:original)
            if(entry.second==intro.component_indices.pressure && std::abs(locations.at(entry.first)[1]-yfault)<1e-12)
              {upper_pressure[entry.first]=next;components[next]=entry.second;locations[next]=locations.at(entry.first);++next;}
          AssertThrow(!upper_pressure.empty(),ExcMessage("Exact jump requires an aligned interface mesh line."));
        }
      std::map<types::global_dof_index,types::global_dof_index> pressure_original;
      for(const auto &entry:upper_pressure)pressure_original[entry.second]=entry.first;
      auto original_index=[&](const types::global_dof_index i){const auto it=pressure_original.find(i);return it==pressure_original.end()?i:it->second;};
      types::global_dof_index gauge=numbers::invalid_dof_index;
      for(const auto &p:components)if(p.second==intro.component_indices.pressure && !constraints.is_constrained(original_index(p.first))){gauge=p.first;break;}
      std::map<types::global_dof_index,unsigned int> masters;
      for(const auto &p:components)if(!constraints.is_constrained(original_index(p.first)) && p.first!=gauge)masters[p.first]=masters.size();
      for(const auto &p:components)
        {
          Row row;
          if(constraints.is_constrained(original_index(p.first)))
            {AssertThrow(constraints.get_inhomogeneity(original_index(p.first))==0.,ExcMessage("Response constraints must be homogeneous."));
             for(const auto &e:*constraints.get_constraint_entries(original_index(p.first)))if(e.first!=gauge)
               {auto master=e.first;if(pressure_original.count(p.first) && upper_pressure.count(master))master=upper_pressure.at(master);row.emplace_back(masters.at(master),e.second);}}
          else if(p.first!=gauge)row.emplace_back(masters.at(p.first),1.);
          expansion[p.first]=row;
        }
      const unsigned int nm=masters.size();DynamicSparsityPattern dsp(nm);
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {cell_indices(cell,dofs);for(const auto i:dofs)for(const auto j:dofs)
          for(const auto &a:expansion[i])for(const auto &b:expansion[j])dsp.add(a.first,b.first);}
      sparsity.copy_from(dsp);A.reinit(sparsity);B.reinit(nm,ns);normal.reinit(ns,nm);FullMatrix<double>D(ns,ns);
      mean_pressure.reinit(nm);weights.assign(ns,0.);
      const QGauss<dim> quadrature(setting("CI_QUADRATURE",3));
      FEValues<dim> values(this->get_mapping(),fe,quadrature,update_values|update_gradients|update_quadrature_points|update_JxW_values);
      SymmetricTensor<2,3>S;S[0][1]=.5;
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {
          values.reinit(cell);cell_indices(cell,dofs);FullMatrix<double> local(dofs.size(),dofs.size());
          for(unsigned int q=0;q<quadrature.size();++q)
            {
              const double w=values.JxW(q);const auto x=values.quadrature_point(q);const double c=chi(x[1]);
              std::vector<SymmetricTensor<2,3>> E(dofs.size());std::vector<double>P(dofs.size()),div(dofs.size());
              for(unsigned int i=0;i<dofs.size();++i)if(!expansion[dofs[i]].empty())
                {E[i]=dev(values[velocity].symmetric_gradient(i,q));P[i]=values[pressure].value(i,q);div[i]=values[velocity].divergence(i,q);
                 for(const auto &a:expansion[dofs[i]])mean_pressure[a.first]+=w*a.second*P[i]/(L*H);}
              for(unsigned int i=0;i<dofs.size();++i)if(!expansion[dofs[i]].empty())for(unsigned int j=0;j<dofs.size();++j)if(!expansion[dofs[j]].empty())
                local(i,j)+=w*(2*scalar_product(E[i],E[j])-div[i]*P[j]-P[i]*div[j]);
              if(c==0 && !exact_jump)continue;
              unsigned int ids[2];double shape[2],gradient[2];surface_basis(x,ids,shape,gradient);
              if(!exact_jump){const auto projection=m.project_to_normal_profiles(x);AssertThrow(projection.active,ExcMessage("Positive lift has no native normal-profile association."));}

              for(unsigned int a=0;a<2;++a)
                {
                  weights[ids[a]]+=w*c*shape[a];
                  for(unsigned int b=0;b<2;++b)D(ids[a],ids[b])+=w*c*c*shape[a]*shape[b];
                  for(unsigned int i=0;i<dofs.size();++i)for(const auto &e:expansion[dofs[i]])
                    {B(e.first,ids[a])+=w*e.second*(-2*scalar_product(E[i],source_strain(x,shape[a],gradient[a]))+P[i]*(exact_jump?jump_lift(x[1])*gradient[a]:0.));
                     normal(ids[a],e.first)+=w*e.second*c*shape[a]*(P[i]-2*E[i][1][1]);}
                }
            }
          for(unsigned int i=0;i<dofs.size();++i)for(unsigned int j=0;j<dofs.size();++j)
            for(const auto &a:expansion[dofs[i]])for(const auto &b:expansion[dofs[j]])A.add(a.first,b.first,a.second*b.second*local(i,j));
        }
      for(unsigned int i=0;i<ns;++i)
        {AssertThrow(std::abs(weights[i]-L/ns)<1e-12,ExcMessage("Geometric lift normalization failed."));
         for(unsigned int j=0;j<nm;++j)normal(i,j)-=weights[i]*mean_pressure[j];}
      SparseDirectUMFPACK inverse;inverse.initialize(A);W.reinit(nm,ns);Vector<double> rhs(nm),solution(nm),residual(nm);double bulk_error=0.;
      for(unsigned int j=0;j<ns;++j)
        {for(unsigned int i=0;i<nm;++i)rhs[i]=B(i,j);inverse.vmult(solution,rhs);
         // Fixed-factor iterative refinement, residual accumulated independently
         // in extended precision. This changes neither operator nor tolerance.
         for(unsigned int refinement=0;refinement<3;++refinement)
           {for(unsigned int row=0;row<nm;++row){long double value=-rhs[row];for(auto entry=A.begin(row);entry!=A.end(row);++entry)value+=static_cast<long double>(entry->value())*solution[entry->column()];residual[row]=value;}
            Vector<double> correction(nm);inverse.vmult(correction,residual);solution-=correction;}
         A.vmult(residual,solution);residual-=rhs;
         bulk_error=std::max(bulk_error,residual.l2_norm()/rhs.l2_norm());for(unsigned int i=0;i<nm;++i)W(i,j)=solution[i];}
      // The constant source has a represented exact Q2 solution: the integral
      // of the piecewise-linear hat minus the imposed affine shear. Separate
      // this invariant mode from numerically solved nonuniform responses.
      Vector<double> constant(nm);
      for(const auto &entry:masters)if(components.at(entry.first)==0)
        {const double y=locations.at(entry.first)[1];
         const double primitive=this->primitive(y);
         constant[entry.second]=primitive-y/H;}
      double correction_norm=0.;
      for(unsigned int i=0;i<nm;++i){double sum=0.;for(unsigned int j=0;j<ns;++j)sum+=W(i,j);const double correction=(constant[i]-sum)/ns;correction_norm=std::max(correction_norm,std::abs(correction));for(unsigned int j=0;j<ns;++j)W(i,j)+=correction;}
      AssertThrow(correction_norm<1e-8,ExcMessage("Constant mode correction is not roundoff."));
      for(unsigned int j=0;j<ns;++j){for(unsigned int i=0;i<nm;++i){solution[i]=W(i,j);rhs[i]=B(i,j);}A.vmult(residual,solution);residual-=rhs;bulk_error=std::max(bulk_error,residual.l2_norm()/rhs.l2_norm());}
      K.reinit(ns,ns);N.reinit(ns,ns);double reciprocity=0.,uniform=0.;
      // Form energy from reconstructed elastic strains, avoiding subtraction
      // of two large source/stiffness contributions in D-B^T A^-1 B.
      std::vector<long double> energy_matrix(ns*ns);
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {values.reinit(cell);cell_indices(cell,dofs);
         for(unsigned int q=0;q<quadrature.size();++q)
           {std::vector<SymmetricTensor<2,3>> strains(ns);
            for(unsigned int i=0;i<dofs.size();++i){const auto strain=dev(values[velocity].symmetric_gradient(i,q));for(const auto &e:expansion[dofs[i]])for(unsigned int j=0;j<ns;++j)strains[j]+=e.second*W(e.first,j)*strain;}
            const auto x=values.quadrature_point(q);unsigned int ids[2];double shape[2],gradient[2];surface_basis(x,ids,shape,gradient);for(unsigned int a=0;a<2;++a)strains[ids[a]]+=source_strain(x,shape[a],gradient[a]);
            for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)energy_matrix[i*ns+j]+=static_cast<long double>(2*law.G*values.JxW(q))*scalar_product(strains[i],strains[j]);}}
      for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)
        {double n=0.;for(unsigned int k=0;k<nm;++k)n+=normal(i,k)*W(k,j);K(i,j)=energy_matrix[i*ns+j]/weights[i];N(i,j)=law.G*n/weights[i];}
      if(exact_jump) build_trace_operator();
      auto arithmetic=output("operator_arithmetic.csv");arithmetic<<"constant_response_correction_m_per_m\n"<<correction_norm<<'\n';
      for(unsigned int i=0;i<ns;++i)
        {double sum=0.;for(unsigned int j=0;j<ns;++j){sum+=K(i,j);reciprocity=std::max(reciprocity,std::abs(weights[i]*K(i,j)-weights[j]*K(j,i)));}
         uniform=std::max(uniform,std::abs(sum-law.G/H));}
      auto raw_checks=output("raw_operator_checks.csv");raw_checks<<"bulk_relative_residual,reciprocity_Pa,uniform_error_Pa_m\n"<<bulk_error<<','<<reciprocity<<','<<uniform<<std::endl;
      AssertThrow(bulk_error<1e-10 && reciprocity<1e-7 && uniform<1e-5,ExcMessage("Native mixed Schur verification failed."));
      auto op=output("operators.csv");op<<"i,j,K_Pa_m,N_Pa_m,weight_m\n";for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)op<<i<<','<<j<<','<<K(i,j)<<','<<N(i,j)<<','<<weights[i]<<'\n';
      setup_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
      auto meta=output("operator_checks.csv");meta<<"masters,cells,surface_unknowns,width_m,quadrature,bulk_relative_residual,reciprocity_Pa,uniform_error_Pa_m,setup_seconds\n"<<nm<<','<<this->get_triangulation().n_global_active_cells()<<','<<ns<<','<<width<<','<<quadrature.size()<<','<<bulk_error<<','<<reciprocity<<','<<uniform<<','<<setup_seconds<<'\n';
      std::vector<PointState> probe(ns);for(unsigned int i=0;i<ns;++i)probe[i].relaxed_jump=1e-4*(1+.3*std::cos(2*numbers::PI*i/ns));
      field_check(probe,2e-4,"frozen_fields.csv");
      // Read-only reconstruction of saved physical states on this identical
      // native FE space, used for bulk maximum-shear and work diagnostics.
      if(const char *path=std::getenv("CI_FIELD_INPUT"))
        {
          std::ifstream samples(path);double sample_time,sample_wall;
          while(samples>>sample_time>>sample_wall)
            {std::vector<PointState> state(ns);for(auto &p:state)AssertThrow(bool(samples>>p.relaxed_jump),ExcMessage("Truncated physical field sample."));
             field_check(state,sample_wall,"physical-fields-"+std::to_string(std::lround(1e6*sample_time))+".csv");}
        }
      ready=true;
    }
    // Evaluate both physical traces independently. No band average enters the
    // exact-jump normal pressure. Their mismatch is reported, not hidden.
    void build_trace_operator()
    {
      const auto &fe=this->get_fe();const auto &intro=this->introspection();
      const auto velocity=intro.extractors.velocities;const auto pressure=intro.extractors.pressure;
      std::vector<FullMatrix<double>> trace_s(2,FullMatrix<double>(ns)),trace_n(2,FullMatrix<double>(ns));
      std::vector<FullMatrix<double>> jump_x(2,FullMatrix<double>(ns)),jump_y(2,FullMatrix<double>(ns));
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      double quadrature_error=0.;
      for(const unsigned int order:{3u,6u})
        {
          std::vector<FullMatrix<double>> shear(2,FullMatrix<double>(ns)),pn(2,FullMatrix<double>(ns));
          FEFaceValues<dim> face(this->get_mapping(),fe,QGauss<dim-1>(order),update_values|update_gradients|update_quadrature_points|update_JxW_values);
          std::vector<double> pmean(ns);
          for(unsigned int j=0;j<ns;++j)for(unsigned int i=0;i<W.m();++i)pmean[j]+=mean_pressure[i]*W(i,j);
          for(const auto &cell:this->get_dof_handler().active_cell_iterators())
            for(unsigned int f=0;f<GeometryInfo<dim>::faces_per_cell;++f)
              if(std::abs(cell->face(f)->center()[1]-yfault)<1e-12)
                {
                  face.reinit(cell,f);cell_indices(cell,dofs);const unsigned int side=cell->center()[1]>yfault?1:0;
                  for(unsigned int q=0;q<face.n_quadrature_points;++q)
                    {
                      auto x=face.quadrature_point(q);x[1]=yfault;
                      unsigned int ids[2];double shape[2],gradient[2];surface_basis(x,ids,shape,gradient);
                      std::vector<SymmetricTensor<2,3>> strains(ns);std::vector<double> pressures(ns);
                      for(unsigned int j=0;j<ns;++j)pressures[j]=-pmean[j];
                      for(unsigned int i=0;i<dofs.size();++i)
                        for(const auto &e:expansion[dofs[i]])for(unsigned int j=0;j<ns;++j)
                          {strains[j]+=e.second*W(e.first,j)*dev(face[velocity].symmetric_gradient(i,q));pressures[j]+=e.second*W(e.first,j)*face[pressure].value(i,q);}
                      if(order==3)
                        {
                          std::vector<Tensor<1,dim>> displacement(ns);
                          for(unsigned int i=0;i<dofs.size();++i)for(const auto &e:expansion[dofs[i]])for(unsigned int j=0;j<ns;++j)
                            displacement[j]+=e.second*W(e.first,j)*face[velocity].value(i,q);
                          for(unsigned int a=0;a<2;++a)displacement[ids[a]][0]+=jump_lift(x[1],side==1)*shape[a];
                          for(unsigned int a=0;a<2;++a)for(unsigned int j=0;j<ns;++j)
                            {jump_x[side](ids[a],j)+=face.JxW(q)*shape[a]*displacement[j][0]/weights[ids[a]];jump_y[side](ids[a],j)+=face.JxW(q)*shape[a]*displacement[j][1]/weights[ids[a]];}
                        }
                      for(unsigned int a=0;a<2;++a)strains[ids[a]]+=source_strain(x,shape[a],gradient[a],side==1);
                      for(unsigned int a=0;a<2;++a)for(unsigned int j=0;j<ns;++j)
                        {const double w=face.JxW(q)*shape[a]/weights[ids[a]];
                         shear[side](ids[a],j)+=w*2*law.G*strains[j][0][1];
                         pn[side](ids[a],j)+=w*law.G*(pressures[j]-2*strains[j][1][1]);}
                    }
                }
          if(order==3){trace_s=shear;trace_n=pn;}
          else for(unsigned int side=0;side<2;++side)for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)
            quadrature_error=std::max({quadrature_error,std::abs(shear[side](i,j)-trace_s[side](i,j)),std::abs(pn[side](i,j)-trace_n[side](i,j))});
        }
      auto csv=output("trace_operators.csv");csv<<"i,j,shear_lower_Pa_m,shear_upper_Pa_m,normal_lower_Pa_m,normal_upper_Pa_m\n";
      double norm=0.,shear_error=0.,normal_jump=0.,jump_error=0.;
      normal_lower=trace_n[0];normal_upper=trace_n[1];
      for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)
        {
          N(i,j)=.5*(trace_n[0](i,j)+trace_n[1](i,j));
          double mass=0.;for(unsigned int q=0;q<3*ns;++q)mass+=qweights[q]*Q(q,i)*Q(q,j)/weights[i];
          jump_error=std::max({jump_error,std::abs(jump_x[1](i,j)-jump_x[0](i,j)-mass),std::abs(jump_y[1](i,j)-jump_y[0](i,j))});
          csv<<i<<','<<j<<','<<trace_s[0](i,j)<<','<<trace_s[1](i,j)<<','<<trace_n[0](i,j)<<','<<trace_n[1](i,j)<<'\n';
          norm+=K(i,j)*K(i,j);shear_error+=std::pow(.5*(trace_s[0](i,j)+trace_s[1](i,j))+K(i,j),2);
          normal_jump+=std::pow(trace_n[0](i,j)-trace_n[1](i,j),2);
        }
      auto checks=output("trace_checks.csv");checks<<"surface_quadrature_error_Pa_m,mean_shear_trace_vs_weak_relative,normal_trace_jump_over_K,displacement_jump_error\n"<<quadrature_error<<','<<std::sqrt(shear_error/norm)<<','<<std::sqrt(normal_jump/norm)<<','<<jump_error<<'\n';
      AssertThrow(quadrature_error<1e-5 && jump_error<1e-12,ExcMessage("Sharp trace quadrature disagreement."));
    }
    void field_check(const std::vector<PointState>&p,const double wall_value,const std::string &name,const bool all_points=false)const
    {
      const auto &fe=this->get_fe();const auto &intro=this->introspection();
      const auto velocity=intro.extractors.velocities;const auto pressure=intro.extractors.pressure;
      const QGauss<dim> quadrature(6);
      FEValues<dim> f(this->get_mapping(),fe,quadrature,update_values|update_gradients|update_quadrature_points|update_JxW_values);
      Vector<double> displacement(W.m());for(unsigned int i=0;i<W.m();++i)for(unsigned int j=0;j<ns;++j)displacement[i]+=W(i,j)*p[j].relaxed_jump;
      const double pmean=mean_pressure*displacement;
      std::vector<double> shear(ns),pn(ns);double energy=0.,minimum=law.p0,trace_error=0.;
      auto file=output(name);file<<"x_m,y_m,JxW_m2,chi_inv_m,tau_xx_Pa,tau_yy_Pa,tau_zz_Pa,tau_xy_Pa,p_perturbation_Pa,pn_Pa\n";
      std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
      for(const auto &cell:this->get_dof_handler().active_cell_iterators())
        {
          f.reinit(cell);cell_indices(cell,dofs);
          for(unsigned int q=0;q<quadrature.size();++q)
            {
              SymmetricTensor<2,3> strain;strain[0][1]=.5*wall_value/H;double pressure_value=-pmean;
              for(unsigned int i=0;i<dofs.size();++i)
                {const auto it=expansion.find(dofs[i]);if(it==expansion.end())continue;double v=0.;for(const auto &e:it->second)v+=e.second*displacement[e.first];
                 strain+=v*dev(f[velocity].symmetric_gradient(i,q));pressure_value+=v*f[pressure].value(i,q);}
              const auto x=f.quadrature_point(q);const double c=chi(x[1]);
              const auto a=manager().project_to_normal_profiles(x);double z=0.;
              if(c>0 && !exact_jump){AssertThrow(a.active,ExcMessage("Missing field association."));z=(1-a.xi)*p[a.segment_index%ns].relaxed_jump+a.xi*p[(a.segment_index+1)%ns].relaxed_jump;}
              unsigned int ids[2];double shape[2],gradient[2];surface_basis(x,ids,shape,gradient);
              if(exact_jump){for(unsigned int j=0;j<2;++j)strain+=p[ids[j]].relaxed_jump*source_strain(x,shape[j],gradient[j]);}
              else strain[0][1]-=.5*c*z;
              const auto stress=2*law.G*strain;
              const double normal_pressure=law.p0+law.G*pressure_value-stress[1][1];minimum=std::min(minimum,normal_pressure);
              trace_error=std::max(trace_error,std::abs(trace(stress)));energy+=law.G*scalar_product(strain,strain)*f.JxW(q);
              if(exact_jump){for(unsigned int j=0;j<2;++j)shear[ids[j]]+=f.JxW(q)*(-scalar_product(stress,source_strain(x,shape[j],gradient[j]))+law.G*pressure_value*jump_lift(x[1])*gradient[j]);}
              else if(c>0){for(unsigned int j=0;j<2;++j){shear[ids[j]]+=f.JxW(q)*c*shape[j]*stress[0][1];pn[ids[j]]+=f.JxW(q)*c*shape[j]*normal_pressure;}}
              if(q==quadrature.size()/2 || all_points || std::getenv("CI_ALL_FIELDS"))file<<x[0]<<','<<x[1]<<','<<f.JxW(q)<<','<<c<<','<<stress[0][0]<<','<<stress[1][1]<<','<<stress[2][2]<<','<<stress[0][1]<<','<<law.G*pressure_value<<','<<normal_pressure<<'\n';
            }
        }
      double shear_error=0.,normal_error=0.;for(unsigned int i=0;i<ns;++i)
        {double s=law.G/H*wall_value,n=law.p0;for(unsigned int j=0;j<ns;++j){s-=K(i,j)*p[j].relaxed_jump;n+=N(i,j)*p[j].relaxed_jump;}
         shear_error=std::max(shear_error,std::abs(shear[i]/weights[i]-s));if(!exact_jump)normal_error=std::max(normal_error,std::abs(pn[i]/weights[i]-n));}
      const double energy_error=std::abs(energy-bulk_energy(wall_value,p));
      auto checks=output("field_checks.csv");checks<<"shear_work_error_Pa,normal_work_error_Pa,energy_error_J_m,trace_error_Pa,min_point_pn_Pa,band_normal_check_applicable\n"<<shear_error<<','<<normal_error<<','<<energy_error<<','<<trace_error<<','<<minimum<<','<<(!exact_jump)<<'\n';
      AssertThrow(shear_error<1e-8 && normal_error<1e-8 && energy_error<1e-10 && trace_error<1e-10 && minimum>0,ExcMessage("Full FE stress/work/pressure check failed."));
    }
    struct Trial
    {std::vector<PointTrial> point;std::vector<PointState> nodal;std::vector<double>increment,stress,pressure,qstress,qpressure;std::vector<bool> active,tied;double weak_error=0.,release_curvature=0.;unsigned int certificate_iterations=0;FullMatrix<double>jacobian;double wall,error,curvature=0.,minimum_trace_pressure=std::numeric_limits<double>::max();unsigned int iterations,base_publications,base_geometry_publications=0;};
#include "cohesive_interface_front.inc"
#include "cohesive_interface_signed_front.inc"
#include "cohesive_interface_insertion.inc"
#include "cohesive_interface_domain_diagnostics.inc"
#include "cohesive_interface_derivative_reference.inc"
    Trial trial(const std::vector<PointState>&old,double dt,double du)const
    {
      if(mode=="front")return std::getenv("CI_SIGNED_SOLVER")?signed_front_trial(old,dt,du):front_trial(old,dt,du);
      const auto oldq=integration_points();const double beta=1/(1+law.G/law.eta*dt);Trial out;out.base_publications=publications;out.base_geometry_publications=geometry_publications;out.wall=beta*(wall+du);out.increment.assign(ns,0.);
      std::vector<bool>active(ns);std::vector<double>sign(ns);Vector<double>f(ns);
      auto evaluate=[&](const std::vector<double>&inc)
      {
        out.point.resize(3*ns);out.stress.assign(ns,law.G/H*out.wall);out.pressure.assign(ns,law.p0);out.jacobian.reinit(ns,ns);out.nodal=old;
        for(unsigned int i=0;i<ns;++i){out.nodal[i].jump+=inc[i];out.nodal[i].relaxed_jump=beta*(old[i].relaxed_jump+inc[i]);out.nodal[i].absolute_slip+=std::abs(inc[i]);out.nodal[i].damage=0.;
          for(unsigned int j=0;j<ns;++j){const double z=beta*(old[j].relaxed_jump+inc[j]);out.stress[i]-=K(i,j)*z;out.pressure[i]+=N(i,j)*z;out.jacobian(i,j)=-beta*K(i,j);}f[i]=out.stress[i];}
        out.qstress.assign(3*ns,0.);out.qpressure.assign(3*ns,0.);out.minimum_trace_pressure=std::numeric_limits<double>::max();
        for(unsigned int q=0;q<3*ns;++q)
          {double increment=0.;for(unsigned int j=0;j<ns;++j){increment+=Q(q,j)*inc[j];out.qstress[q]+=QP(q,j)*out.stress[j];out.qpressure[q]+=QP(q,j)*out.pressure[j];}
           if(exact_jump)
             for(const auto *trace_matrix:{&normal_lower,&normal_upper})
               {double pn=law.p0;for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)pn+=QP(q,i)*(*trace_matrix)(i,j)*beta*(old[j].relaxed_jump+inc[j]);out.minimum_trace_pressure=std::min(out.minimum_trace_pressure,pn);}
           else out.minimum_trace_pressure=std::min(out.minimum_trace_pressure,out.qpressure[q]);
           AssertThrow(out.minimum_trace_pressure>0,ExcMessage("Lost closed contact on actual interface trace."));
           out.point[q]=CohesiveInterface::evaluate(law,oldq[q],increment,out.qpressure[q],dt,sign[q/3]);
           for(unsigned int i=0;i<ns;++i){const double weight=qweights[q]*Q(q,i)/weights[i];f[i]-=weight*out.point[q].resistance;out.nodal[i].damage+=weight*out.point[q].state.damage;
             for(unsigned int j=0;j<ns;++j){double dn=0.;for(unsigned int k=0;k<ns;++k)dn+=QP(q,k)*N(k,j);out.jacobian(i,j)-=weight*(out.point[q].derivative*Q(q,j)+out.point[q].pressure_derivative*beta*dn);}}}
        // Nodal damage slot is an activity flag only. Physical irreversible
        // damage and age live at surface Gauss points.
        for(unsigned int i=0;i<ns;++i)out.nodal[i].damage=std::max(old[i].damage,out.nodal[i].damage>0?1.:0.);
      };
      for(unsigned int i=0;i<ns;++i)
        {double stress=law.G/H*out.wall;for(unsigned int j=0;j<ns;++j)stress-=K(i,j)*beta*old[j].relaxed_jump;
         active[i]=old[i].damage>0 || std::abs(stress)>law.peak;sign[i]=stress>=0?1.:-1.;}
      for(unsigned int outer=0;outer<8;++outer)
        {
          std::vector<unsigned int> ids;for(unsigned int i=0;i<ns;++i)if(active[i])ids.push_back(i);
          for(unsigned int iteration=0;iteration<50;++iteration)
            {
              evaluate(out.increment);out.error=0.;for(auto i:ids)out.error=std::max(out.error,std::abs(f[i]));out.iterations=iteration+1;
              if(out.error<=1e-8)break;
              AssertThrow(iteration<49,ExcMessage("Native interface Newton limit."));
              FullMatrix<double> jac(ids.size());Vector<double>rhs(ids.size()),step(ids.size());
              for(unsigned int i=0;i<ids.size();++i){rhs[i]=-f[ids[i]];for(unsigned int j=0;j<ids.size();++j)jac(i,j)=out.jacobian(ids[i],ids[j]);}
              jac.gauss_jordan();jac.vmult(step,rhs);const double error=out.error;bool found=false;
              for(unsigned int ls=0;ls<25;++ls)
                {auto candidate=out.increment;double scale=std::pow(.5,ls);bool admissible=true;
                 for(unsigned int i=0;i<ids.size();++i){candidate[ids[i]]+=scale*step[i];if(old[ids[i]].damage==0 && candidate[ids[i]]*sign[ids[i]]<0)admissible=false;}
                 if(!admissible)continue;
                 evaluate(candidate);double e=0.;for(auto i:ids)e=std::max(e,std::abs(f[i]));
                 if(e<error || e<=1e-8){out.increment=candidate;found=true;break;}}
              AssertThrow(found,ExcMessage("Native interface line search failed."));
            }
          evaluate(out.increment);bool added=false;
          for(unsigned int i=0;i<ns;++i)if(!active[i] && std::abs(out.stress[i])>law.peak+1e-8){active[i]=true;sign[i]=std::copysign(1.,out.stress[i]);added=true;}
          if(!added)
            {out.error=0.;for(unsigned int i=0;i<ns;++i)if(active[i])out.error=std::max(out.error,std::abs(f[i]));
             for(unsigned int q=0;q<3*ns;++q)if(out.point[q].state.damage==0)out.point[q].q=out.qstress[q];
             out.active=active;
             if(!ids.empty()){LAPACKFullMatrix<double> symmetric(ids.size());for(unsigned int i=0;i<ids.size();++i)for(unsigned int j=0;j<ids.size();++j)symmetric(i,j)=-.5*(out.jacobian(ids[i],ids[j])+out.jacobian(ids[j],ids[i]));symmetric.compute_eigenvalues();out.curvature=std::numeric_limits<double>::max();for(unsigned int i=0;i<ids.size();++i)out.curvature=std::min(out.curvature,symmetric.eigenvalue(i).real());AssertThrow(out.curvature>0,ExcMessage("Native interface candidate fails relaxed incremental curvature; not published."));}
             return out;}
        }
      AssertThrow(false,ExcMessage("Native interface active set failed."));return out;
    }
    double bulk_energy(double w,const std::vector<PointState>&p)const
    {
      double E=.5*law.G*L/H*w*w;
      for(unsigned int i=0;i<ns;++i){E-=law.G/H*w*weights[i]*p[i].relaxed_jump;for(unsigned int j=0;j<ns;++j)E+=.5*weights[i]*p[i].relaxed_jump*K(i,j)*p[j].relaxed_jump;}return E;
    }
    void publish(const Trial&t,double time)
    {
      AssertThrow(time>accepted_time && t.base_publications==publications && t.base_geometry_publications==geometry_publications && t.error<=1e-8 && t.curvature>=0,ExcMessage("Stale or unconverged interface publication."));
      auto staged=manager().clone_prescribed_geometry();
      for(unsigned int i=first_vertex();i<=last_vertex();++i)
        {const auto &s=t.nodal[i%ns];
         AssertThrow(s.damage>=points()[i%ns].damage && s.damage<=1 && s.theta>0 && std::isfinite(s.jump) && std::isfinite(s.relaxed_jump),ExcMessage("Invalid candidate before interface publication."));
         auto p=staged->get_fault(0).get_properties(i-first_vertex());p[rate_property]=t.increment[i%ns]/(time-accepted_time);p[property]=s.jump;p[property+1]=s.damage;p[property+2]=s.theta;p[property+3]=s.reference;p[property+4]=s.relaxed_jump;p[property+5]=s.absolute_slip;}
      const auto oldq=integration_points();for(unsigned int i=first_vertex();i<=last_vertex();++i)for(unsigned int q=0;q<3;++q)
        {const auto &s=t.point[3*(i%ns)+q].state;AssertThrow(s.damage>=oldq[3*(i%ns)+q].damage && s.damage<=1 && s.theta>0,ExcMessage("Invalid Gauss history before publication."));
         auto p=staged->get_fault(0).get_properties(i-first_vertex());const auto offset=gauss_property+6*q;p[offset]=s.jump;p[offset+1]=s.damage;p[offset+2]=s.theta;p[offset+3]=s.reference;p[offset+4]=s.relaxed_jump;p[offset+5]=s.absolute_slip;}
      manager().swap_prescribed_geometry(*staged);wall=t.wall;virgin_age+=law.V0/law.Dc*(time-accepted_time);accepted_time=time;++publications;
    }
    double speed(double t)const
    {return mode=="front"?(t<load_end?.001:0.):mode=="mature"?(t<2?.001:t<4?.002:t<4.5?0.:-.001):.001;}
    public:
    unsigned cutback_depth=0;
    std::pair<std::string,std::string> execute(TableHandler&)override
    {
      if(!ready)build();
      double time=this->get_time();
      if(std::getenv("CI_SIGNED_SOLVER") && std::abs(time-load_end)<=32*std::numeric_limits<double>::epsilon()*std::max(1.,load_end))time=load_end;
      return advance_to(time);
    }
    std::pair<std::string,std::string> advance_to(const double time)
    {
      if(!ready)build();
      if(time<=accepted_time)return {"Native interface","initialized"};
      const auto started=std::chrono::steady_clock::now();const double dt=time-accepted_time,du=mode=="front"?.001*(std::min(time,load_end)-std::min(accepted_time,load_end)):dt*speed(.5*(time+accepted_time));
      if(finite_surface && geometry_publications==0 && insertion_time>=0 && accepted_time>=insertion_time-1e-12)insert_surface();
      const auto old=points();const double oldwall=wall;const auto before_trial=archive();const unsigned before_publications=publications;Trial t;
      dealii::deal_II_exceptions::disable_abort_on_exception();
      try
        {t=trial(old,dt,du);
         if(const char *forced=std::getenv("CI_TEST_CUTBACK_TIME"))
           if(cutback_depth==0 && std::abs(time-std::stod(forced))<1e-10)
             {auto f=output("injected-rejected-candidate.csv");f<<"target_s,q,jump_m,damage,theta\n";for(unsigned q=0;q<3*ns;++q)f<<time<<','<<q<<','<<t.point[q].state.jump<<','<<t.point[q].state.damage<<','<<t.point[q].state.theta<<'\n';AssertThrow(false,ExcMessage("Injected nonlinear line search failure for cutback lifecycle test"));}}

      catch(const std::exception &e)
        {
          dealii::deal_II_exceptions::enable_abort_on_exception();
          AssertThrow(archive()==before_trial && wall==oldwall && publications==before_publications,ExcMessage("Rejected nonlinear trial changed accepted native history"));
          if(std::getenv("CI_SIGNED_SOLVER"))
            {std::map<std::string,std::string> state;save(state);std::ofstream f(this->get_output_directory()+"last-accepted-on-rejection.archive");aspect::oarchive ar(f);ar<<state;}
          const std::string message=e.what();const bool numerical=message.find("line search")!=std::string::npos || message.find("iteration limit")!=std::string::npos || message.find("certificate failed")!=std::string::npos;
          if(!std::getenv("CI_SIGNED_SOLVER") || !numerical || cutback_depth>=4 || dt<.000625)throw;
          auto log=output("native-cutbacks.csv",true);if(log.tellp()==0)log<<"incoming_s,rejected_target_s,dt_s,depth,accepted_unchanged\n";log<<accepted_time<<','<<time<<','<<dt<<','<<cutback_depth<<",1\n";log.close();
          ++cutback_depth;const double middle=accepted_time+.5*dt;
          try{advance_to(middle);auto result=advance_to(time);--cutback_depth;return result;}
          catch(...){--cutback_depth;throw;}
        }
      dealii::deal_II_exceptions::enable_abort_on_exception();
      if(finite_surface && setting("CI_BUFFER_M",0)>0)
        {
          bool reached=false;for(unsigned int q=0;q<3*ns;++q)if(t.point[q].state.damage>=.01 && (qpositions[q]<L*first_segment/ns+setting("CI_BUFFER_M",0) || qpositions[q]>L*last_segment/ns-setting("CI_BUFFER_M",0)))reached=true;
          if(reached && std::getenv("CI_SIGNED_SOLVER"))
            {std::map<std::string,std::string> accepted;save(accepted);std::ofstream a(this->get_output_directory()+"last-accepted-at-buffer.archive");aspect::oarchive ar(a);ar<<accepted;
             auto f=output("rejected-buffer-trial.csv");f<<"target_s,incoming_s,publications,q,x_m,damage,jump_m,V_m_s,shear_Pa,pn_Pa\n";
             for(unsigned q=0;q<3*ns;++q)f<<time<<','<<accepted_time<<','<<publications<<','<<q<<','<<qpositions[q]<<','<<t.point[q].state.damage<<','<<t.point[q].state.jump<<','<<t.point[q].velocity<<','<<t.qstress[q]<<','<<t.qpressure[q]<<'\n';}
          AssertThrow(!reached,ExcMessage("Cohesive zone reached declared finite candidate buffer; stop before publication."));
        }
      if(std::getenv("CI_REMOTE_GUARD"))
        {double maximum=0.;for(unsigned q=0;q<3*ns;++q)if(!installed(q))maximum=std::max(maximum,std::abs(t.qstress[q]));
         if(maximum>=law.peak){auto f=output("rejected-remote.csv");f<<"target_s,max_remote_weak_shear_Pa\n"<<time<<','<<maximum<<'\n';AssertThrow(false,ExcMessage("Remote weak shear reached intact strength; not published"));}}
      const auto oldq=integration_points();auto fresh=t.nodal;double bond=0.,oldbond=0.,fracture=0.,friction=0.,radiation=0.,surface_error=0.,work=0.,trap=0.;
      for(unsigned int q=0;q<3*ns;++q)
        {const auto&s=t.point[q];double increment=0.;for(unsigned int i=0;i<ns;++i)increment+=Q(q,i)*t.increment[i];bond+=qweights[q]*s.storage;
         const double e=oldq[q].jump-oldq[q].reference;const double eb=oldq[q].damage==0 || oldq[q].damage==1?0:.5*e*e*(1-oldq[q].damage)/(2*law.Gc/(law.peak*law.peak)*oldq[q].damage);
         oldbond+=qweights[q]*eb;fracture+=qweights[q]*law.Gc*(s.state.damage-oldq[q].damage);friction+=qweights[q]*s.friction;radiation+=qweights[q]*s.radiation;
         surface_error+=qweights[q]*(s.q*increment-s.storage+eb-law.Gc*(s.state.damage-oldq[q].damage));}
      for(unsigned int i=0;i<ns;++i){double os=law.G/H*oldwall;for(unsigned int j=0;j<ns;++j)os-=K(i,j)*old[j].relaxed_jump;work+=weights[i]*t.stress[i]*du;trap+=weights[i]*.5*(os+t.stress[i])*du;}
      const double E=bulk_energy(t.wall,fresh),oldE=bulk_energy(oldwall,old),maxwell=2*law.G/law.eta*dt*E;
      auto difference=fresh;for(unsigned int i=0;i<ns;++i)difference[i].relaxed_jump-=old[i].relaxed_jump;
      const double be=bulk_energy(t.wall-oldwall,difference),defect=work-(E-oldE+bond-oldbond+fracture+maxwell+friction+radiation+be+surface_error);
      AssertThrow(std::abs(defect)<1e-10*std::max(1.,std::abs(work)),ExcMessage("Native interface work ledger failed."));
      bool activation=false;for(unsigned int q=0;q<3*ns;++q)activation=activation || (oldq[q].damage==0 && t.point[q].state.damage>0);
      if(publications==0 || (activation && !activation_checked))
        {
          const auto before=archive();const auto retry=trial(old,dt,du);
          for(unsigned int i=0;i<ns;++i)AssertThrow(retry.increment[i]==t.increment[i],ExcMessage("Retry changed interface candidate."));
          AssertThrow(before==archive(),ExcMessage("Trial modified accepted manager."));
          auto bad=t;bad.point[3*first_vertex()].state.damage=-1;bool rejected=false;
          dealii::deal_II_exceptions::disable_abort_on_exception();try{publish(bad,time);}catch(const std::exception&){rejected=true;}dealii::deal_II_exceptions::enable_abort_on_exception();
          AssertThrow(rejected && before==archive() && wall==oldwall && publications==t.base_publications,ExcMessage("Atomic rejection failed."));
          auto checks=output("lifecycle.csv",true);if(checks.tellp()==0)checks<<"time_s,activation,rejected_unchanged,retry_bitwise\n";checks<<time<<','<<activation<<",1,1\n";
        }
      if(((publications==0 || publications==100) && mode=="nonuniform") || (mode=="front" && (activation || publications==150)))
        {
          const double beta=1/(1+law.G/law.eta*dt);auto report=output("derivative_checks.csv",true);if(report.tellp()==0)report<<"time_s,epsilon_m,relative_error\n";
          auto force=[&](const std::vector<double>&inc){Vector<double>f(ns),pn(ns);for(unsigned int i=0;i<ns;++i){f[i]=law.G/H*t.wall;pn[i]=law.p0;for(unsigned int j=0;j<ns;++j){f[i]-=K(i,j)*beta*(old[j].relaxed_jump+inc[j]);pn[i]+=N(i,j)*beta*(old[j].relaxed_jump+inc[j]);}}for(unsigned int q=0;q<3*ns;++q){double dq=0,pq=0;for(unsigned int i=0;i<ns;++i){dq+=Q(q,i)*inc[i];pq+=QP(q,i)*pn[i];}const auto r=CohesiveInterface::evaluate(law,oldq[q],dq,pq,dt);for(unsigned int i=0;i<ns;++i)f[i]-=qweights[q]*Q(q,i)*r.resistance/weights[i];}return f;};
          double final_error=0.;
          double rate_scale=std::numeric_limits<double>::max();
          for(unsigned int i=0;i<3*ns;++i)
            {bool responsive=mode!="front";for(unsigned int j=0;j<ns && mode=="front";++j)responsive=responsive || (t.active[j] && Q(i,j)>0);if(!responsive)continue;
             const double transition=2*law.V0*std::exp(-(law.mu0+law.evolution*std::log(t.point[i].state.theta))/law.direct);
             rate_scale=std::min(rate_scale,dt*std::hypot(t.point[i].velocity,transition));}
          for(const double eps:{rate_scale*.001,rate_scale*.0005,rate_scale*.00025})
            {double error=0.,norm=0.;for(unsigned int j=0;j<ns;++j){if(mode=="front" && !t.active[j])continue;auto plus=t.increment,minus=t.increment;plus[j]+=eps;minus[j]-=eps;const auto fp=force(plus),fm=force(minus);for(unsigned int i=0;i<ns;++i){const double d=(fp[i]-fm[i])/(2*eps);error+=std::pow(d-t.jacobian(i,j),2);norm+=d*d;}}final_error=std::sqrt(error/norm);report<<time<<','<<eps<<','<<final_error<<'\n';}
          if(final_error>=1e-6 && std::getenv("CI_DERIVATIVE_ROUNDOFF_REFERENCE"))
            final_error=derivative_reference(old,oldq,t,dt,rate_scale,time);
          AssertThrow(final_error<1e-6,ExcMessage("Complete interface derivative check failed."));
        }
      domain_diagnostics(t,time);
      publish(t,time);
      if(std::getenv("CI_STATE_AUDIT")){std::map<std::string,std::string> state;save(state);std::ofstream f(this->get_output_directory()+"accepted-native.archive");aspect::oarchive ar(f);ar<<state;}
      if(activation)activation_checked=true;
      if(publications==1)
        {
          const auto before=archive();bool rejected=false;dealii::deal_II_exceptions::disable_abort_on_exception();try{publish(t,time+dt);}catch(const std::exception&){rejected=true;}dealii::deal_II_exceptions::enable_abort_on_exception();
          AssertThrow(rejected && archive()==before,ExcMessage("Duplicate candidate published."));
        }
      auto history=output(resumed?"history-resume.csv":"history.csv",true);if(history.tellp()==0)history<<"time_s,dt_s,publications,wall_memory_m,bulk_J_m,bond_J_m,fracture_increment_J_m,friction_J_m,radiation_J_m,Maxwell_J_m,endpoint_work_J_m,trapezoid_work_J_m,bulk_BE_J_m,surface_remainder_J_m,ledger_error_J_m,force_error_Pa,step_seconds,min_curvature_Pa_m,min_side_contact_pressure_Pa,weak_certificate_Pa,certificate_iterations,release_curvature_Pa_m\n";
      history<<time<<','<<dt<<','<<publications<<','<<wall<<','<<E<<','<<bond<<','<<fracture<<','<<friction<<','<<radiation<<','<<maxwell<<','<<work<<','<<trap<<','<<be<<','<<surface_error<<','<<defect<<','<<t.error<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<','<<t.curvature<<','<<t.minimum_trace_pressure<<','<<t.weak_error<<','<<t.certificate_iterations<<','<<t.release_curvature<<'\n';
      auto state=output(resumed?"states-resume.csv":"states.csv",true);if(state.tellp()==0)state<<"time_s,node,x_m,jump_m,damage,Theta,reference_m,relaxed_jump_m,absolute_slip_m,shear_Pa,pn_Pa,q_Pa,V_m_s,tied,installed\n";
      for(unsigned int i=0;i<3*ns;++i){const auto&s=t.point[i].state;state<<time<<','<<i<<','<<qpositions[i]<<','<<s.jump<<','<<s.damage<<','<<s.theta<<','<<s.reference<<','<<s.relaxed_jump<<','<<s.absolute_slip<<','<<t.qstress[i]<<','<<t.qpressure[i]<<','<<t.point[i].q<<','<<t.point[i].velocity<<','<<(s.damage==0)<<','<<installed(i)<<'\n';}
      auto nodal=output(resumed?"nodal_history-resume.csv":"nodal_history.csv",true);if(nodal.tellp()==0)nodal<<"time_s,node,jump_m,relaxed_jump_m,signed_rate_m_s\n";for(unsigned int i=0;i<ns;++i)nodal<<time<<','<<i<<','<<fresh[i].jump<<','<<fresh[i].relaxed_jump<<','<<(movable_node(i)?properties(i)[rate_property]:0.)<<'\n';
      return {"Native interface accepted",std::to_string(publications)};
    }
    void save(std::map<std::string,std::string>&data)const override
    {std::ostringstream s;s<<std::setprecision(17)<<wall<<' '<<accepted_time<<' '<<publications<<' '<<ns<<' '<<width<<' '<<mode<<' '<<exact_jump<<' '<<load_end<<' '<<activation_checked<<' '<<finite_surface<<' '<<first_segment<<' '<<last_segment<<' '<<geometry_publications<<' '<<virgin_age;data["experimental geometric interface"]=s.str();data["experimental geometric interface manager"]=archive();if(finite_surface)data["experimental finite interface configuration"]=finite_configuration();}
    void load(const std::map<std::string,std::string>&data)override
    {const auto config=data.find("experimental finite interface configuration");if(config!=data.end())saved_finite_configuration=config->second;std::istringstream s(data.at("experimental geometric interface"));s>>wall>>accepted_time>>publications>>saved_ns>>saved_width>>saved_mode;AssertThrow(bool(s),ExcMessage("Invalid interface checkpoint."));if(!(s>>saved_exact_jump))saved_exact_jump=false;else if(s>>saved_load_end){s>>activation_checked;s>>finite_surface>>first_segment>>last_segment>>geometry_publications>>virgin_age;}std::istringstream bytes(data.at("experimental geometric interface manager"));{aspect::iarchive ar(bytes);ar>>interface_manager;}resumed=true;auto loaded=output("restart_loaded.csv");loaded<<"accepted_time_s,publications,wall_memory_m\n"<<accepted_time<<','<<publications<<','<<wall<<'\n';ready=false;}
  };
  ASPECT_REGISTER_POSTPROCESSOR(CohesiveInterfaceNative,"cohesive interface native checks","Isolated fixed-Gamma mixed FE and native property/checkpoint qualification.")
}}
