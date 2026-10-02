#include <aspect/postprocess/interface.h>
#include <aspect/particle/manager.h>
#include <aspect/particle/interpolator/linear_least_squares.h>
#include <aspect/simulator_signals.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>

namespace aspect
{
  namespace ReplenishmentTest
  {
    bool measuring=false;
    std::string stage="initial";
    double scale=1e8;
    template<int dim> double exact(const Point<dim> &p,const double time,const unsigned int family)
    {
      const double x=p[0]-time,y=p[1]-.6*time;
      if(family==0) return scale;
      if(family==1) return scale*(1+.02*x+.03*y);
      return scale*(1+.1*std::sin(numbers::PI*x/4)*std::cos(numbers::PI*y/4));
    }
    template<int dim> std::array<double,4> geometry(const Particles::ParticleHandler<dim> &ph,
      const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell)
    {
      const unsigned int n=ph.n_particles_in_cell(cell);
      if(n==0)return {{0,0,0,4}};
      LAPACKFullMatrix<double> A(n,dim+1);
      Point<dim> lo,hi;for(unsigned int d=0;d<dim;++d){lo[d]=1;hi[d]=0;}
      std::set<unsigned int> quadrants;unsigned int i=0;
      for(const auto &p:ph.particles_in_cell(cell))
      {
        auto x=p.get_reference_location();A(i,0)=1;unsigned int quad=0;
        for(unsigned int d=0;d<dim;++d){A(i,d+1)=x[d]-.5;lo[d]=std::min(lo[d],x[d]);hi[d]=std::max(hi[d],x[d]);if(x[d]>.5)quad|=1<<d;}
        quadrants.insert(quad);++i;
      }
      double ratio=0;if(n>=dim+1){A.compute_svd();ratio=A.singular_value(dim)/A.singular_value(0);}
      return {{ratio,hi[0]-lo[0],hi[1]-lo[1],double((1<<dim)-quadrants.size())}};
    }
  }
  namespace Particle { namespace Interpolator {
    template<int dim> class ObservedLLS : public LinearLeastSquares<dim>
    {
      public:
      static void declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Interpolator");prm.enter_subsection("Observed native LLS");
        prm.declare_entry("Limit histories","false",Patterns::Bool(),"Test-only runtime-name mask for Maxwell and analytic histories.");
        prm.declare_entry("Limit crack driving history","false",Patterns::Bool(),"Explicit test of native nonnegative H limiting; no internal properties selected.");
        prm.leave_subsection();prm.leave_subsection();
      }
      void parse_parameters(ParameterHandler &prm) override
      {
        prm.enter_subsection("Interpolator");prm.enter_subsection("Observed native LLS");
        const bool limit=prm.get_bool("Limit histories"),limit_H=prm.get_bool("Limit crack driving history");prm.leave_subsection();
        if(limit || limit_H)
        {
          const auto &info=this->get_particle_manager(this->get_particle_manager_index()).get_property_manager().get_data_info();
          std::vector<bool> flags(info.n_components()-info.get_components_by_field_name("internal: integrator properties"),false);
          AssertThrow(info.get_position_by_field_name("internal: integrator properties")==flags.size(),ExcMessage("Native LLS expects internal properties last."));
          for(const std::string name:{"maxwell stress","function"}) if(limit && info.fieldname_exists(name))
            for(unsigned int k=0;k<info.get_components_by_field_name(name);++k)
              flags.at(info.get_position_by_field_name(name)+k)=true;
          if(limit_H && info.fieldname_exists("crack_driving_force"))flags.at(info.get_position_by_field_name("crack_driving_force"))=true;
          std::string mask;for(bool x:flags){if(!mask.empty())mask+=",";mask+=(x?"true":"false");}
          prm.enter_subsection("Linear least squares");prm.set("Use linear least squares limiter",mask);prm.set("Use boundary extrapolation","false");prm.leave_subsection();
          this->get_pcout()<<"Test LLS runtime history mask: "<<mask<<std::endl;
        }
        prm.leave_subsection();LinearLeastSquares<dim>::parse_parameters(prm);
      }
      std::vector<std::vector<double>> properties_at_points(const ParticleHandler<dim> &ph,
        const std::vector<Point<dim>> &points,const ComponentMask &mask,
        const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const override
      {
        auto values=LinearLeastSquares<dim>::properties_at_points(ph,points,mask,cell);
        if(!ReplenishmentTest::measuring && points.size()==1)
        {
          const auto &info=this->get_particle_manager(this->get_particle_manager_index()).get_property_manager().get_data_info();
          const unsigned int index=info.get_position_by_field_name("maxwell stress");
          const unsigned int n=ph.n_particles_in_cell(cell);
          auto g=ReplenishmentTest::geometry(ph,cell);
          const std::string path=this->get_output_directory()+"insertion_rank"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv";
          const bool header=!std::ifstream(path).good();std::ofstream out(path,std::ios::app);out<<std::setprecision(17);
          if(header)out<<"step,time,stage,cell,n_before,sigma_ratio,x,y,property,input_min,input_max,proposal,reference\n";
          for(unsigned int k=0;k<info.n_components();++k) if(mask[k] && ((k>=index && k<index+(info.fieldname_exists("function")?9:3)) || (info.fieldname_exists("crack_driving_force") && k==info.get_position_by_field_name("crack_driving_force"))))
          {
            double lo=std::numeric_limits<double>::max(),hi=-lo;
            for(const auto &p:ph.particles_in_cell(cell)){lo=std::min(lo,p.get_properties()[k]);hi=std::max(hi,p.get_properties()[k]);}
            if(values[0][k]<0 && info.fieldname_exists("crack_driving_force") && k==info.get_position_by_field_name("crack_driving_force"))
            {
              const std::string dump=this->get_output_directory()+"negative_H_cloud_rank"+std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".csv";
              if(!std::ifstream(dump).good())
              {std::ofstream cloud(dump);cloud<<std::setprecision(17)<<"id,x,y,xi,eta,H\n";for(const auto &p:ph.particles_in_cell(cell))cloud<<p.get_id()<<','<<p.get_location()[0]<<','<<p.get_location()[1]<<','<<p.get_reference_location()[0]<<','<<p.get_reference_location()[1]<<','<<p.get_properties()[k]<<'\n';}
            }
            const unsigned int family=(k-index)/3,component=(k-index)%3;
            out<<this->get_timestep_number()<<','<<this->get_time()<<','<<ReplenishmentTest::stage<<','<<cell->id()<<','<<n<<','<<g[0]<<','<<points[0][0]<<','<<points[0][1]<<','<<k<<','<<lo<<','<<hi<<','<<values[0][k]<<','<<(info.fieldname_exists("function")?(component==0?1:component==1?-1:.5)*ReplenishmentTest::exact(points[0],this->get_time(),family):std::numeric_limits<double>::quiet_NaN())<<'\n';
          }
        }
        return values;
      }
    };
    ASPECT_REGISTER_PARTICLE_INTERPOLATOR(ObservedLLS,"observed native LLS","Test-only observation of unmodified native LLS results.")
  }}
  namespace Postprocess
  {
    template<int dim> class Replenishment : public Interface<dim>, public SimulatorAccess<dim>
    {
      using Snapshot=std::map<types::particle_index,std::vector<double>>;
      Snapshot previous;
      Snapshot snapshot() const
      {
        Snapshot s;for(const auto &p:this->get_particle_manager(0).get_particle_handler())
        {auto &v=s[p.get_id()];for(unsigned int d=0;d<dim;++d)v.push_back(p.get_location()[d]);for(double x:p.get_properties())v.push_back(x);}
        return s;
      }
      void capture(const std::string &stage,const bool published)
      {
        using namespace ReplenishmentTest;
        auto &pm=this->get_particle_manager(0);auto &ph=pm.get_particle_handler();auto now=snapshot();
        unsigned int born=0,lost=0;for(auto &p:now)born+=!previous.count(p.first);for(auto &p:previous)lost+=!now.count(p.first);
        if(previous.empty())born=0;
        const auto rank=Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
        std::string path=this->get_output_directory()+"events_rank"+std::to_string(rank)+".csv";
        bool head=!std::ifstream(path).good();std::ofstream events(path,std::ios::app);events<<std::setprecision(17);
        if(head)events<<"stage,step,time,owned,total,born_or_received,lost_or_sent,next_id\n";
        events<<stage<<','<<this->get_timestep_number()<<','<<this->get_time()<<','<<now.size()<<','<<ph.n_global_particles()<<','<<born<<','<<lost<<','<<ph.get_next_free_particle_index()<<'\n';
        previous=std::move(now);
        path=this->get_output_directory()+"cells_rank"+std::to_string(rank)+".csv";head=!std::ifstream(path).good();std::ofstream out(path,std::ios::app);out<<std::setprecision(17);
        if(head)out<<"stage,step,time,cell,n,sigma_ratio,xspan,yspan,empty_quadrants,family,site,particle_min,particle_max,min,max,rms_error,max_error,mean,reference_mean,volume\n";
        const auto &info=pm.get_property_manager().get_data_info();
        std::array<unsigned int,3> index={{info.get_position_by_field_name("maxwell stress"),info.get_position_by_field_name("function"),info.get_position_by_field_name("function")+3}};
        std::vector<Point<dim>> supports=FE_Q<dim>(2).get_unit_support_points();
        QGauss<dim> gauss(3);Quadrature<dim> support_q(supports);
        double worst=2;typename DoFHandler<dim>::active_cell_iterator worst_cell;
        for(const auto &cell:this->get_dof_handler().active_cell_iterators())if(cell->is_locally_owned())
          {const double ratio=geometry(ph,cell)[0];if(ratio<worst){worst=ratio;worst_cell=cell;}}
        std::ofstream cloud(this->get_output_directory()+"worst_"+stage+"_step"+std::to_string(this->get_timestep_number())+"_rank"+std::to_string(rank)+".csv");
        cloud<<std::setprecision(17)<<"cell,id,x,y,xi,eta,constant,affine,curved\n";
        for(const auto &p:ph.particles_in_cell(worst_cell))cloud<<worst_cell->id()<<','<<p.get_id()<<','<<p.get_location()[0]<<','<<p.get_location()[1]<<','<<p.get_reference_location()[0]<<','<<p.get_reference_location()[1]<<','<<p.get_properties()[index[0]]<<','<<p.get_properties()[index[1]]<<','<<p.get_properties()[index[2]]<<'\n';
        for(const auto &cell:this->get_dof_handler().active_cell_iterators()) if(cell->is_locally_owned())
        {
          auto g=geometry(ph,cell);unsigned int n=ph.n_particles_in_cell(cell);
          for(unsigned int site=0;site<2;++site)
          {
            const Quadrature<dim> &q=site==0?support_q:static_cast<const Quadrature<dim>&>(gauss);
            FEValues<dim> fe(this->get_mapping(),this->get_fe(),q,update_values|update_quadrature_points|update_JxW_values);fe.reinit(cell);
            measuring=true;auto prop=pm.get_interpolator().properties_at_points(ph,fe.get_quadrature_points(),ComponentMask(info.n_components(),true),cell);measuring=false;
            std::vector<Vector<double>> field(q.size(),Vector<double>(this->get_fe().n_components()));
            if(published)fe.get_function_values(this->get_solution(),field);
            for(unsigned int family=0;family<3;++family)
            {
              double pmin=std::numeric_limits<double>::max(),pmax=-pmin;
              for(const auto &p:ph.particles_in_cell(cell)){pmin=std::min(pmin,p.get_properties()[index[family]]);pmax=std::max(pmax,p.get_properties()[index[family]]);}
              for(unsigned int source=0;source<(published?2:1);++source)
              {
                double lo=std::numeric_limits<double>::max(),hi=-lo,err=0,merr=0,mean=0,refmean=0,wtotal=0;
                for(unsigned int j=0;j<q.size();++j)
                {
                  const double value=source==0?prop[j][index[family]]:field[j][this->introspection().component_indices.compositional_fields[3*family]];
                  double ref=exact(fe.quadrature_point(j),this->get_time(),family),w=site==0?1./q.size():q.weight(j);
                  lo=std::min(lo,value);hi=std::max(hi,value);err+=w*std::pow(value-ref,2);merr=std::max(merr,std::abs(value-ref));mean+=w*value;refmean+=w*ref;wtotal+=w;
                }
                out<<stage<<','<<this->get_timestep_number()<<','<<this->get_time()<<','<<cell->id()<<','<<n<<','<<g[0]<<','<<g[1]<<','<<g[2]<<','<<g[3]<<','<<family<<','<<(source==0?"LLS_":"Q2_")<<(site==0?"support":"gauss3")<<','<<pmin<<','<<pmax<<','<<lo<<','<<hi<<','<<std::sqrt(err/wtotal)<<','<<merr<<','<<mean/wtotal<<','<<refmean/wtotal<<','<<cell->measure()<<'\n';
              }
            }
          }
        }
      }
      public:
      void initialize() override
      {
        this->get_signals().post_simulator_initialization.connect([this](const SimulatorAccess<dim>&)
        {
        this->get_signals().post_set_initial_state.connect([this](const SimulatorAccess<dim>&)
        {
          capture("generated",false);
          ReplenishmentTest::stage="initial_management";
          this->get_particle_manager(0).advance_timestep(); // zero time; native bounds and late initialization
          capture("initial_management",false);
          ReplenishmentTest::stage="transport";
        });
        });
      }
      std::pair<std::string,std::string> execute(TableHandler &) override
      {capture("transport",true);return {"Replenishment fixture:","sampled"};}
    };
    ASPECT_REGISTER_POSTPROCESSOR(Replenishment,"replenishment observer","Test-only native generation/replenishment/Q2 observer.")
  }
}
