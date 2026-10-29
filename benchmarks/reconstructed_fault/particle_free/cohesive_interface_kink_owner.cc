/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
// Native owner/checkpoint adapter for the independently verified fitted kernel.
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/reconstructed_fault/cohesive_interface.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <numeric>
#include "cohesive_interface_evolution.h"
#include <sstream>
namespace aspect {namespace Postprocess {
template<int dim> class CohesiveKinkOwner:public Interface<dim>,public SimulatorAccess<dim>
{
 ReconstructedFaultManager<dim> owner;
 unsigned nx=0,nj=0,nm=0,nq=0,generation=0,publications=0;
 bool ready=false,resumed=false;
 std::string input_text;
 std::vector<double> coefficients,bulk,delta,memory,hist,K,Q,w,Nm,Np,lambda;
 std::vector<unsigned> layout;
 std::string read_input() const {const char *path=std::getenv("CI_KINK_INPUT");AssertThrow(path,ExcMessage("Missing kink fixture input"));std::ifstream f(path);AssertThrow(f,ExcMessage("Missing kink input file"));return std::string(std::istreambuf_iterator<char>(f),{});}
 void parse()
 {
  input_text=read_input();std::istringstream s(input_text);s>>nx>>nj>>nm>>nq;
  auto read=[&](std::vector<double>&v,unsigned size){v.resize(size);for(auto &x:v)s>>x;};
  read(coefficients,nj+1);read(bulk,nm);read(delta,nj);read(memory,nj);read(hist,6*nq);read(K,(nj+1)*(nj+1));read(Q,nq*nj);read(w,nq);read(Nm,nq*(nj+1));read(Np,nq*(nj+1));read(lambda,nq/2);
  AssertThrow(s && nx%8==0 && nq==3*nx/2 && nj==nx-2,ExcMessage("Bad kink input layout"));
  for(unsigned j=0;j<nj/2;++j)layout.push_back(j);
  auto initial=make(false);owner.swap_prescribed_geometry(*initial);ready=true;
 }
 std::shared_ptr<ReconstructedFaultManager<dim>> make(bool extended)const
 {
  auto m=std::make_shared<ReconstructedFaultManager<dim>>();m->register_property("kink material Gauss3 jump damage age reference Maxwell absolute slip",18);
  unsigned segments=extended?nx/2:nx/4;std::vector<Point<dim>>vertices(segments+1);
  for(unsigned i=0;i<=segments;++i){double x=.25+double(i)/nx;vertices[i][0]=x;vertices[i][1]=.5*std::max(0.,x-.5);}
  m->add_reconstructed_fault(vertices,std::vector<double>(vertices.size(),.02));
  m->set_shear_sense(0,1); // Explicit material sign, identical on fresh/restart paths.
  for(unsigned i=0;i<=segments;++i){auto p=m->get_fault(0).get_properties(i);for(unsigned j=0;j<18;++j)p[j]=0.;
   if(i<segments)for(unsigned j=0;j<18;++j)p[j]=hist[18*i+j];}
  return m;
 }
 std::string bytes()const
 {std::ostringstream s;{aspect::oarchive a(s);a<<owner<<nx<<nj<<nm<<nq<<generation<<publications<<coefficients<<bulk<<delta<<memory<<hist<<K<<Q<<w<<Nm<<Np<<lambda<<layout<<input_text;}return s.str();}
 void restore(const std::string &s)
 {std::istringstream f(s);aspect::iarchive a(f);a>>owner>>nx>>nj>>nm>>nq>>generation>>publications>>coefficients>>bulk>>delta>>memory>>hist>>K>>Q>>w>>Nm>>Np>>lambda>>layout>>input_text;ready=true;}
 void validate()const
 {
  AssertThrow(generation==0 && publications==0,ExcMessage("Stale kink geometry publication"));
  AssertThrow(owner.get_fault(0).n_vertices()==nx/2+1 && layout.size()==nj,ExcMessage("Kink trace layout mismatch"));
  for(unsigned i=0;i<nx/2;++i){auto p=owner.get_fault(0).get_properties(i);for(unsigned j=0;j<18;++j)AssertThrow(p[j]==hist[18*i+j],ExcMessage("Kink material history changed"));}
  for(unsigned q=nq/2;q<nq;++q){AssertThrow(hist[6*q]==0 && hist[6*q+1]==0 && hist[6*q+3]==0 && hist[6*q+4]==0 && hist[6*q+5]==0 && hist[6*q+2]>0,ExcMessage("New kink record is not virgin tied material"));AssertThrow(std::abs(lambda[q-nq/2])<=3000.,ExcMessage("Kink needs physical damage admission; tied insertion rejected"));}
  for(unsigned q=0;q<nq;++q){double pm=10000.,pp=10000.;for(unsigned k=0;k<=nj;++k){pm+=Nm[q*(nj+1)+k]*coefficients[k];pp+=Np[q*(nj+1)+k]*coefficients[k];}AssertThrow(pm>0 && pp>0,ExcMessage("Kink contact not closed"));}
  // Imported snapshots are stationary mechanically equilibrated old states.
  // Check the old branch as well; an inadmissible incoming state must not be
  // misclassified as a failure caused only by the new segment.
  for(unsigned j=0;j<nj/2;++j)
   {
    double f=0.,mass=0.;for(unsigned k=0;k<=nj;++k)f+=K[(j+1)*(nj+1)+k]*coefficients[k];
    for(unsigned q=0;q<nq/2;++q)
     {const double a=hist[6*q+1],e=hist[6*q]-hist[6*q+3];const double traction=a==0 || a==1?0.:(1-a)*e/((2./(3000.*3000.))*a);f+=Q[q*nj+j]*w[q]*traction;mass+=Q[q*nj+j]*w[q];}
    AssertThrow(mass>0 && std::abs(f)/mass<=1e-8,ExcMessage("Old kink material is not in unchanged mechanical equilibrium"));
   }
  for(unsigned j=nj/2;j<nj;++j){AssertThrow(delta[j]==0 && memory[j]==0,ExcMessage("New kink jump is nonzero"));double f=0.,mass=0.;for(unsigned k=0;k<=nj;++k)f+=K[(j+1)*(nj+1)+k]*coefficients[k];for(unsigned q=nq/2;q<nq;++q){f+=Q[q*nj+j]*w[q]*lambda[q-nq/2];mass+=Q[q*nj+j]*w[q];}AssertThrow(mass>0 && std::abs(f)/mass<=1e-8,ExcMessage("Kink tied weak force failed"));}
 }
 void insertion()
 {
  const auto before=bytes();const auto old_layout=layout;
  auto attempt=[&](bool corrupt){auto candidate=make(true);owner.swap_prescribed_geometry(*candidate);layout.resize(nj);std::iota(layout.begin(),layout.end(),0);
   try{if(corrupt)owner.get_fault(0).get_properties(0)[0]+=1.;validate();}
   catch(...){owner.swap_prescribed_geometry(*candidate);layout=old_layout;throw;}
  };
  bool rejected=false;dealii::deal_II_exceptions::disable_abort_on_exception();try{attempt(true);}catch(const std::exception&){rejected=true;}dealii::deal_II_exceptions::enable_abort_on_exception();
  AssertThrow(rejected && bytes()==before,ExcMessage("Kink owner rejection changed accepted state"));
  attempt(false);const auto staged=bytes();restore(before);attempt(false);AssertThrow(bytes()==staged,ExcMessage("Kink retry changed candidate"));
  ++generation;++publications;const auto after=bytes();bool duplicate=false;dealii::deal_II_exceptions::disable_abort_on_exception();try{validate();}catch(const std::exception&){duplicate=true;}dealii::deal_II_exceptions::enable_abort_on_exception();
  AssertThrow(duplicate && bytes()==after,ExcMessage("Duplicate kink publication"));
  std::ofstream f(this->get_output_directory()+"kink-owner-checks.csv");f<<"rollback_bitwise,retry_bitwise,duplicate_rejected,geometry_publications,physical_clock_s,physical_steps\n1,1,1,"<<publications<<','<<(evolution_ready?evolution_clock:admission_ready?physical_clock:.2)<<','<<(evolution_ready?evolution_publications:0)<<'\n';
 }
#include "cohesive_interface_admission_owner.inc"
#include "cohesive_interface_evolution_owner.inc"
public:
 std::pair<std::string,std::string> execute(TableHandler&)override
 {
  AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,ExcMessage("Kink owner control is serial 2D"));
  if(!ready)parse();
  if(std::getenv("CI_NATIVE_EVOLUTION"))
   {if(!evolution_ready)initialize_evolution();evolve(.2+this->get_time());std::ofstream out(this->get_output_directory()+"evolution-final.archive");out<<bytes()<<evolution_bytes();return {"Native evolved steps",std::to_string(evolution_publications)};}
  if(std::getenv("CI_ADMISSION_INPUT") && !admission_ready)initialize_admission();
  if(this->get_time()>=.001 && generation==0)
   {
    if(std::getenv("CI_EXPECT_KINK_REJECTION"))
     {
      const auto before=bytes();bool rejected=false;
      dealii::deal_II_exceptions::disable_abort_on_exception();
      try{insertion();}catch(const std::exception &e){rejected=std::string(e.what()).find("Kink needs physical damage admission")!=std::string::npos;}
      dealii::deal_II_exceptions::enable_abort_on_exception();
      AssertThrow(rejected && bytes()==before,ExcMessage("Overloaded kink was not atomically rejected"));
      std::ofstream checks(this->get_output_directory()+"kink-overload-checks.csv");checks<<"rejected,unchanged,geometry_publications\n1,1,0\n";
     }
    else insertion();
   }
  if(admission_ready && generation==1 && this->get_time()>=.002)advance_admission();
  std::ofstream f(this->get_output_directory()+"kink-owner-final.archive");f<<bytes();
  return {"Kink geometry publications",std::to_string(publications)};
 }
 void save(std::map<std::string,std::string>&data)const override {data["experimental fitted kink owner v1"]=bytes();if(admission_ready)data["experimental fitted kink admission v1"]=admission_bytes();if(evolution_ready)data["experimental fitted native evolution v1"]=evolution_bytes();}
 void load(const std::map<std::string,std::string>&data)override {restore(data.at("experimental fitted kink owner v1"));AssertThrow(input_text==read_input(),ExcMessage("Kink restart input changed"));resumed=true;const auto it=data.find("experimental fitted kink admission v1");if(it!=data.end())restore_admission(it->second);const auto ev=data.find("experimental fitted native evolution v1");if(ev!=data.end())restore_evolution(ev->second);}
};
ASPECT_REGISTER_POSTPROCESSOR(CohesiveKinkOwner,"cohesive kink owner checks","Isolated fitted mechanical-state transfer through native geometry/history and checkpoint ownership.")
}}
