 // Native verification/publication of a bounded independently solved sequence.
 // No outer mechanical solver or production checkpoint format is changed.
 bool admission_ready=false;
 unsigned admission_index=0,physical_publications=0;
 double physical_clock=.2,physical_gamma=0.;
 std::string admission_text;
 struct AdmissionRecord
 {
  double time=0.,gamma=0.;std::vector<double> c,displacement,memory,history,bulk,tied,lambda;
 };
 std::vector<AdmissionRecord> admission_records;
 std::vector<double> native_response;
 std::string read_admission()const
 {const char *path=std::getenv("CI_ADMISSION_INPUT");AssertThrow(path,ExcMessage("Missing admission input"));std::ifstream f(path);AssertThrow(f,ExcMessage("Cannot read admission sequence"));return std::string(std::istreambuf_iterator<char>(f),{});}
 void initialize_admission()
 {
  admission_text=read_admission();std::istringstream f(admission_text);unsigned count;f>>count>>physical_clock>>physical_gamma;
  auto read=[&](std::vector<double>&v,unsigned size){v.resize(size);for(auto &x:v)f>>x;};
  read(native_response,nm*(nj+1));admission_records.resize(count);
  for(auto &r:admission_records){f>>r.time>>r.gamma;read(r.c,nj+1);read(r.displacement,nj);read(r.memory,nj);read(r.history,6*nq);read(r.bulk,nm);read(r.tied,nq);read(r.lambda,nq);}
  AssertThrow(f && count>0,ExcMessage("Invalid admission sequence"));admission_ready=true;
 }
 std::string admission_bytes()const
 {std::ostringstream s;{aspect::oarchive a(s);a<<admission_text<<admission_index<<physical_publications<<physical_clock<<physical_gamma;}return s.str();}
 void restore_admission(const std::string &s)
 {
  std::string expected;unsigned index,pubs;double clock,gamma;{std::istringstream f(s);aspect::iarchive a(f);a>>expected>>index>>pubs>>clock>>gamma;}
  initialize_admission();AssertThrow(expected==admission_text,ExcMessage("Admission restart sequence changed"));
  admission_index=index;physical_publications=pubs;physical_clock=clock;physical_gamma=gamma;
 }
 struct AdmissionCheck{double source=0.,bulk=0.,weak=0.,ledger=0.,contact=1e100;};
 AdmissionCheck check_admission(const AdmissionRecord &r,unsigned base)const
 {
  AssertThrow(base==admission_index && r.time>physical_clock && generation==1,ExcMessage("Stale admission publication"));
  const double dt=r.time-physical_clock,beta=1./(1.+1e6/1e8*dt),df=2./3000.;
  const double roundoff=64*std::numeric_limits<double>::epsilon();
  auto close=[&](double a,double b,double scale,const char *message){AssertThrow(std::abs(a-b)<=roundoff*std::max(scale,1e-300),ExcMessage(message));};
  AdmissionCheck result;std::vector<double> inc(nj),traction(nq),rate(nq);
  double du=r.gamma-physical_gamma;close(r.c[0],beta*(coefficients[0]+du),std::abs(coefficients[0])+std::abs(du),"Native affine Maxwell update mismatch");
  for(unsigned j=0;j<nj;++j){inc[j]=r.displacement[j]-delta[j];close(r.memory[j],beta*(memory[j]+inc[j]),df,"Maxwell jump update mismatch");close(r.c[j+1],r.memory[j],df,"Mixed coefficient memory mismatch");}
  for(unsigned i=0;i<nm;++i){double value=0.,operand=0.;for(unsigned j=0;j<=nj;++j){value+=native_response[i*(nj+1)+j]*r.c[j];operand+=std::abs(native_response[i*(nj+1)+j]*r.c[j]);}result.bulk=std::max(result.bulk,std::abs(value-r.bulk[i]));close(value,r.bulk[i],operand,"Native bulk response mismatch");}
  CohesiveInterface::Parameters law;double bond0=0.,bond1=0.,fracture=0.,sliding=0.,bondwork=0.;
  for(unsigned q=0;q<nq;++q)
   {
    double increment=0.,scale=0.,pm=10000.,pp=10000.;
    for(unsigned j=0;j<nj;++j){increment+=Q[q*nj+j]*inc[j];scale+=std::abs(Q[q*nj+j]*inc[j]);}
    for(unsigned j=0;j<=nj;++j){pm+=Nm[q*(nj+1)+j]*r.c[j];pp+=Np[q*(nj+1)+j]*r.c[j];}
    result.contact=std::min({result.contact,pm,pp});AssertThrow(pm>0 && pp>0,ExcMessage("Admission lost contact"));
    if(r.tied[q]){close(increment,0.,std::max(scale,df),"Tied Gauss displacement mismatch");increment=0.;AssertThrow(hist[6*q+1]==0 && r.history[6*q+1]==0 && std::abs(r.lambda[q])<=3000.+1e-8,ExcMessage("Invalid intact admission multiplier"));}
    CohesiveInterface::PointState old{hist[6*q],hist[6*q+1],hist[6*q+2],hist[6*q+3],hist[6*q+4],hist[6*q+5]};
    const auto value=CohesiveInterface::evaluate(law,old,increment,.5*(pm+pp),dt);
    const auto &s=value.state;const double expected[]={s.jump,s.damage,s.theta,s.reference,s.relaxed_jump,s.absolute_slip};
    for(unsigned j=0;j<6;++j){const double unit=j==1?1.:j==2?std::max(1.,old.theta):df;result.source=std::max(result.source,std::abs(expected[j]-r.history[6*q+j])/unit);close(expected[j],r.history[6*q+j],unit,"Independent native point return mismatch");}
    AssertThrow(r.history[6*q+1]>=old.damage && r.history[6*q+1]<=1 && r.history[6*q+2]>0,ExcMessage("Invalid irreversible Gauss history"));
    traction[q]=r.tied[q]?r.lambda[q]:value.resistance;rate[q]=increment/dt;
    const double e=old.jump-old.reference,old_storage=old.damage==0 || old.damage==1?0.:.5*e*e*(1-old.damage)/((2./9e6)*old.damage);
    bond0+=w[q]*old_storage;bond1+=w[q]*value.storage;fracture+=w[q]*(s.damage-old.damage);sliding+=w[q]*(value.friction+value.radiation);bondwork+=w[q]*value.q*increment;
   }
  for(unsigned j=0;j<nj;++j){double force=0.,mass=0.;for(unsigned l=0;l<=nj;++l)force+=K[(j+1)*(nj+1)+l]*r.c[l];for(unsigned q=0;q<nq;++q){force+=Q[q*nj+j]*w[q]*traction[q];mass+=Q[q*nj+j]*w[q];}result.weak=std::max(result.weak,std::abs(force)/mass);}
  AssertThrow(result.weak<=1e-8,ExcMessage("Native complete admission weak residual failed"));
  auto energy=[&](const std::vector<double>&c){double e=0.;for(unsigned i=0;i<=nj;++i)for(unsigned j=0;j<=nj;++j)e+=.5*c[i]*K[i*(nj+1)+j]*c[j];return e;};
  auto difference=r.c;for(unsigned i=0;i<=nj;++i)difference[i]-=coefficients[i];double reaction=0.;for(unsigned i=0;i<=nj;++i)reaction+=K[i]*r.c[i];
  const double E=energy(r.c),work=reaction*du;
  result.ledger=work-(E-energy(coefficients)+bond1-bond0+fracture+sliding+2*1e6/1e8*dt*E+energy(difference)+bondwork-(bond1-bond0)-fracture);
  AssertThrow(std::abs(result.ledger)<=1e-10*std::max(1.,std::abs(work)),ExcMessage("Native admission BE ledger failed"));
  return result;
 }
 void publish_admission(const AdmissionRecord &r,unsigned base)
 {
  const auto check=check_admission(r,base);
  auto candidate=owner.clone_prescribed_geometry();
  for(unsigned i=0;i<nx/2;++i){auto p=candidate->get_fault(0).get_properties(i);for(unsigned j=0;j<18;++j)p[j]=r.history[18*i+j];}
  owner.swap_prescribed_geometry(*candidate);coefficients=r.c;bulk=r.bulk;delta=r.displacement;memory=r.memory;hist=r.history;physical_clock=r.time;physical_gamma=r.gamma;++admission_index;++physical_publications;
  std::ofstream f(this->get_output_directory()+"admission-native.csv",std::ios::app);f<<std::setprecision(17);if(f.tellp()==0)f<<"time_s,publications,source_scaled_error,bulk_error,weak_error_Pa,ledger_error_J_m,min_normal_Pa\n";f<<physical_clock<<','<<physical_publications<<','<<check.source<<','<<check.bulk<<','<<check.weak<<','<<check.ledger<<','<<check.contact<<'\n';
 }
 void advance_admission()
 {
  const unsigned target=std::min<unsigned>(admission_records.size(),std::lround(this->get_time()/.001)-1);
  while(admission_index<target)
   {
    const unsigned base=admission_index;const auto &r=admission_records[base];const auto before=bytes()+admission_bytes();
    if(base==0)
     {
      auto bad=r;bad.history[1]=-1.;bool rejected=false;
      dealii::deal_II_exceptions::disable_abort_on_exception();try{publish_admission(bad,base);}catch(const std::exception&){rejected=true;}dealii::deal_II_exceptions::enable_abort_on_exception();
      AssertThrow(rejected && before==bytes()+admission_bytes(),ExcMessage("Admission rejection changed accepted state"));
      check_admission(r,base);check_admission(r,base);AssertThrow(before==bytes()+admission_bytes(),ExcMessage("Admission trial mutated accepted state"));
     }
    publish_admission(r,base);
    const auto after=bytes()+admission_bytes();bool duplicate=false;dealii::deal_II_exceptions::disable_abort_on_exception();try{publish_admission(r,base);}catch(const std::exception&){duplicate=true;}dealii::deal_II_exceptions::enable_abort_on_exception();
    AssertThrow(duplicate && after==bytes()+admission_bytes(),ExcMessage("Admission duplicate published"));
   }
  std::ofstream f(this->get_output_directory()+"admission-final.archive");f<<bytes()<<admission_bytes();
 }
