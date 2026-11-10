/* Independent small oracle and captured scalar-QP replay; no coupled solver. */
#include <aspect/phase_field/obstacle.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>

namespace aspect
{
  namespace PhaseField
  {
    ObstacleResult solve_obstacle_reference(const LinearAlgebra::SparseMatrix &,
                                            const LinearAlgebra::Vector &,const LinearAlgebra::Vector &,const LinearAlgebra::Vector &,
                                            LinearAlgebra::Vector &,const IndexSet &,MPI_Comm,double,unsigned int,unsigned int);
  }
}

int main(int argc,char **argv)
{
  using namespace aspect;
  Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  const MPI_Comm comm=MPI_COMM_WORLD;
  const auto rank=Utilities::MPI::this_mpi_process(comm);
  const bool toy=argc==1;
  unsigned int n=3,maximum=100,linear_maximum=1000;
  double tolerance=1e-16;
  std::vector<std::tuple<unsigned int,unsigned int,double>> entries;
  std::vector<std::array<double,6>> values;
  IndexSet owned;
  if (toy)
    {
      owned=Utilities::MPI::create_evenly_distributed_partitioning(comm,n);
      const double b[3]= {.95,1e-15,.3},l[3]= {.95,0,0},u[3]= {1,1,.2},x[3]= {.95,0,.2};
      for (const auto i:owned)
        {
          entries.emplace_back(i,i,1.);
          values.push_back({{double(i),b[i],l[i],u[i],x[i],1.}});
        }
    }
  else
    {
      const std::string prefix=std::string(argv[1])+"-r"+std::to_string(rank);
      std::ifstream meta(prefix+"-meta.txt");
      AssertThrow(meta,ExcMessage("Missing scalar-QP metadata."));
      meta>>n>>tolerance>>maximum>>linear_maximum;
      owned=IndexSet(n);
      std::ifstream vectors(prefix+"-vectors.csv"),matrix(prefix+"-matrix.csv");
      AssertThrow(vectors && matrix,ExcMessage("Missing scalar-QP rows."));
      std::string line;
      std::getline(vectors,line);
      while (std::getline(vectors,line))
        {
          std::replace(line.begin(),line.end(),',',' ');
          std::istringstream row(line);
          std::array<double,6> v;
          for (auto &x:v)row>>x;
          AssertThrow(!row.fail(),ExcMessage("Invalid captured vector row."));
          owned.add_index(static_cast<unsigned int>(v[0]));
          values.push_back(v);
        }
      owned.compress();
      std::getline(matrix,line);
      while (std::getline(matrix,line))
        {
          std::replace(line.begin(),line.end(),',',' ');
          std::istringstream row(line);
          unsigned int i,j;
          double v;
          row>>i>>j>>v;
          AssertThrow(!row.fail(),ExcMessage("Invalid captured matrix row."));
          entries.emplace_back(i,j,v);
        }
    }
  AssertThrow(Utilities::MPI::sum(owned.n_elements(),comm)==n,ExcMessage("Capture requires its original rank partition."));
  IndexSet relevant=owned;
  for (const auto &[i,j,v]:entries)relevant.add_index(j);
  relevant.compress();
  DynamicSparsityPattern pattern(relevant);
  for (const auto &[i,j,v]:entries)pattern.add(i,j);
  aspect::LinearAlgebra::SparseMatrix matrix;
  matrix.reinit(owned,owned,pattern,comm);
  for (const auto &[i,j,v]:entries)matrix.set(i,j,v);
  matrix.compress(VectorOperation::insert);
  aspect::LinearAlgebra::Vector rhs(owned,comm),lower(owned,comm),upper(owned,comm),initial(owned,comm),state(owned,comm),gradient(owned,comm);
  for (const auto &v:values)
    {
      const unsigned int i=v[0];
      rhs[i]=v[1];
      lower[i]=v[2];
      upper[i]=v[3];
      initial[i]=v[4];
    }
  for (auto *v:
       {
         &rhs,&lower,&upper,&initial
       })v->compress(VectorOperation::insert);
  bool original_failed=false,original_budget_failure=false;
  for (const bool reference:
  {
    true,false
  })
  {
    state=initial;
    const auto start=std::chrono::steady_clock::now();
    bool success=true,budget_failure=false;
    PhaseField::ObstacleResult result;
    try
      {
        result=reference ? PhaseField::solve_obstacle_reference(matrix,rhs,lower,upper,state,relevant,comm,tolerance,maximum,linear_maximum)
               : PhaseField::solve_obstacle(matrix,rhs,lower,upper,state,relevant,comm,tolerance,maximum,linear_maximum);
      }
    catch (const std::exception &e)
      {
        success=false;
        budget_failure=std::string(e.what()).find("active-set iteration budget exhausted")!=std::string::npos;
      }
    matrix.vmult(gradient,state);
    gradient-=rhs;
    double primal=0.,dual=0.,error=0.,projected=0.;
    std::vector<std::pair<double,std::string>> offenders;
    for (const auto i:owned)
      {
        primal=std::max(primal,std::max(double(lower[i]-state[i]),double(state[i]-upper[i])));
        const double g=gradient[i];
        const double diagonal=matrix.diag_element(i);
        const double certificate=std::abs(double(state[i])-std::clamp(double(state[i])-g/diagonal,double(lower[i]),double(upper[i])))*diagonal;
        projected=std::max(projected,certificate);
        const double violation=std::max({0.,double(lower[i]-state[i]),double(state[i]-upper[i])});
        const double priority=std::max(certificate,diagonal*violation);
        if (!toy && priority>0.)
          {
            std::ostringstream row;
            row<<std::setprecision(17)<<i<<','<<state[i]<<','<<lower[i]<<','<<upper[i]<<','<<rhs[i]<<','<<g<<','<<diagonal<<','<<violation<<','<<certificate<<'\n';
            offenders.emplace_back(priority,row.str());
          }
        dual=std::max(dual,lower[i]==upper[i] ? 0. : state[i]==lower[i] ? std::max(0.,-g) : state[i]==upper[i] ? std::max(0.,g) : std::abs(g));
        if (toy)
          {
            const double exact[3]= {.95,1e-15,.2};
            error=std::max(error,std::abs(state[i]-exact[i]));
          }
      }
    primal=Utilities::MPI::max(primal,comm);
    dual=Utilities::MPI::max(dual,comm);
    error=Utilities::MPI::max(error,comm);
    projected=Utilities::MPI::max(projected,comm);
    if (!toy)
      {
        std::sort(offenders.begin(),offenders.end(),[](const auto &a,const auto &b)
        {
          return a.first>b.first;
        });
        const std::string name=std::string(argv[1])+(reference?"-reference":"-corrected")+"-final-r"+std::to_string(rank)+".csv";
        AssertThrow(!std::ifstream(name),ExcMessage("Refusing to overwrite scalar-replay evidence."));
        std::ofstream out(name);
        out<<"i,state,lower,upper,rhs,gradient,diagonal,primal_violation,projected_certificate\n";
        for (unsigned int k=0; k<std::min<std::size_t>(16,offenders.size()); ++k)out<<offenders[k].second;
      }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if (rank==0)std::cout<<std::setprecision(17)<<"{\"reference\":"<<(reference?"true":"false")<<",\"success\":"<<(success?"true":"false")<<",\"budget_exhausted\":"<<(budget_failure?"true":"false")<<",\"seconds\":"<<seconds<<",\"max_active_iterations\":"<<maximum<<",\"iterations\":"<<(success?std::to_string(result.iterations):"null")<<",\"primal_violation\":"<<primal<<",\"projected_certificate\":"<<projected<<",\"dual_violation\":"<<dual<<",\"analytic_error\":"<<error<<"}\n";
    if (reference)
      {
        original_failed=!success;
        original_budget_failure=budget_failure;
      }
    if (toy && !reference)AssertThrow(success && original_failed && original_budget_failure && error<1e-26 && primal==0. && dual<tolerance,ExcMessage("Residual-correction oracle failed."));
  }
}
