#include "../../../source/material_model/fault_theta_history_diagnostic.h"
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>

struct Fault
{
  unsigned int n_vertices() const { return 2; }
  std::array<double,2> vertex(unsigned int i) const { return {double(i),0.}; }
};

int main(int argc,char **argv)
{
  assert(argc==2);
  const auto age=[](double v,double t,double dt)
  { const double x=v*dt/.008;return t*std::exp(-x)-.008/v*std::expm1(-x); };
  const double t0[2]={8e6,2e8},v1[2]={1e-9,1e-20},v2[2]={2e-10,8e-10};
  const double d1=4e8,d2=6e7;
  auto append=[&](unsigned int step,double dt,const double *v)
  { std::ofstream f(argv[1],std::ios::app);f<<std::setprecision(17)<<step<<' '<<dt<<' '<<v[0]<<' '<<v[1]<<'\n'; };
  {std::ofstream f(argv[1]);f<<"2\n0 0 8000000\n1 0 200000000\n";}
  Fault fault;aspect::MaterialModel::internal::FaultThetaHistoryDiagnostic history;
  history.load(argv[1],0,fault);
  assert(history.value(0,.37,age)==.63*t0[0]+.37*t0[1]);
  history.load(argv[1],1,fault);
  assert(history.value(0,.37,age)==.63*t0[0]+.37*t0[1]);
  append(1,d1,v1);history.load(argv[1],2,fault);
  const double prior=history.value(0,.37,age);
  assert(std::abs(prior/age(v1[0]+.37*(v1[1]-v1[0]),.63*t0[0]+.37*t0[1],d1)-1)<1e-14);
  append(2,d2,v2);history.load(argv[1],3,fault);
  for (double x:{0.,.37,.61,1.})
    {
      double expected=(1-x)*t0[0]+x*t0[1];
      expected=age(x==1?v1[1]:v1[0]+x*(v1[1]-v1[0]),expected,d1);
      expected=age(x==1?v2[1]:v2[0]+x*(v2[1]-v2[0]),expected,d2);
      const double actual=history.value(0,x,age);
      assert(std::abs(actual/expected-1)<1e-14);
      assert(actual==history.value(0,x,age));
    }
  const double nodal=.63*age(v2[0],age(v1[0],t0[0],d1),d2)+.37*age(v2[1],age(v1[1],t0[1],d1),d2);
  assert(std::abs(history.value(0,.37,age)/nodal-1)>.01);
  bool caught=false;
  try {history.load(argv[1],4,fault);}catch(const std::runtime_error &){caught=true;}
  assert(caught);
  std::cout<<"Functional Theta: initialization, two independent updates, new coordinates, nodal endpoints, cache and wrong-clock rejection passed.\n";
}
