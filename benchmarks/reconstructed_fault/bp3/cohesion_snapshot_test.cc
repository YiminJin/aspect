// Standalone check of the diagnostic snapshot; no ASPECT solve required.
#include "../../../source/material_model/fault_cohesion_diagnostic.h"
#include <cassert>
#include <cmath>
#include <iostream>

struct TwoVertices
{
  unsigned int n_vertices() const { return 2; }
  std::array<double,2> vertex(unsigned int i) const { return {{double(i),0.}}; }
};

int main(int argc,char **argv)
{
  assert(argc==2);
  {
    std::ofstream out(argv[1]);
    out<<"2\n0 0 3 2 4\n1 0 7 6 9\n";
  }
  aspect::MaterialModel::internal::FaultCohesionDiagnostic field;
  field.load(argv[1],TwoVertices());
  const double beta=.8,kappa=5.;
  for (double xi:{0.,.17,.5,.91,1.})
    {
      const double C=(1-xi)*3+xi*7,I=(1-xi)*4+xi*9,V=2+xi*4;
      const double expected=(kappa*V+beta*I*C)/I;
      assert(field.value(0,xi,beta,kappa)==expected);
    }
  // A ratio of interpolated fields is not its nodal Q1 interpolation.
  assert(std::abs(field.value(0,.5,beta,kappa)
                  -.5*(field.value(0,0,beta,kappa)+field.value(0,1,beta,kappa)))>1e-2);
  {
    std::ofstream out(argv[1]);out<<"2\n0 0 3 2 4\n2 0 7 6 9\n";
  }
  bool rejected=false;
  try
    {
      aspect::MaterialModel::internal::FaultCohesionDiagnostic bad;
      bad.load(argv[1],TwoVertices());
    }
  catch(const std::runtime_error &) { rejected=true; }
  assert(rejected);
  std::cout<<"PASS: initial evaluated resistance, endpoints, nonlinear ratio, geometry mismatch\n";
}
