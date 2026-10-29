/* Independent small mixed-mechanics controls for the publication safeguard. */
#include <aspect/material_model/phase_field_fault/bulk_stability.h>
#include <iostream>

int main()
{
  using namespace dealii;
  namespace BC=aspect::MaterialModel::BulkCohesive;
  DynamicSparsityPattern ds(7,7);
  for(unsigned int i=0;i<7;++i)for(unsigned int j=0;j<7;++j)ds.add(i,j);
  SparsityPattern sp;sp.copy_from(ds);SparseMatrix<double> j(sp);
  // Mixed block: u1=0 from pressure, independent relaxed u2. An artificial
  // constrained row and unused phase row have arbitrary negative diagonals.
  const double wm=7.,wp=3.;
  j.set(0,0,4/wm);j.set(1,1,2/wm);j.set(0,2,1/wm);j.set(2,0,1/wm);
  j.set(1,3,2/wm);j.set(3,1,2/wp);j.set(3,3,1.5/wp);j.set(4,4,4/wp);
  j.set(5,5,-1e12);j.set(6,6,-1e15);
  Vector<double> mass(2);mass[0]=2.;mass[1]=4.;
  const auto negative=BC::relaxed_curvature(j,{0,1,2},{3,4},mass,wm,wp,1e-9);
  // Independent pencil: u2=-delta_phi, so D-B^T A^-1 B=diag(-.5,4).
  AssertThrow(std::abs(negative.minimum+.25)<1e-13 && negative.minimum < -negative.uncertainty,ExcInternalError());
  j.set(3,3,3/wp);
  const auto positive=BC::relaxed_curvature(j,{0,1,2},{3,4},mass,wm,wp,1e-9);
  AssertThrow(std::abs(positive.minimum-.5)<1e-13 && positive.minimum>positive.uncertainty,ExcInternalError());
  // A zero mode cannot be reported as stable.
  j.set(3,3,2/wp);
  const auto zero=BC::relaxed_curvature(j,{0,1,2},{3,4},mass,wm,wp,1e-9);
  AssertThrow(std::abs(zero.minimum)<=zero.uncertainty,ExcInternalError());
  j.set(1,3,2.01/wm);
  bool rejected=false;
  try {BC::relaxed_curvature(j,{0,1,2},{3,4},mass,wm,wp,1e-9);}
  catch(const std::exception &){rejected=true;}
  AssertThrow(rejected,ExcInternalError());
  std::cout<<"{\"negative\":"<<negative.minimum<<",\"positive\":"<<positive.minimum
           <<",\"zero\":"<<zero.minimum<<",\"passed\":true}\n";
}
