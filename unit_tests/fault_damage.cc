/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include "common.h"
#include <aspect/mesh_refinement/fault_damage.h>

namespace
{
  using namespace aspect::MeshRefinement::FaultDamage;
  using namespace dealii;

  BernsteinPatch<2> constant(const double value)
  {
    BernsteinPatch<2> patch;
    patch.fill(value);
    return patch;
  }

  Cell cell(const unsigned int id, const Point<2> &lo, const Point<2> &hi,
            const BernsteinPatch<2> &phase)
  {
    return {CellId(id,{}),BoundingBox<2>({lo,hi}),phase};
  }

  // On two ranks the seed and its first neighbor are owned by different
  // processes. Even a rank with no local excursion takes part in the gather.
  std::vector<Cell> partition(const std::vector<Cell> &cells)
  {
    std::vector<Cell> owned;
    const auto rank=dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    const auto size=dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    for (unsigned int i=0; i<cells.size(); ++i)
      if (i%size==rank)
        owned.push_back(cells[i]);
    return owned;
  }

  Component connected(const std::vector<Cell> &cells, const Point<2> &anchor=Point<2>(-.2,.25))
  {
    const Tensor<1,2> tangent({1.,0.});
    return connected_component(partition(cells),anchor,anchor+.05*tangent,tangent,MPI_COMM_WORLD);
  }
}

TEST_CASE("Fault damage detects slow and subcell excursions on the coarse endpoint")
{
  const auto seed=cell(0, {-.5,0}, {0,.5},constant(.02));
  const auto receiver=[&](const BernsteinPatch<2> &phase)
  {
    return cell(1, {0,0}, {1,1},phase);
  };
  const auto coarse_id=CellId(1, {});
  // With the old 1e-4 increment screen both accepted intervals escape.
  const auto old_decision=[](const double old_phase, const double phase)
  {
    return old_phase<.01 && phase>=.01 && phase-old_phase>=1.e-4;
  };
  CHECK_FALSE(old_decision(.00999,.01001));
  CHECK_FALSE(old_decision(.01001,.0102));
  CHECK(connected({seed,receiver(constant(.00999))}).cells.count(coarse_id)==0);
  CHECK(connected({seed,receiver(constant(.01001))}).cells.count(coarse_id)==1);
  CHECK(connected({seed,receiver(constant(.0102))}).cells.count(coarse_id)==1);

  // phi=.012(1-x)^2, supported on 8.7129% of the coarse width.
  const BernsteinPatch<2> sliver {{.012,0.,0.,.012,0.,0.,.012,0.,0.}};
  CHECK(.012*std::pow(1.-.125,2)==Approx(.0091875));
  CHECK_FALSE(old_decision(0.,.012*std::pow(1.-.125,2)));
  CHECK(connected({seed,receiver(sliver)}).cells.count(coarse_id)==1);
  CHECK(connected({seed,receiver(constant(0.))}).cells.count(coarse_id)==0);
  // The same FE polynomial anchored from its high edge gives an independent
  // measurement of the contour root in the negative-x seed frame.
  const Tensor<1,2> back({-1.,0.});
  const auto line=seed_on_line(partition({receiver(sliver)}), {.02,.4}, {0.,.4},back,MPI_COMM_WORLD);
  REQUIRE(line.anchored);
  CHECK(std::abs(line.rear[0]-(1.-std::sqrt(.01/.012)))<1.e-14);
}

TEST_CASE("Fault damage recomputes connectivity of existing coarse damage across a fine bridge")
{
  const auto seed=cell(0, {-.5,0}, {0,.5},constant(.02));
  const auto island=cell(2, {.25,0}, {1.25,1.},constant(.02));
  auto bridge=cell(1, {0,0}, {.25,.25},constant(.00999));
  const Point<2> anchor(-.2,.15);
  CHECK(connected({seed,bridge,island},anchor).cells.count(island.id)==0);
  bridge.phase=constant(.01001);
  CHECK(connected({seed,bridge,island},anchor).cells.count(island.id)==1);
  // A stand-alone island leaves the specified incoming anchor unsupported.
  CHECK_FALSE(connected({island},anchor).anchored);
}

TEST_CASE("Fault damage matches actual DG traces across translated hanging faces")
{
  // Left coarse rectangle x=[10,11], y=[20,22]. Its right trace reaches
  // the contour only for y < 20+2*(1-sqrt(1/2)) = 20.585786...
  const auto seed=cell(0, {10,20}, {11,22}, {{.02,.02,.02,0.,0.,0.,0.,0.,0.}});
  auto lower=cell(1, {11,20}, {11.5,21},constant(.08));
  const auto upper=cell(2, {11,21}, {11.5,22},constant(.08));
  const Point<2> anchor(10.4,20.2);
  CHECK(connected({seed,lower},anchor).cells.count(lower.id)==1);
  CHECK(connected({seed,upper},anchor).cells.count(upper.id)==0);
  // Both traces exceed .01 somewhere on the shared lower half-face, but
  // the second trace requires y >= 20+sqrt(.01/.02) = 20.707106...
  lower.phase= {{0.,0.,0.,0.,0.,0.,.02,.02,.02}};
  CHECK(connected({seed,lower},anchor).cells.count(lower.id)==0);
  const auto corner=cell(3, {11,22}, {11.5,23},constant(.08));
  auto full=seed;
  full.phase=constant(.08);
  CHECK(connected({full,corner},anchor).cells.count(corner.id)==0);

  // Equality at just one point on a shared face also has zero length.
  full.phase= {{.02,.02,.02,.01,.01,.01,0.,0.,0.}};
  auto other=cell(4, {11,20}, {12,22}, {{0.,0.,0.,.01,.01,.01,.02,.02,.02}});
  CHECK(connected({full,other},anchor).cells.count(other.id)==0);
}

TEST_CASE("Fault damage resolves two islands inside one Q2 cell")
{
  // Bilinear saddle phi=.01+.024(x-.5)(y-.5)-.001. Opposite corners reach
  // .015 but their positive components are separated by subthreshold phase.
  BernsteinPatch<2> saddle;
  for (unsigned int j=0; j<3; ++j)
    for (unsigned int i=0; i<3; ++i)
      saddle[3*j+i]=.009+.024*(.5*i-.5)*(.5*j-.5);
  const auto middle=cell(0, {0,0}, {1,1},saddle);
  const auto receiver=cell(1, {1,.5}, {1.5,1},constant(.04));
  const auto comp=connected({middle,receiver}, {.1,.1});
  REQUIRE(comp.anchored);
  CHECK(comp.cells.count(receiver.id)==0);

  // The coefficients of a transverse quadratic can exceed the contour
  // when its true maximum only touches it. A whole seed line at that
  // maximum still represents zero damaged area. A plateau over the cell
  // has positive area and must be retained, even at the same threshold.
  const double delta=std::ldexp(1.,-9);
  auto touch=constant(damage_contour);
  for (unsigned int i=0; i<9; ++i)
    touch[i]+=i/3==1 ? delta : -delta;
  CHECK_FALSE(connected({cell(0,{0,0},{1,1},touch)},{.25,.5}).anchored);
  CHECK(connected({cell(0,{0,0},{1,1},constant(damage_contour))},{.25,.5}).anchored);
}

TEST_CASE("Fault damage finds the contiguous tilted seed extent from its quartic trace")
{
  // On the diagonal of the translated unit square, phi=.16*t^4 first
  // reaches .01 at t=.5, strictly inside the cell. A second active cell
  // farther back on the ray cannot lengthen this seed across the intact gap.
  auto phase=constant(0.);
  phase[8]=.16;
  const auto seed=cell(0, {3,-2}, {4,-1},phase);
  const auto detached=cell(1, {2,-3}, {3,-2},constant(.04));
  const auto cells=partition({seed,detached});
  const Point<2> anchor(3.7,-1.3),tip(3.9,-1.1),expected(3.5,-1.5);
  const Tensor<1,2> tangent((tip-anchor)/(tip-anchor).norm());
  const auto comp=connected_component(cells,anchor,tip,tangent,MPI_COMM_WORLD);
  REQUIRE(comp.anchored);
  CHECK(comp.cells.size()==1);
  CHECK(comp.rear.distance(expected)<2.e-14);
  const auto line=seed_on_line(cells,anchor,tip,tangent,MPI_COMM_WORLD);
  REQUIRE(line.anchored);
  CHECK(line.rear.distance(expected)<2.e-14);
}

TEST_CASE("Fault damage anchors the trial to the accepted component on its original DG trace")
{
  const Tensor<1,2> tangent({1.,0.});
  const Point<2> anchor(.1,.25),tip(3.7,.25);
  std::vector<Cell> cells = {cell(0,{0,0},{1,1},constant(0.)),
                             cell(1,{1,0},{2,1},constant(0.)),
                             cell(2,{2,0},{3,1},constant(.02)),
                             cell(3,{3,0},{4,1},constant(0.))
                            };
  const auto parent=seed_on_line(partition(cells),anchor,tip,tangent,MPI_COMM_WORLD);
  REQUIRE(parent.source.has_value());
  CHECK(parent.source->cell==cells[2].id);
  cells[0].phase=constant(.02);
  cells[3].phase=constant(.02);
  const auto unpinned=connected_component(partition(cells),anchor,tip,tangent,MPI_COMM_WORLD);
  CHECK(unpinned.cells.count(cells[0].id)==1);
  CHECK(unpinned.cells.count(cells[3].id)==0);
  const auto trial=connected_component(partition(cells),anchor,tip,tangent,MPI_COMM_WORLD,parent.source);
  CHECK(trial.cells.count(cells[0].id)==0);
  CHECK(trial.cells.count(cells[3].id)==1);
  const auto cheap_trial=seed_on_line(partition(cells),anchor,tip,tangent,MPI_COMM_WORLD,parent.source);
  REQUIRE(cheap_trial.anchored);
  CHECK(cheap_trial.rear==trial.rear);
}

TEST_CASE("Fault damage follows an inclined line through a face-connected mesh vertex")
{
  const Point<2> anchor(1.5,1.5),tip(1.75,1.75);
  const auto tangent=(tip-anchor)/(tip-anchor).norm();
  std::vector<Cell> cells= {cell(0,{0,0},{1,1},constant(.03)),cell(1,{1,1},{2,2},constant(.03))};
  const auto isolated=seed_on_line(partition(cells),anchor,tip,tangent,MPI_COMM_WORLD);
  REQUIRE(isolated.anchored);
  CHECK(isolated.rear.distance({1,1})<1.e-14);
  // This third cell meets the diagonal only at (1,1), but its positive
  // trace overlaps both neighbors' faces over positive length.
  cells.push_back(cell(2, {0,1}, {1,2},constant(.03)));
  const auto connected_line=seed_on_line(partition(cells),anchor,tip,tangent,MPI_COMM_WORLD);
  REQUIRE(connected_line.anchored);
  CHECK(connected_line.rear.distance({0,0})<1.e-14);
}
