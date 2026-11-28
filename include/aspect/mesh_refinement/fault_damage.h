/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#ifndef _aspect_mesh_refinement_fault_damage_h
#define _aspect_mesh_refinement_fault_damage_h

#include <aspect/phase_field/bernstein_bounds.h>
#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/grid/cell_id.h>
#include <map>
#include <optional>

namespace aspect::MeshRefinement::FaultDamage
{
  using dealii::BoundingBox;
  using dealii::CellId;
  using dealii::Point;
  using dealii::Tensor;
  using PhaseField::BernsteinPatch;

  /** The dimensionless damage contour used to request a resolved bulk mesh. */
  constexpr double damage_contour = 0.01;

  /** A single physical Cartesian cell and its endpoint DGQ2 phase. Bernstein
   * coefficients are in reference tensor order with x varying fastest. */
  struct Cell
  {
    CellId id;
    BoundingBox<2> bounds;
    BernsteinPatch<2> phase;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
      ar &id &bounds &phase;
    }
  };

  /** A point of supported phase on the incoming seed line in one physical
   * cell. Its position is anchor + distance_from_anchor * tangent (metres).
   * The original cell identifies the one-sided DG trace at a mesh face.
   * Unit coordinates in that cell identify the same trace after refinement;
   * ties on a new bisection plane use the lower child. */
  struct Source
  {
    CellId cell;
    long double distance_from_anchor;
    Point<2> unit_point;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
      ar &cell &distance_from_anchor &unit_point;
    }
  };

  /** Replicated cells of the seed-connected damage excursion. If anchored,
   * rear is the backward end (m) of the contiguous excursion on the seed line. */
  struct Component
  {
    bool anchored = false;
    // False when separate excursion components touch the incoming segment.
    bool unique_incoming_component = true;
    Point<2> rear;
    std::optional<Source> source;
    std::map<CellId,BoundingBox<2>> cells;
  };

  /** Local convex-hull exclusion. A positive result requires a connectivity check. */
  bool may_contain_damage(const BernsteinPatch<2> &phase);

  /**
   * Locate the connected endpoint excursion touching the incoming segment
   * from anchor to tip, preferring support nearest anchor. tangent is the
   * physical unit seed tangent pointing forward. Supply each locally owned
   * Cartesian cell once. Boxes and points are in metres. After initialization,
   * pass the original source for every accepted or trial field, including
   * after refinement. Only that supported one-sided trace seeds the search. Collective.
   *
   * Excursions occupy positive area in the open cell, including a plateau
   * at the contour. Isolated equalities carry no area. Independent DG traces
   * must both reach the contour on a shared face interval
   * of positive length. Interior connectivity is exact on sign-definite or
   * monotone patches with a connected high edge. Otherwise bisect each axis
   * up to five times. Unresolved pieces inside one terminal rectangle may
   * connect conservatively. Each such local join spans at most its diagonal
   * (parent diagonal / 32). Only the actual face and seed-line polynomials
   * connect across rectangles or anchor the result.
   */
  Component connected_component(const std::vector<Cell> &locally_owned,
                                const Point<2>          &anchor,
                                const Point<2>          &tip,
                                const Tensor<1,2>       &tangent,
                                MPI_Comm                 comm,
                                std::optional<Source>    parent_source = {});

  /** Find the backward extent on the incoming line. Collective, exchanging
   * only cells meeting the closed physical line at or behind the seed tip.
   * Pass the same accepted source as for connected_component() when
   * querying a trial. Returns the source and rear with an empty cell map. */
  Component seed_on_line(const std::vector<Cell> &locally_owned,
                         const Point<2>          &anchor,
                         const Point<2>          &tip,
                         const Tensor<1,2>       &tangent,
                         MPI_Comm                 comm,
                         std::optional<Source>    parent_source = {});
}

#endif
