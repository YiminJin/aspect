/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef aspect_cohesive_system_internal_h
#define aspect_cohesive_system_internal_h
#include <aspect/reconstructed_fault/cohesive_system.h>
#include <aspect/reconstructed_fault/cohesive_bulk_history.h>
#include <aspect/reconstructed_fault/cohesive_geometry.h>
#include <aspect/reconstructed_fault/cohesive_evolution.h>
#include <aspect/reconstructed_fault/manager.h>
#include <deal.II/base/parsed_function.h>
#include <map>

namespace aspect
{
  namespace CohesiveInterface
  {
    struct System::Implementation
    {
      using Tensor = SymmetricTensor<2, 3>;
      struct Metadata
      {
        double selected_time = -1., admitted_time = -1., preferred = 0.;
        template <class Archive>
        void
        serialize (Archive &ar, const unsigned int)
        {
          ar & selected_time & admitted_time & preferred;
        }
      };
      struct State
      {
        Geometry geometry;
        unsigned int formed = 0, step = 0, registry_publications = 0;
        std::uint64_t publications = 0, identity_clock = 0;
        double time = 0., virgin_theta = 1.;
        int candidate_epoch = -1;
        unsigned int candidate_anchor = 0;
        std::vector<Metadata> metadata;
        std::vector<std::string> ids;
        std::vector<double> jump, memory, rate, offset;
        std::shared_ptr<BulkHistory<2>> bulk;
        std::shared_ptr<ReconstructedFaultManager<2>> manager;
        std::map<std::string, double> diagnostics;
      };
      struct Column
      {
        std::shared_ptr<Field> load, response, scaled;
      };
      struct Prepared
      {
        Geometry geometry;
        std::vector<Geometry::Point> points;
        CohesiveEvolution::Operators surface;
        std::vector<Column> columns, normals;
        std::vector<double> normal_mass;
        std::vector<double> incoming_memory;
        std::vector<PointState> incoming_points;
        std::shared_ptr<Field> particular, correction, base;
        double transfer_stress = 0., transfer_energy = 0.;
      };

      explicit Implementation (System &);
      static Tensor dev3 (const SymmetricTensor<2, 2> &);
      static Tensor dev3 (const dealii::Tensor<2, 2> &);
      std::shared_ptr<Field> owned () const;
      std::shared_ptr<Field> ghosted () const;
      std::shared_ptr<Field> scaled (const Field &) const;
      std::vector<PointState> points (const State &, const Geometry &) const;
      unsigned int formed_cells (const State &) const;
      std::vector<Tensor> physical_stress (const State &, const BulkHistory<2>::Cell &,
                                           const Quadrature<2> &) const;
      std::shared_ptr<Prepared> prepare (const State &, Geometry, const BulkSolve &);
      void forcing (Prepared &, const State &, const Field &, const BulkSolve &);
      std::shared_ptr<Trial> trial (const State &, std::shared_ptr<Prepared>, const Field &);
      std::pair<Tensor, double> sample (const State &) const;
      std::shared_ptr<ReconstructedFaultManager<2>>
      manager (const State &, const std::vector<PointState> &,
               const std::vector<std::array<double, 3>> &) const;
      void validate (const State &) const;
      double clock_bound (const State &) const;
      std::shared_ptr<State> import (const BulkSolve &);
      std::shared_ptr<State> initialize_seed ();
      State objective_anchor(const State &) const;
      std::shared_ptr<Field> elastic (const Geometry &, const Field &, const std::vector<double> &,
                                      const BulkSolve &) const;
      void conversion_check (State &, const std::string &) const;
      void write (const State &) const;
      std::vector<std::array<double, 6>> traces (const State &, const Field &) const;
      std::string configuration () const;
      void find_domain ();
      Geometry::Clearance protect_domain (const Geometry &, unsigned int formed,
                                          const std::string &stage) const;

      System &system;
      Parameters law;
      double R = .75, support = .75, spacing = .09375, candidate_extent = .1875, cadence = .0025,
             max_length = 7.5;
      double epoch_origin=.410, max_candidate_extent=1.5, boundary_warning=.75;
      unsigned int max_completion_retries=4, seed_cells=32;
      double seed_x=3.,seed_y=0.,seed_length=3.,seed_angle=0.;
      double virgin_age=1.;
      Geometry::Bounds domain;
      bool domain_ready=false;
      std::string setup_signature;
      unsigned int bulk_order = 6, selection_order = 8;
      double end_time = std::numeric_limits<double>::max ();
      double pressure_scaling = 1.;
      bool matrix_ready = false;
      std::string initial_mode = "physical seed", initial_file, initial_expression;
      Functions::ParsedFunction<2> independent_tensor;
      Functions::ParsedFunction<1> initial_surface;
      Functions::ParsedFunction<2> initial_native;
      std::shared_ptr<State> accepted;
      std::weak_ptr<Trial> active_trial;
      bool solving = false;
      bool jaumann_stress=false;
      unsigned int clock_attempts = 0;
      std::string pending_checkpoint;
      std::map<std::string, Column> cached_columns, cached_normals;
      std::string prepared_signature;
      std::shared_ptr<Prepared> cached_prepared;
      std::shared_ptr<Field> initial_pressure;
      double setup_seconds = 0., solve_seconds = 0., history_seconds = 0.;
      unsigned int bulk_solves = 0, cache_hits = 0;
      // Physical histories are stored once in manager left-vertex Gauss3 rows.
      // The last vertex has no cell/history. Diagnostic rows are derived data.
      static constexpr unsigned int history_position = 0, diagnostic_position = 18;
    };

    struct System::Trial
    {
      std::shared_ptr<Implementation::State> parent, state;
      std::shared_ptr<Field> native_solution;
      std::shared_ptr<Implementation::Prepared> prepared;
      CohesiveEvolution::Result surface;
    };
  }
}
#endif
