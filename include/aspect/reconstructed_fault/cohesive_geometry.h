/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef aspect_cohesive_geometry_h
#define aspect_cohesive_geometry_h
#include <aspect/global.h>
#include <deal.II/base/quadrature.h>
#include <boost/serialization/vector.hpp>

namespace aspect
{
  namespace CohesiveInterface
  {
    /** Retained smooth graph path/lifting, not a new direction-selection law.
     * Existing pieces are immutable; zero-history candidate pieces may retire.
     * The fixed Cartesian bulk mesh does not move. */
    class Geometry
    {
    public:
      struct Piece
      {
        double x = 0., y = 0., end = 0., a = 0., b = 0., c1 = 0., c2 = 0., c3 = 0.;
        unsigned int cells = 1;
        bool hermite = false;
        template <class Archive>
        void
        serialize (Archive &ar, const unsigned int)
        {
          ar & x & y & end & a & b & c1 & c2 & c3 & cells & hermite;
        }
      };
      struct Sample
      {
        double y = 0., slope = 0., second = 0.;
        Tensor<1, 2> tangent, tangent_derivative;
      };
      struct Mode
      {
        unsigned int index = 0;
        Tensor<1, 2> value;
        Tensor<2, 2> gradient;
      };
      struct Point
      {
        dealii::Point<2> position;
        Tensor<1, 2> tangent;
        double weight = 0., xi = 0.;
        unsigned int cell = 0, left = 0, right = 0;
      };
      struct Bounds
      {
        dealii::Point<2> lower, upper;
      };
      struct Clearance
      {
        double sampling = 0., lifting = 0.;
        unsigned int sampling_wall = 0, lifting_wall = 0;
      };

      Geometry () = default;
      explicit Geometry (std::vector<Piece> pieces, double width);
      void rebuild ();
      Sample sample (double x) const;
      std::vector<Mode> lifting (const dealii::Point<2> &, int side = 0) const;
      Mode normal_test (const dealii::Point<2> &) const;
      bool normal_active (const dealii::Point<2> &) const;
      Quadrature<2> quadrature (const dealii::Point<2> &lo, const dealii::Point<2> &hi,
                                unsigned int order) const;
      std::vector<Point> surface_points () const;
      /** Complete regions, not a test of quadrature-point coverage. Lifting
       * uses the actual vertical graph strip, not a Euclidean normal tube.
       * Polynomial interval bounds are conservative and outward rounded. */
      Bounds lifting_bounds () const;
      Bounds sampling_bounds (unsigned int formed_pieces, double radius) const;
      Clearance clearance (const Bounds &domain, unsigned int formed_pieces, double radius) const;
      std::string mode_signature (unsigned int index) const;
      void append (double preferred, double arclength, double R, unsigned int cells = 1);
      unsigned int
      modes () const
      {
        return nodes.size () - 2;
      }
      const std::vector<double> &
      vertices () const
      {
        return nodes;
      }
      const std::vector<Piece> &
      pieces () const
      {
        return curve;
      }
      std::vector<Piece> &
      candidate_pieces ()
      {
        return curve;
      }
      double
      width () const
      {
        return support;
      }
      static double preferred_direction (const SymmetricTensor<2, 3> &, double old_tangent);
      template <class Archive>
      void
      serialize (Archive &ar, const unsigned int)
      {
        ar & curve & support;
        if constexpr (Archive::is_loading::value)
          rebuild ();
      }

    private:
      std::vector<double> cuts (double l, double r, double bottom, double top) const;
      std::vector<Piece> curve;
      std::vector<double> nodes;
      double support = .75;
    };
  }
}
#endif
