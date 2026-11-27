/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/boundary_velocity/interface.h>
#include <aspect/geometry_model/box.h>
#include <aspect/initial_composition/interface.h>
#include <aspect/mesh_refinement/interface.h>
#include <aspect/phase_field/degradation.h>

namespace aspect
{
  namespace
  {
    constexpr double sine = 0.86602540378443864676;
    constexpr double peak = .6, ell = 20., plate_speed = 1.e-9;

    // The AT1 first integral is tabulated in theta, phi=peak*cos(theta)^2.
    // The common scale of h=1/g-1 cancels in both the ODE and normalized ramp.
    struct TransverseProfile
    {
      std::vector<double> radius, phase, primitive;
      QGauss<1> gauss {8};

      TransverseProfile()
      {
        constexpr unsigned int count=5000;
        const double step=numbers::PI_2/(count-1);
        const double h=PhaseField::excess_compliance(peak,1.);
        double previous=0.;
        for (unsigned int i=0; i<count; ++i)
          {
            const double theta=i*step;
            const double phi=peak*std::pow(std::cos(theta),2);
            const double dh=(1.+3.*peak)/std::pow(1.-peak,3);
            const double derivative=(i==0 ? 2.*ell*std::sqrt(peak*h/(peak*dh-h)) :
                                     i+1==count ? 2.*ell*std::sqrt(peak*h/(h-peak)) :
                                     2.*ell*peak*std::sin(theta)*std::cos(theta)*
                                     std::sqrt(h/(h*phi-peak*PhaseField::excess_compliance(phi,1.))));
            radius.push_back(i==0 ? 0. : radius.back()+.5*step*(previous+derivative));
            phase.push_back(phi);
            primitive.push_back(i==0 ? 0. : primitive.back()+panel(i,1.));
            previous=derivative;
          }
      }

      // Integral of h over a fraction of one piecewise-linear phase interval.
      double panel(unsigned int right,double fraction) const
      {
        double sum=0.;
        for (unsigned int q=0; q<gauss.size(); ++q)
          sum+=gauss.weight(q)*PhaseField::excess_compliance(phase[right-1]+fraction*gauss.point(q)[0]*
                                                             (phase[right]-phase[right-1]),1.);
        return sum*fraction*(radius[right]-radius[right-1]);
      }

      double value(double distance,bool integrated=false) const
      {
        const double r=std::abs(distance);
        if (r>=radius.back())
          return integrated ? std::copysign(.5,distance) : 0.;
        const unsigned int j=std::upper_bound(radius.begin(),radius.end(),r)-radius.begin();
        const double f=(r-radius[j-1])/(radius[j]-radius[j-1]);
        if (integrated)
          return std::copysign((primitive[j-1]+panel(j,f))/(2.*primitive.back()),distance);
        return phase[j-1]+f*(phase[j]-phase[j-1]);
      }
    };

    const TransverseProfile &profile()
    {
      static const TransverseProfile table;
      return table;
    }

    template <int dim>
    const GeometryModel::Box<dim> &box(const SimulatorAccess<dim> &sim)
    {
      return Plugins::get_plugin_as_type<const GeometryModel::Box<dim>>(sim.get_geometry_model());
    }

    template <int dim>
    double distance(const Point<dim> &p,const GeometryModel::Box<dim> &geometry)
    {
      return sine*p[0]+.5*(p[1]-geometry.get_origin()[1]-geometry.get_extents()[1]);
    }

    template <int dim>
    double finest(const SimulatorAccess<dim> &sim)
    {
      const auto &geometry=box(sim);
      return std::ldexp(geometry.get_extents()[0]/geometry.get_repetitions()[0],
                        -static_cast<int>(sim.get_parameters().initial_global_refinement));
    }
  }

  namespace InitialComposition
  {
    template <int dim>
    class HistoricalPhase : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        double initial_composition(const Point<dim> &point,const unsigned int field) const override
        {
          if (field!=this->introspection().compositional_index_for_name("phase_field"))
            return 0.;
          const auto &geometry=box(*this);
          const double h=finest(*this);
          // The donor stores vertex-interpolated Q1 phase. Embed exactly that
          // polynomial in the branch DGQ2 field. All nonzero cells are finest.
          Point<dim> lower=point;
          for (unsigned int d=0; d<2; ++d)
            lower[d]=geometry.get_origin()[d]+h*std::floor((point[d]-geometry.get_origin()[d])/h);
          double result=0.;
          const auto xi=(point-lower)/h;
          for (unsigned int i=0; i<2; ++i)
            for (unsigned int j=0; j<2; ++j)
              {
                auto node=lower;
                node[0]+=h*i;
                node[1]+=h*j;
                result+=(i ? xi[0] : 1.-xi[0])*(j ? xi[1] : 1.-xi[1])*profile().value(distance(node,geometry));
              }
          return result;
        }
    };
    ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(HistoricalPhase,"BP3 historical AT1",
                                              "Stationary vertex-interpolated transverse AT1 phase with zero initial stress.")
  }

  namespace BoundaryVelocity
  {
    template <int dim>
    class HistoricalLoading : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        Tensor<1,dim> boundary_velocity(const types::boundary_id,const Point<dim> &position) const override
        {
          const double speed=-plate_speed*profile().value(distance(position,box(*this)),true);
          Tensor<1,dim> velocity;
          velocity[0]=.5*speed;
          velocity[1]=-sine*speed;
          return velocity;
        }
    };
    ASPECT_REGISTER_BOUNDARY_VELOCITY_MODEL(HistoricalLoading,"BP3 integrated loading",
                                            "Plate velocity along dip from the normalized integral of diffuse slip.")
  }

  namespace MeshRefinement
  {
    template <int dim>
    class HistoricalMesh : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        void tag_additional_cells() const override
        {
          AssertThrow(dim==2,ExcMessage("The historical BP3 experiment requires a two-dimensional box."));
          const auto &g=box(*this);
          const double fine=finest(*this);
          const double coarse=std::ldexp(g.get_extents()[0]/g.get_repetitions()[0],
                                         -static_cast<int>(this->get_parameters().min_grid_level));
          const double band=std::max(2.*ell,profile().radius.back()+2.*fine);
          const double footprint=(profile().radius.back()+coarse*(sine+.5))/sine+fine;
          const double top=g.get_origin()[1]+g.get_extents()[1];
          for (const auto &cell:this->get_triangulation().active_cell_iterators())
            if (cell->is_locally_owned())
              {
                const auto lo=cell->vertex(0), hi=cell->vertex(GeometryInfo<dim>::vertices_per_cell-1);
                const double h=hi[0]-lo[0];
                AssertThrow(std::abs(h-(hi[1]-lo[1]))<1.e-10*h,ExcMessage("BP3 grading requires square cells."));
                const double d=std::abs(distance(cell->center(),g))-.5*h*(sine+.5);
                double target=d<=band ? fine : std::min(coarse,2.*fine+(d-band)/4.);
                for (double y :
                     {
                       top,g.get_origin()[1]
                     })
                  {
                    const double x=(top-y)*.5/sine;
                    if (lo[0]<=x+footprint && hi[0]>=x-footprint && lo[1]<=y+2.*fine && hi[1]>=y-2.*fine)
                      target=fine;
                  }
                cell->clear_refine_flag();
                cell->clear_coarsen_flag();
                if (h>target*(1.+1.e-12))
                  cell->set_refine_flag();
              }
        }
    };
    ASPECT_REGISTER_MESH_REFINEMENT_CRITERION(HistoricalMesh,"BP3 historical footprint",
                                              "Startup-only dyadic support and contact refinement of the historical BP3 box.")
  }
}
