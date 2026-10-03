#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>
#include <cmath>
#include <limits>

namespace aspect { namespace Postprocess {
  template <int dim>
  class EndpointCheck : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
    std::pair<std::string,std::string> execute(TableHandler &) override
    {
      if constexpr (dim==2)
        {
          const auto &manager=this->get_reconstructed_fault_manager();
          const auto check=[&](const Point<dim> &p)
          {
            const auto strip=manager.project_to_normal_profiles(p);
            const auto source=manager.project_to_bulk_source(p);
            AssertThrow(source.active,ExcMessage("Endpoint handoff left an admissible point unassigned."));
            if (strip.active)
              AssertThrow(source.fault_index==strip.fault_index
                          && source.segment_index==strip.segment_index && source.xi==strip.xi
                          && source.signed_distance==strip.signed_distance,
                          ExcMessage("Endpoint handoff replaced a valid strip association."));
            else
              {
                const auto cached=manager.project_to_bulk_source(p,true);
                AssertThrow(cached.active && cached.fault_index==source.fault_index
                            && cached.segment_index==source.segment_index && cached.xi==source.xi
                            && cached.signed_distance==source.signed_distance,
                            ExcMessage("Cached inactive and uncached source paths differ."));
              }
          };
          // The four positive-phase quadrature points in the reported 45-degree gap.
          for (const auto &p: {Point<2>(51000.,1000.),Point<2>(53000.,3000.),
                              Point<2>(53774.59666924148,3774.5966692414836),Point<2>(-5000.,45000.)})
            {
              check(p);
              // Well outside floating-point ambiguity, on both sides of the plane.
              Tensor<1,2> t({-std::sqrt(.5),std::sqrt(.5)});
              check(p+1e-6*t);
              check(p-1e-6*t);
              for (unsigned int d=0;d<2;++d)
                for (double direction: {-std::numeric_limits<double>::infinity(),
                                         std::numeric_limits<double>::infinity()})
                  {
                    auto adjacent=p;
                    adjacent[d]=std::nextafter(p[d],direction);
                    check(adjacent);
                  }
            }
          for (const auto &c:manager.get_boundary_contacts())
            {
              const double sign=std::copysign(1.,c.normal*c.inward_boundary_normal);
              const auto exterior=c.position-1e-4*c.inward_tangent;
              AssertThrow(!manager.project_to_bulk_source(exterior+sign*(c.transverse_extent+1.)*c.normal).active,
                          ExcMessage("Endpoint handoff enlarged transverse support."));
              AssertThrow(!manager.project_to_bulk_source(exterior-sign*c.transverse_extent*.5*c.normal).active,
                          ExcMessage("Endpoint handoff crossed the physical boundary."));
              // Inside the physical domain, beyond the strip radius but still
              // inside the continuation enclosure: the cached-inactive shortcut
              // must not grow continuation into the positive-s strip side.
              const auto interior=c.position+1e-3*c.inward_tangent
                                  +sign*.99*c.transverse_extent*c.normal;
              AssertThrow(!manager.project_to_normal_profiles(interior).active,
                          ExcMessage("Endpoint fixture needs an inactive strip association."));
              AssertThrow(!manager.project_to_bulk_source(interior,true).active
                          && !manager.project_to_bulk_source(interior).active,
                          ExcMessage("Endpoint handoff grew beyond endpoint roundoff."));
            }
          std::cout<<"ENDPOINT HANDOFF PASS rank "
                   <<Utilities::MPI::this_mpi_process(this->get_mpi_communicator())<<std::endl;
        }
      return {};
    }
  };
  ASPECT_REGISTER_POSTPROCESSOR(EndpointCheck,"endpoint handoff check","Focused automatic endpoint-plane regression.")
}}
