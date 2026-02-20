/*
  Copyright (C) 2025 - by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
 */

#ifndef _aspect_particle_particle_domain_h
#define _aspect_particle_particle_domain_h

#include <deal.II/particles/particle_handler.h>

namespace aspect
{
  namespace Particle
  {
    using namespace dealii::Particles;

    namespace ParticleDomain
    {
      /**
       * Class holding the edge lengths/face areas and neighbors of particle 
       * domains.
       */
      template <int dim>
      class FaceData
      {
        public:
          using particle_iterator = typename ParticleHandler<dim>::particle_iterator;

          /**
           * Clear the data and resize the arrays.
           * @param[in] max_local_particle_index The upper limit of the local 
           *  indices of particles.
           */
          void
          reinit(const types::particle_index max_local_particle_index);

          /**
           * Push back the face data of a particle domain.
           * @param[in] particle_index The local index of the particle.
           * @param[in] face_data An array storing the measure and the 
           *  corresponding neighbor particle of each face. If a face is at 
           *  boundary, i.e. it does not have a neighbor, then the 
           *  corresponding position will be filled with an invalid iterator.
           * @param[in] particle_handler The particle handler that handles the
           *  input particle.
           */
          void
          push_back(const types::particle_index                              particle_index,
                    const std::vector<std::pair<double, particle_iterator>> &face_data,
                    const ParticleHandler<dim>                              &particle_handler);

          /**
           * Returns the number of faces of the particle domain.
           * @param[in] particle_index The local index of the particle.
           */
          unsigned int
          n_faces(const types::particle_index particle_index) const;

          /**
           * Returns the length/area of a edge/face of the particle domain.
           * @param[in] particle_index The local index of the particle.
           * @param[in] face_index The index of the face within the particle
           *  domain.
           */
          double
          face_measure(const types::particle_index particle_index,
                       const unsigned int          face_index) const;

          /**
           * Returns the neighbor particle of a edge/face of the particle
           * domain.
           * @param[in] particle_index The local index of the particle.
           * @param[in] face_index The index of the face within the particle
           *  domain.
           * @param[in] particle_handler The particle handler that handles the
           *  input particle.
           * @param[in] triangulation The background triangulation.
           */
          particle_iterator
          neighbor_particle(const types::particle_index  particle_index,
                            const unsigned int           face_index,
                            const ParticleHandler<dim>  &particle_handler,
                            const Triangulation<dim>    &triangulation) const;

        private:
          /**
           * Struct storing the minimum information required to create a 
           * particle iterator.
           */
          struct ParticleIndicator
          {
            std::uint32_t cell_index;
            std::uint16_t cell_level;
            std::uint16_t particle_index_within_cell;

            ParticleIndicator()
              : cell_index(std::numeric_limits<std::uint32_t>::max())
              , cell_level(std::numeric_limits<std::uint16_t>::max())
              , particle_index_within_cell(std::numeric_limits<std::uint16_t>::max())
            {}
          };

          /**
           * Array of the handles that indicate the start position in the
           * data sets for each particle.
           */
          std::vector<unsigned int> handles;

          /**
           * Array of the number of faces for each Voronoi cell.
           */
          std::vector<std::uint16_t> n_faces_per_vorocell;

          /**
           * Collection of the face measures of all the Voronoi cells.
           */
          std::vector<double> face_measures;

          /**
           * Collection of the neighbor particles of all the Voronoi cells.
           */
          std::vector<ParticleIndicator> neighbor_particles;
      };


      /**
       * Class holding the data required by convected particle domain 
       * interpolation (CPDI).
       */
      template <int dim>
      struct CPDIData
      {
        public:
          /**
           * Clear the data and resize the arrays.
           * @param[in] max_local_particle_index The upper limit of the local 
           *  indices of particles.
           */
          void 
          reinit(const types::particle_index max_local_particle_index);

          /**
           * Push back the CPDI data of a particle domain.
           * @param[in] particle_index The local index of the particle.
           * @param[in] cpdi_data A map storing the CPDI data of the current
           *  particle. The keys of the map are the indices of relevant
           *  vertices, while the values of the map are the values and 
           *  gradients of the CPDI weighting functions.
           */
          void 
          push_back(const types::particle_index                                      particle_index,
                    const std::map<unsigned int, std::pair<double, Tensor<1, dim>>> &cpdi_data);

          /**
           * Returns the number of relevant vertices of the particle domain.
           * The "relevant vertices" are vertices corresponding to those 
           * FE_Q(1) DoFs whose supports have non-empty intersections with
           * the particle domain.
           * @param[in] particle_index The local index of the particle.
           */
          unsigned int
          n_relevant_vertices(const types::particle_index particle_index) const;

          /**
           * Returns the index of a relevant vertex of the particle domain.
           * @param[in] particle_index The local index of the particle.
           * @param[in] i The rank of the relevant vertex within the particle
           *  domain.
           */
          unsigned int
          relevant_vertex_index(const types::particle_index particle_index,
                                const unsigned int          i) const;

          /**
           * Returns the value of the CPDI weighting function corresponding to
           * a relevant vertex.
           * @param[in] particle_index The local index of the particle.
           * @param[in] i The rank of the relevant vertex within the particle
           * domain.
           */
          double
          weighting_function_value(const types::particle_index particle_index,
                                   const unsigned int          i) const;

          /**
           * Returns the gradient of the CPDI weighting function corresponding
           * to a relevant vertex.
           * @param[in] particle_index The local index of the particle.
           * @param[in] i The rank of the relevant vertex within the particle
           * domain.
           */
          Tensor<1, dim>
          weighting_function_gradient(const types::particle_index particle_index,
                                      const unsigned int          i) const;

        private:
          std::vector<unsigned int> handles;

          std::vector<std::uint8_t> n_relevant_vertices_per_vorocell;

          std::vector<unsigned int> relevant_vertices;

          std::vector<double> weighting_function_data;
      };
    }

    template <int dim> class ParticleDomainAccessor;

    template <int dim>
    class ParticleDomainHandler
    {
      public:
        using particle_iterator = ParticleIterator<dim>;

        /**
         * Default constructor.
         */
        ParticleDomainHandler();

#if DEAL_II_VERSION_GTE(9,8,0)
        ParticleDomainHandler(const ParticleHandler<dim> &particle_handler,
                              const bool                  generate_face_data,
                              const bool                  generate_cpdi_data);
#else
        ParticleDomainHandler(const ParticleHandler<dim> &particle_handler,
                              const Triangulation<dim>   &triangulation,
                              const Mapping<dim>         &mapping,
                              const bool                  generate_face_data,
                              const bool                  generate_cpdi_data);
#endif

        void generate_particle_domains();

        ParticleDomainAccessor<dim>
        get_particle_domain(const types::particle_index particle_index) const;

        const Particles::ParticleHandler<dim> &
        get_particle_handler() const;

        bool face_data_requested() const;

        bool cpdi_data_requested() const;

      private:
        /**
         *  Address of the particle handler to work on.
         */
#if DEAL_II_VERSION_GTE(9,7,0)
        ObserverPointer<const ParticleHandler<dim>, ParticleDomainHandler<dim>> particle_handler;
#if !DEAL_II_VERSION_GTE(9,8,0)
        ObserverPointer<const Triangulation<dim>, ParticleDomainHandler<dim>>   triangulation;
        ObserverPointer<const Mapping<dim>, ParticleDomainHandler<dim>>         mapping;
#endif /*!DEAL_II_VERSION_GTE(9,8,0)*/
#else /*DEAL_II_VERSION_GTE(9,7,0)*/
        SmartPointer<const ParticleHandler<dim>, ParticleDomainHandler<dim>>    particle_handler;
        SmartPointer<const Triangulation<dim>, ParticleDomainHandler<dim>>      triangulation;
        SmartPointer<const Mapping<dim>, ParticleDomainHandler<dim>>            mapping;
#endif

        std::vector<double> volumes;

        ParticleDomain::FaceData<dim> face_data;

        ParticleDomain::CPDIData<dim> cpdi_data;

        bool generate_face_data;

        bool generate_cpdi_data;

        friend class ParticleDomainAccessor<dim>;
    };

    template <int dim>
    class ParticleDomainAccessor
    {
      public:
        using particle_iterator = typename ParticleHandler<dim>::particle_iterator;

        ParticleDomainAccessor(const ParticleDomainHandler<dim> &particle_domain_handler,
                               const types::particle_index       particle_index);

        double volume() const;

        unsigned int n_faces() const;

        double face_measure(const unsigned int face_index) const;

        particle_iterator
        neighbor_particle(const unsigned int face_index) const;

        unsigned int n_relevant_vertices() const;

        unsigned int
        relevant_vertex_index(const unsigned int i) const;

        double
        weighting_function_value(const unsigned int i) const;

        Tensor<1, dim>
        weighting_function_gradient(const unsigned int i) const;

        DeclExceptionMsg(
          ExcFaceDataNotGenerated,
          "Face data is requested but not generated. Please set "
          "`generate_face_data' to true when constructing this object.");

        DeclExceptionMsg(
          ExcCPDIDataNotGenerated,
          "CPDI data is requested but not generated. Please set "
          "`generate_cpdi_data' to true when constructing this object.");
        
      private:
        const ParticleDomainHandler<dim> *handler;

        const types::particle_index particle_index;
    };

    /*------------------------ inline functions ------------------------*/

    template <int dim>
    inline ParticleDomainAccessor<dim>
    ParticleDomainHandler<dim>::
    get_particle_domain(const types::particle_index particle_index) const
    {
      AssertIndexRange(particle_index, volumes.size());
      return ParticleDomainAccessor<dim>(*this, particle_index);
    }



    template <int dim>
    inline const Particles::ParticleHandler<dim> &
    ParticleDomainHandler<dim>::get_particle_handler() const
    {
      AssertThrow(particle_handler != nullptr,
                  ExcMessage("This particle domain handler has not been "
                             "associated with a particle handler"));

      return *particle_handler;
    }



    template <int dim>
    inline bool
    ParticleDomainHandler<dim>::face_data_requested() const
    {
      return generate_face_data;
    }



    template <int dim>
    inline bool
    ParticleDomainHandler<dim>::cpdi_data_requested() const
    {
      return generate_cpdi_data;
    }



    template <int dim>
    inline ParticleDomainAccessor<dim>::
    ParticleDomainAccessor(const ParticleDomainHandler<dim> &handler_object,
                           const types::particle_index       particle_index)
      : handler(&handler_object)
      , particle_index(particle_index)
    {}



    template <int dim>
    inline double
    ParticleDomainAccessor<dim>::volume() const
    {
      AssertIndexRange(particle_index, handler->volumes.size());
      return handler->volumes[particle_index];
    }



    template <int dim>
    inline unsigned int
    ParticleDomainAccessor<dim>::n_faces() const
    {
      Assert(handler->generate_face_data, ExcFaceDataNotGenerated());
      return handler->face_data.n_faces(particle_index);
    }



    template <int dim>
    inline double
    ParticleDomainAccessor<dim>::
    face_measure(const unsigned int face_index) const
    {
      Assert(handler->generate_face_data, ExcFaceDataNotGenerated());
      return handler->face_data.face_measure(particle_index, face_index);
    }



    template <int dim>
    inline
    typename ParticleDomainAccessor<dim>::particle_iterator
    ParticleDomainAccessor<dim>::
    neighbor_particle(const unsigned int face_index) const
    {
      Assert(handler->generate_face_data, ExcFaceDataNotGenerated());
      return handler->face_data.neighbor_particle(particle_index,
                                                  face_index,
                                                  *handler->particle_handler,
                                                  *handler->triangulation);
    }



    template <int dim>
    inline unsigned int
    ParticleDomainAccessor<dim>::n_relevant_vertices() const
    {
      Assert(handler->generate_cpdi_data, ExcCPDIDataNotGenerated());
      return handler->cpdi_data.n_relevant_vertices(particle_index);
    }



    template <int dim>
    inline unsigned int
    ParticleDomainAccessor<dim>::
    relevant_vertex_index(const unsigned int i) const
    {
      Assert(handler->generate_cpdi_data, ExcCPDIDataNotGenerated());
      return handler->cpdi_data.relevant_vertex_index(particle_index, i);
    }



    template <int dim>
    inline double
    ParticleDomainAccessor<dim>::
    weighting_function_value(const unsigned int i) const
    {
      Assert(handler->generate_cpdi_data, ExcCPDIDataNotGenerated());
      return handler->cpdi_data.weighting_function_value(particle_index, i);
    }



    template <int dim>
    inline Tensor<1, dim>
    ParticleDomainAccessor<dim>::
    weighting_function_gradient(const unsigned int i) const
    {
      Assert(handler->generate_cpdi_data, ExcCPDIDataNotGenerated());
      return handler->cpdi_data.weighting_function_gradient(particle_index, i);
    }
  }
}

#endif
