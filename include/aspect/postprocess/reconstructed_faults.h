/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#ifndef _aspect_postprocess_reconstructed_faults_h
#define _aspect_postprocess_reconstructed_faults_h

#include <aspect/postprocess/interface.h>
#include <aspect/reconstructed_fault/manager.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace internal
    {
      /** Build and write a flattened output-only view of reconstructed faults. */
      template <int dim>
      class ReconstructedFaultOutput
      {
        public:
          explicit ReconstructedFaultOutput(
            const std::vector<ReconstructedFault<dim>> &faults,
            const std::vector<typename ReconstructedFaultManager<dim>::PropertyInformation>
            &property_information,
            const std::vector<std::vector<double>> *slip_rates = nullptr,
            const std::vector<std::string> &excluded_properties = {});

          void write_vtu(std::ostream &output,
                         const double time,
                         const unsigned int timestep_number) const;

        private:
          struct PropertyOutput
          {
            std::string name;
            unsigned int n_components;
            std::vector<double> values;
          };

          std::vector<Point<dim>> points;
          std::vector<unsigned int> point_fault_ids;
          std::vector<std::array<unsigned int,2>> cells;
          std::vector<unsigned int> cell_fault_ids;
          std::vector<double> slip_rates;
          std::vector<PropertyOutput> properties;
      };
    }


    /** Write replicated reconstructed-fault line geometry in VTU format. */
    template <int dim>
    class ReconstructedFaults : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm);
        void parse_parameters(ParameterHandler &prm) override;

        void initialize() override;

        std::pair<std::string,std::string>
        execute(TableHandler &statistics) override;

        void save(std::map<std::string,std::string> &status) const override;
        void load(const std::map<std::string,std::string> &status) override;

      private:
        std::vector<std::string> excluded_properties;
        std::vector<std::pair<double,std::string>> times_and_vtu_file_names;
    };
  }
}

#endif
