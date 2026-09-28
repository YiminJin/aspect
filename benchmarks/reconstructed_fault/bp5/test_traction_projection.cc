// Offline test harness, selected only by loading this separate test plugin.
#include <aspect/reconstructed_fault/utilities.h>
#include <aspect/simulator_signals.h>
#include <deal.II/base/mpi.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>

namespace
{
  void audit_projection(const unsigned int, dealii::ParameterHandler &)
  {
    using namespace aspect;
    const char *root=std::getenv("ASPECT_TRACTION_PROJECTION_AUDIT");
    AssertThrow(root, dealii::ExcMessage("Set ASPECT_TRACTION_PROJECTION_AUDIT to the prepared offline directory."));
    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)!=0) return;
    std::ifstream manifest(std::string(root)+"/manifest.txt");
    AssertThrow(manifest, dealii::ExcMessage("Missing projection-audit manifest."));
    std::string name;
    while (manifest>>name)
      {
        std::ifstream in(std::string(root)+"/"+name+".input");
        unsigned int n=0,nrhs=0;in>>n>>nrhs;
        AssertThrow(n>1 && nrhs>0, dealii::ExcMessage("Invalid audit dimensions."));
        std::vector<double> diagonal(n),off(n-1),rhs(n);
        for (auto &x:diagonal) in>>x;
        for (auto &x:off) in>>x;
        std::ofstream out(std::string(root)+"/"+name+".projected");
        out.exceptions(std::ios::failbit|std::ios::badbit);
        out<<std::setprecision(17);
        for (unsigned int column=0;column<nrhs;++column)
          {
            for (auto &x:rhs) in>>x;
            AssertThrow(in, dealii::ExcMessage("Incomplete projection RHS."));
            // Identical routine to BP3/BP5's consistent traction postprocessor.
            const auto values=ReconstructedFaultUtilities::solve_tridiagonal_system(diagonal,off,rhs);
            for (const auto x:values) out<<x<<' ';
            out<<'\n';
          }
      }
  }

  void connect_audit()
  {
    aspect::SimulatorSignals<2>::declare_additional_parameters.connect(&audit_projection);
    aspect::SimulatorSignals<3>::declare_additional_parameters.connect(&audit_projection);
  }
}

ASPECT_REGISTER_SIGNALS_PARAMETER_CONNECTOR(connect_audit)
