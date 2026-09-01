/*
  Copyright (C) 2026 - by the authors of the ASPECT code.

  This file is part of ASPECT.
*/

#include "common.h"

#include <aspect/postprocess/reconstructed_faults.h>

#include <sstream>


TEST_CASE("Reconstructed fault output writes line geometry and identifiers")
{
  const std::vector<aspect::ReconstructedFault<2>> faults =
  {
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0),
                                   dealii::Point<2>(1,0),
                                   dealii::Point<2>(2,0)}),
    aspect::ReconstructedFault<2>({dealii::Point<2>(4,1),
                                   dealii::Point<2>(5,2)})
  };

  const aspect::Postprocess::internal::ReconstructedFaultOutput<2> data_out(faults, {});
  std::ostringstream output;
  data_out.write_vtu(output, 2.5, 7);
  const std::string vtu = output.str();

  REQUIRE(vtu.find("type=\"UnstructuredGrid\"") != std::string::npos);
  REQUIRE(vtu.find("NumberOfPoints=\"5\" NumberOfCells=\"3\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"connectivity\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"fault_id\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"vertex_id\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"cell_id\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"TIME\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"CYCLE\"") != std::string::npos);
  REQUIRE(vtu.find("Name=\"slip_rate\"") == std::string::npos);
  REQUIRE(vtu.find("3 ") != std::string::npos); // VTK_LINE
}


TEST_CASE("Reconstructed fault output writes distinguished slip rate")
{
  const std::vector<aspect::ReconstructedFault<2>> faults =
  {
    aspect::ReconstructedFault<2>({dealii::Point<2>(0,0),
                                   dealii::Point<2>(1,0),
                                   dealii::Point<2>(2,0)})
  };
  const std::vector<std::vector<double>> slip_rates = {{1.25, 2.5, 5.0}};

  const aspect::Postprocess::internal::ReconstructedFaultOutput<2> data_out(
    faults, {}, &slip_rates);
  std::ostringstream output;
  data_out.write_vtu(output, 0.0, 0);
  const std::string vtu = output.str();

  REQUIRE(vtu.find("Name=\"slip_rate\"") != std::string::npos);
  REQUIRE(vtu.find("1.25 2.5 5 ") != std::string::npos);
}
