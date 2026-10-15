#ifndef ASPECT_BP3_OUTPUT_FILES_H
#define ASPECT_BP3_OUTPUT_FILES_H

#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace BP3
{
  inline bool needs_header(const std::string &path)
  {
    return !std::filesystem::exists(path) || std::filesystem::file_size(path)==0;
  }

  // Root-only filesystem failures must be reported before another rank enters
  // the next model collective. A failed writer never advances a schedule.
  template <typename Writer>
  void collective_root_write(const MPI_Comm comm, const Writer &write)
  {
    std::string error;
    if (dealii::Utilities::MPI::this_mpi_process(comm)==0)
      try { write(); }
      catch (const std::exception &exception) { error=exception.what(); }
    error=dealii::Utilities::MPI::broadcast(comm,error,0);
    AssertThrow(error.empty(),dealii::ExcMessage("BP3 output: "+error));
  }

  // Validate the preserved output prefix before resuming checkpoint history.
  inline void check_series_prefix(const std::string &path, const unsigned int step,
                                  const double time, const bool profiles=false)
  {
    if (needs_header(path)) return;
    std::ifstream input(path);
    AssertThrow(input,dealii::ExcMessage("Cannot read output index: "+path));
    std::string line, previous;
    std::getline(input,line);
    long long last_step=-1;
    double last_time=-1.;
    while (std::getline(input,line))
      {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::string number, clock, file;
        std::getline(row,number,','); std::getline(row,clock,',');
        const auto current=std::stoll(number);
        const auto current_time=std::stod(clock);
        AssertThrow(current>=0 && std::isfinite(current_time) && current_time>=0.,
                    dealii::ExcMessage("Invalid output state: "+path));
        AssertThrow(current<=step && current_time<=time,
                    dealii::ExcMessage("Output is newer than the loaded checkpoint: "+path+
                      ". Resume in a new branch using the selected checkpoint's metadata prefix."));
        // Old outputs sometimes duplicated initialization. Preserve that
        // evidence; new writers never append another copy of the loaded state.
        if (profiles && line!=previous)
          {
            AssertThrow(current>last_step && current_time>last_time,
                        dealii::ExcMessage("Conflicting profile index states: "+path));
            std::getline(row,file,',');
            AssertThrow(std::filesystem::is_regular_file(std::filesystem::path(path).parent_path()/file),
                        dealii::ExcMessage("Missing indexed profile: "+file+
                          ". Restore the payload from the parent run; an index alone is not history."));
          }
        last_step=current; last_time=current_time; previous=line;
      }
    AssertThrow(input.eof(),dealii::ExcMessage("Cannot finish reading output index: "+path));
  }

  inline void check_restart_output(const std::string &directory, const unsigned int step,
                                   const double time)
  {
    for (const auto *name : {"accepted_steps.csv","stations.csv","heavy_outputs.csv","restored_growth.csv","particle_summary.csv"})
      check_series_prefix(directory+name,step,time);
    check_series_prefix(directory+"profiles.csv",step,time,true);
    const auto payloads=std::filesystem::path(directory)/"profiles";
    if (std::filesystem::exists(payloads))
      for (const auto &entry : std::filesystem::directory_iterator(payloads))
        {
          const auto name=entry.path().filename().string();
          if (name.rfind("fault_",0)==0 && entry.path().extension()==".csv")
            AssertThrow(std::stoul(name.substr(6))<=step,
                        dealii::ExcMessage("Profile payload is newer than the checkpoint: "+entry.path().string()+
                          ". Preserve it and prepare a clean restart branch."));
        }
  }
}
#endif
