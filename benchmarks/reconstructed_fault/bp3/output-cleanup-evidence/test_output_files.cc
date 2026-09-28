#include "../plugin/output_files.h"
#include "../plugin/output_schedule.h"
#include <deal.II/base/mpi.h>
#include <iostream>
int main(int argc,char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc,argv,1);
  dealii::deal_II_exceptions::disable_abort_on_exception();
  const auto comm=MPI_COMM_WORLD;
  const std::string dir=std::string(argv[1])+"/";
  BP3::collective_root_write(comm,[&]() {
    std::filesystem::create_directories(dir+"profiles");
    AssertThrow(BP3::needs_header(dir+"growth.csv"),dealii::ExcInternalError());
    {std::ofstream out(dir+"growth.csv");}
    AssertThrow(BP3::needs_header(dir+"growth.csv"),dealii::ExcInternalError());
    {std::ofstream out(dir+"growth.csv");out<<"step,time\n";}
    AssertThrow(!BP3::needs_header(dir+"growth.csv"),dealii::ExcInternalError());
    {std::ofstream out(dir+"profiles/fault_3.csv");out<<"payload\n";}
    {std::ofstream out(dir+"profiles.csv");out<<"step,time_s,file\n3,300,profiles/fault_3.csv\n";}
    BP3::check_restart_output(dir,4,400.);
  });
  unsigned int caught=0;
  BP3::OutputSchedule schedule;schedule.written(0.,{0.});
  try {
    BP3::collective_root_write(comm,[&]() {
      std::ofstream out(dir+"missing/index.csv");
      out.exceptions(std::ios::failbit|std::ios::badbit);out<<"failure\n";out.close();
    });
    schedule.written(100.,{1.});
  } catch(const std::exception &) {++caught;}
  AssertThrow(schedule.last_time==0. && schedule.reference[0]==0.,dealii::ExcInternalError());
  try {BP3::collective_root_write(comm,[&](){BP3::check_restart_output(dir,2,200.);});}
  catch(const std::exception &) {++caught;}
  BP3::collective_root_write(comm,[&]() {
    std::ofstream out(dir+"profiles.csv");out<<"step,time_s,file\n3,300,profiles/missing.csv\n";
  });
  try {BP3::collective_root_write(comm,[&](){BP3::check_restart_output(dir,4,400.);});}
  catch(const std::exception &) {++caught;}
  AssertThrow(dealii::Utilities::MPI::min(caught,comm)==3,dealii::ExcInternalError());
  if(dealii::Utilities::MPI::this_mpi_process(comm)==0)
    std::cout<<"PASS headers, checkpoint prefix, missing payload, collective write failure, unchanged schedule on "
             <<dealii::Utilities::MPI::n_mpi_processes(comm)<<" ranks\n";
}
