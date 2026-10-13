// Benchmark-only PMPI timing. No solver settings or communicated data change.
// Build with mpicxx -O2 -shared -fPIC; preload only for matched timing controls.
#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace
{
  struct Counter {const char *name; unsigned long long calls=0; double seconds=0.;};
  double seconds()
  {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  Counter counters[]={{"Allreduce"},{"Reduce"},{"Allgather"},{"Allgatherv"},{"Bcast"},
    {"Alltoall"},{"Alltoallv"},{"Barrier"},{"Wait"},{"Waitall"},{"Waitany"},
    {"Send"},{"Recv"},{"Isend"},{"Irecv"},{"Test"},{"Testall"}};
  unsigned int epoch=0;
  void write_counters(const bool reset)
  {
    if(const char *prefix=std::getenv("ASPECT_BENCHMARK_MPI_PROFILE"))
      {
        int rank=0;PMPI_Comm_rank(MPI_COMM_WORLD,&rank);
        const std::string path=std::string(prefix)+"-epoch-"+std::to_string(epoch++)+"-rank-"+std::to_string(rank)+".csv";
        if(FILE *out=std::fopen(path.c_str(),"w"))
          {
            std::fprintf(out,"operation,calls,seconds\n");
            for(const auto &c:counters)std::fprintf(out,"%s,%llu,%.17g\n",c.name,c.calls,c.seconds);
            std::fclose(out);
          }
        if(reset)for(auto &c:counters){c.calls=0;c.seconds=0.;}
      }
  }
}
#define TIMED(index,name,args,values) \
  extern "C" int MPI_##name args {const double start=seconds(); \
    const int status=PMPI_##name values;auto &counter=counters[index]; \
    ++counter.calls;counter.seconds+=seconds()-start;return status;}
TIMED(0,Allreduce,(const void *s,void *r,int n,MPI_Datatype t,MPI_Op o,MPI_Comm c),(s,r,n,t,o,c))
TIMED(1,Reduce,(const void *s,void *r,int n,MPI_Datatype t,MPI_Op o,int root,MPI_Comm c),(s,r,n,t,o,root,c))
TIMED(2,Allgather,(const void *s,int ns,MPI_Datatype ts,void *r,int nr,MPI_Datatype tr,MPI_Comm c),(s,ns,ts,r,nr,tr,c))
TIMED(3,Allgatherv,(const void *s,int ns,MPI_Datatype ts,void *r,const int *nr,const int *d,MPI_Datatype tr,MPI_Comm c),(s,ns,ts,r,nr,d,tr,c))
TIMED(4,Bcast,(void *s,int n,MPI_Datatype t,int root,MPI_Comm c),(s,n,t,root,c))
TIMED(5,Alltoall,(const void *s,int ns,MPI_Datatype ts,void *r,int nr,MPI_Datatype tr,MPI_Comm c),(s,ns,ts,r,nr,tr,c))
TIMED(6,Alltoallv,(const void *s,const int *ns,const int *ds,MPI_Datatype ts,void *r,const int *nr,const int *dr,MPI_Datatype tr,MPI_Comm c),(s,ns,ds,ts,r,nr,dr,tr,c))
TIMED(7,Barrier,(MPI_Comm c),(c))
TIMED(8,Wait,(MPI_Request *r,MPI_Status *s),(r,s))
TIMED(9,Waitall,(int n,MPI_Request *r,MPI_Status *s),(n,r,s))
TIMED(10,Waitany,(int n,MPI_Request *r,int *i,MPI_Status *s),(n,r,i,s))
TIMED(11,Send,(const void *s,int n,MPI_Datatype t,int dest,int tag,MPI_Comm c),(s,n,t,dest,tag,c))
TIMED(12,Recv,(void *r,int n,MPI_Datatype t,int src,int tag,MPI_Comm c,MPI_Status *s),(r,n,t,src,tag,c,s))
TIMED(13,Isend,(const void *s,int n,MPI_Datatype t,int dest,int tag,MPI_Comm c,MPI_Request *r),(s,n,t,dest,tag,c,r))
TIMED(14,Irecv,(void *s,int n,MPI_Datatype t,int src,int tag,MPI_Comm c,MPI_Request *r),(s,n,t,src,tag,c,r))
TIMED(15,Test,(MPI_Request *r,int *f,MPI_Status *s),(r,f,s))
TIMED(16,Testall,(int n,MPI_Request *r,int *f,MPI_Status *s),(n,r,f,s))
#undef TIMED
// Reserved benchmark epoch marker, called only when this profiler is loaded.
extern "C" int MPI_Pcontrol(const int level,...)
{if(level==101)write_counters(true);return MPI_SUCCESS;}
extern "C" int MPI_Finalize()
{
  write_counters(false);
  return PMPI_Finalize();
}
