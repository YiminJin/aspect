/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
// Experimental column ownership: replicated fixed bulk matrix, no sum of
// contributions to an owned column. Broadcasts preserve its serial arithmetic.
#include <deal.II/base/mpi.h>
#include <deal.II/lac/full_matrix.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <sys/resource.h>

struct PathColumnParallel
{
  bool enabled=false;
  unsigned rank=0, ranks=1;

  void initialize(const bool selected)
  {
    ranks=dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    rank=dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    enabled=selected;
    AssertThrow(enabled || ranks==1,dealii::ExcMessage("Multiple ranks require the explicit column-mpi option"));
  }

  bool owns(const unsigned column) const {return !enabled || column%ranks==rank;}
  bool coordinator() const {return rank==0;}

  void from_root(dealii::FullMatrix<double> &matrix) const
  {
    if(!enabled || ranks==1)return;
    std::vector<double> buffer(matrix.m()*matrix.n());unsigned n=0;
    if(coordinator())for(unsigned i=0;i<matrix.m();++i)for(unsigned j=0;j<matrix.n();++j)buffer[n++]=matrix(i,j);
    broadcast(buffer.data(),buffer.size(),0);n=0;
    if(!coordinator())for(unsigned i=0;i<matrix.m();++i)for(unsigned j=0;j<matrix.n();++j)matrix(i,j)=buffer[n++];
  }

  template <class Sparse> void sparse_from_root(Sparse &matrix) const
  {
    if(!enabled || ranks==1)return;
    std::vector<double> buffer(matrix.n_nonzero_elements());unsigned n=0;
    if(coordinator())for(auto i=matrix.begin();i!=matrix.end();++i)buffer[n++]=i->value();
    broadcast(buffer.data(),buffer.size(),0);n=0;
    if(!coordinator())for(auto i=matrix.begin();i!=matrix.end();++i)i->value()=buffer[n++];
  }

  void sum(double *values,const unsigned n) const
  {if(enabled && ranks>1)MPI_Allreduce(MPI_IN_PLACE,values,n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);}

  void sum(dealii::FullMatrix<double> &matrix) const
  {
    if(!enabled || ranks==1)return;
    std::vector<double>buffer(matrix.m()*matrix.n());unsigned n=0;
    for(unsigned i=0;i<matrix.m();++i)for(unsigned j=0;j<matrix.n();++j)buffer[n++]=matrix(i,j);
    sum(buffer.data(),buffer.size());n=0;
    for(unsigned i=0;i<matrix.m();++i)for(unsigned j=0;j<matrix.n();++j)matrix(i,j)=buffer[n++];
  }

  void cell_fields(std::vector<double> &values,const unsigned cells,const unsigned stride) const
  {
    if(!enabled || ranks==1)return;
    for(unsigned owner=0;owner<ranks;++owner)
      {
        const unsigned count=owner<cells?(cells+ranks-1-owner)/ranks:0;
        std::vector<double>buffer(count*stride);unsigned n=0;
        if(rank==owner)for(unsigned cell=owner;cell<cells;cell+=ranks)for(unsigned j=0;j<stride;++j)buffer[n++]=values[cell*stride+j];
        broadcast(buffer.data(),buffer.size(),owner);n=0;
        if(coordinator())for(unsigned cell=owner;cell<cells;cell+=ranks)for(unsigned j=0;j<stride;++j)values[cell*stride+j]=buffer[n++];
      }
  }

  // Gather sparse per-cell quadrature records without summing them. Folding
  // in CellId/index order makes the selector independent of MPI ownership.
  std::map<unsigned,std::vector<double>> cells(const std::map<unsigned,std::vector<double>> &local,const unsigned width) const
  {
    if(!enabled || ranks==1)return local;
    std::vector<double>packed;packed.reserve(local.size()*(width+1));
    for(const auto &entry:local){AssertThrow(entry.second.size()==width,dealii::ExcMessage("Invalid cell record"));packed.push_back(entry.first);packed.insert(packed.end(),entry.second.begin(),entry.second.end());}
    AssertThrow(packed.size()<std::numeric_limits<int>::max(),dealii::ExcMessage("Cell gather exceeds prototype capacity"));
    const int count=packed.size();std::vector<int>counts(ranks),offsets(ranks);
    MPI_Gather(&count,1,MPI_INT,counts.data(),1,MPI_INT,0,MPI_COMM_WORLD);
    unsigned total=0;if(coordinator())for(unsigned r=0;r<ranks;++r){offsets[r]=total;total+=counts[r];}
    std::vector<double>all(total);MPI_Gatherv(packed.data(),count,MPI_DOUBLE,all.data(),counts.data(),offsets.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    std::map<unsigned,std::vector<double>>result;
    if(coordinator())for(unsigned r=0;r<ranks;++r)for(unsigned p=offsets[r];p<unsigned(offsets[r]+counts[r]);p+=width+1)
      {const unsigned cell=all[p];AssertThrow(cell%ranks==r && result.count(cell)==0,dealii::ExcMessage("Duplicate/nonowned cell record"));result[cell]=std::vector<double>(all.begin()+p+1,all.begin()+p+1+width);}
    return result;
  }

  void broadcast(double *values,const std::size_t n,const unsigned owner) const
  {
    if(!enabled || ranks==1)return;
    AssertThrow(n<=std::numeric_limits<int>::max(),dealii::ExcMessage("MPI payload exceeds bounded prototype capacity"));
    const int status=MPI_Bcast(values,n,MPI_DOUBLE,owner,MPI_COMM_WORLD);
    AssertThrow(status==MPI_SUCCESS,dealii::ExcMessage("Column broadcast failed"));
  }

  void columns(dealii::FullMatrix<double> &matrix) const
  {
    if(!enabled || ranks==1)return;
    std::vector<double> values(matrix.m());
    for(unsigned k=0;k<matrix.n();++k)
      {
        if(owns(k))for(unsigned i=0;i<matrix.m();++i)values[i]=matrix(i,k);
        broadcast(values.data(),values.size(),k%ranks);
        for(unsigned i=0;i<matrix.m();++i)matrix(i,k)=values[i];
      }
  }

  void scalars(std::vector<double> &values) const
  {if(enabled)for(unsigned k=0;k<values.size();++k)broadcast(&values[k],1,k%ranks);}

  void fields(std::vector<double> &values,const unsigned cells,const unsigned nc) const
  {
    if(!enabled || ranks==1)return;
    std::vector<double> buffer(5*cells);
    for(unsigned k=0;k<nc;++k)
      {
        if(owns(k))for(unsigned cell=0;cell<cells;++cell)for(unsigned a=0;a<5;++a)
          buffer[5*cell+a]=values[5*(cell*nc+k)+a];
        broadcast(buffer.data(),buffer.size(),k%ranks);
        if(coordinator())for(unsigned cell=0;cell<cells;++cell)for(unsigned a=0;a<5;++a)
          values[5*(cell*nc+k)+a]=buffer[5*cell+a];
      }
  }

  double maximum(const double value) const
  {return enabled?dealii::Utilities::MPI::max(value,MPI_COMM_WORLD):value;}

  void injected_failure() const
  {
    const char *requested=std::getenv("ASPECT_COHESIVE_PATH_FAIL_RANK");
    AssertThrow(!enabled || !requested || rank!=std::stoul(requested),
                dealii::ExcMessage("Test-only nonroot native column failure before completed export"));
  }
};
