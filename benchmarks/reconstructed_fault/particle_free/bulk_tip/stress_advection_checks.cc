/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/postprocess/interface.h>
#include <aspect/material_model/history/tensor_advection.h>
#include "../support/serial_tensor_advection.h"
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace Postprocess
  {
    /** Independent retained serial assembly versus the one-rank production path.
     * Owns only manufactured numerical fields; never accesses material history. */
    template <int dim>
    class StressAdvectionComparison : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          if (this->get_timestep_number()!=0) return {};
          AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
                      ExcMessage("The independent serial transport oracle requires one rank."));
          using namespace MaterialModel::HistoryDiscretization;
          using Field=TensorField<dim>;
          using Operator=TensorAdvection<dim>;
          using Reference=SerialTensorAdvection<dim>;
          Field prototype;
          prototype.space=std::make_shared<typename Field::Space>(this->get_triangulation());
          typename Field::Cells cells;
          for (const auto &cell:this->get_dof_handler().active_cell_iterators())
            if (cell->is_locally_owned())cells.emplace(cell->id(),cell);
          std::ofstream out(this->get_output_directory()+"stress-advection-comparison.csv");
          out<<"flow,projection,step,constant_error\n"<<std::setprecision(17);
          for (unsigned int flow=0;flow<3;++flow)
            {
              class Velocity : public Function<dim>
              {
                unsigned int kind;
                public:
                  Velocity(unsigned int n,unsigned int k):Function<dim>(n),kind(k) {}
                  double value(const Point<dim> &p,unsigned int c) const override
                  {
                    if(c==0)return kind==0 ? .2 : kind==1 ? -.2+.1*p[1] : .2+.3*p[0];
                    if(c==1)return kind==2 ? -.1*p[1] : .05;
                    return 0.;
                  }
              } velocity_function(this->get_fe().n_components(),flow);
              LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,this->get_mpi_communicator());
              VectorTools::interpolate(this->get_mapping(),this->get_dof_handler(),velocity_function,owned,
                                       this->introspection().component_masks.velocities);
              LinearAlgebra::BlockVector velocity(this->introspection().index_sets.system_partitioning,
                                                   this->introspection().index_sets.system_relevant_partitioning,this->get_mpi_communicator());
              velocity=owned;
              const auto op=Operator::assemble(*this,prototype,cells,velocity,4,0.);
              const auto ref=Reference::assemble(*this,prototype,cells,velocity,4,0.);
              const typename Operator::Field field=[](const Point<dim> &p,double t)
              {
                SymmetricTensor<2,dim> v;
                v[0][0]=1.+.1*p[0]*p[0]+.03*t;
                v[1][1]=-.2+.1*p[1];v[0][1]=.3+.02*p[0]*p[1];return v;
              };
              const typename Operator::Field zero=[](const Point<dim> &,double){return SymmetricTensor<2,dim>();};
              const typename Operator::Field constant=[](const Point<dim> &,double)
              {SymmetricTensor<2,dim> v;v[0][0]=1.;v[1][1]=-.3;v[0][1]=.2;return v;};
              auto a=Operator::project(*this,*op,field,0.,4),b=Reference::project(*this,*ref,field,0.,4);
              const auto difference=[](const Field &x,const Field &y)
              {double error=0.;for(unsigned int c=0;c<x.coefficients.size();++c)for(const auto i:x.space->dofs.locally_owned_dofs())error=std::max(error,std::abs(x.coefficients[c][i]-y.coefficients[c][i]));return error;};
              const double projection=difference(*a,*b),dt=.05/std::max(1.,op->inverse_advection_time);
              a=Operator::advance(*this,*op,*a,field,zero,0.,dt);
              b=Reference::advance(*this,*ref,*b,field,zero,0.,dt);
              const double step=difference(*a,*b);
              const auto c=Operator::project(*this,*op,constant,0.,4);
              const auto next=Operator::advance(*this,*op,*c,constant,zero,0.,dt);
              const double preserve=difference(*c,*next);
              AssertThrow(projection<2e-12 && step<2e-12 && preserve<2e-12,
                          ExcMessage("Owned-row transport differs from the retained serial oracle or loses material constants."));
              out<<flow<<','<<projection<<','<<step<<','<<preserve<<'\n';
            }
          return {"Stress advection", "serial reference / owned rows passed"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(StressAdvectionComparison,"stress advection comparison",
                                 "Small independent serial/owned-row transport comparison; one rank only.")
  }
}
