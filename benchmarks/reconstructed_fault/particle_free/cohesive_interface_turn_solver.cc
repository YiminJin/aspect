/* Copyright (C) 2026 by the authors of ASPECT. GPL v2 or later. */
// Stateless experimental C interface to the retained native signed solver.
// Python owns provisional/accepted archives; no production owner is replaced.
#include "cohesive_interface_evolution.h"
#include <cstring>

namespace
{
  void copy(dealii::FullMatrix<double> &A, unsigned m, unsigned n, const double *p)
  {
    A.reinit(m,n);
    for(unsigned i=0;i<m;++i)for(unsigned j=0;j<n;++j)A(i,j)=p[i*n+j];
  }
}

extern "C" void *cohesive_turn_create(unsigned n,unsigned nq,const double *K,const double *Q,
                                      const double *Nm,const double *Np,const double *weights,const double *mass)
{
  auto *op=new aspect::CohesiveEvolution::Operators;
  copy(op->K,n+1,n+1,K);copy(op->Q,nq,n,Q);
  copy(op->Nm,nq,n+1,Nm);copy(op->Np,nq,n+1,Np);
  op->weights.assign(weights,weights+nq);op->mass.assign(mass,mass+n);
  return op;
}

extern "C" void cohesive_turn_destroy(void *pointer)
{
  delete static_cast<aspect::CohesiveEvolution::Operators *>(pointer);
}

extern "C" void cohesive_turn_use_structural_q1_ties(void *pointer)
{
  static_cast<aspect::CohesiveEvolution::Operators *>(pointer)->structural_q1_ties=true;
}

extern "C" void *cohesive_turn_create_multi(unsigned n,unsigned nq,unsigned na,
                                            const double *K,const double *Q,const double *Nm,const double *Np,
                                            const double *weights,const double *mass)
{
  auto *op=new aspect::CohesiveEvolution::Operators;
  op->affine_modes=na;
  copy(op->K,n+na,n+na,K);copy(op->Q,nq,n,Q);
  copy(op->Nm,nq,n+na,Nm);copy(op->Np,nq,n+na,Np);
  op->weights.assign(weights,weights+nq);op->mass.assign(mass,mass+n);
  return op;
}

// history columns: jump, damage, theta, reference, relaxed_jump, absolute_slip.
// output columns append q, friction, radiation, velocity, resistance,
// derivative, pressure_derivative, storage to those six persistent fields.
static int solve_impl(void *pointer,const double *incoming,const double *history,
                                    double dt,double daffine,const double *guess,const int *ties,
                                    double *coefficients,double *increment,
                                    double *point,double *tangent,int *tied,double *meta,char *error,unsigned capacity,
                                    const double *multiple_increments=nullptr)
{
  try
    {
      const auto &op=*static_cast<aspect::CohesiveEvolution::Operators *>(pointer);
      const unsigned n=op.Q.n(),nq=op.Q.m(),na=op.affine_modes;
      aspect::CohesiveEvolution::Incoming old;
      old.affine=incoming[0];old.memory.assign(incoming+na,incoming+n+na);old.history.resize(nq);
      std::vector<double> affine_increments;
      if(na>1){old.affine_memory.assign(incoming,incoming+na);AssertThrow(multiple_increments,dealii::ExcMessage("Missing affine loading increments"));affine_increments.assign(multiple_increments,multiple_increments+na);}
      for(unsigned q=0;q<nq;++q)
        {
          auto &s=old.history[q];s.jump=history[6*q];s.damage=history[6*q+1];s.theta=history[6*q+2];
          s.reference=history[6*q+3];s.relaxed_jump=history[6*q+4];s.absolute_slip=history[6*q+5];
        }
      dealii::Vector<double> predictor(n);std::vector<bool> predictor_ties(nq);
      if(guess)for(unsigned j=0;j<n;++j)predictor[j]=guess[j];
      if(ties)for(unsigned q=0;q<nq;++q)predictor_ties[q]=ties[q];
      const auto result=aspect::CohesiveEvolution::solve(op,old,dt,daffine,
                         guess?&predictor:nullptr,ties?&predictor_ties:nullptr,na>1?&affine_increments:nullptr);
      for(unsigned j=0;j<n+na;++j)coefficients[j]=result.coefficients[j];
      for(unsigned j=0;j<n;++j)increment[j]=result.increment[j];
      for(unsigned i=0;i<n;++i)for(unsigned j=0;j<n;++j)tangent[i*n+j]=result.tangent(i,j);
      for(unsigned q=0;q<nq;++q)
        {
          const auto &r=result.point[q];const auto &s=r.state;
          const double values[14]={s.jump,s.damage,s.theta,s.reference,s.relaxed_jump,s.absolute_slip,
                                   r.q,r.friction,r.radiation,r.velocity,r.resistance,r.derivative,r.pressure_derivative,r.storage};
          std::copy(values,values+14,point+14*q);tied[q]=result.tied[q];
        }
      meta[0]=result.weak;meta[1]=result.curvature;meta[2]=result.contact;
      meta[3]=result.iterations;meta[4]=result.certificate_iterations;meta[5]=result.free;
      return 0;
    }
  catch(const std::exception &e)
    {
      if(capacity){std::strncpy(error,e.what(),capacity-1);error[capacity-1]=0;}
      return 1;
    }
}

extern "C" int cohesive_turn_solve(void *pointer,const double *incoming,const double *history,
                                  double dt,double daffine,const double *guess,const int *ties,
                                  double *coefficients,double *increment,double *point,double *tangent,
                                  int *tied,double *meta,char *error,unsigned capacity)
{return solve_impl(pointer,incoming,history,dt,daffine,guess,ties,coefficients,increment,point,tangent,tied,meta,error,capacity);}

extern "C" int cohesive_turn_solve_multi(void *pointer,const double *incoming,const double *history,
                                        double dt,const double *daffine,const double *guess,const int *ties,
                                        double *coefficients,double *increment,double *point,double *tangent,
                                        int *tied,double *meta,char *error,unsigned capacity)
{return solve_impl(pointer,incoming,history,dt,0.,guess,ties,coefficients,increment,point,tangent,tied,meta,error,capacity,daffine);}
