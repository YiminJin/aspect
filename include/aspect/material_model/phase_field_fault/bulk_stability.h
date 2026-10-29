/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#ifndef ASPECT_BULK_COHESIVE_STABILITY_H
#define ASPECT_BULK_COHESIVE_STABILITY_H

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <limits>

namespace aspect { namespace MaterialModel { namespace BulkCohesive
{
  /** Small critical-space check for the homogeneous matched-Maxwell bulk law.
   * Input is the already condensed native tangent: no artificial constrained
   * rows and no second C^T H C. Pressure belongs to the eliminated mixed block,
   * whose saddle eigenvalues are NOT phase stability eigenvalues. All factors
   * refer to the current incremental potential, not the physical energy ledger.
   * Dense storage is bounded deliberately; an unsupported size fails closed.
   */
  struct RelaxedCurvature
  {
    double minimum=0.,uncertainty=0.,eigen_residual=0.,linear_residual=0.;
    double asymmetry=0.,cross_symmetry_relative=0.;
    dealii::Vector<double> mode;
  };

  inline RelaxedCurvature relaxed_curvature(
    const dealii::SparseMatrix<double> &jacobian,
    const std::vector<unsigned int> &mechanics,
    const std::vector<unsigned int> &phase,
    const dealii::Vector<double> &metric,
    const double mechanical_factor,const double phase_factor,
    const double mechanical_tolerance)
  {
    using namespace dealii;
    AssertThrow(!phase.empty() && phase.size()<=1024,
                ExcMessage("Bulk stability critical space exceeds the bounded 1024-master implementation; no publication."));
    const unsigned int n=jacobian.m(),m=phase.size();
    std::vector<unsigned int> index(n,numbers::invalid_unsigned_int);
    for(unsigned int k=0;k<mechanics.size();++k)index[mechanics[k]]=k;
    DynamicSparsityPattern ds(mechanics.size(),mechanics.size());
    for(unsigned int k=0;k<mechanics.size();++k)
      for(auto e=jacobian.begin(mechanics[k]);e!=jacobian.end(mechanics[k]);++e)
        if(index[e->column()]!=numbers::invalid_unsigned_int)ds.add(k,index[e->column()]);
    SparsityPattern sp;sp.copy_from(ds);SparseMatrix<double> A(sp);
    for(unsigned int k=0;k<mechanics.size();++k)
      for(auto e=jacobian.begin(mechanics[k]);e!=jacobian.end(mechanics[k]);++e)
        if(index[e->column()]!=numbers::invalid_unsigned_int)
          A.set(k,index[e->column()],mechanical_factor*e->value());
    SparseDirectUMFPACK inverse;inverse.initialize(A);
    std::vector<std::vector<std::pair<unsigned int,double>>> B(m),mechanical_columns(m);
    std::vector<unsigned int> phase_index(n,numbers::invalid_unsigned_int);
    for(unsigned int j=0;j<m;++j)phase_index[phase[j]]=j;
    double cross_scale=1.,cross_difference=0.;
    for(unsigned int k=0;k<mechanics.size();++k)
      for(auto e=jacobian.begin(mechanics[k]);e!=jacobian.end(mechanics[k]);++e)
        if(phase_index[e->column()]!=numbers::invalid_unsigned_int && e->value()!=0.)
          {
            const double value=mechanical_factor*e->value();
            mechanical_columns[phase_index[e->column()]].emplace_back(k,value);
            cross_scale=std::max(cross_scale,std::abs(value));
            cross_difference=std::max(cross_difference,std::abs(value-phase_factor*jacobian.el(e->column(),mechanics[k])));
          }
    for(unsigned int j=0;j<m;++j)
      {
        AssertThrow(metric[j]>0.,ExcInternalError());
        // Read the physical phase rows, i.e. B^T. Verify their transpose
        // against the differently scaled native mechanics rows explicitly.
        for(auto e=jacobian.begin(phase[j]);e!=jacobian.end(phase[j]);++e)
          if(index[e->column()]!=numbers::invalid_unsigned_int && e->value()!=0.)
            {
              const double b=phase_factor*e->value();
              cross_scale=std::max(cross_scale,std::abs(b));
              cross_difference=std::max(cross_difference,std::abs(b-mechanical_factor*jacobian.el(e->column(),phase[j])));
              B[j].emplace_back(index[e->column()],b);
            }
      }
    FullMatrix<double> reduced(m,m);
    RelaxedCurvature result;
    result.cross_symmetry_relative=cross_difference/cross_scale;
    // Use the existing independent native-tangent analysis norm check. An
    // entrywise relative comparison is inappropriate for cancelled entries.
    // Both actual cross actions are retained below, without transposing one
    // to conceal their arithmetic difference; the Schur asymmetry is measured.
    AssertThrow(result.cross_symmetry_relative<1e-11,
                ExcMessage("Bulk potential cross tangent fails the existing symmetry check; no stability qualification."));
    Vector<double> rhs(mechanics.size()),response(mechanics.size()),residual(mechanics.size());
    for(unsigned int j=0;j<m;++j)
      {
        rhs=0.;for(const auto &b:mechanical_columns[j])rhs[b.first]=b.second;
        inverse.vmult(response,rhs);A.vmult(residual,response);residual-=rhs;
        result.linear_residual=std::max(result.linear_residual,residual.l2_norm()/std::max(1.,rhs.l2_norm()));
        AssertThrow(result.linear_residual<=mechanical_tolerance,
                    ExcMessage("Bulk relaxed-curvature mechanics solve failed its residual check."));
        for(unsigned int i=0;i<m;++i)
          {
            double value=phase_factor*jacobian.el(phase[i],phase[j]);
            for(const auto &b:B[i])value-=b.second*response[b.first];
            reduced(i,j)=value/std::sqrt(metric[i]*metric[j]);
          }
      }
    double norm=0.;
    for(unsigned int i=0;i<m;++i)
      {
        double row=0.,asymmetry=0.;
        for(unsigned int j=0;j<m;++j)
          {row+=std::abs(reduced(i,j));asymmetry+=.5*std::abs(reduced(i,j)-reduced(j,i));}
        norm=std::max(norm,row);result.asymmetry=std::max(result.asymmetry,asymmetry);
      }
    LAPACKFullMatrix<double> symmetric(m,m);
    for(unsigned int i=0;i<m;++i)for(unsigned int j=0;j<m;++j)
      symmetric(i,j)=.5*(reduced(i,j)+reduced(j,i));
    Vector<double> values;FullMatrix<double> vectors;
    // Gershgorin bounds contain the complete spectrum. No positive Ritz-only
    // acceptance: compute the entire bounded critical-space spectrum.
    symmetric.compute_eigenvalues_symmetric(-2*std::max(1.,norm),2*std::max(1.,norm),
      2*std::numeric_limits<double>::min(),values,vectors);
    AssertDimension(values.size(),m);
    result.minimum=values[0];result.mode.reinit(m);
    Vector<double> error(m);
    for(unsigned int i=0;i<m;++i)
      {
        result.mode[i]=vectors(i,0)/std::sqrt(metric[i]);
        for(unsigned int j=0;j<m;++j)error[i]+=reduced(i,j)*vectors(j,0);
        error[i]-=result.minimum*vectors(i,0);
      }
    result.eigen_residual=error.l2_norm();
    result.uncertainty=result.eigen_residual+result.asymmetry
      +128*std::numeric_limits<double>::epsilon()*m*std::max(1.,norm);
    // This is a floating-point diagnostic margin, not an interval-arithmetic
    // theorem for an ill-conditioned mixed inverse. Its measured solve and
    // symmetry defects are exported; no physical acceptance tolerance changes.
    return result;
  }
}}}
#endif
