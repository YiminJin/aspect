/* C interface for independent 1-D strip verification of the actual point law. */
#include <aspect/material_model/phase_field_fault/bulk_cohesive.h>
#include <deal.II/base/exceptions.h>
#include <exception>

extern "C" int bulk_strip_points(const unsigned int count,const double *parameters,
                                 const double *old,const double *increment,double *output)
{
  namespace BC=aspect::MaterialModel::BulkCohesive;
  dealii::deal_II_exceptions::disable_abort_on_exception();
  try
    {
      const BC::Parameters p{parameters[0],parameters[1],parameters[2],parameters[3]};
      for(unsigned int i=0;i<count;++i)
        {
          BC::State s;s.T[0][1]=old[6*i];s.a=old[6*i+1];s.signed_shear=old[6*i+2];
          s.absolute_shear=old[6*i+3];s.phi=old[6*i+4];s.q=old[6*i+5];
          BC::Tensor strain;strain[0][1]=increment[3*i]/2.;
          const auto r=BC::update(p,s,strain,increment[3*i+1],increment[3*i+2]);
          const auto ledger=BC::account(p,s,r,strain,increment[3*i+2]);
          const double values[]={r.state.T[0][1],r.state.a,r.state.signed_shear,r.state.absolute_shear,
            r.state.phi,r.state.q,r.stress[0][1],r.w,r.driving,r.bulk_energy,r.cohesive_energy,
            r.bulk_dissipation,r.cohesive_dissipation,r.friction_dissipation,ledger.error,
            r.da_dt,r.da_dphi,r.dq_dt,r.dq_dphi};
          for(unsigned int j=0;j<19;++j)output[19*i+j]=values[j];
        }
      return 0;
    }
  catch(const std::exception &){return 1;}
}
