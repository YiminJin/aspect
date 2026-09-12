// Small production-assembly/solver tests, not a K3 convergence fixture.
#include "phase_field_periodic_domains.cc"

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class VerifyPhasePrecision : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler &) override
        {
          auto &handler=this->get_phase_field_handler();
          auto &pm=handler.get_associated_particle_manager();
          const auto Hindex=pm.get_property_manager().get_data_info().get_position_by_field_name("crack_driving_force");
          const auto block=this->introspection().variable("phase_field").block_index;
          const auto comm=this->get_mpi_communicator();
          const auto before=K3::fingerprint(*this);
          const double E=internal::PhaseFieldTestAccess<dim>::single_material_coefficients(handler,0,0).second;
          const double gp=internal::PhaseFieldTestAccess<dim>::single_material_coefficients(handler,0,1).second-E;
          const double Hc=-E/gp;
          struct Restore
          {
            std::vector<std::pair<double*,double>> saved;
            ~Restore() { for (auto &e:saved) *e.first=e.second; }
          };
          {
            Restore restore;
            for (auto &p:pm.get_particle_handler())
              {
                restore.saved.emplace_back(&p.get_properties()[Hindex],p.get_properties()[Hindex]);
                p.get_properties()[Hindex]=Hc;
              }
            LinearAlgebra::BlockSparseMatrix matrix;
            matrix.reinit(this->get_system_matrix().n_block_rows(),this->get_system_matrix().n_block_cols());
            for (unsigned int r=0;r<matrix.n_block_rows();++r)
              for (unsigned int c=0;c<matrix.n_block_cols();++c)
                matrix.block(r,c).copy_from(this->get_system_matrix().block(r,c));
            matrix.collect_sizes();
            LinearAlgebra::BlockVector owned(this->introspection().index_sets.system_partitioning,comm);
            LinearAlgebra::BlockVector rhs(owned),saved_rhs(owned),state(this->get_solution());
            for (const double value : {0.,1e-18,1e-13,1e-6})
              {
                owned=this->get_solution();
                owned.block(block)=value;
                this->get_current_constraints().distribute(owned);
                state=owned;
                double allowance=0;
                AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler,matrix,rhs,state,false,&allowance),ExcInternalError());
                const double residual=rhs.block(block).l2_norm();
                saved_rhs=rhs;
                AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler,matrix,rhs,state,false),ExcInternalError());
                saved_rhs-=rhs;
                AssertThrow(saved_rhs.block(block).l2_norm()==0,ExcMessage("Scale assembly changed the phase residual."));
                AssertThrow(std::isfinite(allowance) && allowance>0,ExcMessage("Invalid phase precision scale."));
                if (value==0 || value==1e-18)
                  AssertThrow(residual<=allowance,ExcMessage("Zero/tiny phase input must meet the precision criterion."));
                else
                  AssertThrow(residual>1000*allowance,ExcMessage("A material residual was hidden by the precision allowance."));
                const double target=std::max(1e-8*residual,allowance);
                handler.evolve_phase_field(matrix,rhs,state);
                AssertThrow(internal::PhaseFieldTestAccess<dim>::assemble(handler,matrix,rhs,state,false),ExcInternalError());
                AssertThrow(rhs.block(block).l2_norm()<=target,ExcMessage("Mixed phase solver failed its fresh residual check."));
                if (value<=1e-18)
                  {
                    saved_rhs.block(block)=state.block(block);
                    owned.block(block)-=saved_rhs.block(block);
                    AssertThrow(owned.block(block).l2_norm()==0,ExcMessage("Tiny converged phase input was unnecessarily changed."));
                  }
                this->get_pcout() << std::setprecision(17) << "PHASE_PRECISION value=" << value
                  << " initial=" << residual << " allowance=" << allowance
                  << " final=" << rhs.block(block).l2_norm() << " target=" << target << std::endl;
              }
          }
          AssertThrow(K3::fingerprint(*this)==before,ExcMessage("Phase precision test changed live state."));
          return {"Phase mixed precision criterion:","verified"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(VerifyPhasePrecision,"verify phase precision",
                                 "Frozen zero/tiny/material phase residual and stopping tests.")
  }
}
