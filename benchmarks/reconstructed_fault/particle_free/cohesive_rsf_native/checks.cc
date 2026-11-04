/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include <aspect/reconstructed_fault/cohesive_bulk_history.h>
#include <aspect/reconstructed_fault/cohesive_geometry.h>
#include <aspect/postprocess/interface.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <fstream>
#include <iomanip>

namespace aspect
{
  namespace GeometryModel
  {
    template <int dim> class CohesiveHistoryBlock : public Box<dim>
    {
    public:
      void
      create_coarse_mesh (parallel::distributed::Triangulation<dim> &grid) const override
      {
        Box<dim>::create_coarse_mesh (grid);
        if (std::getenv ("ASPECT_COHESIVE_HISTORY_AFFINE"))
          return;
        GridTools::transform (
            [] (const Point<dim> &x)
              {
                Point<dim> p = x;
                const double w = std::sin (numbers::PI * x[0]) * std::sin (numbers::PI * x[1]);
                p[0] += .035 * w;
                p[1] += .02 * w;
                return p;
              },
            grid);
      }
    };
    ASPECT_REGISTER_GEOMETRY_MODEL (
        CohesiveHistoryBlock, "cohesive history block",
        "Small fixed bilinear mesh for independent exact-jump Maxwell history checks.")
  }

  namespace Postprocess
  {
    template <int dim>
    class CohesiveHistoryChecks : public Interface<dim>, public SimulatorAccess<dim>
    {
      class Stress : public Function<dim>
      {
      public:
        Stress () : Function<dim> (3) {}
        double
        value (const Point<dim> &x, const unsigned int c = 0) const override
        {
          if (c == 0)
            return 170. + 90 * x[0] * x[0] * x[1] * x[1] + 35 * x[0] * x[1];
          if (c == 1)
            return -80. + 47 * x[0] * x[0] - 63 * x[1] * x[1] + 18 * x[0] * x[1] * x[1];
          return 260. + 33 * x[0] * x[0] * x[1] - 87 * x[0] * x[1] * x[1];
        }
      };
      class Increment : public Function<dim>
      {
      public:
        Increment (const unsigned int n, const double sign) : Function<dim> (n), sign (sign) {}
        double
        value (const Point<dim> &x, const unsigned int c = 0) const override
        {
          if (c == 0)
            return sign * 1e-5 * (x[0] * x[0] * x[1] + .2 * x[1] * x[1]);
          if (c == 1)
            return 1e-5 * (x[0] * x[1] * x[1] - .3 * x[0] * x[0]);
          return 0.;
        }

      private:
        double sign;
      };

    public:
      std::pair<std::string, std::string>
      execute (TableHandler &) override
      {
        if (this->get_timestep_number () != 0)
          return { "Cohesive native history", "no mutation" };
        if constexpr (dim == 2)
          {
            using History = CohesiveInterface::BulkHistory<dim>;
            const auto communicator = this->get_mpi_communicator ();
            const auto ranks = Utilities::MPI::n_mpi_processes (communicator);
            const auto &intro = this->introspection ();
            LinearAlgebra::BlockVector owned (intro.index_sets.system_partitioning, communicator),
                du;
            du.reinit (this->get_solution ());
            VectorTools::interpolate (this->get_mapping (), this->get_dof_handler (),
                                      Increment (this->get_fe ().n_components (), 1.), owned);
            const bool unmasked_pressure_finite
                = std::isfinite (owned.block (intro.block_indices.pressure).l2_norm ());
            LinearAlgebra::BlockVector masked (intro.index_sets.system_partitioning, communicator);
            masked = 0.;
            VectorTools::interpolate (this->get_mapping (), this->get_dof_handler (),
                                      Increment (this->get_fe ().n_components (), 1.), masked,
                                      intro.component_masks.velocities);
            const bool masked_pressure_zero
                = masked.block (intro.block_indices.pressure).l2_norm () == 0.;
            auto delta = masked.block (intro.block_indices.velocities);
            delta -= owned.block (intro.block_indices.velocities);
            AssertThrow (delta.l2_norm () == 0. && masked_pressure_zero,
                         ExcMessage ("Velocity-only interpolation changed its native field or "
                                     "pressure initialization."));
            du = owned;
            auto h = std::make_shared<History> (this->get_simulator (), 1e6, 1e8);
            h->initialize (Stress ());
            const auto before = h->pack ();
            const double E0 = h->storage (QGauss<dim> (8));
            auto rejected = h->advance (du, .2);
            const auto check = h->check_advance (*rejected, du, .2, QGauss<dim> (8));
            AssertThrow (h->pack () == before,
                         ExcMessage ("Rejected Maxwell history changed its accepted input."));
            auto retry = h->advance (du, .2);
            AssertThrow (rejected->pack () == retry->pack (),
                         ExcMessage ("Maxwell retry differs from the rejected trial."));
            const auto checkpoint = retry->pack ();
            History restored (this->get_simulator (), 1e6, 1e8);
            restored.restore (checkpoint);
            AssertThrow (restored.pack () == checkpoint,
                         ExcMessage ("Physical CellId Maxwell restart differs."));
            const double E1 = retry->storage (QGauss<dim> (8)),
                         Er = restored.storage (QGauss<dim> (8));
            AssertThrow (E1 == Er, ExcMessage ("Maxwell restored physical energy differs."));
            VectorTools::interpolate (this->get_mapping (), this->get_dof_handler (),
                                      Increment (this->get_fe ().n_components (), -1.), owned);
            du = owned;
            auto next = retry->advance (du, .125), continued = restored.advance (du, .125);
            const auto second = retry->check_advance (*next, du, .125, QGauss<dim> (8));
            AssertThrow (next->pack () == continued->pack (),
                         ExcMessage ("Non-proportional Maxwell restart continuation differs."));
            double norm = 0., remainder = 0.;
            const FE_DGQ<dim> dg (2);
            const QGauss<dim> samples (8);
            FEValues<dim> fe (this->get_mapping (), this->get_fe (), QGauss<dim> (8),
                              update_gradients);
            std::vector<SymmetricTensor<2, dim>> strain (fe.n_quadrature_points);
            for (const auto &cell : this->get_dof_handler ().active_cell_iterators ())
              if (cell->is_locally_owned ())
                {
                  // The complete native tensor is retained at points other than
                  // the source rule; no claim that its DG projection is exact.
                  const auto stress = next->evaluate (cell, QGauss<dim> (8));
                  for (const auto &t : stress)
                    norm = std::max (norm, t.norm ());
                  const auto &coefficients = next->dg_coefficients ().at (cell->id ());
                  for (unsigned int q = 0; q < samples.size (); ++q)
                    {
                      SymmetricTensor<2, 3> projected;
                      for (unsigned int i = 0; i < 9; ++i)
                        {
                          projected[0][0]
                              += coefficients[i] * dg.shape_value (i, samples.point (q));
                          projected[1][1]
                              += coefficients[9 + i] * dg.shape_value (i, samples.point (q));
                          projected[0][1]
                              += coefficients[18 + i] * dg.shape_value (i, samples.point (q));
                        }
                      projected[2][2] = -projected[0][0] - projected[1][1];
                      remainder = std::max (remainder, (stress[q] - projected).norm ());
                    }
                }
            norm = Utilities::MPI::max (norm, communicator);
            remainder = Utilities::MPI::max (remainder, communicator);
            const auto records = next->pack ();
            if(std::getenv("ASPECT_COHESIVE_OBJECTIVE_CHECK"))
              {
                using Full=SymmetricTensor<2,3>;
                CohesiveInterface::Geometry::Piece piece;
                piece.x=.15;piece.y=.43;piece.end=.85;piece.a=piece.b=.2;piece.cells=8;
                const CohesiveInterface::Geometry geometry({piece},.125);
                const bool affine=std::getenv("ASPECT_COHESIVE_HISTORY_AFFINE");
                std::vector<double> memory(geometry.modes()),rate(memory.size());
                for(unsigned int j=0;j<memory.size();++j)
                  {memory[j]=1e-4*std::sin(numbers::PI*(j+1)/(memory.size()+1));rate[j]=.2*memory[j];}
                const auto full=[](const dealii::Tensor<2,2> &L)
                  {
                    Full a;for(unsigned int i=0;i<2;++i)for(unsigned int j=i;j<2;++j)a[i][j]=.5*(L[i][j]+L[j][i]);
                    const double tr=trace(a)/3.;for(unsigned int i=0;i<3;++i)a[i][i]-=tr;return a;
                  };
                const auto physical=[&](const typename History::Cell &cell,const Quadrature<dim> &rule)
                  {
                    auto value=next->evaluate(cell,rule);
                    for(unsigned int q=0;q<rule.size();++q)
                      {
                        const auto x=this->get_mapping().transform_unit_to_real_cell(cell,rule.point(q));
                        if(affine)for(const auto &m:geometry.lifting(x))value[q]+=2e6*memory[m.index]*full(m.gradient);
                      }
                    return value;
                  };
                const auto gradient=[&](const typename History::Cell &cell,const Quadrature<dim> &rule)
                  {
                    FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_gradients|update_quadrature_points);
                    fe.reinit(cell);std::vector<dealii::Tensor<2,dim>> L(rule.size());
                    fe[intro.extractors.velocities].get_function_gradients(du,L);
                    for(unsigned int q=0;q<rule.size();++q)
                      {
                        L[q][1][0]+=.37;L[q][0][1]-=.37;
                        if(affine)for(const auto &m:geometry.lifting(fe.quadrature_point(q)))L[q]+=rate[m.index]*m.gradient;
                      }
                    return L;
                  };
                const auto projection_rule=[&](const typename History::Cell &cell)
                  {return affine ? geometry.quadrature(cell->vertex(0),cell->vertex(3),8) : Quadrature<dim>(QGauss<dim>(8));};
                const auto parent=next->pack();
                const auto rotated=next->rotated_anchor(physical,gradient,.2,8,projection_rule);
                std::shared_ptr<History> cut_rotated;
                if(affine)
                  {
                    cut_rotated=next->cut_rotated_anchor(physical,gradient,.2,8,geometry);
                    const auto checkpoint=cut_rotated->pack();
                    History restored_cut(this->get_simulator(),1e6,1e8);restored_cut.restore(checkpoint);
                    AssertThrow(restored_cut.pack()==checkpoint,ExcMessage("Cut tensor/partition/native checkpoint did not restore exactly."));
                    const auto a=cut_rotated->advance(du,.125),b=restored_cut.advance(du,.125);
                    cut_rotated->check_advance(*a,du,.125,QGauss<dim>(8));
                    AssertThrow(a->pack()==b->pack(),ExcMessage("Cut-history BE continuation changed on restart."));
                    const auto zero_gradient=[](const typename History::Cell &,const Quadrature<dim> &q)
                      {return std::vector<dealii::Tensor<2,dim>>(q.size());};
                    const auto identity=cut_rotated->cut_rotated_anchor(physical,zero_gradient,.2,8,geometry);
                    AssertThrow(identity->pack()==checkpoint,ExcMessage("Zero-spin cut-history identity changed accepted state."));
                  }
                double moment=0.,error=0.,reference=0.,energy=0.,scale=1.,increment_reference=0.,cut_error=0.;
                double native_cut_error=0.;
                unsigned long long cut_coefficients=0;
                for(const auto &cell:this->get_dof_handler().active_cell_iterators())
                  if(cell->is_locally_owned())
                    {
                      const auto rule=affine ? geometry.quadrature(cell->vertex(0),cell->vertex(3),12) : Quadrature<dim>(QGauss<dim>(12));
                      FEValues<dim> fe(this->get_mapping(),this->get_fe(),rule,update_JxW_values);
                      fe.reinit(cell);
                      const auto before=physical(cell,rule),A0=next->evaluate(cell,rule),A1=rotated->evaluate(cell,rule);
                      const auto L=gradient(cell,rule);
                      const auto cut_A=cut_rotated ? cut_rotated->evaluate(cell,rule) : std::vector<Full>();
                      std::array<Full,9> moments;
                      struct FitPoint {Point<dim> u;double weight;Full increment;};
                      std::map<unsigned int,std::vector<FitPoint>> regions;
                      for(unsigned int q=0;q<rule.size();++q)
                        {
                          const double angle=.1*(L[q][1][0]-L[q][0][1]),c=std::cos(2*angle),s=std::sin(2*angle);
                          const auto &a=before[q];Full expected=a;
                          const double mean=.5*(a[0][0]+a[1][1]),diff=.5*(a[0][0]-a[1][1]);
                          expected[0][0]=mean+diff*c-a[0][1]*s;expected[1][1]=mean-diff*c+a[0][1]*s;
                          expected[0][1]=diff*s+a[0][1]*c;
                          const auto actual=before[q]+A1[q]-A0[q],defect=actual-expected;
                          if(cut_rotated)
                            {
                              const auto defect=before[q]+cut_A[q]-A0[q]-expected;
                              native_cut_error+=fe.JxW(q)*(defect*defect);
                            }
                          if(affine)
                            {
                              const auto x=this->get_mapping().transform_unit_to_real_cell(cell,rule.point(q));
                              unsigned int region=0;
                              if(x[0]>=geometry.vertices().front() && x[0]<=geometry.vertices().back())
                                {
                                  const double d=x[1]-geometry.sample(x[0]).y;
                                  if(std::abs(d)<geometry.width())
                                    {
                                      const auto j=std::upper_bound(geometry.vertices().begin(),geometry.vertices().end(),x[0])-geometry.vertices().begin()-1;
                                      region=1+2*j+(d>=0.);
                                    }
                                }
                              regions[region].push_back({rule.point(q),fe.JxW(q),expected-before[q]});
                            }
                          error+=fe.JxW(q)*(defect*defect);reference+=fe.JxW(q)*(expected*expected);
                          increment_reference+=fe.JxW(q)*((expected-before[q])*(expected-before[q]));
                          energy+=fe.JxW(q)*((actual+a)*(actual-a))/(4e6);
                          scale=std::max(scale,a.norm());
                          for(unsigned int ix=0;ix<3;++ix)for(unsigned int iy=0;iy<3;++iy)
                            moments[3*ix+iy]+=fe.JxW(q)/cell->measure()*std::pow(rule.point(q)[0],ix)*std::pow(rule.point(q)[1],iy)*defect;
                        }
                      for(const auto &m:moments)moment=std::max(moment,m.norm());
                      // Best-approximation capacity only: genuine independent
                      // Q2 modes on each material lifting piece/side, not more
                      // quadrature samples or a reset of scalar slip memory.
                      for(const auto &region:regions)
                        {
                          const auto &points=region.second;
                          Point<dim> lo=points.front().u,hi=lo;
                          for(const auto &p:points)for(unsigned int d=0;d<2;++d)
                            {lo[d]=std::min(lo[d],p.u[d]);hi[d]=std::max(hi[d],p.u[d]);}
                          LAPACKFullMatrix<double> basis(points.size(),9);
                          FullMatrix<double> original(points.size(),9);
                          for(unsigned int q=0;q<points.size();++q)
                            {
                              const double u=2*(points[q].u[0]-lo[0])/(hi[0]-lo[0])-1.,v=2*(points[q].u[1]-lo[1])/(hi[1]-lo[1])-1.;
                              for(unsigned int ix=0;ix<3;++ix)for(unsigned int iy=0;iy<3;++iy)
                                original(q,3*ix+iy)=basis(q,3*ix+iy)=std::sqrt(points[q].weight)*std::pow(u,ix)*std::pow(v,iy);
                            }
                          basis.compute_svd();
                          AssertThrow(basis.singular_value(8)>1e-10*basis.singular_value(0),
                                      ExcMessage("Cut-space capacity fit is rank deficient; do not hide it with regularization."));
                          basis.compute_inverse_svd(1e-12);
                          std::vector<Full> defects(points.size());
                          for(unsigned int c=0;c<3;++c)
                            {
                              const unsigned int i=c==1 ? 1 : 0,j=c==0 ? 0 : 1;
                              Vector<double> rhs(points.size()),coefficients(9),fit(points.size());
                              for(unsigned int q=0;q<points.size();++q)rhs[q]=std::sqrt(points[q].weight)*points[q].increment[i][j];
                              basis.vmult(coefficients,rhs);original.vmult(fit,coefficients);
                              for(unsigned int q=0;q<points.size();++q)defects[q][i][j]=fit[q]-rhs[q];
                            }
                          for(auto &a:defects){a[2][2]=-a[0][0]-a[1][1];cut_error+=a*a;}
                          cut_coefficients+=27;
                        }
                    }
                moment=Utilities::MPI::max(moment,communicator);scale=Utilities::MPI::max(scale,communicator);
                error=Utilities::MPI::sum(error,communicator);reference=Utilities::MPI::sum(reference,communicator);
                increment_reference=Utilities::MPI::sum(increment_reference,communicator);energy=Utilities::MPI::sum(energy,communicator);
                cut_error=Utilities::MPI::sum(cut_error,communicator);cut_coefficients=Utilities::MPI::sum(cut_coefficients,communicator);
                native_cut_error=Utilities::MPI::sum(native_cut_error,communicator);
                AssertThrow(next->pack()==parent,ExcMessage("Cohesive objective anchor modified accepted native/tensor history."));
                AssertThrow(moment<2e-11*scale,ExcMessage("Cohesive independent full-tensor rotation moments failed."));
                AssertThrow(!affine || std::sqrt(native_cut_error/increment_reference)<1e-3,
                            ExcMessage("Native cut-tensor objective update fails the unchanged representation target."));
                const auto reacted=rotated->advance(du,.125);
                rotated->check_advance(*reacted,du,.125,QGauss<dim>(8));
                if(Utilities::MPI::this_mpi_process(communicator)==0)
                  {
                    std::ofstream out(this->get_output_directory()+"cohesive-objective-check.csv");
                    out<<std::setprecision(17)<<"cells,moment_error_Pa,scale_Pa,relative_L2,rotation_increment_relative_L2,projection_J_m,native_remainder_Pa,cut_Q2_increment_relative_L2,cut_Q2_tensor_coefficients,native_cut_increment_relative_L2\n"
                       <<records.size()<<','<<moment<<','<<scale<<','<<std::sqrt(error/reference)<<','<<std::sqrt(error/increment_reference)<<','<<energy<<','<<remainder<<','<<std::sqrt(cut_error/increment_reference)<<','<<cut_coefficients<<','<<std::sqrt(native_cut_error/increment_reference)<<'\n';
                  }
              }
            if (Utilities::MPI::this_mpi_process (communicator) == 0)
              {
                std::ofstream f (this->get_output_directory () + "maxwell-history-checks.csv");
                f << std::setprecision (17)
                  << "ranks,cells,source_Pa,source_scaled,second_Pa,second_scaled,initial_J_m,"
                     "trial_J_m,restart_J_m,max_full_tensor_Pa,max_retained_non_DG_remainder_Pa,"
                     "retry_exact,restart_exact,unmasked_pressure_finite,masked_pressure_zero\n"
                  << ranks << ',' << before.size () << ',' << check.first << ',' << check.second
                  << ',' << second.first << ',' << second.second << ',' << E0 << ',' << E1 << ','
                  << Er << ',' << norm << ',' << remainder << ",1,1," << unmasked_pressure_finite
                  << ',' << masked_pressure_zero << '\n';
                std::ofstream rows (this->get_output_directory ()
                                    + "maxwell-history-cell-records.csv");
                rows << std::setprecision (17);
                for (const auto &entry : records)
                  {
                    rows << entry.first;
                    for (const double value : entry.second)
                      rows << ',' << value;
                    rows << '\n';
                  }
              }
            return { "Cohesive native history", "strict recurrence/retry/restart pass" };
          }
        return { "Cohesive native history", "2D only" };
      }
    };
    ASPECT_REGISTER_POSTPROCESSOR (
        CohesiveHistoryChecks, "cohesive native history checks",
        "Independent general tensor/native remainder checks on native owned cells and ghosts.")
  }
}
