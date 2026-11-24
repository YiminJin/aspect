/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/simulator/solver/midpoint_system.h>
#include <aspect/phase_field/bernstein_bounds.h>
#include <numeric>
#include <aspect/simulator.h>
#include <aspect/gravity_model/interface.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_direct.h>

namespace aspect::GrowingFault
{
  namespace
  {
    Stress plane_strain(const SymmetricTensor<2,2> &d)
    {
      Stress embedded;
      embedded[0][0]=d[0][0];
      embedded[1][1]=d[1][1];
      embedded[0][1]=d[0][1];
      return deviator(embedded);
    }

  }

  MidpointSystem::MidpointSystem(const SimulatorAccess<2> &simulator,const MidpointMaterial &m,const LinearAlgebra::BlockVector &solution)
    : sim(simulator),material(m),comm(sim.get_mpi_communicator()),h(.5*sim.get_timestep()),G(m.maxwell().G),B(m.maxwell().relaxation(h)),
      Kf(sim.get_parameters().surface_energy_model.fracture_energy/(sim.get_parameters().surface_energy_model.geometry.normalization()
                                                                    *sim.get_parameters().surface_energy_model.length_scale)),field(m.phase_field()),phase_block(sim.introspection().block_indices.compositional_fields[field]),
      nv(solution.block(0).size()),phase_begin(nv+solution.block(1).size()),n(phase_begin+solution.block(phase_block).size()),fracture(sim,field)
  {
    AssertThrow(h>0.,ExcMessage("Midpoint mechanics needs a positive stage duration."));
    // Both admissibility bounds depend on this attempt's half-duration and
    // hold on every rank. Let the time-step manager retry from the same parent.
    if (h*G>m.maxwell().eta ||
        !m.friction().admits_duration(h,sim.get_parameters().surface_pressure))
      throw ExcNonlinearSolverNoConvergence();
    const auto &old=sim.get_old_solution();
    const auto &fe=sim.get_fe();
    const auto &intro=sim.introspection();
    const auto c=intro.component_indices.compositional_fields[field];
    const auto offset=solution.get_block_indices().block_start(phase_block);
    initial.assign(n,0.);
    for (const unsigned int block:
    {
      0u,1u,phase_block
    })
    for (const auto i:intro.index_sets.system_partitioning[block])
      initial[(block==0?0:block==1?nv:phase_begin)+i]=old.block(block)[i]*(block==0?h:block==1?1./G:1.);
    Utilities::MPI::sum(initial,comm,initial);
    lower=initial;
    upper=lower;
    for (Index i=phase_begin; i<n; ++i)
      {
        upper[i]=.5*(1.+lower[i]);
        AssertThrow(lower[i]<=upper[i],ExcMessage("Infeasible incoming Bernstein obstacle in growing fault."));
      }
    mass.assign(n,0.);
    for (const auto &[row,value]:fracture.mass) mass[phase_begin+row]=value;
    Utilities::MPI::sum(mass,comm,mass);

    const auto &constraints=sim.get_current_constraints();
    const auto boundary=DoFTools::extract_boundary_dofs(sim.get_dof_handler(),intro.component_masks.velocities);
    unsigned int invalid=0;
    for (const auto i:boundary)
      if (intro.index_sets.system_partitioning[0].is_element(i)) invalid|=!constraints.is_constrained(i);
    std::map<Index,double> owned_constraints;
    for (const auto &line:constraints.get_lines())
      {
        if (line.index<nv && intro.index_sets.system_partitioning[0].is_element(line.index))
          {
            invalid|=!line.entries.empty();
            owned_constraints[line.index]=h*line.inhomogeneity;
          }
        if (line.index>=nv && line.index<phase_begin && intro.index_sets.system_partitioning[1].is_element(line.index-nv))
          {
            invalid|=!line.entries.empty();
            owned_constraints[line.index]=line.inhomogeneity/G;
          }
        if (line.index>=offset && line.index<offset+n-phase_begin) invalid=1;
      }
    AssertThrow(!Utilities::MPI::max(invalid,comm),ExcMessage("Midpoint mechanics requires fully prescribed wall velocity with simple prescribed values, at most one pressure datum and unconstrained DG phase."));
    for (auto part:Utilities::MPI::all_gather(comm,owned_constraints)) fixed.merge(part);
    auto datum=fixed.lower_bound(nv);
    if (datum==fixed.end()) datum=fixed.emplace(nv,0.).first;
    AssertThrow(std::next(datum)==fixed.end(),ExcMessage("Midpoint fault supports one prescribed physical pressure datum."));
    const double gauge=initial[datum->first]-datum->second;
    for (Index i=nv; i<phase_begin; ++i) initial[i]-=gauge;
    for (const auto &[i,v]:fixed) initial[i]=v;

    unsigned int nonzero_gravity=0;
    double flux=0.,flux_scale=0.;
    const auto &sections=m.sections();
    for (const auto &s:m.history().sections)
      {
        angles.push_back(s.angle);
        formed.push_back(s.formed);
      }
    eligible.assign(angles.size(),0);
    std::vector<Index> ids(fe.n_dofs_per_cell());
    for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const auto &rule=sections.cells.at(cell->id());
          FEValues<2> values(sim.get_mapping(),fe,rule.quadrature,update_values|update_gradients|update_JxW_values|update_quadrature_points);
          values.reinit(cell);
          cell->get_dof_indices(ids);
          auto &data=cells[cell->id()];
          std::vector<unsigned int> local;
          for (unsigned int j=0; j<ids.size(); ++j)
            if (ids[j]<phase_begin || fe.system_to_component_index(j).first==c)
              {
                local.push_back(j);
                data.indices.push_back(ids[j]<phase_begin?ids[j]:phase_begin+ids[j]-offset);
              }
          for (unsigned int q=0; q<rule.quadrature.size(); ++q)
            {
              Sample p {rule.section[q],values.quadrature_point(q),values.JxW(q),0.,0.,{}};
              nonzero_gravity|=sim.get_gravity_model().gravity_vector(p.position)!=Tensor<1,2>();
              for (const auto j:local)
                {
                  p.shapes.push_back({plane_strain(values[intro.extractors.velocities].symmetric_gradient(j,q)),
                                      values[intro.extractors.velocities].divergence(j,q),values[intro.extractors.pressure].value(j,q),
                                      values.shape_value_component(j,q,c)
                                     });
                  if (ids[j]<nv)
                    {
                      const double term=p.area*p.shapes.back().div*(initial[ids[j]]/h);
                      flux+=term;
                      flux_scale+=std::abs(term);
                    }
                  if (ids[j]>=phase_begin) p.old_phase+=p.shapes.back().phase*old[ids[j]];
                }
              data.points.push_back(std::move(p));
            }
        }
    const unsigned int gravity_owners=Utilities::MPI::sum(nonzero_gravity,comm);
    AssertThrow(gravity_owners==0,ExcMessage("Growing fault mechanics requires zero gravity at every native integration point; nonzero on "+
                                             std::to_string(gravity_owners)+" of "+std::to_string(Utilities::MPI::n_mpi_processes(comm))+" ranks."));
    flux=Utilities::MPI::sum(flux,comm);
    flux_scale=Utilities::MPI::sum(flux_scale,comm);
    // Sum of all Q1 continuity tests is the constant test. Its integral
    // equals the prescribed net wall flux regardless of the interior values.
    AssertThrow(std::abs(flux)<=1.e-12*flux_scale,ExcMessage("Incompatible prescribed net velocity flux: "+Utilities::to_string(flux)+" m^2/s."));
  }

  MidpointSystem::Evaluation MidpointSystem::assemble_stage(const Values &x,const bool tangent,const bool publish) const
  {
    const auto &mat=material.maxwell();
    const auto &parent=material.history();
    const auto &geometry=material.sections();
    const unsigned int ns=angles.size();
    std::vector<Stress> planes(ns);
    Values lengths(ns);
    for (unsigned int k=0; k<ns; ++k)
      {
        planes[k]=geometry.shear(angles[k]);
        lengths[k]=geometry.length(k,angles[k]);
      }
    Evaluation e;
    e.residual.assign(n,0.);
    SectionTransfer transfer(ns);
    struct Trial
    {
      Stress strain;
      double div=0.,pressure=0.,phase=0.;
    };
    std::map<CellId,std::vector<Trial>> trials;
    for (const auto &[id,cell]:cells)
      {
        auto &states=trials[id];
        for (unsigned int q=0; q<cell.points.size(); ++q)
          {
            const auto &p=cell.points[q];
            const auto k=p.strip;
            Trial t;
            for (unsigned int j=0; j<cell.indices.size(); ++j)
              {
                const double v=x[cell.indices[j]];
                t.strain+=v*p.shapes[j].strain;
                t.div+=v*p.shapes[j].div;
                t.pressure+=v*p.shapes[j].pressure;
                t.phase+=v*p.shapes[j].phase;
              }
            const auto free=mat.response(parent.cells.at(id)[q],t.strain,t.phase,h,planes[k],0.);
            transfer.add(k,p.area/lengths[k],free.rate,free.fraction/(G*B*h));
            states.push_back(t);
          }
      }
    transfer.sum(comm);
    for (unsigned int k=0; k<ns; ++k)
      e.roots.push_back(material.friction().solve_strip(k,transfer.free_rate[k],transfer.mobility[k],
                                                        h,sim.get_parameters().surface_pressure));
    // In exact arithmetic the weighted sum of the returned point rates is V.
    // In cold creep even the rounding error of the scalar resistance can be
    // much larger than V/M. Recover that low part of the traction before
    // forming the native endpoint tensors and shear inventories.
    Values traction_remainder(ns);
    if (publish)
      {
        std::vector<long double> native_rate(ns), magnitude(ns);
        std::vector<unsigned int> count(ns);
        for (const auto &[id,cell]:cells)
          for (unsigned int q=0; q<cell.points.size(); ++q)
            {
              const auto &p=cell.points[q];
              const auto k=p.strip;
              const auto &t=trials.at(id)[q];
              const auto &old=parent.cells.at(id)[q];
              const auto &root=e.roots[k];
              const auto r=mat.response(old,t.strain,t.phase,h,planes[k],root.resistance);
              const long double weight=p.area/lengths[k];
              native_rate[k]+=weight*r.rate;
              magnitude[k]+=weight*(std::abs(r.rate)+2.*std::abs(root.resistance)*r.fraction/(G*B*h)
                                    +4.*std::abs(old.cohesive_strain*planes[k])/h);
              ++count[k];
            }
        Utilities::MPI::sum(native_rate,comm,native_rate);
        Utilities::MPI::sum(magnitude,comm,magnitude);
        Utilities::MPI::sum(count,comm,count);
        for (unsigned int k=0; k<ns; ++k)
          {
            const long double defect=e.roots[k].rate-native_rate[k];
            // n positive-weight additions, plus the 32-epsilon root certificate.
            const long double budget=(64.+4.*count[k])*std::numeric_limits<double>::epsilon()
                                     *(magnitude[k]+std::abs(e.roots[k].rate));
            AssertThrow(std::abs(defect)<=budget,ExcMessage("Native transfer disagrees with the signed section root."));
            if (transfer.mobility[k]>0.) traction_remainder[k]=defect/transfer.mobility[k];
          }
      }
    std::vector<unsigned int> admitted(ns,0);
    unsigned int detached=0;
    if (publish) e.candidate.sections=parent.sections;
    for (const auto &[id,cell]:cells)
      {
        const auto &states=trials.at(id);
        const auto &history=parent.cells.at(id);
        const unsigned int size=cell.indices.size();
        FullMatrix<double> block(tangent?size:0,tangent?size:0);
        for (unsigned int q=0; q<cell.points.size(); ++q)
          {
            const auto &p=cell.points[q];
            const auto k=p.strip;
            const auto &S=planes[k];
            const auto &t=states[q];
            const auto &old=history[q];
            const double f=e.roots[k].resistance;
            const auto r=mat.response(old,t.strain,t.phase,h,S,f,traction_remainder[k]);
            if (publish)
              {
                PointHistory end {2.*r.stress-old.stress,2.*r.cohesive_strain-old.cohesive_strain,
                                  old.signed_shear+2.*h*r.rate,old.absolute_shear+2.*h *std::abs(r.rate)
                                 };
                e.candidate.cells[id].push_back(end);
                admitted[k]|=(formed[k] || eligible[k]) && t.phase>0.;
                detached|=!formed[k] && !eligible[k] && (end.absolute_shear!=0. || end.cohesive_strain.norm()!=0.);
                const double phase1=2.*t.phase-p.old_phase;
                const double cohesive=mat.cohesive_energy(end.cohesive_strain,phase1);
                const double change=cohesive-mat.cohesive_energy(old.cohesive_strain,p.old_phase);
                auto &b=e.ledger;
                b.bulk+=p.area*(end.stress*end.stress)/(4.*G);
                b.cohesive+=p.area*cohesive;
                b.storage_change+=p.area*((end.stress*end.stress-old.stress*old.stress)/(4.*G)+change);
                b.bulk_loss+=p.area*2.*h*(r.stress*r.stress)/(2.*mat.eta);
                const auto reaction=2.*(1.-r.fraction)*r.force*S;
                b.cohesive_loss+=p.area*2.*h*G/mat.eta*(reaction*r.cohesive_strain);
                b.adjoint_defect+=p.area*2.*h*f*r.rate;
                b.compliance_remainder+=p.area*(reaction*(end.cohesive_strain-old.cohesive_strain)-change-r.driving*(phase1-p.old_phase));
              }
            const auto df=mat.derivative(r, {},0.,h,S,1.);
            for (unsigned int j=0; j<size; ++j)
              {
                const auto &v=p.shapes[j];
                const auto row=cell.indices[j];
                e.residual[row]+=p.area*(r.stress*v.strain/G-t.pressure*v.div-v.pressure*t.div-v.phase*r.driving/Kf);
                if (!tangent) continue;
                const auto d=mat.derivative(r,v.strain,v.phase,h,S,0.);
                // The last row represents df = condensed slope * integral(dw|f).
                // Its Schur complement is the cross-cell, cross-rank derivative.
                e.jacobian[ {n+k,row}]-=e.roots[k].slope*p.area/lengths[k]*d.rate;
                e.jacobian[ {row,n+k}]+=p.area*(df.stress*v.strain/G-v.phase*df.driving/Kf);
                for (unsigned int i=0; i<size; ++i)
                  {
                    const auto &test=p.shapes[i];
                    block(i,j)+=p.area*(d.stress*test.strain/G-v.pressure*test.div-test.pressure*v.div-test.phase*d.driving/Kf);
                  }
              }
          }
        if (tangent)
          for (unsigned int i=0; i<size; ++i)
            for (unsigned int j=0; j<size; ++j)
              if (block(i,j)!=0.) e.jacobian[ {cell.indices[i],cell.indices[j]}]+=block(i,j);
      }
    const auto endpoint=[&](const Index i)
    {
      return 2.*x[i]-lower[i];
    };
    for (const auto &[row,value]:fracture.mass)
      {
        const auto i=phase_begin+row;
        e.residual[i]+=value;
        if (publish)
          {
            e.ledger.fracture+=Kf*value*endpoint(i);
            e.ledger.storage_change+=Kf*value*(endpoint(i)-lower[i]);
          }
      }
    for (const auto &[indices,value]:fracture.stiffness)
      {
        const auto i=phase_begin+indices.first,j=phase_begin+indices.second;
        e.residual[i]+=value*x[j];
        if (tangent) e.jacobian[ {i,j}]+=value;
        if (publish)
          {
            e.ledger.fracture+=.5*Kf*value*endpoint(i)*endpoint(j);
            e.ledger.storage_change+=.5*Kf*value*(endpoint(i)*endpoint(j)-lower[i]*lower[j]);
          }
      }
    Utilities::MPI::sum(e.residual,comm,e.residual);
    double squared=0.;
    for (Index i=0; i<phase_begin; ++i)
      if (!fixed.count(i)) squared+=e.residual[i]*e.residual[i];
    e.ledger.mechanical=std::sqrt(squared);
    for (Index i=phase_begin; i<n; ++i)
      e.ledger.kkt=std::max(e.ledger.kkt,std::abs(PhaseField::projected_gradient(x[i],e.residual[i],mass[i],lower[i],upper[i])));
    if (publish)
      {
        AssertThrow(!Utilities::MPI::max(detached,comm),ExcMessage("Detached exterior cohesive history requires a formation law."));
        Utilities::MPI::max(admitted,comm,admitted);
        auto &b=e.ledger;
        for (auto *v:
             {
               &b.bulk,&b.cohesive,&b.storage_change,&b.bulk_loss,&b.cohesive_loss,&b.adjoint_defect,&b.compliance_remainder,&b.fracture
             })
          *v=Utilities::MPI::sum(*v,comm);
        for (unsigned int k=0; k<ns; ++k)
          {
            auto &s=e.candidate.sections[k];
            s.angle=angles[k];
            s.formed=formed[k] || admitted[k];
            s.rate=e.roots[k].rate;
            s.resistance=e.roots[k].resistance;
            s.slip+=2.*h*s.rate;
            b.friction_work+=2.*h*lengths[k]*s.rate*s.resistance;
          }
        b.adjoint_defect-=b.friction_work;
        for (const auto &[i,v]:fixed) if (i<nv) b.prescribed_velocity_work+=2.*G*v*e.residual[i];
        for (Index i=phase_begin; i<n; ++i) b.phase_work+=Kf*(endpoint(i)-lower[i])*e.residual[i];
        e.candidate.step=parent.step+1;
        e.candidate.time=sim.get_time();
      }
    return e;
  }

  void MidpointSystem::restore_continuity(Values &x) const
  {
    Evaluation projection;
    projection.residual.assign(n,0.);
    for (const auto &[id,cell]:cells)
      {
        const unsigned int size=cell.indices.size();
        FullMatrix<double> block(size,size);
        for (const auto &p:cell.points)
          {
            double divergence=0.;
            for (unsigned int j=0; j<size; ++j) divergence+=p.shapes[j].div*x[cell.indices[j]];
            for (unsigned int i=0; i<size; ++i)
              {
                const auto &v=p.shapes[i];
                if (cell.indices[i]>=phase_begin) continue;
                projection.residual[cell.indices[i]]-=p.area*v.pressure*divergence;
                for (unsigned int j=0; j<size; ++j)
                  if (cell.indices[j]<phase_begin)
                    {
                      const auto &u=p.shapes[j];
                      // 3-D deviatoric product plus div-div/3 recovers
                      // the full symmetric gradient of the Stokes metric.
                      block(i,j)+=p.area*(2.*B*(v.strain*u.strain+v.div*u.div/3.)-v.div*u.pressure-v.pressure*u.div);
                    }
              }
          }
        for (unsigned int i=0; i<size; ++i)
          for (unsigned int j=0; j<size; ++j)
            if (block(i,j)!=0.) projection.jacobian[ {cell.indices[i],cell.indices[j]}]+=block(i,j);
      }
    Utilities::MPI::sum(projection.residual,comm,projection.residual);
    const auto delta=solve_linearized_obstacle(projection,x);
    for (Index i=0; i<phase_begin; ++i) x[i]+=delta[i];
  }

  void MidpointSystem::use_phase_search_metric(Evaluation &e) const
  {
    for (auto it=e.jacobian.begin(); it!=e.jacobian.end();)
      if (it->first.first>=phase_begin && it->first.first<n)
        it=e.jacobian.erase(it);
      else ++it;
    for (const auto &[ij,value]:fracture.stiffness)
      e.jacobian[ {phase_begin+ij.first,phase_begin+ij.second}]+=value;
    for (const auto &[i,value]:fracture.mass)
      e.jacobian[ {phase_begin+i,phase_begin+i}]+=value;
  }

  MidpointSystem::Values MidpointSystem::solve_linearized_obstacle(const Evaluation &e,const Values &x) const
  {
    // Gather sparse element blocks only for the direct solve. Strip borders
    // keep its graph sparse; eliminating their linear increments gives the
    // exact condensed tangent without forming any dense strip outer products.
    const auto parts=Utilities::MPI::gather(comm,e.jacobian);
    Values update(n);
    std::string error;
    if (Utilities::MPI::this_mpi_process(comm)==0)
      try
        {
          const Index rows=n+angles.size();
          Matrix entries;
          for (const auto &part:parts)
            for (const auto &[ij,v]:part) entries[ij]+=v;
          for (Index i=n; i<rows; ++i) entries[ {i,i}]=1.;
          DynamicSparsityPattern graph(rows,rows);
          for (const auto &[ij,v]:entries) if (v!=0.) graph.add(ij.first,ij.second);
          for (Index i=0; i<rows; ++i) graph.add(i,i);
          SparsityPattern sp;
          sp.copy_from(graph);
          // Solve the linearized VI to feasibility before taking a nonlinear
          // step. A row leaves the working set only when its multiplier has
          // the wrong sign. Shorten a search to the first blocking bound.
          std::map<Index,double> constraints;
          for (const auto &[i,v]:fixed) constraints[i]=v-x[i];
          for (Index i=phase_begin; i<n; ++i)
            if (x[i]==lower[i] || x[i]==upper[i]) constraints[i]=0.;
          Vector<double> feasible(rows),result(rows),rhs(rows);
          SparseMatrix<double> matrix(sp);
          bool solved=false;
          for (Index pass=0; pass<4*(n-phase_begin)+20; ++pass)
            {
              matrix=0.;
              rhs=0.;
              for (Index i=0; i<n; ++i) rhs[i]=-e.residual[i];
              for (const auto &[ij,v]:entries)
                if (v!=0. && !constraints.count(ij.first))
                  {
                    const auto column=constraints.find(ij.second);
                    if (column==constraints.end()) matrix.set(ij.first,ij.second,v);
                    else rhs[ij.first]-=v*column->second;
                  }
              for (const auto &[i,v]:constraints)
                {
                  matrix.set(i,i,1.);
                  rhs[i]=v;
                }
              SparseDirectUMFPACK direct;
              direct.initialize(matrix);
              direct.vmult(result,rhs);
              double alpha=1., target=0.;
              Index blocking=n;
              for (Index i=phase_begin; i<n; ++i)
                if (!constraints.count(i))
                  {
                    const double lo=lower[i]-x[i],hi=upper[i]-x[i];
                    if (result[i]>=lo && result[i]<=hi) continue;
                    const double bound=result[i]<lo?lo:hi;
                    const double step=(bound-feasible[i])/(result[i]-feasible[i]);
                    if (step<alpha)
                      {
                        alpha=std::max(0.,step);
                        blocking=i;
                        target=bound;
                      }
                  }
              if (blocking<n)
                {
                  feasible.sadd(1.-alpha,alpha,result);
                  feasible[blocking]=target;
                  constraints[blocking]=target;
                  continue;
                }
              feasible=result;
              Vector<double> gradient(n);
              std::copy(e.residual.begin(),e.residual.end(),gradient.begin());
              for (const auto &[ij,v]:entries)
                if (ij.first>=phase_begin && ij.first<n) gradient[ij.first]+=v*result[ij.second];
              Index release=n;
              double largest=1.e-12;
              for (const auto &[i,v]:constraints)
                if (i>=phase_begin && lower[i]!=upper[i])
                  {
                    const double violation=(v==lower[i]-x[i]?-gradient[i]:gradient[i])/mass[i];
                    if (violation>largest)
                      {
                        largest=violation;
                        release=i;
                      }
                  }
              if (release==n)
                {
                  solved=true;
                  break;
                }
              constraints.erase(release);
            }
          AssertThrow(solved,ExcMessage("The linearized midpoint obstacle failed to find a feasible working set."));
          std::copy_n(feasible.begin(),n,update.begin());
        }
      catch (const std::exception &ex)
        {
          error=ex.what();
        }
    error=Utilities::MPI::broadcast(comm,error);
    AssertThrow(error.empty(),ExcMessage("Sparse midpoint correction failed: "+error));
    return Utilities::MPI::broadcast(comm,update);
  }

  void MidpointSystem::write_candidate_fields(const Values &x,LinearAlgebra::BlockVector &solution) const
  {
    const auto &intro=sim.introspection();
    const auto c=intro.component_indices.compositional_fields[field];
    const auto offset=solution.get_block_indices().block_start(phase_block);
    std::vector<Index> ids(sim.get_fe().n_dofs_per_cell());
    unsigned int invalid=0;
    for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(ids);
          PhaseField::BernsteinPatch<2> patch;
          for (unsigned int j=0; j<patch.size(); ++j)
            {
              const auto i=phase_begin+ids[sim.get_fe().component_to_system_index(c,j)]-offset;
              patch[j]=2.*x[i]-lower[i];
              invalid|=x[i]<lower[i] || x[i]>upper[i];
            }
          const auto bounds=PhaseField::enclosure<2>(patch);
          const double tolerance=128.*std::numeric_limits<double>::epsilon();
          invalid|=bounds.first < -tolerance || bounds.second > 1.+tolerance;
        }
    AssertThrow(!Utilities::MPI::max(invalid,comm),ExcMessage("Midpoint endpoint failed phase bounds or irreversibility."));
    for (const unsigned int b:
    {
      0u,1u,phase_block
    })
    {
      LinearAlgebra::Vector owned(intro.index_sets.system_partitioning[b],comm);
      for (const auto j:owned.locally_owned_elements())
        owned[j]=b==0?x[j]/h:b==1?G*x[nv+j]:2.*x[phase_begin+j]-lower[phase_begin+j];
      owned.compress(VectorOperation::insert);
      solution.block(b)=owned;
    }
  }

  namespace
  {
    // Confirm the observable a consumer can reconstruct from stored native
    // increments, including roundoff in adding an increment to old shear.
    void check_published_section_rates(const MidpointSystem &system,const History &candidate)
    {
      const unsigned int n=candidate.sections.size();
      std::vector<long double> rates(n),inventory(n);
      for (const auto &[id,cell]:system.cells)
        for (unsigned int q=0; q<cell.points.size(); ++q)
          {
            const auto &p=cell.points[q];
            const unsigned int k=p.strip;
            const long double old=system.material.history().cells.at(id)[q].signed_shear;
            const long double next=candidate.cells.at(id)[q].signed_shear;
            const long double weight=p.area/(2.*system.h*system.material.sections().length(k,candidate.sections[k].angle));
            rates[k]+=weight*(next-old);
            inventory[k]+=weight*(std::abs(next)+std::abs(old));
          }
      Utilities::MPI::sum(rates,system.comm,rates);
      Utilities::MPI::sum(inventory,system.comm,inventory);
      for (unsigned int k=0; k<n; ++k)
        {
          const double rate=candidate.sections[k].rate;
          const long double error=std::abs(rates[k]-rate);
          const long double budget=32.*std::numeric_limits<double>::epsilon()*inventory[k];
          if (error>budget || (rate!=0. && error>=std::abs(rate)))
            {
              std::ostringstream message;
              message<<"Unresolved native slip rate on strip "<<k<<": stored increments give "<<rates[k]
                     <<" m/s, signed root "<<rate<<" m/s, transfer roundoff budget "<<budget<<" m/s.";
              AssertThrow(false,ExcMessage(message.str()));
            }
        }
    }
  }

  static void solve_midpoint(const SimulatorAccess<2> &sim,const MidpointMaterial &material, LinearAlgebra::BlockVector &solution, History &next, Balance &balance)
  {
    TimerOutput::Scope timer(sim.get_computing_timer(),"Coupled midpoint fault");
    MidpointSystem system(sim,material,solution);
    auto x=system.initial;
    system.freeze_direction_neighborhood();
    system.restore_continuity(x);
    double initial_merit=0.;
    bool use_positive_metric=true;
    unsigned int direction_updates=0;
    const auto &settings=sim.get_parameters();
    for (unsigned int iteration=0; iteration<settings.max_nonlinear_iterations; ++iteration)
      {
        auto stage=system.assemble_stage(x,true,false);
        sim.get_pcout()<<"   Midpoint iteration "<<iteration<<", mechanics "<<stage.ledger.mechanical
                       <<", projected phase "<<stage.ledger.kkt<<std::endl;
        if (stage.ledger.mechanical<=settings.nonlinear_tolerance && stage.ledger.kkt<=1.e-7)
          {
            if (system.select_prospective_planes(x))
              {
                if (++direction_updates>12) throw ExcNonlinearSolverNoConvergence();
                continue;
              }
            // Bind publication to a fresh evaluation on the final plane map.
            auto returned=system.assemble_stage(x,true,true);
            if (returned.ledger.mechanical>settings.nonlinear_tolerance || returned.ledger.kkt>1.e-7)
              throw ExcNonlinearSolverNoConvergence();
            check_published_section_rates(system,returned.candidate);
            system.write_candidate_fields(x,solution);
            next=std::move(returned.candidate);
            balance=returned.ledger;
            balance.iterations=iteration;
            return;
          }
        // Globalize a newly activated phase obstacle with the convex
        // fracture-plus-mass metric. A decade of residual reduction switches
        // back to the full physical Newton matrix for the remaining solve.
        const double merit=std::max(stage.ledger.mechanical/settings.nonlinear_tolerance,stage.ledger.kkt/1.e-7);
        if (iteration==0) initial_merit=merit;
        if (merit<=std::max(10.,.1*initial_merit)) use_positive_metric=false;
        if (use_positive_metric) system.use_phase_search_metric(stage);
        const auto d=system.solve_linearized_obstacle(stage,x);
        for (unsigned int j=0; j<x.size(); ++j)
          {
            x[j]+=d[j];
            if (j>=system.phase_begin) x[j]=std::clamp(x[j],system.lower[j],system.upper[j]);
          }
      }
    throw ExcNonlinearSolverNoConvergence();
  }
  template <int dim>
  void validate_midpoint_material(const Parameters<dim> &p,const MaterialModel::Interface<dim> &model)
  {
    const auto &methods=p.compositional_field_methods;
    const auto selected=std::find(methods.begin(),methods.end(),Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase);
    if (selected==methods.end()) return;
    const auto *provider=dynamic_cast<const MidpointMaterial *>(&model);
    AssertThrow(provider,ExcMessage("Midpoint fault phase requires a material providing the Maxwell/AT1 midpoint response."));
    const auto field=provider->phase_field();
    AssertThrow(dim==2 && !p.enable_elasticity &&
                p.nonlinear_solver==Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes &&
                field==static_cast<unsigned int>(selected-methods.begin()) &&
                std::count(p.compositional_field_methods.begin(),p.compositional_field_methods.end(),
                           Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase)==1 &&
                !p.explicit_compositions.selected[field] &&
                p.nonlinear_solver_failure_strategy!=Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep &&
                p.surface_pressure>0. && p.pressure_normalization=="no",
                ExcMessage("Growing fault requires a single midpoint fault phase, plane strain, native elasticity off, single Advection, iterated Newton Stokes, prescribed positive surface pressure and no pressure normalization. Nonconvergence must abort or request a retry."));
    AssertThrow(!p.mesh_deformation_enabled && !p.initial_adaptive_refinement && !p.adaptive_refinement_interval &&
                p.additional_refinement_times.empty() && p.stokes_velocity_degree==2 && !p.use_equal_order_interpolation_for_stokes &&
                !p.use_locally_conservative_discretization && !p.include_melt_transport,
                ExcMessage("Growing fault requires a fixed conforming Q2/Q1 mesh."));
    AssertThrow(!p.use_direct_stokes_solver,ExcMessage("Midpoint fault mechanics requires separate velocity and pressure blocks."));
    AssertThrow(!p.run_postprocessors_on_nonlinear_iterations,ExcMessage("Midpoint fault output requires an accepted step; iteration postprocessing is unsupported."));
    AssertThrow(!p.enable_prescribed_dilation &&
                (p.formulation_mass_conservation==Parameters<dim>::Formulation::MassConservation::incompressible ||
                 p.formulation_mass_conservation==Parameters<dim>::Formulation::MassConservation::ask_material_model),
                ExcMessage("Midpoint fault mechanics requires incompressible continuity without prescribed dilation."));
  }
  template void validate_midpoint_material(const Parameters<2> &,const MaterialModel::Interface<2> &);
  template void validate_midpoint_material(const Parameters<3> &,const MaterialModel::Interface<3> &);

  SolverControl advance_midpoint(const SimulatorAccess<2> &sim,MidpointMaterial &material,LinearAlgebra::BlockVector &solution)
  {
    SolverControl control(sim.get_parameters().max_nonlinear_iterations,sim.get_parameters().nonlinear_tolerance);
    material.begin_attempt();
    const auto field=material.phase_field();
    if (sim.get_timestep_number()==0)
      {
        // Initialize the physical Q2 field with the inherited whole-cell
        // scaling. An admissible polynomial keeps every input coefficient.
        const auto &intro=sim.introspection();
        const auto c=intro.component_indices.compositional_fields[field], b=intro.block_indices.compositional_fields[field];
        const auto offset=solution.get_block_indices().block_start(b);
        LinearAlgebra::Vector owned(intro.index_sets.system_partitioning[b],sim.get_mpi_communicator());
        std::vector<types::global_dof_index> ids(sim.get_fe().n_dofs_per_cell());
        unsigned int invalid=0;
        for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(ids);
              std::vector<double> patch(9);
              for (unsigned int j=0; j<9; ++j) patch[j]=solution[ids[sim.get_fe().component_to_system_index(c,j)]];
              invalid|=!PhaseField::limit_bernstein<2>(patch,std::accumulate(patch.begin(),patch.end(),0.)/9.);
              for (unsigned int j=0; j<9; ++j) owned[ids[sim.get_fe().component_to_system_index(c,j)]-offset]=patch[j];
            }
        AssertThrow(!Utilities::MPI::max(invalid,sim.get_mpi_communicator()),ExcMessage("Invalid initial phase mean."));
        owned.compress(VectorOperation::insert);
        solution.block(b)=owned;
        control.check(0,0.);
        return control;
      }
    AssertThrow(material.history().step+1==sim.get_timestep_number(),ExcMessage("Growing fault accepted parent clock mismatch."));
    History next;
    Balance ledger;
    solve_midpoint(sim,material,solution,next,ledger);
    material.stage_candidate(std::move(next),ledger);
    control.check(ledger.iterations,ledger.mechanical);
    return control;
  }

}
