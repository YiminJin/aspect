    // Experimental monotone-formation predictor. Acceptance is NOT restricted
    // to this cone: bounded Gauss multipliers certify the complete weak law.
    Trial front_trial(const std::vector<PointState> &old,const double dt,const double du) const
    {
      const auto oldq=integration_points();
      const double beta=1/(1+law.G/law.eta*dt);
      // Geometry-only contraction; retain the identical summation order, but
      // do not repeat it for every test-function row and Newton evaluation.
      FullMatrix<double> projected_normal(3*ns,ns);
      for(unsigned int q=0;q<3*ns;++q)for(unsigned int j=0;j<ns;++j)
        for(unsigned int k=0;k<ns;++k)projected_normal(q,j)+=QP(q,k)*N(k,j);
      Trial out;out.base_publications=publications;out.base_geometry_publications=geometry_publications;out.wall=beta*(wall+du);
      out.increment.resize(ns);std::vector<bool> bound(ns,false);Vector<double> f(ns);
      for(unsigned int q=0;q<3*ns;++q)if(oldq[q].damage==0.)
        for(unsigned int i=0;i<ns;++i)if(Q(q,i)>0)bound[i]=true;
      for(unsigned int i=0;i<ns;++i)
        {out.increment[i]=movable_node(i)?properties(i)[rate_property]*dt:0.;
         if(bound[i])out.increment[i]=std::max(0.,out.increment[i]);}
      auto evaluate=[&](const std::vector<double> &increment)
      {
        out.nodal=old;out.point.resize(3*ns);out.stress.assign(ns,law.G/H*out.wall);
        out.pressure.assign(ns,law.p0);out.jacobian.reinit(ns,ns);
        for(unsigned int i=0;i<ns;++i)
          {
            out.nodal[i].jump+=increment[i];out.nodal[i].relaxed_jump=beta*(old[i].relaxed_jump+increment[i]);
            out.nodal[i].absolute_slip+=std::abs(increment[i]);out.nodal[i].damage=old[i].damage;
            for(unsigned int j=0;j<ns;++j)
              {const double z=beta*(old[j].relaxed_jump+increment[j]);out.stress[i]-=K(i,j)*z;out.pressure[i]+=N(i,j)*z;out.jacobian(i,j)=-beta*K(i,j);}
            f[i]=out.stress[i];
          }
        out.qstress.assign(3*ns,0.);out.qpressure.assign(3*ns,0.);out.tied.assign(3*ns,false);
        out.minimum_trace_pressure=std::numeric_limits<double>::max();
        for(unsigned int q=0;q<3*ns;++q)
          {
            double delta=0.;for(unsigned int i=0;i<ns;++i)
              {delta+=Q(q,i)*increment[i];out.qstress[q]+=QP(q,i)*out.stress[i];out.qpressure[q]+=QP(q,i)*out.pressure[i];}
            out.point[q]=CohesiveInterface::evaluate(law,oldq[q],delta,out.qpressure[q],dt,1.);
            out.tied[q]=oldq[q].damage==0. && delta==0.;
            for(const auto *side:{&normal_lower,&normal_upper})
              {double pn=law.p0;for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)pn+=QP(q,i)*(*side)(i,j)*beta*(old[j].relaxed_jump+increment[j]);out.minimum_trace_pressure=std::min(out.minimum_trace_pressure,pn);}
            for(unsigned int i=0;i<ns;++i)
              {
                const double r=qweights[q]*Q(q,i)/weights[i];f[i]-=r*out.point[q].resistance;
                if(Q(q,i)>0 && out.point[q].state.damage>0)out.nodal[i].damage=1.;
                for(unsigned int j=0;j<ns;++j)
                  {const double dn=projected_normal(q,j);
                   out.jacobian(i,j)-=r*(out.point[q].derivative*Q(q,j)+out.point[q].pressure_derivative*beta*dn);}
              }
          }
        AssertThrow(out.minimum_trace_pressure>0,ExcMessage("Front lost closed contact."));
      };
      for(unsigned int iteration=0;iteration<100;++iteration)
        {
          evaluate(out.increment);std::vector<unsigned int> free;
          out.error=0.;double norm=0.;
          for(unsigned int i=0;i<ns;++i)
            {const double r=!movable_node(i)?0.:bound[i] && out.increment[i]==0.?std::max(0.,f[i]):f[i];norm+=r*r;out.error=std::max(out.error,std::abs(r));
             if(movable_node(i) && (!bound[i] || out.increment[i]>0. || f[i]>0.))free.push_back(i);}
          out.iterations=iteration+1;if(out.error<=1e-8)break;
          AssertThrow(iteration<99,ExcMessage("Front predictor iteration limit; not published."));
          FullMatrix<double> matrix(free.size());Vector<double> rhs(free.size()),step(free.size());
          for(unsigned int i=0;i<free.size();++i)
            {rhs[i]=-f[free[i]];for(unsigned int j=0;j<free.size();++j)matrix(i,j)=out.jacobian(free[i],free[j]);}
          matrix.gauss_jordan();matrix.vmult(step,rhs);double scale=1.;
          for(unsigned int i=0;i<free.size();++i)if(bound[free[i]] && step[i]<0.)scale=std::min(scale,-out.increment[free[i]]/step[i]);
          bool found=false;
          for(unsigned int ls=0;ls<30;++ls)
            {
              auto candidate=out.increment;for(unsigned int i=0;i<free.size();++i)
                {candidate[free[i]]+=scale*step[i];if(bound[free[i]])candidate[free[i]]=std::max(0.,candidate[free[i]]);}
              // The experimental signed route reuses this as a predictor.
              // Reject inadmissible Newton proposals, not the accepted state.
              if(std::getenv("CI_SIGNED_SOLVER"))
                {
                  double minimum=law.p0;
                  for(unsigned q=0;q<3*ns;++q)for(const auto *side:{&signed_op.Nm,&signed_op.Np})
                    {double pn=law.p0;for(unsigned j=0;j<signed_ids.size();++j)pn+=(*side)(q,j+1)*beta*(old[signed_ids[j]].relaxed_jump+candidate[signed_ids[j]]);minimum=std::min(minimum,pn);}
                  if(minimum<=0.)
                    {auto log=output("rejected-contact-proposals.csv",true);if(log.tellp()==0)log<<"target_s,scale,minimum_trial_Pa\n";log<<accepted_time+dt<<','<<scale<<','<<minimum<<'\n';scale*=.5;continue;}
                }
              evaluate(candidate);double next=0.,maximum=0.;
              for(unsigned int i=0;i<ns;++i)
                {const double r=!movable_node(i)?0.:bound[i] && candidate[i]==0.?std::max(0.,f[i]):f[i];next+=r*r;maximum=std::max(maximum,std::abs(r));}
              if(next<norm || maximum<=1e-8){out.increment=candidate;found=true;break;}scale*=.5;
            }
          AssertThrow(found,ExcMessage("Front predictor line search; not published."));
        }
      evaluate(out.increment);
      certify_front(out);
      // In this monotone-front fixture, an intact Q1 element has all three
      // Gauss equalities. They imply both endpoint variations are zero.
      std::vector<bool> constrained(ns,false);for(unsigned int i=0;i<ns;++i)constrained[i]=!movable_node(i);
      for(unsigned int i=0;i<ns;++i)
        {unsigned int count=0;for(unsigned int q=0;q<3;++q)count+=out.tied[3*i+q];
         AssertThrow(count==0 || count==3,ExcMessage("Partial Gauss tie requires general nullspace treatment."));
         if(count==3){constrained[i]=true;constrained[(i+1)%ns]=true;}}
      std::vector<unsigned int> free;for(unsigned int i=0;i<ns;++i)if(!constrained[i])free.push_back(i);
      auto curvature=[&](const std::vector<unsigned int> &ids)
      {
        if(ids.empty())return 0.;
        LAPACKFullMatrix<double> matrix(ids.size());
        for(unsigned int i=0;i<ids.size();++i)for(unsigned int j=0;j<ids.size();++j)matrix(i,j)=-.5*(out.jacobian(ids[i],ids[j])+out.jacobian(ids[j],ids[i]));
        matrix.compute_eigenvalues();double value=std::numeric_limits<double>::max();
        for(unsigned int i=0;i<ids.size();++i)value=std::min(value,matrix.eigenvalue(i).real());
        return value;
      };
      out.curvature=curvature(free);std::vector<unsigned int> all;for(unsigned int i=0;i<ns;++i)if(movable_node(i))all.push_back(i);out.release_curvature=curvature(all);
      if(out.curvature<0.)
        {
          auto rejected=output("rejected_front.csv");rejected<<"time_s,x_m,old_damage,trial_damage,jump_m,increment_m,shear_Pa,pn_Pa,curvature_Pa_m,release_curvature_Pa_m\n";
          for(unsigned int q=0;q<3*ns;++q)rejected<<accepted_time+dt<<','<<qpositions[q]<<','<<oldq[q].damage<<','<<out.point[q].state.damage<<','<<out.point[q].state.jump<<','<<out.point[q].velocity*dt<<','<<out.qstress[q]<<','<<out.qpressure[q]<<','<<out.curvature<<','<<out.release_curvature<<'\n';
          auto tangent=output("rejected_tangent.csv");tangent<<"i,j,J_Pa_m,free_i,free_j\n";
          for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)tangent<<i<<','<<j<<','<<out.jacobian(i,j)<<','<<(!constrained[i])<<','<<(!constrained[j])<<'\n';
        }
      AssertThrow(out.curvature>=0.,ExcMessage("Negative admissible front curvature; candidate saved before publication."));
      out.active.resize(ns);for(unsigned int i=0;i<ns;++i)out.active[i]=!constrained[i];
      return out;
    }

    void certify_front(Trial &out) const
    {
      // Certify unrestricted weak equilibrium. FISTA solves the convex box
      // residual problem; its step is bounded by ||R||_1 ||R||_infinity.
      std::vector<unsigned int> tied;for(unsigned int q=0;q<3*ns;++q)if(out.tied[q] && installed(q))tied.push_back(q);
      std::vector<double> lambda(tied.size()),y(tied.size()),target(ns);
      for(unsigned int i=0;i<ns;++i)
        {target[i]=out.stress[i]/law.peak;for(unsigned int q=0;q<3*ns;++q)if(!out.tied[q])target[i]-=qweights[q]*Q(q,i)*out.point[q].resistance/(weights[i]*law.peak);}
      double max_row=0.,max_column=0.;std::vector<double> rows(ns,0.);
      for(unsigned int k=0;k<tied.size();++k)
        {
          const auto q=tied[k];lambda[k]=std::max(-1.,std::min(1.,out.qstress[q]/law.peak));y[k]=lambda[k];double col=0.;
          for(unsigned int i=0;i<ns;++i){const double r=qweights[q]*Q(q,i)/weights[i];rows[i]+=r;col+=r;}max_column=std::max(max_column,col);
        }
      for(const double r:rows)max_row=std::max(max_row,r);
      const double lipschitz=max_row*max_column;double acceleration=1.;
      std::vector<double> residual(ns);
      auto box_residual=[&](const std::vector<double> &values)
      {
        for(unsigned int i=0;i<ns;++i)residual[i]=movable_node(i)?-target[i]:0.;
        for(unsigned int k=0;k<tied.size();++k)
          {const auto q=tied[k];const unsigned int ids[2]={q/3,(q/3+1)%ns};for(const auto i:ids)if(movable_node(i))residual[i]+=qweights[q]*Q(q,i)*values[k]/weights[i];}
        double error=0.;for(const double r:residual)error=std::max(error,std::abs(r)*law.peak);return error;
      };
      out.weak_error=box_residual(lambda);
      for(unsigned int iteration=0;out.weak_error>1e-8 && iteration<20000 && !tied.empty();++iteration)
        {
          box_residual(y);auto next=lambda;
          for(unsigned int k=0;k<tied.size();++k)
            {const auto q=tied[k];const unsigned int ids[2]={q/3,(q/3+1)%ns};double gradient=0.;for(const auto i:ids)gradient+=qweights[q]*Q(q,i)*residual[i]/weights[i];next[k]=std::max(-1.,std::min(1.,y[k]-gradient/lipschitz));}
          const double a=.5*(1+std::sqrt(1+4*acceleration*acceleration));
          for(unsigned int k=0;k<tied.size();++k)y[k]=next[k]+(acceleration-1)/a*(next[k]-lambda[k]);
          lambda=std::move(next);acceleration=a;out.weak_error=box_residual(lambda);out.certificate_iterations=iteration+1;
        }
      AssertThrow(out.weak_error<=1e-8,ExcMessage("Unrestricted weak front admission certificate failed; not published."));
      for(unsigned int k=0;k<tied.size();++k)
        {out.point[tied[k]].q=law.peak*lambda[k];out.point[tied[k]].resistance=out.point[tied[k]].q;}
      out.error=out.weak_error;
    }
