    // Finite straight record birth. The bulk mixed space and its Schur response
    // are fixed; the actual jump trace space and native geometry owner change.
    void insert_surface()
    {
      AssertThrow(geometry_publications==0,ExcMessage("Duplicate finite geometry publication."));
      const unsigned int left=std::lround(ns*setting("CI_APPEND_FIRST",.125));
      const unsigned int right=std::lround(ns*setting("CI_APPEND_LAST",.875));
      AssertThrow(left<=first_segment && right>=last_segment && left<right && right<ns,
                  ExcMessage("Insertion must extend the selected finite segment."));
      const auto old_nodes=points(),old_gauss=integration_points();
      const auto old_bytes=archive();const auto old_left=first_segment,old_right=last_segment;
      const auto old_publications=publications;const double energy=bulk_energy(wall,old_nodes);
      field_check(old_nodes,wall,"insertion-before-fields.csv");
      auto make_candidate=[&]()
      {
        auto candidate=std::make_shared<ReconstructedFaultManager<dim>>();
        candidate->register_property("experimental interface jump damage age reference relaxed-jump absolute-slip",6);
        candidate->register_property("experimental interface Gauss3 physical histories v1",18);
        candidate->register_property("experimental signed geometric interface velocity",1);
        std::vector<Point<dim>> vertices(right-left+1);
        for(unsigned int i=left;i<=right;++i){vertices[i-left][0]=L*i/ns;vertices[i-left][1]=yfault;}
        candidate->add_reconstructed_fault(vertices,std::vector<double>(vertices.size(),width));
        for(unsigned int i=left;i<=right;++i)
          {
            auto p=candidate->get_fault(0).get_properties(i-left);const auto &v=old_nodes[i];
            const double values[]={v.jump,v.damage,v.theta,v.reference,v.relaxed_jump,v.absolute_slip};
            for(unsigned int j=0;j<6;++j)p[property+j]=values[j];
            p[rate_property]=(i>=old_left && i<=old_right)?properties(i)[rate_property]:0.;
            for(unsigned int q=0;q<3;++q)
              {const auto &g=old_gauss[3*i+q];const double data[]={g.jump,g.damage,g.theta,g.reference,g.relaxed_jump,g.absolute_slip};
               for(unsigned int j=0;j<6;++j)p[gauss_property+6*q+j]=data[j];}
          }
        return candidate;
      };
      auto candidate=make_candidate();
      // Exercise the real geometry/trace-map rollback, not only a temporary
      // history vector. swap_prescribed_geometry invalidates native associations.
      manager().swap_prescribed_geometry(*candidate);first_segment=left;last_segment=right;
      const auto candidate_bytes=archive();
      manager().swap_prescribed_geometry(*candidate);first_segment=old_left;last_segment=old_right;
      AssertThrow(archive()==old_bytes && publications==old_publications,ExcMessage("Geometry rejection modified accepted state."));
      candidate=make_candidate();manager().swap_prescribed_geometry(*candidate);first_segment=left;last_segment=right;
      AssertThrow(archive()==candidate_bytes,ExcMessage("Geometry retry is not deterministic."));
      double history_error=0.,force_error=0.,pressure_error=0.,energy_error=0.,birth_weak_error=0.;
      try
      {
      const auto new_nodes=points(),new_gauss=integration_points();
      for(unsigned int i=0;i<3*ns;++i)
        {const auto &a=old_gauss[i],&b=new_gauss[i];history_error=std::max({history_error,std::abs(a.jump-b.jump),std::abs(a.damage-b.damage),std::abs(a.theta-b.theta),std::abs(a.reference-b.reference),std::abs(a.relaxed_jump-b.relaxed_jump),std::abs(a.absolute_slip-b.absolute_slip)});}

      for(unsigned int i=0;i<ns;++i)
        {double ds=0.,dn=0.;for(unsigned int j=0;j<ns;++j){ds+=K(i,j)*(new_nodes[j].relaxed_jump-old_nodes[j].relaxed_jump);dn+=N(i,j)*(new_nodes[j].relaxed_jump-old_nodes[j].relaxed_jump);}force_error=std::max(force_error,std::abs(ds));pressure_error=std::max(pressure_error,std::abs(dn));}
      energy_error=bulk_energy(wall,new_nodes)-energy;
      AssertThrow(history_error==0 && force_error<=1e-8 && pressure_error<=1e-8 && std::abs(energy_error)<=1e-10,ExcMessage("Finite insertion changed physical history; not accepted."));
      // Certify the enlarged tied space at the SAME physical state and
      // accepted rate. No constitutive update, clock advance or rate reset.
      Trial equilibrium;equilibrium.stress.assign(ns,law.G/H*wall);
      equilibrium.pressure.assign(ns,law.p0);equilibrium.qstress.assign(3*ns,0.);
      equilibrium.point.resize(3*ns);equilibrium.tied.resize(3*ns);
      for(unsigned int i=0;i<ns;++i)for(unsigned int j=0;j<ns;++j)
        {equilibrium.stress[i]-=K(i,j)*new_nodes[j].relaxed_jump;equilibrium.pressure[i]+=N(i,j)*new_nodes[j].relaxed_jump;}
      for(unsigned int q=0;q<3*ns;++q)
        {
          const auto &s=new_gauss[q];double pn=0.,velocity=0.;
          for(unsigned int i=0;i<ns;++i)
            {equilibrium.qstress[q]+=QP(q,i)*equilibrium.stress[i];pn+=QP(q,i)*equilibrium.pressure[i];if(movable_node(i))velocity+=Q(q,i)*properties(i)[rate_property];}
          AssertThrow(pn>0,ExcMessage("Inserted contact is not closed."));
          const double speed=std::abs(velocity),sign=velocity==0.?0.:std::copysign(1.,velocity);
          const double mu=law.direct*std::asinh(speed/(2*law.V0)*std::exp((law.mu0+law.evolution*std::log(s.theta))/law.direct));
          const double bond=(s.damage==0. || s.damage==1.)?0.:(1-s.damage)/(2*law.Gc/(law.peak*law.peak)*s.damage)*(s.jump-s.reference);
          equilibrium.point[q].state=s;equilibrium.point[q].resistance=bond+s.damage*sign*(pn*mu+law.radiation*speed);
          equilibrium.tied[q]=s.damage==0.;
        }
      certify_front(equilibrium);birth_weak_error=equilibrium.weak_error;
      field_check(new_nodes,wall,"insertion-after-fields.csv");
      }
      catch(...)
      {manager().swap_prescribed_geometry(*candidate);first_segment=old_left;last_segment=old_right;throw;}
      // One geometry commit, no physical timestep/publication and no energy.
      ++geometry_publications;
      Trial stale;stale.base_publications=publications;stale.base_geometry_publications=geometry_publications-1;stale.error=0.;stale.curvature=0.;
      const auto committed=archive();bool stale_rejected=false;
      dealii::deal_II_exceptions::disable_abort_on_exception();
      try{publish(stale,accepted_time+1e-3);}catch(const std::exception&){stale_rejected=true;}
      dealii::deal_II_exceptions::enable_abort_on_exception();
      AssertThrow(stale_rejected && archive()==committed && publications==old_publications,ExcMessage("Old-geometry trial was published."));
      auto f=output("insertion.csv",true);if(f.tellp()==0)f<<"time_s,old_left_m,old_right_m,new_left_m,new_right_m,old_jump_dofs,new_jump_dofs,geometry_publications,physical_publications,history_error,load_error_Pa,pressure_error_Pa,storage_change_J_m,fracture_change_J_m,virgin_Theta,rollback_bitwise,retry_bitwise,stale_geometry_rejected,birth_weak_error_Pa\n";
      f<<accepted_time<<','<<L*old_left/ns<<','<<L*old_right/ns<<','<<L*left/ns<<','<<L*right/ns<<','<<old_right-old_left-1<<','<<right-left-1<<','<<geometry_publications<<','<<publications<<','<<history_error<<','<<force_error<<','<<pressure_error<<','<<energy_error<<",0,"<<virgin_age<<",1,1,1,"<<birth_weak_error<<'\n';
    }
