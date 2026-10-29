    // Optional observation/stop hooks for the horizontal-domain control.
    // The supplied Trial is still uncommitted. No constitutive values change.
    void domain_diagnostics(const Trial &candidate,const double time) const
    {
      if(std::getenv("CI_DETACHED_GUARD"))
        {
          const double center=L*.5*(setting("CI_SEED_FIRST",.375)+setting("CI_SEED_LAST",.625));
          unsigned middle=0;for(unsigned q=1;q<3*ns;++q)if(std::abs(qpositions[q]-center)<std::abs(qpositions[middle]-center))middle=q;
          AssertThrow(candidate.point[middle].state.damage>=.01,ExcMessage("Domain control lost its mature seed"));
          unsigned left=middle,right=middle;
          while(left>0 && candidate.point[left-1].state.damage>=.01)--left;
          while(right+1<3*ns && candidate.point[right+1].state.damage>=.01)++right;
          bool detached=false;for(unsigned q=0;q<3*ns;++q)if((q<left || q>right) && candidate.point[q].state.damage>=.01)detached=true;
          if(detached)
            {
              std::map<std::string,std::string> accepted;save(accepted);
              std::ofstream archive(this->get_output_directory()+"last-accepted-before-detached.archive");aspect::oarchive ar(archive);ar<<accepted;
              auto f=output("rejected-detached.csv");f<<"target_s,incoming_s,q,x_m,damage,jump_m\n";
              for(unsigned q=0;q<3*ns;++q)f<<time<<','<<accepted_time<<','<<q<<','<<qpositions[q]<<','<<candidate.point[q].state.damage<<','<<candidate.point[q].state.jump<<'\n';
              AssertThrow(false,ExcMessage("Detached damage reached domain-control stop; not published"));
            }
        }
      if(const char *times=std::getenv("CI_SNAPSHOT_TIMES"))
        {
          std::string text(times);std::replace(text.begin(),text.end(),',',' ');std::istringstream input(text);double target;
          while(input>>target)
            if(std::abs(time-target)<=1e-10)
              field_check(candidate.nodal,candidate.wall,"physical-fields-"+std::to_string(std::lround(target*1e6))+".csv",true);
        }
    }
