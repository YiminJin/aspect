// Diagnostic replay of a single recorded step, from its own accepted parent.
// Its endpoint export supplies the final prospective angles only.
void prescribe_replay_angles() const
{
  const char *file=std::getenv("ASPECT_TIP_PRESCRIBED_REPLAY");
  if (!file || !sim().bulk_tip_coupling) return;
  auto &op=*sim().bulk_tip_coupling;
  if (op.prescribed_direction_replay) return;
  AssertThrow(model().localize_tip_direction,ExcMessage("Replay requires the checkpoint's localized formation law."));
  std::ifstream input(file);
  std::string line;
  std::getline(input,line);
  for (unsigned int k=0;k<op.section_angles.size();++k)
    {
      AssertThrow(std::getline(input,line),ExcMessage("Incomplete prescribed-angle replay table."));
      std::replace(line.begin(),line.end(),',',' ');
      std::istringstream row(line);
      unsigned int id,formed,parent_formed;
      double xi,dxi,ds,angle,parent_age,age,parent_slip,slip;
      row>>id>>xi>>dxi>>ds>>angle>>formed>>parent_formed>>parent_age>>age>>parent_slip>>slip;
      const auto &old=op.parent->sections[k];
      AssertThrow(row && id==k && xi==old.coordinate && parent_age==old.age && parent_slip==old.slip
                  && bool(parent_formed)==old.formed && (!old.formed || angle==old.angle)
                  && std::abs(angle)<numbers::PI/4. && std::abs(ds*std::cos(angle)-dxi)<=1e-13*dxi
                  && std::abs(dxi-old.weight*std::cos(old.angle))<=1e-13*dxi,
                  ExcMessage("Prescribed-angle replay does not match the accepted parent."));
      op.section_angles[k]=angle;
      op.section_measures[k]=old.weight*std::cos(old.angle)/std::cos(angle);
    }
  op.prescribed_direction_replay=true;
}

// Probe the native center calculation on a read-only copy of the accepted
// state. Piecewise constant DG data separate the graph connectivity test
// from mechanics. The 2 x 1 Cartesian manufactured fixture has h=1/8.
void check_accepted_front() const
{
  const auto &op=*sim().bulk_tip_coupling;
  auto state=*op.parent;
  state.step=1;
  for (auto &section:state.sections)
    {
      section.formed=true;
      section.angle=0.;
    }
  const auto &fe=this->get_fe();
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  for (bool gap:{false,true})
    {
      for (const auto &entry:state.native_cells)
        {
          const double x=entry.second->center()[0];
          const bool connected=x<1.75 && !(gap && x>1.5);
          const double value=connected || x>1.875 ? .1 : 1e-16;
          entry.second->get_dof_indices(ids);
          for (unsigned int j=0;j<ids.size();++j)
            if (fe.system_to_component_index(j).first==op.pc)
              state.tip_phase[ids[j]-op.po]=value;
        }
      state.tip_phase.compress(VectorOperation::insert);
      const auto center=op.accepted_selection_center(state);
      double expected=1.5;
      if (!gap)
        for (const auto &section:state.sections)
          if (section.coordinate<1.75) expected=std::max(expected,section.coordinate);
      gate(gap ? "front excludes detached superlevel and tiny positive tail" : "front advances along connected accepted phase",
           std::abs(center[0]-expected)<1e-12 && center[1]==.5);
    }
}

// A frozen two-band section. Its local predictor vanishes throughout the
// entire disk. Only slip mobility and predictor stress in a remote band
// can produce a nonzero resistance and hence a nonconstant local score.
void check_remote_direction() const
{
  internal::BulkTipCoupling<dim> probe(sim(),model(),sim().current_constraints);
  auto state=std::make_shared<typename internal::BulkTipCoupling<dim>::State>(*probe.parent);
  probe.parent=state;
  auto x=probe.coordinates(sim().solution);
  x=0.;
  const auto &fe=this->get_fe();
  std::vector<types::global_dof_index> ids(fe.n_dofs_per_cell());
  for (const auto &entry:state->native_cells)
    {
      const bool remote=entry.second->center()[1]<.125;
      entry.second->get_dof_indices(ids);
      for (unsigned int j=0;j<ids.size();++j)
        if (fe.system_to_component_index(j).first==probe.pc) x.block(probe.pb)[ids[j]-probe.po]=remote ? .15 : 0.;
      for (auto &p:state->history.at(entry.first))
        {
          p.stress={};
          p.A={};
          p.absolute_shear=0.;
          if (remote) { p.stress[0][1]=1000.; p.stress[0][0]=300.; p.stress[1][1]=-300.; }
        }
    }
  x.compress(VectorOperation::insert);
  probe.prepare_direction();
  bool disjoint=true;
  for (const auto &entry:probe.direction_weights)
    for (unsigned int q=0;q<entry.second.size();++q)
      disjoint &= entry.second[q]==0. || state->history.at(entry.first)[q].stress.norm()==0.;
  gate("remote section has zero direct stress on the entire direction disk",disjoint);
  const bool changed=probe.update_virgin_planes(x);
  gate("remote RSF response affects local score through section resistance",changed && probe.candidate_rsf_solves>0);
}
