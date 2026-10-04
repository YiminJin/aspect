from prepare_inputs import prm,root,base
for policy in 'AB':
 data=base.copy()
 # Reuse the maintained live mesh/profile/geometry sources; no BP3 mechanical callbacks.
 data.update({
  ('Additional shared libraries',):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/build/transport/libbp3_local_transport.release.so',
  ('Output directory',):'$ASPECT_SOURCE_DIR/benchmarks/reconstructed_fault/bp3_local_tests/output-transport-'+policy+'-final',
  ('Nonlinear solver scheme',):'single Advection, no Stokes',('End time',):'1.6',
  ('Maximum time step',):'0.4',('Maximum first time step',):'0.4',
  ('Termination criteria','Termination criteria'):'end time',('Termination criteria','Checkpoint on termination'):'false',
  ('Time stepping','List of model names'):'convection time step',
  ('Mesh refinement','BP3 local comparison','Policy'):policy,
  ('Boundary velocity model','Prescribed velocity boundary indicators'):'',
  ('Prescribed Stokes solution','Model name'):'function',
  ('Prescribed Stokes solution','Velocity function','Function expression'):'1;0.6',
  ('Prescribed Stokes solution','Pressure function','Function expression'):'0',
  ('Compositional fields','Number of fields'):'8',
  ('Compositional fields','Names of fields'):'tau_xx,tau_yy,tau_xy,shadow_xx,shadow_yy,shadow_xy,theta_initial,strengthening',
  ('Compositional fields','Types of fields'):'stress,stress,stress,generic,generic,generic,generic,chemical composition',
  ('Compositional fields','Compositional field methods'):'particles',
  ('Compositional fields','Mapped particle properties'):'tau_xx:maxwell stress[0],tau_yy:maxwell stress[1],tau_xy:maxwell stress[2],shadow_xx:shadow_xx,shadow_yy:shadow_yy,shadow_xy:shadow_xy,theta_initial:phase field fault state,strengthening:initial strengthening',
  ('Initial composition model','Model name'):'local curved history',
  ('Particles','Interpolation scheme'):'local transport LLS',
  ('Particles','Initial composition','List of field names'):'shadow_xx,shadow_yy,shadow_xy,theta_initial,strengthening',
  ('Particles','Initial composition','Spatially refreshed field names'):'shadow_xx,shadow_yy,shadow_xy,strengthening',
  ('Postprocess','List of postprocessors'):'local transport probes, particles',
  ('Checkpointing','Steps between checkpoint'):'0',
 })
 data={k:v for k,v in data.items() if not (len(k)>1 and k[0]=='Postprocess' and k[1]!='List of postprocessors')}
 data[('Postprocess','BP3','Allow truncated transition')]='true'
 data[('Postprocess','Particles','Data output format')]='none'
 prm.write(data,root/'inputs'/('transport-'+policy+'.prm'))
