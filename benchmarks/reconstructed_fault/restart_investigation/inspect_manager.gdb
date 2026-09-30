set pagination off
set confirm off
set breakpoint pending on
set print pretty on
set print elements 30
set debuginfod enabled off
handle SIGSEGV stop print nopass
python
import gdb,json

def vector(v):
    impl=v['_M_impl']
    return impl['_M_start'],int(impl['_M_finish']-impl['_M_start'])

def nested(v):
    p,n=vector(v); result=[]
    for i in range(n):
        q,m=vector(p[i]);result.append(dict(size=m,values=[float(q[j]) for j in range(m)]))
    return dict(size=n,rows=result)

def snapshot(obj,label,frame=None):
    faults,n=vector(obj['reconstructed_faults'])
    data=dict(where=label,faults=n,vertices=[vector(faults[i]['vertices'])[1] for i in range(n)],
        committed=nested(obj['timestep_committed_slip_rates']),
        current=nested(obj['current_newton_slip_rates']),trial=nested(obj['trial_slip_rates']),
        prescribed_outer_size=vector(obj['prescribed_slip_rates'])[1],
        solve_active=bool(obj['slip_rate_nonlinear_solve_active']),
        trial_active=bool(obj['slip_rate_trial_active']))
    if label=='before trial values':data['incoming']=nested(frame.read_var('values'))
    gdb.write('MANAGER_STATE '+json.dumps(data,sort_keys=True)+'\n')
    gdb.execute('bt 6')

class AfterRebuild(gdb.FinishBreakpoint):
    def __init__(self,frame,pointer):
        super().__init__(frame,internal=True);self.pointer=pointer
    def stop(self):
        snapshot(self.pointer.dereference(),'after restart rebuild')
        return False

class Inspect(gdb.Breakpoint):
    def __init__(self,spec,label):
        super().__init__(spec);self.label=label
    def stop(self):
        try:
            frame=gdb.newest_frame();pointer=frame.read_var('this')
            snapshot(pointer.dereference(),self.label,frame)
            if self.label=='restart rebuild entry':AfterRebuild(frame,pointer)
        except Exception as exc:gdb.write('INSPECTION_ERROR '+str(exc)+'\n')
        return False

Inspect('aspect::ReconstructedFaultManager<2>::rebuild_after_deserialization','restart rebuild entry')
Inspect('aspect::ReconstructedFaultManager<2>::set_slip_rate_trial_values','before trial values')
end
run
printf "\nSTOPPED BACKTRACE\n"
bt 16
frame 0
info line
frame 1
info line
frame 2
info line
frame 3
info line
quit
