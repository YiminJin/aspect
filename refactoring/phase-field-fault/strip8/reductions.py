"""GDB observer of actual MPI_Allreduce result buffers during the first selector call."""
import gdb
import os
import struct


class Reduced(gdb.FinishBreakpoint):
    def __init__(self,count,address):
        super().__init__(gdb.newest_frame(),internal=True)
        self.count,self.address=count,address
        self.silent=True

    def stop(self):
        values=struct.unpack(f'={self.count}d',gdb.selected_inferior().read_memory(self.address,self.count*8))
        width=self.count//9
        with open('live-'+str(self.count),'w') as f:
            print(*values[8*width:],file=f)
        return False


class Reductions(gdb.Breakpoint):
    def __init__(self):
        super().__init__('*MPI_Allreduce',internal=True)
        self.silent=True
        self.remaining={81,27}

    def stop(self):
        count=int(gdb.parse_and_eval('$rdx'))
        if count in self.remaining:
            Reduced(count,int(gdb.parse_and_eval('$rsi')))
            self.remaining.remove(count)
            if not self.remaining: self.enabled=False
        return False
