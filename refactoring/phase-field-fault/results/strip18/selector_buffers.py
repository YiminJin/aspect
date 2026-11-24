"""GDB callbacks for the pinned x86-64 Open MPI selector reductions."""
import os
import struct
from pathlib import Path

import gdb


class Reduced(gdb.FinishBreakpoint):
    def __init__(self, count, address, input_dir):
        super().__init__(gdb.newest_frame(), internal=True)
        self.count, self.address, self.input_dir = count, address, input_dir
        self.silent = True

    def stop(self):
        width = self.count // 18
        address = self.address + 15 * width * 8
        inferior = gdb.selected_inferior()
        record = f'live-{self.count}'
        data = bytes(inferior.read_memory(address, width * 8))
        if os.environ['OMPI_COMM_WORLD_RANK'] == '0':
            values = struct.unpack(f'={width}d', data)
            Path(record).write_text(' '.join(map(repr, values))+'\n')
        if self.input_dir:
            # Common-input replay only. Set both returned reductions on each
            # rank. The debugger stops this run at the first selected angle.
            values = (self.input_dir/record).read_text().split()
            replacement = struct.pack(f'={width}d', *map(float, values))
            inferior.write_memory(address, replacement)
            if bytes(inferior.read_memory(address,width*8)) != replacement:
                raise ValueError(f'Failed to set {record}')
            Path(record+'-used-r'+os.environ['OMPI_COMM_WORLD_RANK']).write_bytes(replacement)
        return False


class Reductions(gdb.Breakpoint):
    def __init__(self, input_dir):
        super().__init__('*MPI_Allreduce', internal=True)
        self.remaining = {18*9, 18*3}
        self.input_dir = Path(input_dir) if input_dir else None
        self.silent = True

    def stop(self):
        count = int(gdb.parse_and_eval('$rdx'))
        if count in self.remaining:
            Reduced(count,int(gdb.parse_and_eval('$rsi')),self.input_dir)
            self.remaining.remove(count)
            if not self.remaining:
                self.enabled = False
        return False
