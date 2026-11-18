#!/usr/bin/env python3
"""One frozen spatial diagnostic; no accepted physical interval is advanced."""
from pathlib import Path
import argparse,os,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=argparse.ArgumentParser();p.add_argument('nx',type=int,choices=[128,256,512]);p.add_argument('--suffix',default='');p.add_argument('--seconds',type=int,default=1800)
p.add_argument('--input-dir',type=Path);p.add_argument('--checker',type=Path,default=ROOT/'build-particle-free/stage4-wavelength/libbulk_tip_checks.release.so');a=p.parse_args()
os.environ['ASPECT_WAVELENGTH_INPUT']=str((a.input_dir or ROOT/f'build-particle-free/stage4-wavelength/input-{a.nx}').resolve())
os.environ['ASPECT_TIP_RESTORING_RESPONSE']='1'
command=['python3',str(HERE.parent/'bulk_tip/run.py'),f'stage4-wavelength-{a.nx}{a.suffix}',
 '--fixture',str(Path(os.environ['ASPECT_WAVELENGTH_INPUT'])/'fixture.prm'),
 '--executable',str(ROOT/'build-particle-free/first-free-turn/tested/aspect-release'),
 '--checker',str(a.checker.resolve()),
 '--ranks','4','--bind-cores','--dt','.00125','--end','0','--end-step','0',
 '--seconds',str(a.seconds),'--fixed-phase','--checkpoint-steps','1000000']
raise SystemExit(subprocess.call(command,cwd=ROOT))
