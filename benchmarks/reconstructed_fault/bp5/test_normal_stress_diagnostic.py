"""Cheap independent geometry/projection and restart-staging checks."""
import argparse
import csv
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import struct
import zlib
import numpy as np
from stage_normal_stress_diagnostic import parameters, stage, sha


class NormalDiagnosticTests(unittest.TestCase):
    def test_matched_clock_and_narrow_archive_edit(self):
        from prepare_normal_stress_experiment import half_clock, retime_pending
        start=5310111071.5634108;time=start;rows=[]
        for i,dt in enumerate((.0028219241006433027,.0028144282043784941,.0028069639581435046,.0027995311884659868)):
            time+=dt;rows.append(dict(step=5613+i,time_s=time,dt=dt))
        schedule,adjustments=half_clock(rows,start,5612)
        self.assertEqual(len(schedule),8)
        current=start
        for step,target,dt in schedule:
            self.assertEqual(current+dt,target);current=target
        self.assertEqual(current,float(rows[-1]['time_s']))
        for i in range(4):self.assertEqual(schedule[2*i][2],rows[i]['dt']/2)
        raw=b'x'*237+struct.pack('<dddI',rows[0]['time_s'],rows[0]['dt'],.00283,5613)+b'unchanged history'*20
        compressed=zlib.compress(raw)
        archive=struct.pack('<4I',1,len(raw),len(raw),len(compressed))+compressed
        edited,old=retime_pending(archive,rows[0]['time_s'],rows[0]['dt'],5613,schedule[0][1],schedule[0][2])
        new=zlib.decompress(edited[16:])
        self.assertEqual(raw[:237],new[:237]);self.assertEqual(raw[253:],new[253:])
        with self.assertRaises(ValueError):retime_pending(archive,0.,rows[0]['dt'],5613,schedule[0][1],schedule[0][2])

    def test_plot_reference_removal_preserves_trend(self):
        from plot_normal_stress_diagnostic import perturbation, roughness, read
        a=np.zeros(4,dtype=[(n,float) for n in ('weak_pressure_Pa','weak_reference_normal_Pa',
                                             'existing_production_weak_normal_Pa')])
        a['weak_pressure_Pa']=[1.,3.,5.,7.]
        a['weak_reference_normal_Pa']=[50e6,50e6+1,50e6+2,50e6+3]
        a['existing_production_weak_normal_Pa']=a['weak_reference_normal_Pa']+a['weak_pressure_Pa']
        np.testing.assert_array_equal(perturbation(a,'existing_production_weak_normal_Pa'),a['weak_pressure_Pa'])
        np.testing.assert_array_equal(roughness(np.arange(4),a['weak_pressure_Pa']),[0.,0.])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'empty.csv';path.write_text('cell,qp,p\n')
            self.assertEqual(read(path).size,0)

    def test_inclined_split_and_consistent_mass(self):
        # Nonuniform geometry and nonconstant transverse pressure/stress;
        # positive work quadrature, shared nodes, and off-diagonal contraction.
        x,w=np.polynomial.legendre.leggauss(4);x=(x+1)/2;w=w/2
        n=np.array([np.sqrt(3)/2,.5]);M=np.zeros((4,4));loads=np.zeros((4,4))
        for j,L in enumerate((.7,1.3,.4)):
            for xi,weight in zip(x,w*L):
                N=np.array([1-xi,xi]);p=1e6*np.sin(j+xi)
                tau=np.array([[2e6*(1+xi),3e5*(j-xi)],[3e5*(j-xi),-2e6*(1+xi)]])
                d=-(n[0]**2*tau[0,0]+2*n[0]*n[1]*tau[0,1]+n[1]**2*tau[1,1])
                self.assertAlmostEqual(d,-n@tau@n,places=8)
                values=np.array([p,d,50e6,50e6+p+d])
                M[j:j+2,j:j+2]+=weight*np.outer(N,N)
                loads[:,j:j+2]+=weight*values[:,None]*N
        solved=np.linalg.solve(M,loads.T).T
        np.testing.assert_allclose(solved[:3].sum(axis=0),solved[3],rtol=2e-15)
        np.testing.assert_allclose(M@solved.T,loads.T,rtol=2e-15,atol=1e-8)
        # f/m is a diagnostic, demonstrably not the consistent coefficients.
        self.assertGreater(np.max(abs(loads[0]/M.sum(axis=1)-solved[0])),1.)

    def test_staging_preserves_checkpoint_and_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);checkpoint=job/'output/restart/01';checkpoint.mkdir(parents=True)
            (checkpoint/'bp3_accepted_state.txt').write_text('5612 5310111071.5634108\n')
            for name in ('resume.z','mesh','mesh.info','mesh_fixed.data','mesh_variable.data'):
                (checkpoint/name).write_bytes(('unaltered '+name).encode())
            fixture=job/'fixture';fixture.mkdir()
            for name in ('fault.txt','completion.txt','target_cells.txt'):(fixture/name).write_text(name)
            original=Path(__file__).parent/'server-30km-loading-surface8/first_event.prm'
            source=job/'first_event.prm';source.write_text('# Documentation with $0 and $& stays literal.\n'+original.read_text())
            bp5=job/'libbp5_steady_initialization.release.so';bp5.write_bytes(b'plugin')
            diagnostic=job/'libbp5_normal_stress_diagnostic.release.so';diagnostic.write_bytes(b'observer')
            (checkpoint.parent.parent/'state_startup_predictor.csv').write_text('5613,5310111071.5662327,10000000,.00281,.02,.02\n')
            args=argparse.Namespace(checkpoint=checkpoint,input=source,job=job,destination=job/'branch',
                                   bp5_library=bp5,diagnostic_library=diagnostic,steps=5,control=False,local_verification=False)
            with contextlib.redirect_stdout(io.StringIO()):stage(args)
            for p in checkpoint.iterdir():
                q=args.destination/'output-normal-diagnostic/restart/01'/p.name
                self.assertEqual(sha(p),sha(q));self.assertNotEqual(p.stat().st_ino,q.stat().st_ino)
            self.assertEqual(source.read_bytes(),(args.destination/'production_input.prm').read_bytes())
            executable=(args.destination/'normal_stress_diagnostic_restart.prm').read_text()
            self.assertIn(source.read_text(),executable)
            self.assertNotIn('\ninclude production_input.prm',executable)
            self.assertEqual((args.destination/'output-normal-diagnostic/state_startup_predictor.csv').read_text().strip(),
                             'accepted_step,time,maximum_dt,proposed_dt,measure,limit')
            with self.assertRaises(ValueError):stage(args)
            config=parameters(source.read_text())
            self.assertEqual(config['Material model','Phase field fault','I h surface quadrature subdivisions'],'8')

            # Prepare both real input paths without executing ASPECT. The mock
            # accepted clock is not evidence of a completed mechanical branch.
            from prepare_normal_stress_experiment import prepare
            start=5310111071.5634108;time=start;rows=[]
            for i,dt in enumerate((.0028219241006433027,.0028144282043784941,.0028069639581435046,.0027995311884659868)):
                time+=dt;rows.append(dict(step=5613+i,time_s=time,dt=dt))
            raw=b'x'*237+struct.pack('<dddI',rows[0]['time_s'],rows[0]['dt'],.00283,5613)+b'history remains fixed'
            compressed=zlib.compress(raw)
            (checkpoint/'resume.z').write_bytes(struct.pack('<4I',1,len(raw),len(raw),len(compressed))+compressed)
            args.branch='A';args.a_output=None;args.destination=job/'A'
            with contextlib.redirect_stdout(io.StringIO()):prepare(args)
            a_output=args.destination/'output-normal-diagnostic'
            with (a_output/'normal_summary.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=('step','time_s','dt'));writer.writeheader();writer.writerows(rows)
            args.branch='B';args.a_output=a_output;args.destination=job/'B'
            with contextlib.redirect_stdout(io.StringIO()):prepare(args)
            manifest=json.loads((args.destination/'experiment.json').read_text())
            self.assertEqual(len(manifest['schedule']),8)
            self.assertEqual(manifest['schedule'][-1]['time_s'],rows[-1]['time_s'])
            self.assertEqual(zlib.decompress((checkpoint/'resume.z').read_bytes()[16:]),raw)
            config=parameters((args.destination/'normal_stress_diagnostic_restart.prm').read_text().replace('include production_input.prm',''))
            self.assertEqual(config['Postprocess','BP5 normal diagnostic','New accepted steps'],'8')
            self.assertEqual(config['Postprocess','BP5 normal diagnostic','Expected clock file'],'expected_clock.txt')
            self.assertTrue(config['Time stepping','List of model names'].endswith(', function'))
            resolved=parameters(source.read_text()) | config
            self.assertEqual(resolved['Time stepping','BP5 state startup','Maximum logarithmic state change'],'0.02')


if __name__=='__main__':unittest.main()
