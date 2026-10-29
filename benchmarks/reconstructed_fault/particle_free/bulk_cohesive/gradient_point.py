#!/usr/bin/env python3
"""Benchmark bindings to C++ return; no constitutive law implemented in Python."""
from pathlib import Path
import ctypes as ct
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'build-particle-free/coulomb-gradient'
OUT=ROOT/'benchmarks/reconstructed_fault/particle_free/evidence/coulomb_gradient'
OUT.mkdir(parents=True,exist_ok=True)
NAMES=['Txx','Tyy','Txy','Axx','Axy','Kxx','Kxy','Qxx','Qxy','L','phi',
       'tauxx','tauyy','tauxy','Wxx','Wxy','speed','Y','bulk','cohesive','Db','Dc','Df',
       'active','theta_plus','theta_minus','pn_plus','pn_minus','gamma_plus','gamma_minus','system_speed']
lib=ct.CDLL(str(WORK/'point/libcoulomb_gradient_point.so'))
ptr=ct.POINTER(ct.c_double)
lib.gradient_point.argtypes=[ptr,ptr,ptr,ct.c_double,ptr,ptr,ptr,ct.c_char_p]
lib.gradient_point.restype=ct.c_int
def point(old,increment,dt,pressure=10000.,X=(0.,0.),mu=.1,cr=0.,jac=True):
    p=np.array([1e6,1e8,4.,mu,cr,dt],dtype='d');s=np.array(old,dtype='d');de=np.array(increment,dtype='d');x=np.array(X,dtype='d')
    result=np.empty(len(NAMES));J=np.empty((len(NAMES),6));error=ct.create_string_buffer(4096)
    code=lib.gradient_point(p.ctypes.data_as(ptr),s.ctypes.data_as(ptr),de.ctypes.data_as(ptr),pressure,
                           x.ctypes.data_as(ptr),result.ctypes.data_as(ptr),J.ctypes.data_as(ptr) if jac else None,error)
    if code:raise RuntimeError(error.value.decode())
    return result,J if jac else None
def initial(phi):
    s=np.zeros(11);s[10]=phi;return s
lib.gradient_projection.argtypes=[ptr,ptr,ct.c_double,ct.c_double,ct.c_double,ptr,ptr,ct.POINTER(ct.c_uint),ct.c_char_p]
lib.gradient_projection.restype=ct.c_int
def project(Z,tau,pressure=10000.,mu=.1,cr=0.,jac=True):
    z=np.array(Z,dtype='d');t=np.array(tau,dtype='d');force=np.empty(2);J=np.empty((2,6));active=ct.c_uint();error=ct.create_string_buffer(4096)
    code=lib.gradient_projection(z.ctypes.data_as(ptr),t.ctypes.data_as(ptr),pressure,mu,cr,force.ctypes.data_as(ptr),
        J.ctypes.data_as(ptr) if jac else None,ct.byref(active),error)
    if code:raise RuntimeError(error.value.decode())
    return force,J if jac else None,active.value
def tensor(v):
    xx,xy=v;return np.array([[xx,xy,0],[xy,-xx,0],[0,0,0.]])
def stress(s):return np.array([[s[0]-2e6*s[3],s[2]-2e6*s[4],0],
                              [s[2]-2e6*s[4],s[1]+2e6*s[3],0],[0,0,-s[0]-s[1]]])
def strain(v):
    x,y,shear=v;return np.array([[x-(x+y)/3,shear/2,0],[shear/2,y-(x+y)/3,0],[0,0,-(x+y)/3]])
def ledger(old,result,increment,dt,X):
    h=4*old[10]*(1+old[10])/(1-old[10])**2;tau=stress(result);old_tau=stress(old)
    q=tensor(result[7:9]);old_q=tensor(old[7:9]);delta_tau=tau-old_tau;delta_q=q-old_q
    storage=result[18]+result[19]-(old_tau*old_tau).sum()/4e6-h*(old_q*old_q).sum()/4e6
    work=(tau*strain(increment)).sum();diss=dt*sum(result[20:23]);loss=((delta_tau*delta_tau).sum()+h*(delta_q*delta_q).sum())/4e6
    micro=dt*(tensor(X)*tensor(result[14:16])).sum()
    return {'storage':storage,'work':work,'dissipation':diss,'BE_loss':loss,'micro':micro,
            'error':work-storage-diss-loss-micro}
