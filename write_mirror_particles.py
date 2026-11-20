"""
Created on Fri 19/10/2022
@author: mfrosst
"""
import os
import pickle
import h5py
import numpy as np
import pandas as pd

from colour import Color
import scipy.stats as stats
from scipy.interpolate import interpn

#write to hdf5
from write_to_hdf5 import write_pcls, just_write_halo_pcls, write_pcls_wgas, write_snapshot

#LOAD FUNC#################################################################
def load_file_agama(file):
    df = pd.read_csv(file, sep=r"\s+", header=0)  # whitespace/tab separated (fast C parser)
    #positions
    x = df['#x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    pos = np.array([x,y,z]).T

    #velocities
    vx = df['vx'].to_numpy()
    vy = df['vy'].to_numpy()
    vz = df['vz'].to_numpy()
    vel = np.array([vx,vy,vz]).T

    #masses
    mass = df['mass'].to_numpy()
    return (pos, vel, mass)
    

#DO BIG BOY CALCULATIONS###########################################################
def mirror_agama(star_file_in, dm_file_in, out_file, mass_unit=1.0):
    '''mass_unit: masses are divided by this (e.g. 1e8 for Gadget UnitMass = 1e8 Msun)'''
    (star_pos, star_vel, star_mass) = load_file_agama(star_file_in)
    (halo_pos, halo_vel, halo_mass) = load_file_agama(dm_file_in)
    star_mass = star_mass / mass_unit; halo_mass = halo_mass / mass_unit
    # mirroring keeps 2*floor(N/2) particles; drop a leftover particle so array lengths agree
    star_mass = star_mass[:2*(len(star_mass)//2)]; halo_mass = halo_mass[:2*(len(halo_mass)//2)]

    #MIRROR PARTICLES
    #DISK
    #first, randomly select half of all particles to be deleted
    even_star_pos = star_pos[1::2];  odd_star_pos = star_pos[0::2]
    even_star_vel = star_vel[1::2];  odd_star_vel = star_vel[0::2]
    print('DISK | even half:', len(even_star_pos), 'even/tot:',len(even_star_pos)/len(star_pos))

    #now, take these random half and mirror them
    newodd_star_pos = -even_star_pos; newodd_star_vel = -even_star_vel
    new_star_pos = np.concatenate((newodd_star_pos, even_star_pos))
    new_star_vel = np.concatenate((newodd_star_vel, even_star_vel))

    #HALO
    #first, randomly select half of all particles to be deleted
    even_halo_pos = halo_pos[1::2];  odd_halo_pos = halo_pos[0::2]
    even_halo_vel = halo_vel[1::2];  odd_halo_vel = halo_vel[0::2]
    print('HALO | even half:', len(even_halo_pos), 'even/tot:',len(even_halo_pos)/len(halo_pos))

    #now, take these random half and mirror them
    newodd_halo_pos = -even_halo_pos; newodd_halo_vel = -even_halo_vel
    new_halo_pos = np.concatenate((newodd_halo_pos, even_halo_pos))
    new_halo_vel = np.concatenate((newodd_halo_vel, even_halo_vel))
    
    write_pcls(new_star_pos, new_star_vel, star_mass, new_halo_pos, new_halo_vel, halo_mass, out_file)

def mirror_agama_onecomp(dm_file_in, out_file, mass_unit=1.0):
    (halo_pos, halo_vel, halo_mass) = load_file_agama(dm_file_in)
    halo_mass = halo_mass / mass_unit
    halo_mass = halo_mass[:2*(len(halo_mass)//2)]

    #MIRROR PARTICLES
    #HALO
    #first, randomly select half of all particles to be deleted
    even_halo_pos = halo_pos[1::2];  odd_halo_pos = halo_pos[0::2]
    even_halo_vel = halo_vel[1::2];  odd_halo_vel = halo_vel[0::2]
    print('HALO | even half:', len(even_halo_pos), 'even/tot:',len(even_halo_pos)/len(halo_pos))

    #now, take these random half and mirror them
    newodd_halo_pos = -even_halo_pos; newodd_halo_vel = -even_halo_vel
    new_halo_pos = np.concatenate((newodd_halo_pos, even_halo_pos))
    new_halo_vel = np.concatenate((newodd_halo_vel, even_halo_vel))

    just_write_halo_pcls(new_halo_pos, new_halo_vel, halo_mass, out_file)
    return

def mirror_halves(pos, vel, *extra):
    '''Keep every second particle and add its point-reflected copy (x,v) -> (-x,-v).'''
    keep = slice(1, None, 2)
    p, v = pos[keep], vel[keep]
    out = [np.concatenate((-p, p)), np.concatenate((-v, v))]
    for e in extra:
        out.append(np.concatenate((e[keep], e[keep])))
    return out

def mirror_agama_wgas(star_file_in, dm_file_in, gas_file_in, out_file, mass_unit=1.0):
    '''As mirror_agama, plus a gas component (text file with a "u" column -> InternalEnergy).'''
    sp, sv, sm = load_file_agama(star_file_in)
    hp, hv, hm = load_file_agama(dm_file_in)
    g = pd.read_csv(gas_file_in, sep=r"\s+", header=0)
    gp = g[['#x', 'y', 'z']].to_numpy(); gv = g[['vx', 'vy', 'vz']].to_numpy()
    gm = g['mass'].to_numpy(); gu = g['u'].to_numpy()
    sp, sv, sm = mirror_halves(sp, sv, sm)
    hp, hv, hm = mirror_halves(hp, hv, hm)
    gp, gv, gm, gu = mirror_halves(gp, gv, gm, gu)
    print('mirrored: disc %d, halo %d, gas %d particles' % (len(sm), len(hm), len(gm)))
    write_pcls_wgas(sp, sv, sm / mass_unit, hp, hv, hm / mass_unit, gp, gv, gm / mass_unit,
                    out_file, u_gas=gu)

def mirror_agama_all(star_file_in, dm_file_in, out_file, gas_file_in=None, sub=None, mass_unit=1.0, mirror=True):
    '''Disc (PartType2) and smooth halo (PartType1) [and gas (PartType0)] are mirrored as in
    mirror_agama; subhalo particles (sub = dict(pos, vel, mass), PartType3) are NOT mirrored,
    since that would split each subhalo in two. The whole system is then shifted to zero
    centre of mass and zero net momentum. Returns {ptype: first ParticleID}.
    mirror=False: write disc/halo/gas as read (e.g. quiet-start files that are already mirrored).'''
    mh = mirror_halves if mirror else (lambda *a: list(a))
    sp, sv, sm = mh(*load_file_agama(star_file_in))
    hp, hv, hm = mh(*load_file_agama(dm_file_in))
    parts = {2: dict(pos=sp, vel=sv, mass=sm), 1: dict(pos=hp, vel=hv, mass=hm)}
    if gas_file_in is not None:
        g = pd.read_csv(gas_file_in, sep=r"\s+", header=0)
        gp, gv, gm, gu = mh(g[['#x', 'y', 'z']].to_numpy(), g[['vx', 'vy', 'vz']].to_numpy(),
                                       g['mass'].to_numpy(), g['u'].to_numpy())
        parts[0] = dict(pos=gp, vel=gv, mass=gm, u=gu)
    if sub is not None:
        parts[3] = dict(pos=np.array(sub['pos']), vel=np.array(sub['vel']), mass=np.array(sub['mass']))
    mtot = sum(d['mass'].sum() for d in parts.values())
    xcm = sum((d['pos'] * d['mass'][:, None]).sum(0) for d in parts.values()) / mtot
    vcm = sum((d['vel'] * d['mass'][:, None]).sum(0) for d in parts.values()) / mtot
    for d in parts.values():
        d['pos'] = d['pos'] - xcm; d['vel'] = d['vel'] - vcm; d['mass'] = d['mass'] / mass_unit
    print('recentred by dx=%s kpc, dv=%s km/s; N per type: %s' %
          (np.round(xcm, 4), np.round(vcm, 4), {t: len(d['mass']) for t, d in sorted(parts.items())}))
    return write_snapshot(parts, out_file)

#MAIN#######################
if __name__ == '__main__':
    #dir = '/Users/23229092/Documents/Idealized_Runs/out_Models/agama_test/agama_hernquist/'
    #
    #dm_file_in = dir+'model_dm_final_EAGLEHR.txt'
    #file_out = './model_hernq_final_EAGLEHR.hdf5'
    #mirror_agama_onecomp(dm_file_in, file_out)
    #
    #stop
    #EAGLE MOCKS
    dir = '/Users/23229092/Documents/Idealized_Runs/out_Models/agama_test/agama_medium_disk_test/'

    star_file_in = dir+'model_stars_final_EAGLELR.txt'
    dm_file_in = dir+'model_dm_final_EAGLELR.txt'
    file_out = './model_final_EAGLELR.hdf5'
    mirror_agama(star_file_in, dm_file_in, file_out)

    star_file_in = dir+'model_stars_final_EAGLEHR.txt'
    dm_file_in = dir+'model_dm_final_EAGLEHR.txt'
    file_out = './model_final_EAGLEHR.hdf5'
    mirror_agama(star_file_in, dm_file_in, file_out)

    stop
    
    #AGAMA
    star_file_in = './medium_disk/model_stars_final_fiducial1e6.txt'
    mus = np.array([5,10,20])
    for i, mu in enumerate(mus):
        print(mu)
        dm_file_in = './medium_disk/model_dm_final_mu'+str(mu)+'.txt'
        file_out = './model_final_fiducialdisk_mu'+str(mu)+'.hdf5'
        mirror_agama(star_file_in, dm_file_in, file_out)

    #FIDUCIAL
    star_file_in = './medium_disk/model_stars_final_fiducial1e6.txt'
    dm_file_in = './medium_disk/model_dm_final_fiducial1e7.txt'
    file_out = './model_final_fiducial.hdf5'
    mirror_agama(star_file_in, dm_file_in, file_out)

    #HALF FIDUCIAL
    star_file_in = './medium_disk/model_stars_final_fiducial5e5.txt'
    dm_file_in = './medium_disk/model_dm_final_fiducial5e6.txt'
    file_out = './model_final_fiducialhalf.hdf5'
    mirror_agama(star_file_in, dm_file_in, file_out)
