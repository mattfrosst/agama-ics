"""
Created on Fri 19/10/2022
@author: mfrosst
"""

import os
import h5py
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from colour import Color

#matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap

G = 1.0

#DENSITY SCATTER FUNC######################################################
def density_scatter(x, y, ax = None, bins = 20):
    if ax is None:
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    z[np.where(np.isnan(z))] = 0.0
    return z

def make_Ramp( ramp_colors ):
    color_ramp = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in ramp_colors ] )
    return color_ramp

#LOAD FUNC#################################################################
def load_file(data_destination):
    """Reads data and returns coords and mass
    """
    f1 = data_destination
    print('file: ',f1)
    
    f =  h5py.File(f1, "r")
    #Disk
    star_pos  = np.array(f['PartType2']['Coordinates']) #kpc
    star_vel  = np.array(f['PartType2']['Velocities']) #km/s
    star_mass = np.array(f['PartType2']['Masses']) #Msol
    star_pos  = star_pos.reshape((len(star_mass),3))
    star_vel  = star_vel.reshape((len(star_mass),3))

    return (star_pos, star_vel, star_mass)

def load_file_agama(file):
    df = pd.read_csv(file, sep="\t+", header=0)
    print(df.columns)
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
    
    #mass
    mass = df['mass'].to_numpy()

    return (pos, vel, mass)

def get_cylindrical(PosStars, VelStars):
  '''Calculates cylindrical coordinates
  '''
  rho      = np.sqrt(PosStars[:,0]**2 + PosStars[:,1]**2)
  varphi   = np.arctan2(PosStars[:,1], PosStars[:,0])
  z        = PosStars[:,2]

  v_rho    = VelStars[:,0] * np.cos(varphi) + VelStars[:,1] * np.sin(varphi)
  v_varphi = -VelStars[:,0]* np.sin(varphi) + VelStars[:,1] * np.cos(varphi)

  v_z      = VelStars[:,2]
  return(rho, varphi, z, v_rho, v_varphi, v_z)

def disk_density(rho, z, Rd, z0, M):
    term1 = M/(4*np.pi*z0*Rd**2)
    return term1 * (1/np.cosh(z/z0))**2 * np.exp(-rho/Rd)

#DO BIG BOY CALCULATIONS###########################################################
def calculate():
    end_tag = 'TEST' #'Halton'
    file = './test/model_stars_final.txt'
    G = 1.0

    Rd = 5.0; z0 = Rd * 0.1
    
    #colors
    blue_color = make_Ramp(['#08519c','#3182bd','#6baed6','#bdd7e7','#eff3ff'][::-1])
    red_color = make_Ramp(['#a50f15','#de2d26','#fb6a4a','#fcae91','#fee5d9'][::-1])
    RdBu_color = make_Ramp(['#fb6a4a','#fcae91','#fee5d9','#eff3ff','#bdd7e7','#6baed6'][::-1])
    
    #Model
    #(star_pos, star_vel, star_mass) = load_file(file)
    (star_pos, star_vel, star_mass) = load_file_agama(file)
    #star_pos = star_pos / Rd
    
    #star
    x = star_pos.T[0]; y = star_pos.T[1]; z = star_pos.T[2]
    vx = star_vel.T[0]; vy = star_vel.T[1]; vz = star_vel.T[2]
    (rho, varphi, z, v_rho, v_varphi, v_z) = get_cylindrical(star_pos, star_vel)

    plt.rcParams.update({
    "lines.color": "black",
    "patch.edgecolor": "black",
    "text.color": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "lightgray",
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white"})

    #points
    los_density = density_scatter(x, z, ax = None, bins = 20)
    tt = np.argsort(los_density)
    
    plt.scatter(x[tt],z[tt],c=los_density[tt],cmap=red_color,marker='.')
    norm = Normalize(vmin=np.min(los_density),vmax=np.max(los_density))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=red_color))
    cbar.ax.set_ylabel('Density')

    #contours of expected density profile
    delta = 0.01
    _x = np.arange(-10.0, 10.0, 20*0.005)
    _z = np.arange(-2, 2, 4*0.005)
    X, Z = np.meshgrid(_x, _z)
    RHO = np.zeros((len(_x),len(_z)))
    for i, x_axis in enumerate(_x):
        for j, z_axis in enumerate(_z):
            RHO[j][i] = disk_density(abs(x_axis), z_axis, Rd, z0, 0.01)
    CS = plt.contour(X, Z, RHO, colors='w',linewidths=1.0)
    
    #misc
    plt.xlabel('R/R$_{d}$',fontsize=18)
    plt.ylabel('Z/R$_{d}$',fontsize=18)
    #plt.gca().set_aspect('equal')
    #plt.xlim(-12,12)
    #plt.ylim(-12,12)
    plt.savefig('thickdisk_contours'+end_tag+'.png',dpi=300)
    plt.clf()

    #XY########################################################
    los_density = density_scatter(x, y, ax = None, bins = 20)
    tt = np.argsort(los_density)
    
    plt.scatter(x[tt],y[tt],c=los_density[tt],cmap=red_color,marker='.')
    norm = Normalize(vmin=np.min(los_density),vmax=np.max(los_density))
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=red_color))
    cbar.ax.set_ylabel('Density')

    plt.xlabel('X/R$_{d}$',fontsize=18)
    plt.ylabel('Y/R$_{d}$',fontsize=18)
    #plt.gca().set_aspect('equal')
    plt.axhline(y=0,c='w',lw=1)
    plt.axvline(x=0,c='w',lw=1)
    #plt.xlim(-12,12)
    #plt.ylim(-12,12)
    plt.savefig('thickdisk_xy'+end_tag+'.png',dpi=300)
    plt.clf()

    #Colour by V###############################################
    #points
    #tt = np.argsort(y) 
    print(len(vy),len(x), len(z))
    my_norm = Normalize(vmin=np.min(vy),vmax=np.max(vy))
    map = cm.ScalarMappable(norm=my_norm,cmap='jet')
    plt.scatter(x,z,c=vy,cmap='jet',marker='.')
    cbar = plt.colorbar(cm.ScalarMappable(norm=my_norm, cmap='jet'))
    cbar.ax.set_ylabel('V$_{los}$')

    #misc
    plt.axvline(x=1.0,c='w',linestyle='--')
    plt.axvline(x=-1.0,c='w',linestyle='--')
    plt.xlabel('X/R$_{d}$',fontsize=18)
    plt.ylabel('Z/R$_{d}$',fontsize=18)
    #plt.gca().set_aspect('equal')
    #plt.xlim(-12,12)
    #plt.ylim(-12,12)
    plt.savefig('thickdisk_edgeon_vlos'+end_tag+'.png',dpi=300)
    plt.clf()

    #points
    tt = np.argsort(z)
    my_norm = Normalize(vmin=np.min(v_varphi),vmax=np.max(v_varphi))
    map = cm.ScalarMappable(norm=my_norm,cmap='jet')
    plt.scatter(x[tt],y[tt],c=v_varphi[tt],cmap='jet',marker='.')
    cbar = plt.colorbar(cm.ScalarMappable(norm=my_norm, cmap='jet'))
    cbar.ax.set_ylabel('V$_{\phi}$')

    #misc
    plt.xlabel('X/R$_{d}$',fontsize=18)
    plt.ylabel('Y/R$_{d}$',fontsize=18)    
    #plt.xlim(-12,12)
    #plt.ylim(-12,12)
    plt.savefig('thickdisk_faceon_vPhi'+end_tag+'.png',dpi=300)
    plt.clf()    
    return



#MAIN#######################
if __name__ == '__main__':
  calculate()
