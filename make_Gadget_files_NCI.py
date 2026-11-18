import numpy as np

from scipy.special   import kn, iv
from scipy.special   import lambertw
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize  import brentq

import matplotlib.pylab as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

########################################################################


def work_out_all_inputs(MD_in, C_in,  save=False):
    '''Makes Gadget param.txt and Submit.sh files
    '''
    model_name = 'C%s_MD%s'%(str(round(C_in,3)).replace('.','p'),str(round(MD_in,3)).replace('.','p'))

    C_str = str(round(C_in,3)).replace('.','p')
    MD_str = str(round(MD_in,3)).replace('.','p')

    #Make submit file (for magnus)
    submit_file = '''#!/bin/bash
    #PBS -P ft29
    #PBS -N 'C{0}_MD{1}_seed3'
    #PBS -q normal
    #PBS -l walltime=24:00:00
    #PBS -l mem=100GB
    #PBS -l ncpus=48
    #PBS -l wd
    #PBS -l storage=scratch/ft29+gdata/ft29
    
    module load openmpi
    module load gsl
    module load hdf5

    echo Start from $(date +%T.%3N)
    mpirun -np $PBS_NCPUS /scratch/ft29/mf9832/gadget3/executable_accel /scratch/ft29/mf9832/Submit/param_C{0}_MD{1}.txt
    echo End from $(date +%T.%3N)
    '''.format(C_str, MD_str)

    #make param_file
    param_file = '''%Relevant files

InitCondFile /home/571/mf9832/Models/suite1/seed3/fs{1}_C{0}
OutputDir    /scratch/ft29/mf9832/suite1_new_accel/seed3/MD{1}/C{0}/

EnergyFile        energy.txt
InfoFile          info.txt
TimingsFile       timings.txt
CpuFile           cpu.txt

RestartFile       restart
SnapshotFileBase  snapshot

OutputListFilename  %/scratch/ft29/mf9832/Submit/output_times.txt  %Not needed

% CPU-time limit
TimeLimitCPU     86400 #Set to under 48 hrs, in s
ResubmitOn        0
ResubmitCommand   %Can use this to autorestart, if ResubmitOn == 1

% Code options
ICFormat                 3   % 3 = hdf5 (Matt - switch to 'long IDs'), 1 == Kate
ComovingIntegrationOn    0
CoolingOn                0
TypeOfTimestepCriterion  0
OutputListOn             0
PeriodicBoundariesOn     0
StarformationOn          0

%  Characteristics of run
TimeBegin             0.
TimeMax               200.0 % 200 1000 = good bar evolution (for 2d, otherwise may need to be raised)
Omega0                1. %params ignored for idealised run
OmegaLambda           0. %
OmegaBaryon           0. %
HubbleParam           1. %
BoxSize               1.e5 %also ignored for idealised run

% Output frequency
TimeBetSnapshot        1 %
TimeOfFirstSnapshot    0.0

CpuTimeBetRestartFile     600.0    ; here in seconds
TimeBetStatistics         0.001

% Accuracy of time integration
ErrTolIntAccuracy      0.01    % used for TypeOfTimestepCritrion==0
CourantFac             0.15       % for SPH
MaxSizeTimestep        0.05    %often 0.01, but higher = faster
MinSizeTimestep        0.0

% Tree algorithm, force accuracy, domain update frequency
ErrTolTheta            0.3
TypeOfOpeningCriterion 1
ErrTolForceAcc         0.025

%  Further parameters of SPH
DesNumNgb              200.
ArtBulkViscConst       1.
ArtBulkDissConst       1.
InitGasTemp            0.          % always ignored if set to 0
MinGasTemp             1000.

% Memory allocation
PartAllocFactor        50.

% System of units
UnitLength_in_cm         3.085678e21        %  1.0 kpc/h
UnitMass_in_g            1.989e41           %  1.0e8/h solar masses
UnitVelocity_in_cm_per_s 9.777922e5         %  1 km/sec - conversion from magi units
GravityConstantInternal  0.

    
% Softening lengths
MinGasHsmlFractional     1. % minimum softening in terms of the gravitational softening length

%change softenings as required.
SofteningGas       0.125 
SofteningHalo      0.125 
SofteningDisk      0.125 
SofteningBulge     0.125  %used as photon/virtual particles
SofteningStars     0.125 
SofteningBndry     0.125 

SofteningGasMaxPhys       0.125 
SofteningHaloMaxPhys      0.125
SofteningDiskMaxPhys      0.125 
SofteningBulgeMaxPhys     0.125  %sets maximum search radius for photons
SofteningStarsMaxPhys     0.125 
SofteningBndryMaxPhys     0.125 

% New Parameters
SnapFormat                         3 %hdf5
TreeDomainUpdateFrequency          0.1
MaxRMSDisplacementFac              0.125
MaxNumNgbDeviation                 2
NumFilesPerSnapshot                1
NumFilesWrittenInParallel          1
BufferSize                         100
    
    '''.format(C_str,MD_str)

    
    if save:
        with open('./param_'+ model_name +'.txt', 'w') as model:
            model.write(param_file)

        with open('./Submit_'+ model_name + '.sh', 'w') as model:
            model.write(submit_file)
    
    return

if __name__ == '__main__':
    MD = [0.03]#[0.01, 0.02, 0.03, 0.04, 0.05]
    C = [10.0]#[5.0, 7.5, 10.0, 15.0, 20.0]
    for i in range(len(MD)):
        for j in range(len(C)):
            work_out_all_inputs(MD[i], C[j], save=True)
    pass

