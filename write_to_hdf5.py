import numpy as np
import h5py

def just_write_disk_pcls(pos_stars, vel_stars, m_stars, file_out, noz=False):
        Nstars = len(m_stars)

        if noz:
                #Transpose array
                vel_stars_T = vel_stars.T; pos_stars_T = pos_stars.T
                #remove all z components, make them 0.0
                vel_stars_T[2] = np.zeros(len(vel_stars_T[2]))
                pos_stars_T[2] = np.zeros(len(pos_stars_T[2]))
                #return array to original status
                vel_stars = vel_stars_T.T; pos_stars = pos_stars_T.T
        
	#You'll need to change the ids - they should run from 0 to the number of pcls in the simulation (and currently this would include the Type 1 pcls).
        id_stars = np.arange(1,Nstars+1)
        file = h5py.File(file_out,'w')
        #npart will be included in the hdf5 header, with 6 numbers indicting the number of each pcl type in the file.
        npart = np.array([0,0,Nstars,0,0,0])

	#Write out GADGET headers in hdf5, don't need to change this.
        h = file.create_group("Header")
        h.attrs['NumPart_ThisFile'] = npart    

        h.attrs['NumPart_Total'] = npart # npart set as above
        h.attrs['NumPart_Total_HighWord'] = 0*npart
        h.attrs['MassTable'] = np.zeros(6)    
        h.attrs['Time'] = 0.0
        h.attrs['Redshift'] = 0.0
        h.attrs['BoxSize'] = 1000.
        h.attrs['NumFilesPerSnapshot'] = 1 # number of files for multi-part snapshots
        h.attrs['Omega0'] = 1.0 # z=0 Omega_matter
        h.attrs['OmegaLambda'] = 0.0 # z=0 Omega_Lambda
        h.attrs['HubbleParam'] = 1.0 # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
        h.attrs['Flag_Sfr'] = 1; # flag indicating whether star formation is on or off
        h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
        h.attrs['Flag_StellarAge'] = 1; # flag indicating whether stellar ages are to be saved
        h.attrs['Flag_Metals'] = 1; # flag indicating whether metallicity are to be saved
        h.attrs['Flag_Feedback'] = 1; # flag indicating whether some parts of springel-hernquist model are active
        h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
        h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs
   

	#Now re-write disk pcls into hdf5 file
        p = file.create_group("PartType2")
        p.create_dataset("Velocities",data=vel_stars)
        p.create_dataset("Coordinates",data=pos_stars)
        p.create_dataset("ParticleIDs",data=id_stars)
        p.create_dataset("Masses",data=m_stars)

	# close the HDF5 file, which saves these outputs
        file.close()
	# all done!

def just_write_halo_pcls(pos_halo, vel_halo, m_halo, file_out):
        Nhalo = len(m_halo)

        #You'll need to change the ids - they should run from 0 to the number of pcls in the simulation (and currently this would include the Type 1 pcls).
        id_halo = np.arange(1,Nhalo+1)
        file = h5py.File(file_out,'w')
        #npart will be included in the hdf5 header, with 6 numbers indicting the number of each pcl type in the file.
        npart = np.array([0,Nhalo,0,0,0,0])

	#Write out GADGET headers in hdf5, don't need to change this.
        h = file.create_group("Header")
        h.attrs['NumPart_ThisFile'] = npart    
        
        h.attrs['NumPart_Total'] = npart # npart set as above
        h.attrs['NumPart_Total_HighWord'] = 0*npart
        h.attrs['MassTable'] = np.zeros(6)    
        h.attrs['Time'] = 0.0
        h.attrs['Redshift'] = 0.0
        h.attrs['BoxSize'] = 1000.
        h.attrs['NumFilesPerSnapshot'] = 1 # number of files for multi-part snapshots
        h.attrs['Omega0'] = 1.0 # z=0 Omega_matter
        h.attrs['OmegaLambda'] = 0.0 # z=0 Omega_Lambda
        h.attrs['HubbleParam'] = 1.0 # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
        h.attrs['Flag_Sfr'] = 1; # flag indicating whether star formation is on or off
        h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
        h.attrs['Flag_StellarAge'] = 1; # flag indicating whether stellar ages are to be saved
        h.attrs['Flag_Metals'] = 1; # flag indicating whether metallicity are to be saved
        h.attrs['Flag_Feedback'] = 1; # flag indicating whether some parts of springel-hernquist model are active
        h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
        h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

        #Now re-write disk pcls into hdf5 file
        p = file.create_group("PartType1")
        p.create_dataset("Velocities",data=vel_halo)
        p.create_dataset("Coordinates",data=pos_halo)
        p.create_dataset("ParticleIDs",data=id_halo)
        p.create_dataset("Masses",data=m_halo)

	# close the HDF5 file, which saves these outputs
        file.close()
	# all done!
        
def write_pcls(pos_stars, vel_stars, m_stars, pos_dm, vel_dm, m_dm, file_out):
        Nstars = len(m_stars)
        Ndm = len(m_dm)
        # unique IDs across types: halo 1..Ndm, disc Ndm+1..Ndm+Nstars
        id_dm = np.arange(1,Ndm+1)
        id_stars = np.arange(Ndm+1,Ndm+Nstars+1)
        
        file = h5py.File(file_out,'w')

        npart = np.array([0,Ndm,Nstars,0,0,0])

        #Write out GADGET headers in hdf5, don't need to change this.
        h = file.create_group("Header")
        h.attrs['NumPart_ThisFile'] = npart
        h.attrs['NumPart_Total'] = npart # npart set as above
        h.attrs['NumPart_Total_HighWord'] = 0*npart
        h.attrs['MassTable'] = np.zeros(6)
        h.attrs['Time'] = 0.0
        h.attrs['Redshift'] = 0.0
        h.attrs['BoxSize'] = 1000.
        h.attrs['NumFilesPerSnapshot'] = 1 # number of files for multi-part snapshots
        h.attrs['Omega0'] = 1.0 # z=0 Omega_matter
        h.attrs['OmegaLambda'] = 0.0 # z=0 Omega_Lambda
        h.attrs['HubbleParam'] = 1.0 # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
        h.attrs['Flag_Sfr'] = 1; # flag indicating whether star formation is on or off
        h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
        h.attrs['Flag_StellarAge'] = 1; # flag indicating whether stellar ages are to be saved
        h.attrs['Flag_Metals'] = 1; # flag indicating whether metallicity are to be saved
        h.attrs['Flag_Feedback'] = 1; # flag indicating whether some parts of springel-hernquist model are active
        h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
        h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

        #Now re-write disk pcls into hdf5 file
        p = file.create_group("PartType2")
        p.create_dataset("Velocities",data=vel_stars)
        p.create_dataset("Coordinates",data=pos_stars)
        p.create_dataset("ParticleIDs",data=id_stars)
        p.create_dataset("Masses",data=m_stars)

        p = file.create_group("PartType1")
        p.create_dataset("Velocities",data=vel_dm)
        p.create_dataset("Coordinates",data=pos_dm)
        p.create_dataset("ParticleIDs",data=id_dm)
        p.create_dataset("Masses",data=m_dm)

        file.close()
        # all done! 

def write_pcls_wgas(pos_stars, vel_stars, m_stars,
                    pos_dm, vel_dm, m_dm,
                    pos_gas, vel_gas, m_gas,
                    file_out, u_gas=None):
        '''Gas (PartType0), halo (PartType1) and disc (PartType2).
        u_gas: specific internal energy per gas particle in (velocity unit)^2, written as
        'InternalEnergy' (Gadget uses it when InitGasTemp = 0).'''
        Nstars = len(m_stars)
        Ndm = len(m_dm)
        Ngas = len(m_gas)
        # unique IDs across types: gas 1..Ngas, halo next, disc last
        id_gas = np.arange(1, Ngas+1)
        id_dm = np.arange(Ngas+1, Ngas+Ndm+1)
        id_stars = np.arange(Ngas+Ndm+1, Ngas+Ndm+Nstars+1)

        file = h5py.File(file_out,'w')

        npart = np.array([Ngas,Ndm,Nstars,0,0,0])

        #Write out GADGET headers in hdf5, don't need to change this.
        h = file.create_group("Header")
        h.attrs['NumPart_ThisFile'] = npart
        h.attrs['NumPart_Total'] = npart # npart set as above
        h.attrs['NumPart_Total_HighWord'] = 0*npart
        h.attrs['MassTable'] = np.zeros(6)
        h.attrs['Time'] = 0.0
        h.attrs['Redshift'] = 0.0
        h.attrs['BoxSize'] = 1000.
        h.attrs['NumFilesPerSnapshot'] = 1 # number of files for multi-part snapshots
        h.attrs['Omega0'] = 1.0 # z=0 Omega_matter
        h.attrs['OmegaLambda'] = 0.0 # z=0 Omega_Lambda
        h.attrs['HubbleParam'] = 1.0 # z=0 hubble parameter (small 'h'=H/100 km/s/Mpc)
        h.attrs['Flag_Sfr'] = 1; # flag indicating whether star formation is on or off
        h.attrs['Flag_Cooling'] = 0; # flag indicating whether cooling is on or off
        h.attrs['Flag_StellarAge'] = 1; # flag indicating whether stellar ages are to be saved
        h.attrs['Flag_Metals'] = 1; # flag indicating whether metallicity are to be saved
        h.attrs['Flag_Feedback'] = 1; # flag indicating whether some parts of springel-hernquist model are active
        h.attrs['Flag_DoublePrecision'] = 0; # flag indicating whether ICs are in single/double precision
        h.attrs['Flag_IC_Info'] = 0; # flag indicating extra options for ICs

        #Now re-write disk pcls into hdf5 file
        p = file.create_group("PartType2")
        p.create_dataset("Velocities",data=vel_stars)
        p.create_dataset("Coordinates",data=pos_stars)
        p.create_dataset("ParticleIDs",data=id_stars)
        p.create_dataset("Masses",data=m_stars)

        p = file.create_group("PartType1")
        p.create_dataset("Velocities",data=vel_dm)
        p.create_dataset("Coordinates",data=pos_dm)
        p.create_dataset("ParticleIDs",data=id_dm)
        p.create_dataset("Masses",data=m_dm)

        p = file.create_group("PartType0")
        p.create_dataset("Velocities",data=vel_gas)
        p.create_dataset("Coordinates",data=pos_gas)
        p.create_dataset("ParticleIDs",data=id_gas)
        p.create_dataset("Masses",data=m_gas)
        if u_gas is not None:
                p.create_dataset("InternalEnergy",data=u_gas)

        file.close()
        # all done! 

def write_snapshot(parts, file_out):
        '''General Gadget HDF5 IC writer.
        parts: dict {ptype: dict(pos=, vel=, mass=[, u=])}, ptype in 0..5.
        Particle IDs are unique across types, assigned in increasing ptype order starting at 1.
        Returns {ptype: first ParticleID of that type}.'''
        npart = np.zeros(6, dtype=np.int64)
        for t, d in parts.items():
                npart[t] = len(d['mass'])
        file = h5py.File(file_out, 'w')
        h = file.create_group("Header")
        h.attrs['NumPart_ThisFile'] = npart
        h.attrs['NumPart_Total'] = npart
        h.attrs['NumPart_Total_HighWord'] = 0*npart
        h.attrs['MassTable'] = np.zeros(6)
        h.attrs['Time'] = 0.0
        h.attrs['Redshift'] = 0.0
        h.attrs['BoxSize'] = 1000.
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['Omega0'] = 1.0
        h.attrs['OmegaLambda'] = 0.0
        h.attrs['HubbleParam'] = 1.0
        h.attrs['Flag_Sfr'] = 1
        h.attrs['Flag_Cooling'] = 0
        h.attrs['Flag_StellarAge'] = 1
        h.attrs['Flag_Metals'] = 1
        h.attrs['Flag_Feedback'] = 1
        h.attrs['Flag_DoublePrecision'] = 0
        h.attrs['Flag_IC_Info'] = 0
        first, nextid = {}, 1
        for t in sorted(parts):
                d = parts[t]
                n = len(d['mass'])
                first[t] = nextid
                p = file.create_group("PartType%d" % t)
                p.create_dataset("Coordinates", data=d['pos'])
                p.create_dataset("Velocities", data=d['vel'])
                p.create_dataset("ParticleIDs", data=np.arange(nextid, nextid + n))
                p.create_dataset("Masses", data=d['mass'])
                if t == 0 and d.get('u') is not None:
                        p.create_dataset("InternalEnergy", data=d['u'])
                nextid += n
        file.close()
        return first
