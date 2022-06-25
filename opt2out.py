# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:39:34 2022

@author: rossc
"""

import os, sys, glob
import numpy as np
from io import StringIO

def readin(fname):
    f = open(fname)
    content = f.read()
    f.close()
    return content

def atom_list_out(outfile):
    # Get list of atoms
    outcontent = readin(outfile)
    atomlist = []
    outcontent = outcontent.split("CARTESIAN COORDINATES (ANGSTROEM)")[1]
    outcontent = outcontent.split("CARTESIAN COORDINATES (A.U.)")[0]
    for line in outcontent.split("\n"):
        line = line.split()
        if len(line) != 4:
            continue
        atomlist.append(line[0])
    return atomlist

def get_opt(optfile):
    # coordinates
    content = readin(optfile)
    coord_data = content.split("$coordinates\n")[1].split("#")[0]
    line0 = coord_data.split("\n")[0]
    frames, atoms = line0.strip().split()
    frames, atoms = int(frames), int(int(atoms)/3)
    coord_data = " ".join(coord_data.split("\n")[1:]) #remove first line & and remove all newlines to make it one long line
    coord_data = coord_data.strip().split() # make into list of texts
    coord_data = np.array(coord_data) # convert list to numpy array
    coord_data = coord_data.astype(np.float64) # convert texts to numbers
    coordinates = np.ndarray((frames, atoms, 3))
    for i,frame in enumerate(np.split(coord_data, frames)): # Split it into the known number of frames we have
        # each frame has the shape atoms*3, we want to convert it to (atoms, 3)
        frame = np.vstack(np.split(frame, atoms))
        coordinates[i] = frame
    
    content = readin(optfile)
    energy_data = content.split("$energies\n")[1].split("\n\n")[0]
    energy_data = " ".join(energy_data.split("\n")[1:]) #remove first line & and remove all newlines to make it one long line
    energy_data = energy_data.strip().split() # make into list of texts
    energy_data = np.array(energy_data) # convert list to numpy array
    energies = energy_data.astype(np.float64) # convert texts to numbers

    content = readin(optfile)
    gradients_data = content.split("$gradients\n")[1].split("$")[0]
    gradients_data = " ".join(gradients_data.split("\n")[1:]) #remove first line & and remove all newlines to make it one long line
    gradients_data = gradients_data.strip().split() # make into list of texts
    gradients_data = np.array(gradients_data) # convert list to numpy array
    gradients_data = gradients_data.astype(np.float64) # convert texts to numbers
    forces = np.ndarray((frames, atoms, 3))    
    for i,frame in enumerate(np.split(gradients_data, frames)): # Split it into the known number of frames we have
        # each frame has the shape atoms*3, we want to convert it to (atoms, 3)
        frame = np.vstack(np.split(frame, atoms))
        forces[i] = frame

    return frames, atoms, energies, coordinates, forces


system = "methane"

atomlist = atom_list_out(f"{system}/{system}.out")
frames, atoms, energies, coordinates, forces = get_opt(f"{system}/{system}.opt")

a="""
# Convert atomic units to our units
coordinates *= 0.52917724900001
forces *= 0.52917724900001

with open(f"{system}/{system}_opt_forces_trj.xyz", 'w') as xyzout:
    for i in range(coordinates.shape[0]):
        xyzout.write(f"{atoms}\nEnergy: {energies[i]}\n")
        
        for j in range(coordinates.shape[1]):
            xyzout.write(f"{atomlist[j]} {coordinates[i][j][0]} {coordinates[i][j][1]} {coordinates[i][j][2]} {forces[i][j][0]} {forces[i][j][1]} {forces[i][j][2]}\n")
#"""