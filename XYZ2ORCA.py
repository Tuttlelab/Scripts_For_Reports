# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:57:18 2022

@author: bwb16179
"""
import os, sys, datetime, glob, time
from ase.io import read, write
import numpy as np


ElectronNos = {
    "H" : 1,
    "B" : 5,
    "C" : 6,
    "N" : 7,
    "O" : 8,
    "F" : 9,
    "P" : 15,
    "S" : 16,
    "Cl" : 17,
    "Ir" : 77
    }



def buffer(string, L, end=True):
    while len(string) < L:
        string = string+"0"
    return string


CPU = """#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name={}
#SBATCH --account=tuttle-rmss
#SBATCH --partition=standard
#SBATCH --time="{}"
#SBATCH --ntasks={} --nodes=1
module purge
module load orca/5.0.3

/opt/software/scripts/job_prologue.sh 

"""

def xyz2orca(fname):
    frames = read(fname, index=":")
    if len(frames) == 0:
        return False
    name = fname.split("/")[-1].replace(".xyz", "")
    species = frames[0].get_chemical_symbols()
    print("".join(species))
    folder = os.path.dirname(fname)
    
    CPUS = 1
    
    strT = '240:00:00'
    print(strT)
    
    for i,frame in enumerate(frames):
        jobname = fname.replace(".xyz", ".inp")
        inp = jobname.split("/")[-1]
        out = jobname.split("/")[-1].replace(".inp", ".out")
        if os.path.exists(folder+"/"+inp):
            print(f"Found: {out}, skipping.")
            return 0
        sbatch = open(jobname.replace(".inp", ".sh"), 'w')
        sbatch.write(CPU.format(jobname.replace(".inp", ""), strT, str(CPUS)))
        
        # Check Multiplicity of Molecule
        
        electrons = 0
        
        for coord, atom in zip(frame.get_positions(), species):
            if atom == "H":
                electrons += ElectronNos["H"]
            if atom == "B":
                electrons += ElectronNos["B"]
            if atom == "C":
                electrons += ElectronNos["C"]
            if atom == "N":
                electrons += ElectronNos["N"]
            if atom == "O":
                electrons += ElectronNos["O"]
            if atom == "F":
                electrons += ElectronNos["F"]
            if atom == "P":
                electrons += ElectronNos["P"]
            if atom == "S":
                electrons += ElectronNos["S"]
            if atom == "Cl":
                electrons += ElectronNos["Cl"]
            if atom == "Ir":
                electrons += ElectronNos["Ir"]
        
        if electrons % 2 == 1:
            multiplicity = 2
        else:
            multiplicity = 1
            
        f = open(jobname, 'w')
        print(jobname)
        f.write(f"""# ORCAForces - {jobname}
# Basic Mode
#
! OPT wB97X D4 Def2-TZVPP SlowConv TightSCF DEFGRID2 Def2/J RIJCOSX Smallprint

%maxcore 3375

%geom 
Maxiter 2000
END

* xyz 0 {multiplicity}
""".format(name))
        
        for coord, atom in zip(frame.get_positions(), species):
            line = [" "]*51
            line[0] = atom
            line[11:19] = buffer(str(coord[0]), 8)
            line[27:35] = buffer(str(coord[1]), 8)
            line[43:51] = buffer(str(coord[2]), 8)
            f.write("".join(line))
            f.write("\n")
        f.write("*") 
        f.close()
        
        sbatch.write("/opt/software/orca/5.0.3/orca {} > {}\n".format(inp, out))
        break
    
    sbatch.write("\n/opt/software/scripts/job_epilogue.sh\n")
    sbatch.close()
    
    print(f"Electrons = {electrons}")
    print(f"Multiplicity = {multiplicity}")
    cdir = os.path.abspath(".")
    os.chdir(folder)
    cmd = "sbatch {}".format(os.path.basename(jobname).replace(".inp", ".sh"))
    print(cmd)
    os.system(cmd)
    os.chdir(cdir)
    return True

    
def FilterChemistry(atomlist):
    species_order = ["H", "C", "N", "O", "F",  "P", "S", "Cl", "Ir"] # MUST BE ORDERED BY ATOMIC NUMBER
    for atom in atomlist:
        if atom not in species_order:
            return False
    return True
    
    

if __name__ == "__main__":
    tasks = glob.glob("PTM/*.xyz") + glob.glob("CSD/*.xyz") + glob.glob("AJD/*.xyz")
    ordered_tasks = []
    ordered_tasks_natoms = []
    tasks_symbols = {}
    skipped = 0
    
    for task in tasks:
        try:
            mol = read(task)
        except:
            continue
        ordered_tasks.append(task)
        ordered_tasks_natoms.append(len(mol.get_chemical_symbols()))
        tasks_symbols[task] = mol.get_chemical_symbols()
    
    ordered_tasks = np.array(ordered_tasks)
    ordered_tasks_natoms = np.array(ordered_tasks_natoms)
    order_small_to_large = np.argsort(ordered_tasks_natoms)
    for task in ordered_tasks[order_small_to_large]:
        if FilterChemistry(tasks_symbols[task]):
            print(task, len(tasks_symbols[task]))
            xyz2orca(task)
        else:
            skipped += 1
            print(task, len(tasks_symbols[task]), "skipping because it contains a bad atom")
            
        print(f"{skipped} jobs were skipped due to bad atoms")