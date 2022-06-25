# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 16:30:03 2022

@author: rossc
"""

import os, sys, datetime, glob, time
from ase.io import read, write
import numpy as np

CPU = """
#======================================================
#
# Job script for running a serial job on parallel core 
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=standard
#
# Job name
#SBATCH --job-name={}
#
# Specify project account
#SBATCH --account=tuttle-rmss
#
# No. of tasks required (ntasks=1 for a single-core job)
# Here, tasks = cores (40 cores per node)
#
#SBATCH --ntasks={} --nodes=1
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time="{}"
#
#======================================================

module purge

#example module load command  
module laod orca/5.0.1  
module load orca/5.0.2

#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

"""

def readin(fname):
    f = open(fname)
    content = f.read()
    f.close()
    return content

def out2inps(fname):
    CPUS = 4
    strT = '240:00:00'    
    
    name = fname.split("/")[-1].replace(".out", "")
    folder = os.path.dirname(fname)
    
    #species = frames[0].get_chemical_symbols()
    #print("".join(species))

    content = readin(fname)
    
    for i,frame in enumerate(content.split("CARTESIAN COORDINATES (ANGSTROEM)")[1:]):
        jobname = fname.replace(".out", f"_opt_step_{i}.inp")
        newOutName = jobname.replace(".inp", ".out")
        shname = jobname.replace(".inp", ".sh")
        QueueName = os.path.basename(jobname).replace(".inp", "")
        print(jobname, shname)

        if os.path.exists(jobname):
            continue
        
        sbatch = open(shname, 'w')
        sbatch.write(CPU.format(QueueName, str(CPUS), strT))
        
        
        cartesianSection = frame.split("CARTESIAN COORDINATES (A.U.)")[0]
        #Filter for just the lines that contain atoms (length = 4)
        cartesian = []
        for line in cartesianSection.split("\n"):
            if len(line.strip().split()) == 4:
                cartesian.append(line)
        #Turn this list of lines into one string of text with a newline between each item in the list
        cartesian = "\n".join(cartesian)
        
        file = open(jobname, 'w')
        file.write(f"""# ORCAForces - {jobname}
# Basic Mode
#
! OPT wB97X D4 Def2-tzvpp  SlowConv TightSCF DEFGRID2  Def2/J RIJCOSX
%maxcore 3375
%PAL NPROCS {CPUS} END
    
* xyz 0 1
""".format(name))
        #print(cartesian)
        file.write(cartesian)
        file.write("\n")
        file.write("*")
        file.close()
        
        sbatch.write("/opt/software/orca/5.0.2/orca {} > {}\n".format(os.path.basename(jobname), os.path.basename(newOutName)))
        sbatch.write("\n/opt/software/scripts/job_epilogue.sh\n")
        sbatch.close()
        
        cdir = os.path.abspath(".")
        os.chdir(folder)
        cmd = "sbatch {}".format(os.path.basename(jobname).replace(".inp", ".sh"))
        print("RUNNING:", cmd)
        print(cmd)
        #os.system(cmd)
        os.chdir(cdir)

        

if __name__ == "__main__":       
    tasks = glob.glob("../engrad/Alex/RandomLigands/*/*.out")
    completed_jobs = []
    for task in tasks:
        with open(task, "r") as f:
            #Validate that this is an optimization that contains all the frames
            content = f.read()
            if "* Geometry Optimization Run *" not in content:
                print(task, "This is not a geometry optimization")
                continue
            if "ReducePrint".upper() not in content.upper():
                print(task, "This is job did not have ReducePrint set to false")
                continue
            completed_jobs.append(task)
            
    for task in completed_jobs:
        print(task)
        #out2inps(task)
