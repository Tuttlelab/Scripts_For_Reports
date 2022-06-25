# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:59:52 2022

@author: rossc
"""
import os, sys, datetime, glob, time, shutil

def readin(fname):
    f = open(fname)
    lines = f.read()
    f.close()
    return lines

def IrChecker(folder, fname):
    
    print(f"{folder}\{fname}")
    
    single_folder = os.path.join(folder, "single")
    multiple_folder = os.path.join(folder, "multiple")
    
    lines = readin(f"{folder}/{fname}").split("\n")
    count = 0
    
    for line in lines:
        if line.startswith("Ir"):
            count += 1
        else:
            continue
    
    if count == 1:
        subsubfolder = single_folder
    else:
        subsubfolder = multiple_folder
        
    if os.path.exists(f"{folder}\{fname}"):
        shutil.move(f"{folder}\{fname}", f"{subsubfolder}\{fname}")
        print(f"No of Ir = {count}. {fname} moved to {subsubfolder}")
        
if __name__ == "__main__":
    tasks = glob.glob("CSD\\*.xyz")
    folder = os.path.join(os.path.abspath("."), "CSD")

    for subfolder in ["single", "multiple"]:
        if not os.path.exists(f"CSD\{subfolder}"):
            os.mkdir(f"CSD\{subfolder}")
        else:
            continue

    for job in tasks:
        IrChecker(folder, os.path.basename(job))

            