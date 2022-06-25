import os, glob, time, shutil

charged = " Charge =  1 Multiplicity = 1"
uncharged = " Charge =  0 Multiplicity = 1"

def readin(fname):
    f = open(fname)
    lines = f.read()
    f.close()
    return lines

def chargechecker(folder, fname):
    # Get list of atoms
    name = fname.split("\\")[-1].replace(".log", "")
    print(name)
    log = f"{name}.log"


    one_folder = os.path.join(folder, "Charged")
    zero_folder = os.path.join(folder, "Uncharged")

    lines = readin(fname)

    if charged in lines:
        subsubfolder = one_folder
    elif uncharged in lines: # Best not to use elif if you can use else
        subsubfolder = zero_folder
    else:
        print(f"Error with {name}")
        subsubfolder = folder
        exit

    if os.path.exists(f"{folder}\\{log}"):
        shutil.move(f"{folder}\\{log}", f"{subsubfolder}\\{log}")

if __name__ == "__main__":
    tasks = glob.glob("AJD_files\\*.log")
    topfolder = os.path.join(os.path.abspath("."), "AJD_files")

    for subfolder in ["Charged", "Uncharged"]:
        if not os.path.exists(f"AJD_files\\{subfolder}"):
            os.mkdir(f"AJD_files\\{subfolder}")

    for job in tasks:
        chargechecker(topfolder, job)
