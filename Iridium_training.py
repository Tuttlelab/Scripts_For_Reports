# -*- coding: utf-8 -*-


###############################################################################
# To begin with, let's first import the modules and setup devices we will use:
import matplotlib, os, sys
if os.name != "nt":
    matplotlib.use('Agg')
import platform
import torch
import torchani
import torchani.nn as TNN
import math, sys, pandas, h5py, pickle, time
#import torch.utils.tensorboard
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from multiprocessing import freeze_support
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

np.random.seed(int(time.time()))

print("Make sure to be running this in the A100_2 env!!!")

config = {
    "AllowCUDA": True,
    "KMP_DUPLICATE_LIB_OK": "True",
    "MultiGPU": False,
    "batch_size": 32,
    "dspath": "N=17228.h5",
    "hard_reset": False,
    "verbose_file": None,
    "logfile": "Training.log",
    "model": 1,
    "reset_lr": True,
    "force_coefficient": 0,
    "preAEV": False,
    "GraphEveryEpochs": 10,
    #"OutputFolder": "../N_vs_error/2000"
}



if config["hard_reset"] == True:
    print("Performing a HARD RESET")
    config["hard_reset"] = False
    if type(config["logfile"]) == str and os.path.exists(config["logfile"]):
        os.remove(config["logfile"])
    for file in [f"{config['OutputFolder']}/Training.log", f"{config['OutputFolder']}/DNN_training.png",
                 f"{config['OutputFolder']}/best.pt", f"{config['OutputFolder']}/latest.pt", f"{config['OutputFolder']}/Verbose.log"]:
        if os.path.exists(file):
            os.remove(file)

class Logger:
    def __init__(self, logfile=None, verbose=True):
        if type(logfile) == str:
            self.Logfile = open(logfile, 'w')
        else:
            self.Logfile = False
        
        self.verbose = bool(verbose)
    
    def Log(self, string):
        string = str(string)
        if self.Logfile != False:
            self.Logfile.write(string)
            self.Logfile.write("\n")
            self.Logfile.flush()
        if self.verbose:
            print(string)
        
    def close(self):
        if self.Logfile != False:
            self.Logfile.close()
            self.Logfile = False
            


Log = Logger(f"{config['OutputFolder']}/config['verbose_file']" if type(config["verbose_file"]) == str else None, verbose=True)
    
    
os.environ["KMP_DUPLICATE_LIB_OK"] = config["KMP_DUPLICATE_LIB_OK"]
plt.ioff()

try:
    reset_lr = config["reset_lr"]
    latest_checkpoint = config["restart_latest"]
except:
    reset_lr = False
    latest_checkpoint = "latest.pt"
    
Log.Log("reset_lr: "+str(reset_lr))
Log.Log("latest_checkpoint:"+latest_checkpoint)


# device to run the training
Log.Log("CUDA availibility: "+str(torch.cuda.is_available()))
if config["AllowCUDA"] == False or torch.cuda.is_available() == False:
    device = torch.device('cpu')   
    print("FORCING TO CPU")
else:
    device = torch.device('cuda')
Log.Log("Running on device: "+str(device))


Rcr = 5.2000e+00  # Cut-off
Rca = 3.5000e+00
EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.1687500e+00, 1.4375000e+00, 1.7062500e+00, 1.9750000e+00, 2.2437500e+00, 2.5125000e+00, 2.7812500e+00, 3.0500000e+00, 3.3187500e+00, 3.5875000e+00, 3.8562500e+00, 4.1250000e+00, 4.3937500e+00, 4.6625000e+00, 4.9312500e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.5500000e+00, 2.8500000e+00, 2.2000000e+00], device=device)

species_order = ["H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Ir"] # MUST HAVE ONE OF EACH
num_species = len(species_order)
cuaev = False if str(device) == "cpu" else True
Log.Log("cuaev: "+str(cuaev))
#aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species, use_cuda_extension=cuaev)
aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
energy_shifter = torchani.utils.EnergyShifter(None)


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()

dspath = config["dspath"]
batch_size = config["batch_size"]


st = time.time()


#a="""
pickled_dataset_path = dspath.replace(".h5", "")
pickled_training = pickled_dataset_path+"_"+str(config["model"])+"_training.pkl"
pickled_validation = pickled_dataset_path+"_"+str(config["model"])+"_validation.pkl"
pickled_SelfEnergies = pickled_dataset_path+"_"+str(config["model"])+"_SelfEnergies.pkl"

if not os.path.exists(dspath) and not os.path.exists(pickled_training):
    print("dataset path does not exist! Exiting.")
    sys.exit()

# We pickle the dataset after loading to ensure we use the same validation set
# each time we restart training, otherwise we risk mixing the validation and
# training sets on each restart.

st = time.time()
if os.path.isfile(pickled_training):
    print(f'Unpickling preprocessed dataset found in {pickled_SelfEnergies}')
    energy_shifter.self_energies = pickle.load(open(pickled_SelfEnergies, 'rb')).to(device)

    print(f'Unpickling preprocessed dataset found in {pickled_validation}')
    validation = pickle.load(open(pickled_validation, 'rb')).collate(config["batch_size"]).cache()

    print(f'Unpickling preprocessed dataset found in {pickled_training}')
    training = pickle.load(open(pickled_training, 'rb')).collate(config["batch_size"]).cache()  

    x =  round(time.time()-st, 3)
    print("Dataset "+pickled_training+" already made")
else:    
    print(f"Processing dataset: {dspath}")
    training, validation = torchani.data.load(dspath, additional_properties=('forces',))\
                                        .subtract_self_energies(energy_shifter, species_order)\
                                        .species_to_indices(species_order)\
                                        .shuffle()\
                                        .split(0.8, None)

    with open(pickled_training, 'wb') as f:
        pickle.dump(training, f)
    with open(pickled_validation, 'wb') as f:
        pickle.dump(validation, f)
    with open(pickled_SelfEnergies, 'wb') as f:
        pickle.dump(energy_shifter.self_energies.cpu(), f)

    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()
    x =  round(time.time()-st, 3)
    print(f"Pickled dataset generated and saved in {x} s")
#"""

    
Log.Log('Self atomic energies: '+str(energy_shifter.self_energies))



aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 256),
    torch.nn.CELU(0.1),
    torch.nn.Linear(256, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)
B_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 224),
    torch.nn.CELU(0.1),
    torch.nn.Linear(224, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)
N_network = torch.nn.Sequential(
    torch.nn.Linear(in_features=aev_dim, out_features=192),
    torch.nn.CELU(alpha=0.1),
    torch.nn.Linear(in_features=192, out_features=160),
    torch.nn.CELU(alpha=0.1),
    torch.nn.Linear(in_features=160, out_features=128),
    torch.nn.CELU(alpha=0.1),
    torch.nn.Linear(in_features=128, out_features=1)
)
O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)
F_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
P_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)
Cl_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

Ir_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 144),
    torch.nn.CELU(0.1),
    torch.nn.Linear(144, 112),
    torch.nn.CELU(0.1),
    torch.nn.Linear(112, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

nn = torchani.ANIModel([H_network, C_network, N_network, O_network, P_network, Cl_network, Ir_network])


def init_params(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, a=1.0)
        torch.nn.init.zeros_(m.bias)

nn.apply(init_params)


if config["preAEV"]:
    print("Setting up neural network for pre-cooked AEV dataset")
    model = nn.to(device)
else:
    model = torchani.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Now let's setup the optimizers. NeuroChem uses Adam with decoupled weight decay
# to updates the weights and Stochastic Gradient Descent (SGD) to update the biases.
# Moreover, we need to specify different weight decay rate for different layes.
#
# .. note::
#
#   The weight decay in `inputtrain.ipt`_ is named "l2", but it is actually not
#   L2 regularization. The confusion between L2 and weight decay is a common
#   mistake in deep learning.  See: `Decoupled Weight Decay Regularization`_
#   Also note that the weight decay only applies to weight in the training
#   of ANI models, not bias.
#
# .. _Decoupled Weight Decay Regularization:
#   https://arxiv.org/abs/1711.05101

AdamW = torch.optim.AdamW([
    # H networks
    {'params': [H_network[0].weight]},
    {'params': [H_network[2].weight], 'weight_decay': 0.00001},
    {'params': [H_network[4].weight], 'weight_decay': 0.000001},
    {'params': [H_network[6].weight]},
    # B Networks
    {'params': [B_network[0].weight]},
    {'params': [B_network[2].weight], 'weight_decay': 0.00001},
    {'params': [B_network[4].weight], 'weight_decay': 0.000001},
    {'params': [B_network[6].weight]},
    # C networks
    {'params': [C_network[0].weight]},
    {'params': [C_network[2].weight], 'weight_decay': 0.00001},
    {'params': [C_network[4].weight], 'weight_decay': 0.000001},
    {'params': [C_network[6].weight]},
    # N networks
    {'params': [N_network[0].weight]},
    {'params': [N_network[2].weight], 'weight_decay': 0.00001},
    {'params': [N_network[4].weight], 'weight_decay': 0.000001},
    {'params': [N_network[6].weight]},
    # O networks
    {'params': [O_network[0].weight]},
    {'params': [O_network[2].weight], 'weight_decay': 0.00001},
    {'params': [O_network[4].weight], 'weight_decay': 0.000001},
    {'params': [O_network[6].weight]},
    # F Networks
    {'params': [F_network[0].weight]},
    {'params': [F_network[2].weight], 'weight_decay': 0.00001},
    {'params': [F_network[4].weight], 'weight_decay': 0.000001},
    {'params': [F_network[6].weight]},
    # P networks
    {'params': [P_network[0].weight]},
    {'params': [P_network[2].weight], 'weight_decay': 0.00001},
    {'params': [P_network[4].weight], 'weight_decay': 0.000001},
    {'params': [P_network[6].weight]},
    # S Networks
    {'params': [S_network[0].weight]},
    {'params': [S_network[2].weight], 'weight_decay': 0.00001},
    {'params': [S_network[4].weight], 'weight_decay': 0.000001},
    {'params': [S_network[6].weight]},
    # Cl networks
    {'params': [Cl_network[0].weight]},
    {'params': [Cl_network[2].weight], 'weight_decay': 0.00001},
    {'params': [Cl_network[4].weight], 'weight_decay': 0.000001},
    {'params': [Cl_network[6].weight]},
    # Ir networks
    {'params': [Ir_network[0].weight]},
    {'params': [Ir_network[2].weight], 'weight_decay': 0.00001},
    {'params': [Ir_network[4].weight], 'weight_decay': 0.000001},
    {'params': [Ir_network[6].weight]},

])

SGD = torch.optim.SGD([
    # H networks
    {'params': [H_network[0].bias]},
    {'params': [H_network[2].bias]},
    {'params': [H_network[4].bias]},
    {'params': [H_network[6].bias]},
    # B networks
    {'params': [B_network[0].bias]},
    {'params': [B_network[2].bias]},
    {'params': [B_network[4].bias]},
    {'params': [B_network[6].bias]},
    # C networks
    {'params': [C_network[0].bias]},
    {'params': [C_network[2].bias]},
    {'params': [C_network[4].bias]},
    {'params': [C_network[6].bias]},
    # N networks
    {'params': [N_network[0].bias]},
    {'params': [N_network[2].bias]},
    {'params': [N_network[4].bias]},
    {'params': [N_network[6].bias]},
    # O networks
    {'params': [O_network[0].bias]},
    {'params': [O_network[2].bias]},
    {'params': [O_network[4].bias]},
    {'params': [O_network[6].bias]},
    # F networks
    {'params': [F_network[0].bias]},
    {'params': [F_network[2].bias]},
    {'params': [F_network[4].bias]},
    {'params': [F_network[6].bias]},
    # P networks
    {'params': [P_network[0].bias]},
    {'params': [P_network[2].bias]},
    {'params': [P_network[4].bias]},
    {'params': [P_network[6].bias]},
    # S networks
    {'params': [S_network[0].bias]},
    {'params': [S_network[2].bias]},
    {'params': [S_network[4].bias]},
    {'params': [S_network[6].bias]},
    # Cl networks
    {'params': [Cl_network[0].bias]},
    {'params': [Cl_network[2].bias]},
    {'params': [Cl_network[4].bias]},
    {'params': [Cl_network[6].bias]},
    # Ir networks
    {'params': [Ir_network[0].bias]},
    {'params': [Ir_network[2].bias]},
    {'params': [Ir_network[4].bias]},
    {'params': [Ir_network[6].bias]},
], lr=1e-3)


###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0, verbose=False)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0, verbose=False)


###############################################################################
# Train the model by minimizing the MSE loss, until validation RMSE no longer
# improves during a certain number of steps, decay the learning rate and repeat
# the same process, stop until the learning rate is smaller than a threshold.
#
# We first read the checkpoint files to restart training. We use `latest.pt`
# to store current training state.


###############################################################################
# Resume training from previously saved checkpoints:
if os.path.isfile(latest_checkpoint):
    if device.type == "cpu":
        checkpoint = torch.load(latest_checkpoint, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(latest_checkpoint)
    nn.load_state_dict(checkpoint['nn'])
    AdamW.load_state_dict(checkpoint['AdamW'])
    SGD.load_state_dict(checkpoint['SGD'])
    AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    SGD_scheduler.load_state_dict(checkpoint['SGD_scheduler'])



if reset_lr:
    Log.Log("Reset learning rates, you should only do this at the begining of a continuation!")
    for x in AdamW.param_groups:
        x["lr"] = 1e-3
    for x in SGD.param_groups:
        x["lr"] = 1e-3
    AdamW_scheduler._last_lr=[]
    SGD_scheduler._last_lr=[]
    AdamW_scheduler.best = 10000

def validate():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    force_rmse = 0.0
    count = 0
    force_count = 0
    model.train(False)
    for properties in validation:
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_forces = properties['forces'].to(device).float()
        true_energies = properties['energies'].to(device).float()
        _, predicted_energies = model((species, coordinates))
        total_mse += mse_sum(predicted_energies, true_energies).item()
        
        #Calculate Forces MSE
        #When only some conformers have real true forces we need to delete the filler
        bad_forces_index = torch.where(true_forces > 9999.00)[0]
        real_forces_index = [x for x in np.arange(0, true_forces.shape[0]) if x not in bad_forces_index]
        real_forces_index = np.array(real_forces_index)
        
        forces = -torch.autograd.grad(predicted_energies.sum().squeeze(), coordinates, create_graph=True, retain_graph=True)[0]
        force_rmse += mse_sum(forces[real_forces_index], true_forces[real_forces_index]).item()


        
        count += predicted_energies.shape[0]
        force_count += real_forces_index.shape[0]
    
    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count)), hartree2kcalmol(math.sqrt(force_rmse / force_count)), hartree2kcalmol(total_mse / count), hartree2kcalmol(force_rmse / force_count)


###############################################################################
# We will also use TensorBoard to visualize our training process
#tensorboard = torch.utils.tensorboard.SummaryWriter()

###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')

Log.Log("training starting from epoch " + str(AdamW_scheduler.last_epoch + 1))
max_epochs = 10000
early_stopping_learning_rate = 1.1E-5
best_model_checkpoint = "best.pt"

if reset_lr:
    TrainingLog = config["restart_latest"].replace("latest.pt", "Training.log")
else:
    TrainingLog = "Training.log"

if os.path.exists(TrainingLog) or reset_lr:
    training_log = pandas.read_csv(TrainingLog, index_col=0)
else:
    training_log = pandas.DataFrame(columns=["Epoch", "Energy RMSE", "Force RMSE", "Loss", "Energy MSE", "Force MSE"])
    

# Nothing has crashed up to this point so write out to the config file that we don't want to restart again next time


# After loading from another models we must change the target of latest_checkpoint so that we don't overwrite the checkpoint of another model!
latest_checkpoint = "latest.pt"
print("Reset latest_checkpoint to:", latest_checkpoint)

best_i = 0


for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    energy_rmse, force_rmse, energy_mse, force_mse = validate()
    EF_coef = energy_rmse + (force_rmse * config["force_coefficient"])
    
    Log.Log(f"ENERGY RMSE: {round(energy_rmse, 3)} FORCE RMSE: {round(force_rmse, 3)} kcal/mol ENERGY MSE: {round(energy_mse, 3)} FORCE MSE: {round(force_mse, 3)} kcal/mol EF_coef: {round(EF_coef, 3)} at epoch {AdamW_scheduler.last_epoch + 1}")
    
    learning_rate = AdamW.param_groups[0]['lr']

    if learning_rate < early_stopping_learning_rate:
        Log.Log("learning_rate < early_stopping_learning_rate, exiting...")
        break

    
    # checkpoint
    if AdamW_scheduler.is_better(EF_coef, AdamW_scheduler.best):
        try:
            torch.save(nn.state_dict(), best_model_checkpoint)#.format(AdamW_scheduler.last_epoch + 1))
        except PermissionError: # happens sometimes on windows for no good reason
            torch.save(nn.state_dict(), best_model_checkpoint)
            
        #torch.save(model, "FullModel_Best")

    AdamW_scheduler.step(EF_coef)
    SGD_scheduler.step(EF_coef)
    

    #tqdm module does the progress bar
    #for i, properties in tqdm.tqdm(enumerate(training), total=len(training)): #,desc="epoch {}".format(AdamW_scheduler.last_epoch)
    for i, properties in enumerate(training): #,desc="epoch {}".format(AdamW_scheduler.last_epoch)
        species = properties['species'].to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).float()
        true_forces = properties['forces'].to(device).float()
        num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates))


        #When only some conformers have real true forces we need to delete the filler
        bad_forces_index = torch.where(true_forces > 9999.00)[0]
        real_forces_index = [x for x in np.arange(0, true_forces.shape[0]) if x not in bad_forces_index]
        real_forces_index = np.array(real_forces_index)
        #print(real_forces_index)
        
        
        # We can use torch.autograd.grad to compute force. Remember to
        # create graph so that the loss of the force can contribute to
        # the gradient of parameters, and also to retain graph so that
        # we can backward through it a second time when computing gradient
        # w.r.t. parameters.
        forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]

        # Now the total loss has two parts, energy loss and force loss
        energy_loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()
        if real_forces_index.shape[0] > 0:
            force_loss = mse(true_forces[real_forces_index], forces[real_forces_index])
            force_loss = force_loss.sum(dim=(1, 2))
            force_loss = force_loss / (num_atoms * (real_forces_index.shape[0] / true_forces.shape[0])).mean()
            force_loss = force_loss.sum()
            force_loss = force_loss * config["force_coefficient"]
            loss = energy_loss + force_loss
            #print("force_loss:", force_loss)
            #print("energy_loss:", energy_loss)
        else:
            loss = energy_loss
            #print("energy_loss:", energy_loss)

        training_log.loc[AdamW_scheduler.last_epoch + 1] = [energy_rmse, force_rmse, energy_mse, force_mse, float(loss.sum().cpu().detach().numpy())] 
        
        AdamW.zero_grad()
        SGD.zero_grad()
        loss.backward()
        AdamW.step()
        SGD.step()

        # write current batch loss to TensorBoard
        #tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)

    try:
        torch.save({
            'nn': nn.state_dict(),
            'AdamW': AdamW.state_dict(),
            'SGD': SGD.state_dict(),
            'AdamW_scheduler': AdamW_scheduler.state_dict(),
            'SGD_scheduler': SGD_scheduler.state_dict(),
        }, latest_checkpoint)
    except PermissionError: # happens sometimes on windows for no good reason
        print("Permission error in saving latest.pt, we'll just skip this one.")
    except OSError: # happens sometimes on windows for no good reason
        print("OSerror in saving latest.pt, we'll just skip this one.")
    
    training_log.to_csv(TrainingLog)
    
    
    if (AdamW_scheduler.last_epoch + 1) % config["GraphEveryEpochs"] == 0:
        fig, axs = plt.subplots(4)
        fig.suptitle(f"Iridium deep learning\nForce coef = {config['force_coefficient']}")
        axs[0].plot(training_log.index, training_log["Energy RMSE"], lw=1.5, label="Energy RMSE")
        axs[1].plot(training_log.index, training_log["Force RMSE"], lw=1.5, label="Force RMSE")
        axs[2].plot(training_log.index, training_log["Energy MSE"], lw=1.5, label="Energy MSE")
        axs[3].plot(training_log.index, training_log["Force MSE"], lw=1.5, label="Force MSE")

        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Force RMSE")
        axs[0].set_ylabel("Energy RMSE")
        axs[2].set_ylabel("Energy MSE")
        axs[3].set_ylabel("Force MSE")
        
        axs[0].set_yscale("log")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig("DNN_training.png")
        plt.show()
        print("Training graph saved to: DNN_training.png")
Log.close()

"ello" + "5"


