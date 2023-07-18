import pandas as pd
import numpy as np
from neat.fields import StellnaQS
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit
import time
from mpi4py import MPI

start_time = time.time()
filename='ScanNA2.csv'

# Initialize the MPI communicator
comm = MPI.COMM_WORLD
size = comm.Get_size()  # total number of processes
rank = comm.Get_rank()  # rank of the current process

# Only one process should handle the file input
if rank == 0:
    # Read the CSV file
    df = pd.read_csv(filename)
    df = df.dropna()
else:
    df = None

# Broadcast df to all processes
df = comm.bcast(df, root=0)

#ARIES-CS specs
B0 = 5.3267  # Tesla, magnetic field on-axis
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044

s_initial=0.7 #normalized
s_max=0.989 #normalized

r_initial = Rminor_ARIES*np.sqrt(s_initial)  # meters
r_max = Rminor_ARIES*np.sqrt(s_max)  # meters

energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 5  # resolution in theta
nphi = 5  # resolution in phi
nlambda_trapped = 10  # number of pitch angles for trapped particles
nlambda_passing = 0  # number of pitch angles for passing particles

constant_b20 = True
nsamples = 10000  # resolution in time
tfinal = 1e-3  # seconds
dist = 0
thetas = np.linspace(0, 2 * np.pi, ntheta)

# Define the function for parallel processing
def perform_particle_tracing(i):
    varphis = np.linspace(0, 2 * np.pi / df['nfp'].iloc[i], nphi)

    g_field_basis = StellnaQS(rc=[1, df['rc1'].iloc[i],df['rc2'].iloc[i]], 
                              zs=[0, df['zs1'].iloc[i],df['zs2'].iloc[i]], 
                              etabar=df['eta'].iloc[i], 
                              B2c=df['b2c'].iloc[i],
                              B0=B0,
                              nfp=df['nfp'].iloc[i], 
                              order='r2', 
                              nphi=101)
    
    g_field = StellnaQS(rc=g_field_basis.rc * Rmajor_ARIES, 
                        zs=g_field_basis.zs * Rmajor_ARIES,
                        etabar=g_field_basis.etabar / Rmajor_ARIES,
                        B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
                        B0=B0,
                        nfp=g_field_basis.nfp, 
                        order='r2', 
                        nphi=101)

    g_particle = ChargedParticleEnsemble(
        r_initial=r_initial,
        r_max=r_max,
        energy=energy,
        charge=charge,
        mass=mass,
        ntheta=ntheta,
        nphi=nphi,
        nlambda_trapped=nlambda_trapped,
        nlambda_passing=nlambda_passing
    )

    # print(f"Starting particle tracer for index {i} with B20 constant")

    g_orbits = ParticleEnsembleOrbit(
        g_particle,
        g_field,
        nsamples=nsamples,
        nthreads=4,
        tfinal=tfinal,
        constant_b20=constant_b20,
        dist=dist,
        thetas=thetas,
        phis=varphis
    )

    loss_fraction = g_orbits.loss_fraction(r_max=r_max, jacobian_weight=True)
    return loss_fraction[-1]

# Split the data among processes
data_split = np.array_split(range(len(df)), size)[rank]

# Perform particle tracing in parallel using MPI
losses = [perform_particle_tracing(i) for i in data_split]

# Gather all results to process 0
all_losses = comm.gather(losses, root=0)

# Only process 0 should handle the file output
if rank == 0:
    # Flatten the list of losses
    all_losses = [item for sublist in all_losses for item in sublist]

    # Attach the losses to the dataframe
    df['Loss Fraction'] = all_losses

    # Save the modified DataFrame back to the CSV file
    df.to_csv('loss_fraction' + filename, index=False)

    print(f"The process with multiprocessing took {time.time() - start_time} seconds.")