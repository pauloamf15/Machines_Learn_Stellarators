import pandas as pd
import numpy as np
from neat.fields import StellnaQS
from neat.tracing import ChargedParticleEnsemble, ParticleEnsembleOrbit
import time
import multiprocessing as mp

start_time = time.time()

# Read the CSV file
df = pd.read_csv('scan_toy.csv')
df = df.dropna()

B0 = 5.3267  # Tesla, magnetic field on-axis
Rmajor_ARIES = 7.7495 * 2
Rminor_ARIES = 1.7044

r_initial = Rminor_ARIES * 0.9  # meters
r_max = Rminor_ARIES * 0.99  # meters

energy = 3.52e6  # electron-volt
charge = 2  # times charge of proton
mass = 4  # times mass of proton
ntheta = 5  # resolution in theta
nphi = 5  # resolution in phi
nlambda_trapped = 10  # number of pitch angles for trapped particles
nlambda_passing = 0  # number of pitch angles for passing particles

constant_b20 = True
nsamples = 10000  # resolution in time
tfinal = 1e-4  # seconds
dist = 0
thetas = np.linspace(0, 2 * np.pi, ntheta)

# Define the function for parallel processing
def perform_particle_tracing(i):
    varphis = np.linspace(0, 2 * np.pi / df['nfp'].iloc[i], nphi)

    g_field_basis = StellnaQS(rc=[1, df['rc1'].iloc[i]], zs=[0, df['zs1'].iloc[i]], etabar=df['eta'].iloc[i], B0=B0,
                              nfp=df['nfp'].iloc[i], order='r1', nphi=401)
    g_field = StellnaQS(rc=g_field_basis.rc * Rmajor_ARIES, zs=g_field_basis.zs * Rmajor_ARIES,
                        etabar=g_field_basis.etabar / Rmajor_ARIES,
                        B2c=g_field_basis.B2c * (B0 / Rmajor_ARIES / Rmajor_ARIES),
                        B0=B0, nfp=g_field_basis.nfp, order='r1', nphi=41)

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

# Perform particle tracing in parallel using multiprocessing
pool = mp.Pool(2)
losses = pool.map(perform_particle_tracing, range(len(df)))
pool.close()
pool.join()

# print(losses)

# Attach the losses to the dataframe
df['Loss Fraction'] = losses

# Save the modified DataFrame back to the CSV file
df.to_csv('your_file_with_loss_fraction.csv', index=False)

print(f"The process with multiprocessing took {time.time() - start_time} seconds.")