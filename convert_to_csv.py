#!/usr/bin/env python

import os
import sys
import numpy as np
from scipy.io import netcdf_file
import pandas as pd

if len(sys.argv) != 2:
   raise RuntimeError("A qsc_out.*.nc file must be provided as an argument")

filename = sys.argv[1]
bare_filename = os.path.basename(filename)
s = 'qsc_out.'
if bare_filename[:len(s)] != s or filename[-3:] != '.nc':
   raise RuntimeError("A qsc_out.*.nc file must be provided as an argument")

f = netcdf_file(filename, 'r', mmap=False)
n_scan = f.variables['n_scan'][()]
iota = np.abs(f.variables['scan_iota'][()])
eta_bar = f.variables['scan_eta_bar'][()]
B2c = f.variables['scan_B2c'][()]
B20_variation = f.variables['scan_B20_variation'][()]
r_singularity = f.variables['scan_r_singularity'][()]
d2_volume_d_psi2 = f.variables['scan_d2_volume_d_psi2'][()]
L_grad_B = f.variables['scan_min_L_grad_B'][()]
L_grad_grad_B = f.variables['scan_min_L_grad_grad_B'][()]
elongation = f.variables['scan_max_elongation'][()]
min_R0 = f.variables['scan_min_R0'][()]
helicity=f.variables['scan_helicity'][()]
R0c = f.variables['scan_R0c'][()]
Z0s = f.variables['scan_Z0s'][()]
nfp = f.variables['nfp'][()]
axlength=nfp*0

heli_bool=[False if helicity[i] == 0 else True for i in range(len(helicity))]

# For r_singularity, replace 1e30 with 1
r_singularity = np.minimum(r_singularity, np.ones_like(r_singularity))

# Create a DataFrame from the extracted variables
data = {}
from qsc import Qsc
axlength=[Qsc(rc=R0c[j], zs=Z0s[j], nfp=nfp).axis_length for j in range(0,len(R0c))]

# Add the rest of the variables to the data dictionary
data.update({
    'axLength': axlength,
    'RotTrans': iota,
    'nfp': nfp,
    'helicity': heli_bool,
})

# Add each column of R0c and Z0s to the data dictionary
for i in range(1,R0c.shape[1]): #assume R0c0=1 and Z0s0=0
    data[f'rc{i}'] = R0c[:, i]
    data[f'zs{i}'] = Z0s[:, i]
    # data[f'x{2*i-1}'] = R0c[:, i]
    # data[f'x{2*i}'] = Z0s[:, i]

data.update({
    'b2c': B2c,
    'r_singularity': r_singularity,
    'd2_volume_d_psi2': d2_volume_d_psi2,
    'B20_var': B20_variation,
    'etabar': eta_bar,
    'max_elong': elongation,
    'lgradB': L_grad_B,
    'lgrad_gradB': L_grad_grad_B,
    'min_R0': min_R0,
})
# data.update({
#     f'x{2*R0c.shape[1]-1}': eta_bar,
#     f'x{2*R0c.shape[1]}': B2c,
#     f'y0': 0.33*np.abs(1/iota),
#     f'y1': 0.09/r_singularity,
#     f'y2': -1e2/d2_volume_d_psi2,
#     f'y3': B20_variation/1.1,
#     f'y4': elongation/8,
#     f'y5': 0.3/L_grad_B,
#     f'y6': 0.3/L_grad_grad_B,
#     f'y7': 0.3/min_R0,
# })

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_filename = os.path.splitext(filename)[0] + '.csv'
df.to_csv(csv_filename, index=False)

print(f"CSV file created: {csv_filename}")