"""
Module choice_data
Contains default data for components and tabulated data for noise empirical models.
"""

import numpy as np


fmin = 50
fmax = 10000
nb = 1  # number of bands
nfreq = int(round(3.0 * nb * np.log10(float(fmax) / float(fmin)) / np.log10(2.0)) + 1)
theta_step = 5.0
nthet = int(180.0 / theta_step) + 1

RH = 70.0
pa2psia = 1/6894.75728

kt2mps = 0.5144
ft2m = 0.3048

# Fan Data
fan_IGV = False
# fan_distortion = True
no_fan_stages = 1
np.seterr(divide='ignore')

# Performance data to be used for airframe
maxNoPts = 3

no_ffs = 1  # number of fuselage fans, default is 1
ff_IGV = False
no_ff_stages = 1

p0 = 2.0000E-05
Tisa = 288.15  # ISA Condition Static Temperature under ISA, sea level condition
Pisa = 101325.0  # ISA Condition Static Pressure under ISA, sea level condition
Risa = 287.05  # ISA Condition gas constant
rhoisa = Pisa / (Risa * Tisa)  # ISA Condition Static Density under ISA, sea level condition

n_rows = 36
n_cols = 7

gamma_air = 1.40
gamma_gas = 1.333

cp_air = 1005.0
cp_gas = 1148.0

