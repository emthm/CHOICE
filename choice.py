"""
Main program choice
CHOICE = CHalmers university nOIse CodE
Python implementation: Marily Thoma
Original (Fortran) code: Tomas Grönstedt
Debugging: Marily Thoma, Xin Zhao
Program based on the NoPred code developed by David Carlsson and Lars Ellprant supervised by Richard Avellán.
Last update: 2023-01-03
"""

import sys

if sys.version_info.major != 3:
    sys.exit('ERROR: The code must be executed using Python 3!')

from choice_read_and_write import ReadFiles, preparse_trajectories, save_noise_points
from choice_interf import *


def calc_noise_points(i, trajectory, modules, mpd, noise_choice, weight_choice):
    if not noise_choice.trajectory_performance:
        mpd = set_rotational_speeds_choice(mpd, weight_choice, 'Fan')
        # estimate absolute speeds from relative rotating speeds (from performance) using several points
        if noise_choice.fuselage_fan:
            mpd = set_rotational_speeds_choice_ff(mpd, weight_choice, 'fuselage_fan')
            # estimate absolute speeds from relative rotating speeds (from performance) using several points

    performance_choice = PerformanceChoice.set(i, modules, trajectory.n_traj_pts, mpd, noise_choice, weight_choice)

    noise_sources = NoiseSources.compute(trajectory, modules, noise_choice, weight_choice, performance_choice, i)
    [SPLi, xsii, Mai, Tai, alphai] = interpolate_to_t_source(trajectory, modules, noise_sources.prms)
    ground_noise = \
        GroundNoise.compute_flight_effects(noise_choice.use_ground_reflection, noise_choice.use_spherical_spreading,
                                           noise_choice.use_atmospheric_attenuation, trajectory, noise_choice.ymic[i],
                                           SPLi, xsii, Mai, Tai, alphai, noise_sources.theta, noise_sources.fband)
    EPNL = CertificationData.compute(trajectory.n_times, ground_noise.fobs, noise_sources.fband, ground_noise.SPLp)
    save_noise_points('Output\choiceOutput.txt', noise_choice.opPnt[i].strip(), noise_choice.fuselage_fan, EPNL)


if __name__ == "__main__":

    perf_file = 'Input\performanceResults.txt'
    dim_file = 'Input\dimensionsWeight.txt'
    weight_file = 'Input\weightAircraft.txt'
    noise_file = 'Input\inputNoise.txt'

    input = ReadFiles(dim_file, weight_file, noise_file, perf_file)
    weight_choice = WeightChoice(input.weightFile)
    noise_choice = NoiseChoice(input.noiseFile)

    if noise_choice.use_trajectory_preparser:
        preparse_trajectories(noise_choice.trajectory_performance, noise_choice.opPnt, input.modules)
        # execution stops after preparse. Must set false to proceed to calculations.

    get_version_num('Output\choiceOutput.txt')

    if noise_choice.nops == 1:
        # read x, y, Va, alpha from trajectory file (Cutback.txt, Take-off.txt or Approach.txt)
        trajectory = Trajectory.set(0, noise_choice.opPnt[0], noise_choice)
        # Compute r, clgr and xsi, time
        calc_noise_points(0, trajectory, input.modules, input.mpd, noise_choice, weight_choice)
    else:
        for i in range(noise_choice.nops):
            trajectory = Trajectory.set(i, noise_choice.opPnt[i], noise_choice)
            calc_noise_points(i, trajectory, input.modules, input.mpd, noise_choice, weight_choice)

    # establish ICAO certification limits for given mass and number of engines. 
    certificationLimits(noise_choice.no_engines, noise_choice.total_weight_airfrm / 1000)
