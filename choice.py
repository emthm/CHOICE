"""
Main program choice
CHOICE = CHalmers university nOIse CodE
Python implementation: Marily Thoma
Original (Fortran) code: Tomas Grönstedt
Debugging: Marily Thoma, Xin Zhao
Program based on the NoPred code developed by David Carlsson and Lars Ellprant supervised by Richard Avellán.
Last update: 2022-12-12
"""

import sys

if sys.version_info.major != 3:
    sys.exit('ERROR: The code must be executed using Python 3!')

import choice_read_and_write
import choice_interf


def calc_noise_points(i, trajectory, modules, mpd, noise_choice, weight_choice):
    if not noise_choice.trajectory_performance:
        mpd = choice_interf.set_rotational_speeds_choice(mpd, weight_choice, 'Fan')
        # estimate absolute speeds from relative rotating speeds (from performance) using several points
        if noise_choice.fuselage_fan:
            mpd = choice_interf.set_rotational_speeds_choice_ff(mpd, weight_choice, 'fuselage_fan')
            # estimate absolute speeds from relative rotating speeds (from performance) using several points

    performance_choice = choice_interf.set_performance_choice(i, modules, trajectory.n_traj_pts, mpd, noise_choice,
                                                              weight_choice)

    [prms, theta, fband] = choice_interf.compute_noise_sources(noise_choice.opPnt[i].strip(), trajectory, modules,
                                                               noise_choice, weight_choice, performance_choice,i)
    [SPLi, xsii, Mai, Tai, alphai] = choice_interf.interpolate_to_t_source(trajectory, modules, prms)
    [SPLp, fobs] = choice_interf.compute_flight_effects(noise_choice.use_ground_reflection,
                                                        noise_choice.use_spherical_spreading,
                                                        noise_choice.use_atmospheric_attenuation, trajectory,
                                                        noise_choice.ymic[i], SPLi, xsii, Mai, Tai, alphai, theta,
                                                        fband)
    EPNL = choice_interf.CertificationData(trajectory.n_times, fobs, fband, SPLp)
    choice_interf.save_noise_points('Output\choiceOutput.txt', noise_choice.opPnt[i].strip(), noise_choice.fuselage_fan,
                                    EPNL)


if __name__ == "__main__":

    perf_file = 'Input\performanceResults.txt'
    dim_file = 'Input\dimensionsWeight.txt'
    weight_file = 'Input\weightAircraft.txt'
    noise_file = 'Input\inputNoise.txt'

    [modules, weightFile, noiseFile, mpd] = choice_read_and_write.read_external_files(dim_file, weight_file, noise_file,
                                                                                      perf_file)
    weight_choice = choice_interf.WeightChoice(weightFile)
    noise_choice = choice_interf.NoiseChoice(noiseFile)

    if noise_choice.use_trajectory_preparser:
        choice_read_and_write.preparse_trajectories(noise_choice.trajectory_performance, noise_choice.opPnt, modules)
        # execution stops after preparse. Must set false to proceed to calculations.

    choice_interf.get_version_num('Output\choiceOutput.txt')

    if noise_choice.nops == 1:
        # read x, y, Va, alpha from trajectory file (Cutback.txt, Take-off.txt or Approach.txt)
        trajectory = choice_interf.set_trajectory(0, noise_choice.opPnt[0], noise_choice)
        # Compute r, clgr and xsi, time
        calc_noise_points(0, trajectory, modules, mpd, noise_choice, weight_choice)
    else:
        for i in range(noise_choice.nops):
            trajectory = choice_interf.set_trajectory(i, noise_choice.opPnt[i], noise_choice)
            calc_noise_points(i, trajectory, modules, mpd, noise_choice, weight_choice)

    # establish ICAO certification limits for given mass and number of engines. 
    choice_interf.certificationLimits(noise_choice.no_engines, noise_choice.total_weight_airfrm / 1000)
