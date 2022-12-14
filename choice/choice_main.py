"""
Main program choice
CHOICE = CHalmers university nOIse CodE
Python implementation: Marily Thoma
Original (Fortran) code: Tomas Grönstedt
Debugging: Marily Thoma, Xin Zhao
Program based on the NoPred code developed by David Carlsson and Lars Ellprant supervised by Richard Avellán.
Last update: 2023-01-09
"""


from choice.choice_read_and_write import ReadFiles, preparse_trajectories, save_noise_points
from choice.choice_interf import *


class CHOICE:
    """
    Instantiate input files and folders to be used for the noise calculation.

    :param str input_folder: input files folder path
    :param str output_folder: output files folder path
    :param str perf_file: file containing the engine performance parameters
    :param str dim_file: file containing the modules of the engine architecture
    :param str weight_file: file containing engine sizing data
    :param str noise_file: file that is used to define the noise calculation study
    """
    def __init__(self, input_folder='Input/', output_folder='Output/', perf_file='performanceResults.txt',
                 dim_file='dimensionsWeight.txt', weight_file='weightAircraft.txt', noise_file='inputNoise.txt'):
        self.perf_file = input_folder + '/' + perf_file
        self.dim_file = input_folder + '/' + dim_file
        self.weight_file = input_folder + '/' + weight_file
        self.noise_file = input_folder + '/' + noise_file
        self.output_folder = output_folder
        self.input_folder = input_folder

    def run_choice(self):
        """ Sets the required parameters and performs the noise calculation"""

        input = ReadFiles(self.dim_file, self.weight_file, self.noise_file, self.perf_file)
        weight_choice = WeightChoice(input.weightFile)
        noise_choice = NoiseChoice(input.noiseFile)

        if noise_choice.use_trajectory_preparser:
            preparse_trajectories(noise_choice.trajectory_performance, noise_choice.opPnt, input.modules,
                                  self.input_folder)
            # execution stops after preparse. Must set false to proceed to calculations.

        get_version_num(self.output_folder + '\choiceOutput.txt')

        if noise_choice.nops == 1:
            # read x, y, Va, alpha from trajectory file (Cutback.txt, Take-off.txt or Approach.txt)
            trajectory = Trajectory.set(0, noise_choice.opPnt[0], noise_choice, self.input_folder)
            # Compute r, clgr and xsi, time
            self.calc_noise_points(0, trajectory, input.modules, input.mpd, noise_choice, weight_choice,
                                   self.input_folder, self.output_folder)
        else:
            for i in range(noise_choice.nops):
                trajectory = Trajectory.set(i, noise_choice.opPnt[i], noise_choice, self.input_folder)
                self.calc_noise_points(i, trajectory, input.modules, input.mpd, noise_choice, weight_choice,
                                       self.input_folder, self.output_folder)

        # establish ICAO certification limits for given mass and number of engines.
        certificationLimits(noise_choice.no_engines, noise_choice.total_weight_airfrm / 1000)

    @staticmethod
    def calc_noise_points(i, trajectory, modules, mpd, noise_choice, weight_choice, input_folder, output_folder):
        if not noise_choice.trajectory_performance:
            mpd = set_rotational_speeds_choice(mpd, weight_choice, 'Fan')
            # estimate absolute speeds from relative rotating speeds (from performance) using several points
            if noise_choice.fuselage_fan:
                mpd = set_rotational_speeds_choice_ff(mpd, weight_choice, 'fuselage_fan')
                # estimate absolute speeds from relative rotating speeds (from performance) using several points

        performance_choice = PerformanceChoice.set(i, modules, trajectory.n_traj_pts, mpd, noise_choice, weight_choice,
                                                   input_folder)

        noise_sources = NoiseSources.compute(trajectory, modules, noise_choice, weight_choice, performance_choice, i,
                                             output_folder)
        [SPLi, xsii, Mai, Tai, alphai] = interpolate_to_t_source(trajectory, modules, noise_sources.prms)
        ground_noise = \
            GroundNoise.compute_flight_effects(noise_choice.use_ground_reflection, noise_choice.use_spherical_spreading,
                                               noise_choice.use_atmospheric_attenuation, trajectory,
                                               noise_choice.ymic[i], SPLi, xsii, Mai, Tai, alphai, noise_sources.theta,
                                               noise_sources.fband)
        EPNL = CertificationData.compute(trajectory.n_times, ground_noise.fobs, noise_sources.fband, ground_noise.SPLp)
        save_noise_points(output_folder + '\choiceOutput.txt', noise_choice.opPnt[i].strip(), noise_choice.fuselage_fan,
                          EPNL)
