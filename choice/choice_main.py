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
    :param str weight_file: file containing engine sizing data
    :param str noise_file: file that is used to define the noise calculation study
    :param str file_type: file type for noise source matrices (csv, m, None)
    """
    def __init__(self, input_folder='Input/', output_folder='Output/', perf_file='performanceResults.txt',
                 weight_file='weightAircraft.txt', noise_file='inputNoise.txt', file_type=None):
        self.perf_file = input_folder + '/' + perf_file
        self.weight_file = input_folder + '/' + weight_file
        self.noise_file = input_folder + '/' + noise_file
        self.output_folder = output_folder
        self.ext = file_type
        self.input_folder = input_folder

    def run_choice(self):
        """ Sets the required parameters and performs the noise calculation"""

        input = ReadFiles(self.weight_file, self.noise_file, self.perf_file)
        self.weight_choice = WeightChoice(input.weightFile)
        self.noise_choice = NoiseChoice(input.noiseFile)

        if self.noise_choice.use_trajectory_preparser:
            preparse_trajectories(self.noise_choice.trajectory_performance, self.noise_choice.opPnt, input.modules,
                                  self.input_folder)
            # execution stops after preparse. Must set false to proceed to calculations.

        get_version_num(self.output_folder + '\choiceOutput.txt')

        if self.noise_choice.nops == 1:
            # read x, y, Va, alpha from trajectory file (Cutback.txt, Take-off.txt or Approach.txt)
            self.trajectory = Trajectory.set(0, self.noise_choice.opPnt[0], self.noise_choice, self.input_folder)
            # Compute r, clgr and xsi, time
            self.calc_noise_points(0, self.trajectory, input.modules, input.mpd, self.noise_choice, self.weight_choice,
                                   self.input_folder, self.output_folder, self.ext)
        else:
            for i in range(self.noise_choice.nops):
                self.trajectory = Trajectory.set(i, self.noise_choice.opPnt[i], self.noise_choice, self.input_folder)
                self.calc_noise_points(i, self.trajectory, input.modules, input.mpd, self.noise_choice, self.weight_choice,
                                       self.input_folder, self.output_folder)

        # establish ICAO certification limits for given mass and number of engines.
        certificationLimits(self.noise_choice.no_engines, self.noise_choice.total_weight_airfrm / 1000)

    @staticmethod
    def calc_noise_points(i, trajectory, modules, mpd, noise_choice, weight_choice, input_folder, output_folder, ext):
        if not noise_choice.trajectory_performance:
            mpd = set_rotational_speeds_choice(mpd, weight_choice, 'Fan')
            # estimate absolute speeds from relative rotating speeds (from performance) using several points
            if noise_choice.fuselage_fan:
                mpd = set_rotational_speeds_choice(mpd, weight_choice, 'Fuselage_fan')
                # estimate absolute speeds from relative rotating speeds (from performance) using several points

        performance_choice = PerformanceChoice.set(i, modules, trajectory.n_traj_pts, mpd, noise_choice, weight_choice,
                                                   input_folder)

        noise_sources = NoiseSources.compute(trajectory, modules, noise_choice, weight_choice, performance_choice, i,
                                             output_folder, ext)
        [SPLi, trajectory] = interpolate_to_t_source(trajectory, modules, noise_sources.prms)
        ground_noise = \
            GroundNoise.compute_flight_effects(noise_choice.use_ground_reflection, noise_choice.use_spherical_spreading,
                                               noise_choice.use_atmospheric_attenuation, trajectory,
                                               noise_choice.ymic[i], noise_choice.dtIsa, noise_choice.elevation,
                                               SPLi, noise_sources.theta, noise_sources.fband)
        EPNL = CertificationData.compute(trajectory.n_times, ground_noise.fobs, noise_sources.fband, ground_noise.SPLp,
                                         modules)
        save_noise_points(output_folder + '\choiceOutput.txt', noise_choice.opPnt[i].strip(), noise_choice.fuselage_fan,
                          EPNL)
