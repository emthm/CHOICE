"""
Module choice_interf
Choice interface: used to set the required parameters for the noise calculation and call the physics-based methods from
the module choice_physics.
"""

import sys
import math
import numpy as np
import choice_physics
import choice_aux
import choice_data
from scipy.interpolate import interp1d

output_file_open = False
maxNoPts = 3  # Maximum number of operating points
version_num = '1.0'


def get_version_num(fname):
    """ Writes CHOICE version number in output file. """
    with open(fname, 'w') as f:
        f.write('! Choice version: ' + version_num)


class WeightChoice:
    """ A class to set the engine size and architecture. """

    def __init__(self, weightFile):
        """ Constructs all the necessary attributes for the noise calculation, from the provided weightAircraft file."""

        self.MtipD_fan = weightFile.get('MtipFan')
        if self.MtipD_fan is not None and (self.MtipD_fan > 3.0 or self.MtipD_fan < 0.7): choice_aux.report_error(
            'unexpected value on MtipD_fan', 'setFan', 'SetWeightChoice')

        self.xnlD_fan = weightFile.get('xnl')
        if self.xnlD_fan is not None and self.xnlD_fan > 600: choice_aux.report_error(
            'xnl from self.weightFile very large. Unit should always be rps!', 'setFan', 'SetWeightChoice')

        self.rss_fan = weightFile.get('FanRss')
        if self.rss_fan is not None and (self.rss_fan > 300.0 or self.rss_fan < 5.0): choice_aux.report_error(
            'unexpected value on rss_fan', 'setFan', 'SetWeightChoice')

        self.N_rotors_fan = weightFile.get('FanR1BNb')
        self.N_stators_fan = weightFile.get('FanVsOgvNb')

        self.A2_fan = weightFile.get('FanA2')
        if self.A2_fan is not None and (self.A2_fan > 50.0 or self.A2_fan < 0.1): choice_aux.report_error(
            'unexpected value on A2_fan', 'setFan', 'SetWeightChoice')

        self.D1_fan = weightFile.get('FanR1BDiaOuter')
        if self.D1_fan is not None and (self.D1_fan < 0.1 or self.D1_fan > 10.0): choice_aux.report_error(
            'unexpected value on D1_fan', 'setFan', 'SetWeightChoice')

        self.n_stages_LPC = weightFile.get('stages_LPC')

        self.gbx_ratio = weightFile.get('GBX_ratio')
        self.MtipD_lpc = weightFile.get('MtipLpc')
        self.xnlD_lpc = self.xnlD_fan * self.gbx_ratio
        self.rss_lpc = weightFile.get('RSS_compr')

        self.N_rotors_lpc = weightFile.get('N_compr')
        self.N_stators_lpc = weightFile.get('S_compr')

        self.D1_lpc = 2 * weightFile.get('r_compr')
        self.Dh1_lpc = 2 * weightFile.get('rh_compr')

        self.gbx_ratio = weightFile.get('GBX_ratio')
        self.N_rotors_lpt = weightFile.get('LptStgLastBNb')
        self.SRS_lpt = weightFile.get('SRS')
        self.n_stages_lpt = weightFile.get('stages_LPT')
        self.De_lpt = weightFile.get('LptStgLastDiaOuter')
        self.Ae_lpt = weightFile.get('LptStgLastAExit')

        self.type_comb = weightFile.get('CombType')
        self.Nfmax_comb = weightFile.get('maxNo_nozzles_dac')  # total number of DAC fuel nozzles
        self.Aec_comb = weightFile.get('A_comb_exit') / (choice_data.ft2m ** 2)  # combustor exit area (ft^2)
        self.De_comb = weightFile.get(
            'Deff_comb') / choice_data.ft2m  # exhaust nozzle exit plane effective diameter (ft)
        self.Dh_comb = weightFile.get(
            'Dhyd_comb') / choice_data.ft2m  # exhaust nozzle exit plane hydraulic diameter (ft)
        self.Lc_comb = weightFile.get('Lc_comb') / choice_data.ft2m  # combustor nominal length (ft)
        self.h_comb = weightFile.get('h_annulus_comb') / choice_data.ft2m  # annulus height at combustor exit (ft)

        self.A_core_caj = weightFile.get('A_core')
        self.A_bypass_caj = weightFile.get('A_bypass')

        self.MtipD_ff = weightFile.get('Mtipff')
        if self.MtipD_ff is not None and (self.MtipD_ff > 3.0 or self.MtipD_ff < 0.7): choice_aux.report_error(
            'unexpected value on MtipD_ff', 'setff', 'SetWeightChoice')

        self.xnlD_ff = weightFile.get('xnlff')
        if self.xnlD_ff is not None and self.xnlD_ff > 600: choice_aux.report_error(
            'xnlff from weightFile very large. Unit should always be rps!', 'setff', 'SetWeightChoice')

        self.rss_ff = weightFile.get('ffRss')
        if self.rss_ff is not None and (self.rss_ff > 300.0 or self.rss_ff) < 5.0: choice_aux.report_error(
            'unexpected value on rss_ff', 'setff', 'SetWeightChoice')

        self.N_rotors_ff = weightFile.get('ffR1BNb')
        self.N_stators_ff = weightFile.get('ffVsOgvNb')

        self.A2_ff = weightFile.get('ffA2')
        if self.A2_ff is not None and (self.A2_ff > 50.0 or self.A2_ff < 0.1): choice_aux.report_error(
            'unexpected value on A2_ff', 'setff', 'SetWeightChoice')

        self.D1_ff = weightFile.get('ffR1BDiaOuter')
        if self.D1_ff is not None and (self.D1_ff < 0.1 or self.D1_ff > 10.0): choice_aux.report_error(
            'unexpected value on D1_ff', 'setff', 'SetWeightChoice')

        self.A_core_caj_ffn = weightFile.get('A_core_ffn')
        self.A_bypass_caj_ffn = weightFile.get('A_bypass_ffn')


class NoiseChoice:
    """ A class that sets the required input arguments for the noise prediction cases to be run. """

    def __init__(self, noiseFile):
        """ Constructs all the necessary attributes for the noise calculation, from the provided inputNoise file. """

        self.opPnt = noiseFile.get('point').split()
        self.nops = len(self.opPnt)

        if self.nops > maxNoPts: sys.exit('number of noise points greater than allowed')

        self.trajectory_performance = False
        if 'true' in noiseFile.get('trajectory_performance'): self.trajectory_performance = True

        self.use_trajectory_preparser = False
        if 'true' in noiseFile.get('use_trajectory_preparser'): self.use_trajectory_preparser = True

        self.gen_noise_source_matr = False
        if 'true' in noiseFile.get('generate_noise_sources'): self.gen_noise_source_matr = True

        self.use_ground_reflection = True
        if 'false' in noiseFile.get('use_ground_reflection'): self.use_ground_reflection = False

        self.use_spherical_spreading = True
        if 'false' in noiseFile.get('use_spherical_spreading'): self.use_spherical_spreading = False

        self.use_atmospheric_attenuation = True
        if 'false' in noiseFile.get('use_atmospheric_attenuation'): self.use_atmospheric_attenuation = False

        self.use_suppression_factor = False
        self.S_fan_inlet = 0
        self.S_fan_dis = 0
        self.S_lpt = 0
        if 'true' in noiseFile.get('use_suppression_factor'):
            self.use_suppression_factor = True
            self.S_fan_inlet = noiseFile.get('fan_inlet_suppression')
            self.S_fan_dis = noiseFile.get('fan_dis_suppression')
            self.S_lpt = noiseFile.get('lpt_suppression')

        # type_nozzles only affects the total EPNL calculation. If it is set to mix, the fan discharge noise source
        # will be removed. The default is separate nozzles.
        self.type_nozzles = "separate"
        self.type_nozzles = noiseFile.get('type_nozzles')  # "separate" or "mix", other type will not be recognized

        # check if fuselage_fan exists
        self.fuselage_fan = False
        if 'true' in noiseFile.get('fuselage_fan'): self.fuselage_fan = True

        self.xmic = np.zeros(3)
        self.ymic = np.zeros(3)
        self.zmic = np.zeros(3)
        self.Nf_comb_ign = np.zeros(3)
        self.Nf_comb_pattern = np.zeros(3)

        if self.nops == 1:
            self.xmic[0] = noiseFile.get('xmic')
            self.ymic[0] = noiseFile.get('ymic')
            self.zmic[0] = noiseFile.get('zmic')
            # Combustor ignited nozzles
            self.Nf_comb_ign[0] = int(noiseFile.get('comb_ign_nozzles'))
            self.Nf_comb_pattern = float(noiseFile.get('dac_nozzle_pattern'))
        elif self.nops > 1:

            if self.nops > maxNoPts: sys.stop('number of noise points greater than allowed')

            self.xmic = np.array([float(xm) for xm in noiseFile.get('xmic').split()])
            if len(self.xmic) != self.nops: sys.exit('xmic data not consistent with number of points')

            self.ymic = np.array([float(ym) for ym in noiseFile.get('ymic').split()])
            if len(self.ymic) != self.nops: sys.exit('ymic data not consistent with number of points')

            self.zmic = np.array([float(zm) for zm in noiseFile.get('zmic').split()])
            if len(self.zmic) != self.nops: sys.exit('zmic data not consistent with number of points')

            # Combustor ignited nozzles
            self.Nf_comb_ign = np.array([int(nf) for nf in noiseFile.get('comb_ign_nozzles').split()])
            self.Nf_comb_pattern = np.array([float(nf) for nf in noiseFile.get('dac_nozzle_pattern').split()])

        self.Sw_airfrm = noiseFile.get('wing_area')  # wing area m2
        self.span_airfrm = noiseFile.get('wing_span')  # wing span m
        self.ND = int(noiseFile.get('jet_aircraft_coef'))  # coefficient for te noise (1 -> +6 dB)
        self.flap_type = noiseFile.get('flap_type')
        self.NoFlap = int(noiseFile.get('noFlaps'))
        if self.NoFlap == 1:
            self.S_flap_airfrm = noiseFile.get('flap_area')  # flap area m2
            self.span_flap_airfrm = noiseFile.get('flap_span')  # flap span m
        else:
            self.S_flap_airfrm = np.array([float(sf) for sf in noiseFile.get('flap_area').split()])
            self.span_flap_airfrm = np.array([float(bf) for bf in noiseFile.get('flap_span').split()])
        self.slat_type = noiseFile.get('leading_edge_type')
        self.span_hor_tail = noiseFile.get('horizontal_tail_span')  # horizontal tail span m
        self.Sht = noiseFile.get('horizontal_tail_area')  # horizontal tail area m2
        self.span_ver_tail = noiseFile.get('vertical_tail_span')  # vertical tail span m
        self.Svt = noiseFile.get('vertical_tail_area')  # vertical tail area m2

        self.no_engines = int(noiseFile.get('no_engines'))
        self.dtIsa = noiseFile.get('dtIsa')
        self.total_weight_airfrm = noiseFile.get('aircraft_mass')

        self.nlg_airfrm = int(noiseFile.get('noLG'))
        if self.nlg_airfrm == 1:
            self.d_wheel_airfrm[0] = noiseFile.get('d_wheel')
            self.d_strut_airfrm[0] = noiseFile.get('d_strut')
            self.d_wire_airfrm[0] = noiseFile.get('d_wire')
        else:
            self.d_wheel_airfrm = np.array([float(dw) for dw in noiseFile.get('d_wheel').split()])
            self.d_strut_airfrm = np.array([float(ds) for ds in noiseFile.get('d_strut').split()])
            self.d_wire_airfrm = np.array([float(dw) for dw in noiseFile.get('d_wire').split()])
            self.N_wheels_airfrm = np.array([float(nw) for nw in noiseFile.get('Nwheels').split()])
            self.N_struts_airfrm = np.array([float(ns) for ns in noiseFile.get('Nstruts').split()])

        # Distortion parameter based on Average axial velocity decrement divided by the blade speed at the place where
        # the average distortion acts, Povinelli, F. P., Dittmar, J. H., & Woodward, R. P. (1972). Effects of
        # installation caused flow distortion on noise from a fan designed for turbofan engines.
        self.fan_distortion = noiseFile.get('fan_distortion')
        self.ff_distortion = noiseFile.get('ff_distortion')

        self.psi_airfrm_vec = np.zeros(3)
        self.defl_flap_airfrm_vec = np.zeros(3)
        self.LandingGear = np.zeros(3)
        if self.nops == 1:
            self.psi_airfrm_vec[0] = noiseFile.get('psi')
            self.defl_flap_airfrm_vec[0] = noiseFile.get('defl_flap')
            self.LandingGear[0] = noiseFile.get('LandingGear_vec')
        else:
            self.psi_airfrm_vec = np.array([float(psi) for psi in noiseFile.get('psi').split()])
            self.defl_flap_airfrm_vec = np.array([float(df) for df in noiseFile.get('defl_flap').split()])
            self.LandingGear = np.array([int(lg) for lg in noiseFile.get('LandingGear_vec').split()])

        self.defl_slat_airfrm_vec = np.array([0.0, 0.0, 22.0])


class Trajectory:
    """ A class that sets the trajectory data of the aircraft. """

    def __init__(self, ipnt, opPnt, noise_choice):
        """
        Constructs all the necessary attributes for the trajectory object to be used in the noise calculation,
        for the selected trajectories. Computes distance to mic (r), flight path angle (clgr), time and angle between
        aircraft and microphone (xsi)
        :param ipnt: Operating point (trajectory) number in order of appearance in the inputNoise.txt
        :param opPnt: Operating point (trajectory), e.g. Approach, Cutback, Sideline
        :param noise_choice: NoiseChoice object
        """

        self.read_trajectory_input(opPnt)  # read trajectory data from file

        if 'Approach' in opPnt: self.x = self.x - self.x[-1]

        self.r = np.sqrt((self.x - noise_choice.xmic[ipnt]) ** 2 + (self.y - noise_choice.ymic[ipnt]) ** 2 +
                         noise_choice.zmic[ipnt] ** 2)  # distance along trajectory to microphone

        self.clgr = np.zeros(self.n_traj_pts)  # flight path angle along trajectory
        for i in range(self.n_traj_pts - 1):
            if self.y[i + 1] != 0:
                self.clgr[i] = np.rad2deg(math.atan((self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])))

            if np.isnan(self.clgr[i]): sys.exit('not a number (NaN) computed for clgr')

        self.clgr[self.n_traj_pts - 1] = self.clgr[self.n_traj_pts - 2]

        self.xsi = choice_physics.get_xsi(self.clgr, self.x, self.y, noise_choice.xmic[ipnt], noise_choice.ymic[ipnt])

        if np.isnan(self.xsi).any(): sys.exit('not a number computed for xsi')
        self.xsid = np.rad2deg(self.xsi)

        # time is computed for 2D trajectory
        self.time = np.zeros(self.n_traj_pts)

        for i in range(1, self.n_traj_pts):
            if math.sqrt((self.y[i] - self.y[i - 1]) ** 2 + (self.x[i] - self.x[i - 1]) ** 2) == 0:
                self.time[i] = self.time[i - 1] + 1
                # aircraft is not moving - impossible to compute time => set a fictitious time
            else:
                # use average velocity
                self.V_avg = (self.Va[i] + self.Va[i - 1]) / 2
                self.time[i] = self.time[i - 1] + math.sqrt(
                    (self.y[i] - self.y[i - 1]) ** 2 + (self.x[i] - self.x[i - 1]) ** 2) / self.V_avg
                if np.isnan(self.time[i]): sys.exit('not a number computed for time')

    def set_time_vector(self, ipnt, noise_choice):
        """
        Computes the retarded time.
        :param ipnt: Operating point (trajectory) number in order of appearance in the inputNoise.txt
        :param noise_choice: NoiseChoice object
        """
        dt = 0.5

        # create tmic (starting from when the first noise reaches the microphone from xstart (i.e. r[0]/a[0]) to when
        # the noise reaches the microphone from the last point)
        self.tmic = np.arange(self.r[0] / self.a[0], self.time[-1] + self.r[-1] / self.a[-1], 0.5)

        self.n_times = len(self.tmic)

        self.t_source = np.zeros(self.n_times)
        self.x_source = np.zeros(self.n_times)
        self.y_source = np.zeros(self.n_times)
        self.va_source = np.zeros(self.n_times)
        self.a_source = np.zeros(self.n_times)
        self.r1 = np.zeros(self.n_times)
        self.clgr_source = np.zeros(self.n_times)

        self.x_source[0] = self.x[0] - noise_choice.xmic[ipnt]  # use mic-related coordinate system
        self.y_source[0] = self.y[0] - noise_choice.ymic[ipnt]  # use mic-related coordinate system
        self.va_source[0] = self.Va[0]
        self.a_source[0] = self.a[0]
        self.clgr_source[0] = self.clgr[0]
        self.z_source = noise_choice.zmic[ipnt]
        self.r1[0] = math.sqrt(self.x_source[0] ** 2 + self.y_source[0] ** 2 + self.z_source ** 2)
        # distance between source and microphone.
        for i in range(1, self.n_times):
            if self.va_source[i - 1] == 0:
                dx = 0
            else:
                dx = choice_physics.get_dx(self.x_source[i - 1], self.y_source[i - 1], self.z_source, dt,
                                           self.clgr_source[i - 1], self.a_source[i - 1], self.va_source[i - 1],
                                           self.r1[i - 1] / self.a_source[i - 1] + dt)

            # update using dx
            self.x_source[i] = self.x_source[i - 1] + dx
            dy = dx * math.tan(np.radians(self.clgr_source[i - 1]))
            self.y_source[i] = self.y_source[i - 1] + dy
            dr = math.sqrt(dx ** 2 + dy ** 2)
            self.r1[i] = math.sqrt(self.x_source[i] ** 2 + self.y_source[i] ** 2 + self.z_source ** 2)
            if dr == 0:
                self.t_source[i] = self.t_source[i - 1] + dt
            else:
                self.t_source[i] = self.t_source[i - 1] + dr / self.va_source[i - 1]

            # interpolate
            self.clgr_source[i] = interp1d(self.x, self.clgr, fill_value="extrapolate")(
                self.x_source[i] + noise_choice.xmic[ipnt])
            self.va_source[i] = interp1d(self.x, self.Va, fill_value="extrapolate")(
                self.x_source[i] + noise_choice.xmic[ipnt])
            self.a_source[i] = interp1d(self.x, self.a, fill_value="extrapolate")(
                self.x_source[i] + noise_choice.xmic[ipnt])

    def set_velocity(self, dtIsa):
        """
        Computes ambient temperature (ta), speed of sound (a) and Mach number (Ma) along the trajectory
        :param dtIsa: Deviation from ISA temperature
        """
        R = choice_aux.get_R(0.0)
        self.ta = np.array([choice_physics.AtmosphericEffects.get_t_ambient(h, dtIsa) for h in self.y])
        self.a = np.sqrt(choice_data.gamma_air * R * self.ta)
        self.Ma = self.Va / self.a

    def read_trajectory_input(self, opPnt):
        """
        Reads and stores the trajectory data (position, velocity and angle of attack) for the provided operating point
        :param opPnt: Operating point
        """
        file = 'Input/' + opPnt.strip() + '.txt'

        x = []
        y = []
        Va = []
        alpha = []
        z = []

        with open(file) as fp:
            for line in fp:
                if line:  # omit blank lines
                    li = line.strip()
                    if not li.startswith('!'):
                        string = li.split()
                        x.append(float(string[0]))
                        y.append(float(string[1]))
                        Va.append(float(string[2]))
                        alpha.append(float(string[3]))
                        if len(string) == 5:
                            z.append(float(string[4]))
                        else:
                            z.append(0)
                    else:
                        continue

        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.Va = np.asarray(Va)
        self.alpha = np.asarray(alpha)
        self.z = np.asarray(z)
        self.n_traj_pts = len(self.x)


def set_trajectory(ipnt, opPnt, noise_choice):
    """
    Sets the required trajectory details for the noise calculation.
    :param ipnt: Operating point (trajectory) number in order of appearance in the inputNoise.txt
    :param opPnt: opPnt: Operating point (trajectory), e.g. Approach, Cutback, Sideline
    :param noise_choice: NoiseChoice object
    :return: A trajectory object
    """
    traj = Trajectory(ipnt, opPnt, noise_choice)  # Compute r, clgr, xsi and time
    traj.set_velocity(noise_choice.dtIsa)  # compute ta, a and Ma
    traj.set_time_vector(ipnt, noise_choice)  # compute "retarded" vectors at source

    return traj


def set_rotational_speeds_choice(mpd, weightChoice, comp):
    """ Estimates absolute speeds from relative rotating speeds (from performance) using several points. """
    N = {}
    if comp == 'Fan':
        param = 'NL'
        xnlD = weightChoice.xnlD_fan
    elif comp == 'fuselage_fan':
        param = 'NFF'
        xnlD = weightChoice.xnlD_ff

    for key in mpd.__dict__:
        N[key] = mpd.__dict__[key].get(param)
    if len(mpd.perf_file_sl) > 0: N["perf_file_to"] = N["perf_file_sl"]  # use sideline data instead of take-off if
    # sideline exists.

    # form ratios and use the WEICO value
    for key in N.keys():
        N[key] = xnlD * (N[key] / N["perf_file_toc"])
        mpd.__dict__[key][param] = N[key]

    return mpd


class PerformanceChoice:
    """ A class that sets the engine performance data. """

    def __init__(self, ptr, trajPerf, point, n_traj, weight_choice, mpd):
        """ Constructs the basic attributes for the engine performance that are required for the noise calculation. """
        self.ptr = ptr
        self.trajPerf = trajPerf
        self.point = point
        self.openRotor = False
        self.n_traj = n_traj
        self.weight_choice = weight_choice
        self.mpd = mpd

    def getIsOpenRotor(self, modules):
        """ Checks if it is an open rotor. """
        for module in modules:
            if module.rstrip() == 'Prop':
                self.openRotor = True
                break
        return self.openRotor

    def coAxialJet(self):
        """ Sets the jet performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_coAxialJet_performance.txt'
        if self.trajPerf:
            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 6)
            self.dmdt_1_caj = storage_mat[:, 0]
            self.dmdt_2_caj = storage_mat[:, 1]
            self.v_1_caj = storage_mat[:, 2]
            self.v_2_caj = storage_mat[:, 3]
            self.T_1_caj = storage_mat[:, 4]
            self.T_2_caj = storage_mat[:, 5]
        else:
            # 2 is cold.
            W8 = self.get_variable(self.point, 'W8')
            W18 = self.get_variable(self.point, 'W18')
            M8 = self.get_variable(self.point, 'M8')
            M18 = self.get_variable(self.point, 'M18')
            T8 = self.get_variable(self.point, 'T8')
            T18 = self.get_variable(self.point, 'T18')

            C18 = choice_physics.AtmosphericEffects.getVel(choice_data.gamma_air, T18, M18)
            C8 = choice_physics.AtmosphericEffects.getVel(choice_data.gamma_gas, T8, M8)

            self.dmdt_1_caj = np.full(self.n_traj, W8)
            self.dmdt_2_caj = np.full(self.n_traj, W18)
            self.v_1_caj = np.full(self.n_traj, C8)
            self.v_2_caj = np.full(self.n_traj, C18)
            self.T_1_caj = np.full(self.n_traj, T8)
            self.T_2_caj = np.full(self.n_traj, T18)

    def setAirfrm(self, noise_choice):
        """ Sets the airframe performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_airfrm_performance.txt'
        if self.trajPerf:
            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 9)
            self.psi_airfrm = storage_mat[:, 0]
            self.defl_flap_airfrm = storage_mat[:, 5]
            self.defl_slat_airfrm = storage_mat[:, 6]
            self.LandingGear = storage_mat[:, 7]
        else:
            self.psi_airfrm = np.full(self.n_traj, noise_choice.psi_airfrm_vec[self.ptr])
            self.defl_flap_airfrm = np.full(self.n_traj, noise_choice.defl_flap_airfrm_vec[self.ptr])
            self.defl_slat_airfrm = np.full(self.n_traj, noise_choice.defl_slat_airfrm_vec[self.ptr])
            self.LandingGear = np.full(self.n_traj, noise_choice.LandingGear[self.ptr])

    def setLpt(self):
        """ Sets the low pressure turbine performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_lpt_performance.txt'
        if self.trajPerf:
            if choice_aux.containsStars(file_path):
                print('Discovered raw trajectory mission file (lpt) - computing *** and ** from other data and '
                      're-writing file (this is done only once)')
                choice_aux.preProcessLptFile(file_path, self.weight_choice.De_lpt, self.weight_choice.Ae_lpt)

            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 5)
            self.Vtr_lpt = storage_mat[:, 0]
            self.Texit_lpt = storage_mat[:, 1]
            self.xnl_lpt = storage_mat[:, 2]
            self.mcore_lpt = storage_mat[:, 3]
            self.Cax_lpt = storage_mat[:, 4]
        else:
            # use performanceResults.txt file and additional data to estimate the trajectory variables.
            xnl = self.get_variable(self.point, 'NL')
            t5 = self.get_variable(self.point, 'T5')
            w5 = self.get_variable(self.point, 'W5')
            p5 = self.get_variable(self.point, 'P5')
            Cax = choice_aux.get_Cax(choice_data.gamma_gas, p5, t5, w5, self.weight_choice.Ae_lpt)

            xnl_lpt = xnl * self.weight_choice.gbx_ratio  # xnl should be fan speed
            self.Vtr_lpt = np.full(self.n_traj, (math.pi * self.weight_choice.De_lpt) * xnl_lpt * 0.70)
            # when the relative tip speed is unknown, use 70% of the blade tip speed
            self.Texit_lpt = np.full(self.n_traj, t5)
            self.xnl_lpt = np.full(self.n_traj, xnl * self.weight_choice.gbx_ratio)  # xnl should be fan speed
            self.mcore_lpt = np.full(self.n_traj, w5)
            self.Cax_lpt = np.full(self.n_traj, Cax)

    def setComb(self):
        """ Sets the combustor performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_comb_performance.txt'
        if self.trajPerf:
            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 7)
            self.P3_comb = storage_mat[:, 0]
            self.P4_comb = storage_mat[:, 1]
            self.P7_comb = storage_mat[:, 2]
            self.T3_comb = storage_mat[:, 3]
            self.T4_comb = storage_mat[:, 4]
            self.T5_comb = storage_mat[:, 5]
            self.W3_comb = storage_mat[:, 6]
        else:
            # use performanceResults.txt file and additional data to estimate the trajectory variables.
            P3 = self.get_variable(self.point, 'P3')
            P4 = self.get_variable(self.point, 'P4')
            P7 = self.get_variable(self.point, 'P5')
            T3 = self.get_variable(self.point, 'T3')
            T4 = self.get_variable(self.point, 'T4')
            T5 = self.get_variable(self.point, 'T5')
            W3 = self.get_variable(self.point, 'W3')

            self.P3_comb = np.full(self.n_traj, P3)
            self.P4_comb = np.full(self.n_traj, P4)
            self.P7_comb = np.full(self.n_traj, P7)
            self.T3_comb = np.full(self.n_traj, T3)
            self.T4_comb = np.full(self.n_traj, T4)
            self.T5_comb = np.full(self.n_traj, T5)
            self.W3_comb = np.full(self.n_traj, W3)

    def setFan(self):
        """ Sets the fan performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_fan_performance.txt'
        if self.trajPerf:
            if choice_aux.containsStars(file_path):
                print('Discovered raw trajectory mission file (fan) - computing Mtip and Mu from other data and '
                      're-writing file (this is done only once)')
                choice_aux.preProcessFanFile(file_path, self.weight_choice.D1_fan, self.weight_choice.A2_fan)

            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 5)
            self.Mtip_fan = storage_mat[:, 0]
            self.Mu_fan = storage_mat[:, 1]
            self.xnl_fan = storage_mat[:, 2]
            self.dt_fan = storage_mat[:, 3]
            self.g1_fan = storage_mat[:, 4]
        else:
            # use performanceResults.txt file and additional data to estimate the trajectory variables.
            xnl = self.get_variable(self.point, 'NL')
            g1 = self.get_variable(self.point, 'W2')
            t1 = self.get_variable(self.point, 'T2')
            dt = self.get_variable(self.point, 'T13') - t1
            p1 = self.get_variable(self.point, 'P2')
            [Mtip, Mu, Umid] = choice_aux.setMachNumbers(p1, t1, g1, self.weight_choice.A2_fan,
                                                         self.weight_choice.D1_fan, xnl, choice_data.gamma_air)

            self.xnl_fan = np.full(self.n_traj, xnl)
            self.dt_fan = np.full(self.n_traj, dt)
            self.g1_fan = np.full(self.n_traj, g1)
            self.Mtip_fan = np.full(self.n_traj, Mtip)
            self.Mu_fan = np.full(self.n_traj, Mu)

    def setLpc(self):
        """ Sets the low pressure compressor performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_lpc_performance.txt'
        if self.trajPerf:
            if choice_aux.containsStars(file_path):
                print('Discovered raw trajectory mission file (lpc) - computing Mtip and Mu from other data and '
                      're-writing file (this is done only once)')
                A2_lpc = math.pi * ((self.weight_choice.D1_lpc / 2) ** 2 - (self.weight_choice.Dh1_lpc / 2) ** 2)
                choice_aux.preProcessFanFile(file_path, self.weight_choice.D1_lpc, A2_lpc)

            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 5)
            self.Mtip_lpc = storage_mat[:, 0]
            self.Mu_lpc = storage_mat[:, 1]
            self.xnl_lpc = storage_mat[:, 2]
            self.dt_lpc = storage_mat[:, 3]
            self.g1_lpc = storage_mat[:, 4]
        else:
            # use performanceResults.txt file and additional data to estimate the trajectory variables.
            if not self.openRotor:
                xnl = self.get_variable(self.point, 'NL')
                xni = xnl * self.weight_choice.gbx_ratio
            else:
                sys.exit('no open rotor method yet - lpc')

            g1 = self.get_variable(self.point, 'W23')
            t1 = self.get_variable(self.point, 'T23')
            p1 = self.get_variable(self.point, 'P23')
            A = (math.pi / 4) * (self.weight_choice.D1_lpc ** 2 - self.weight_choice.Dh1_lpc ** 2)
            [Mtip, Mu, Umid] = choice_aux.setMachNumbers(p1, t1, g1, A, self.weight_choice.D1_lpc, xni,
                                                         choice_data.gamma_air)

            if self.weight_choice.gbx_ratio == 1:
                dt = (0.6 * Umid ** 2) / (choice_data.cp_air * self.weight_choice.n_stages_LPC)
            else:
                dt = (0.45 * Umid ** 2) / (choice_data.cp_air * self.weight_choice.n_stages_LPC)

            self.xnl_lpc = np.full(self.n_traj, xni)
            self.dt_lpc = np.full(self.n_traj, dt)
            self.g1_lpc = np.full(self.n_traj, g1)
            self.Mtip_lpc = np.full(self.n_traj, Mtip)
            self.Mu_lpc = np.full(self.n_traj, Mu)

    def setff(self):
        """ Sets the fuselage fan performance parameters. """
        file_path = 'Input/' + self.point.rstrip() + '_fuselagefan_performance.txt'
        if self.trajPerf:
            if choice_aux.containsStars(file_path):
                print('Discovered raw trajectory mission file (fuselagefan) - computing Mtip and Mu from other data '
                      'and re-writing file (this is done only once)')
                choice_aux.preProcessFanFile(file_path, self.weight_choice.D1_ff, self.weight_choice.A2_ff)

            storage_mat = choice_aux.loadStorageMat(file_path, self.n_traj, 5)
            self.Mtip_ff = storage_mat[:, 0]
            self.Mu_ff = storage_mat[:, 1]
            self.xnl_ff = storage_mat[:, 2]
            self.dt_ff = storage_mat[:, 3]
            self.g1_ff = storage_mat[:, 4]
        else:
            # use performanceResults.txt file and additional data to estimate the trajectory variables.
            xnl = self.get_variable(self.point, 'NFF')
            g1 = self.get_variable(self.point, 'Wff2')
            t1 = self.get_variable(self.point, 'Tff2')
            dt = self.get_variable(self.point, 'Tff3') - t1
            p1 = self.get_variable(self.point, 'Pff2')
            [Mtip, Mu, Umid] = choice_aux.setMachNumbers(p1, t1, g1, self.weight_choice.A2_ff, self.weight_choice.D1_ff,
                                                         xnl, choice_data.gamma_air)

            self.xnl_ff = np.full(self.n_traj, xnl)
            self.dt_ff = np.full(self.n_traj, dt)
            self.g1_ff = np.full(self.n_traj, g1)
            self.Mtip_ff = np.full(self.n_traj, Mtip)
            self.Mu_ff = np.full(self.n_traj, Mu)

    def coAxialJet_ffn(self):
        """ Sets the jet performance parameters for fuselage fan nozzle . """

        if self.trajPerf:
            storage_mat = choice_aux.loadStorageMat(self.point.rstrip() + '_ffnJet_performance.txt', self.n_traj, 6)
            self.dmdt_1_caj_ffn = storage_mat[:, 0]
            self.dmdt_2_caj_ffn = storage_mat[:, 1]
            self.v_1_caj_ffn = storage_mat[:, 2]
            self.v_2_caj_ffn = storage_mat[:, 3]
            self.T_1_caj_ffn = storage_mat[:, 4]
            self.T_2_caj_ffn = storage_mat[:, 5]
        else:
            # 2 is cold.
            Wffn8 = self.get_variable(self.point, 'Wffn8')
            Wffn18 = self.get_variable(self.point, 'Wffn18')
            Mffn8 = self.get_variable(self.point, 'Mffn8')
            Mffn18 = self.get_variable(self.point, 'Mffn18')
            Tffn8 = self.get_variable(self.point, 'Tffn8')
            Tffn18 = self.get_variable(self.point, 'Tffn18')

            Cffn18 = choice_physics.getVel(choice_data.gamma_air, Tffn18, Mffn18)
            Cffn8 = choice_physics.getVel(choice_data.gamma_gas, Tffn8, Mffn8)

            self.dmdt_1_caj_ffn = np.full(self.n_traj, Wffn8)
            self.dmdt_2_caj_ffn = np.full(self.n_traj, Wffn18)
            self.v_1_caj_ffn = np.full(self.n_traj, Cffn8)
            self.v_2_caj_ffn = np.full(self.n_traj, Cffn18)
            self.T_1_caj_ffn = np.full(self.n_traj, Tffn8)
            self.T_2_caj_ffn = np.full(self.n_traj, Tffn18)

    def get_variable(self, point, name):
        """ Returns the value of the item with the specified key for the provided point. """
        if point == 'Take-off':
            variable = self.mpd.perf_file_to.get(name)
        elif point == 'Sideline':
            variable = self.mpd.perf_file_sl.get(name)
        elif point == 'Cutback':
            variable = self.mpd.perf_file_cutback.get(name)
        elif point == 'Approach':
            variable = self.mpd.perf_file_approach.get(name)
        elif point == 'Cruise':
            variable = self.mpd.perf_file_cr.get(name)
        elif point == 'Top-of-climb':
            variable = self.mpd.perf_file_toc.get(name)
        return variable


def set_performance_choice(ptr, modules, n_traj, mpd, noise_choice, weight_choice):
    """ Returns a PerformanceChoice object that contains the performance data for each component. """
    performance_choice = PerformanceChoice(ptr, noise_choice.trajectory_performance, noise_choice.opPnt[ptr].strip(),
                                           n_traj, weight_choice, mpd)
    openRotor = performance_choice.getIsOpenRotor(modules)
    if openRotor: sys.exit('Model is open rotor - noise modelling not implemented yet')

    for module in modules:
        if module == 'Fan': performance_choice.setFan()
        if module == 'Ipc' or module == 'Lpc': performance_choice.setLpc()
        if module == 'Lpt': performance_choice.setLpt()
        if module == 'Comb': performance_choice.setComb()
        if module == 'cold_nozzle': performance_choice.coAxialJet()
        if module == 'fuselage_fan': performance_choice.setff()
        if module == 'ff_nozzle': performance_choice.coAxialJet_ffn()

    performance_choice.setAirfrm(noise_choice)

    return performance_choice


class Prms:
    """ A class that defines the rms acoustic pressure for each aircraft noise component. """

    def __init__(self, npts):
        """ Initializes the prms matrix for each component. """
        nfreq = choice_data.nfreq
        nthet = choice_data.nthet
        self.Fan_inlet_tone = np.zeros((nfreq, nthet, npts))
        self.Fan_discharge_tone = np.zeros((nfreq, nthet, npts))
        self.Fan_inlet_broadband = np.zeros((nfreq, nthet, npts))
        self.Fan_discharge_broadband = np.zeros((nfreq, nthet, npts))
        self.Fan_inlet_combination = np.zeros((nfreq, nthet, npts))
        self.Fan_inlet = np.zeros((nfreq, nthet, npts))
        self.Fan_discharge = np.zeros((nfreq, nthet, npts))
        self.Lpc_inlet_tone = np.zeros((nfreq, nthet, npts))
        self.Lpc_discharge_tone = np.zeros((nfreq, nthet, npts))
        self.Lpc_inlet_broadband = np.zeros((nfreq, nthet, npts))
        self.Lpc_discharge_broadband = np.zeros((nfreq, nthet, npts))
        self.Lpc_inlet_combination = np.zeros((nfreq, nthet, npts))
        self.Lpc_inlet = np.zeros((nfreq, nthet, npts))
        self.Lpc_discharge = np.zeros((nfreq, nthet, npts))
        self.Comb = np.zeros((nfreq, nthet, npts))
        self.Lpt = np.zeros((nfreq, nthet, npts))
        self.Caj = np.zeros((nfreq, nthet, npts))
        self.Airfrm = np.zeros((nfreq, nthet, npts))
        self.Ff_inlet_tone = np.zeros((nfreq, nthet, npts))
        self.Ff_discharge_tone = np.zeros((nfreq, nthet, npts))
        self.Ff_inlet_broadband = np.zeros((nfreq, nthet, npts))
        self.Ff_discharge_broadband = np.zeros((nfreq, nthet, npts))
        self.Ff_inlet_combination = np.zeros((nfreq, nthet, npts))
        self.Ff_inlet = np.zeros((nfreq, nthet, npts))
        self.Ff_discharge = np.zeros((nfreq, nthet, npts))
        self.Caj_ffn = np.zeros((nfreq, nthet, npts))


def compute_noise_sources(operatingPoint, traj, modules, noise, weight, performance, nop):
    """
    Compute the acoustic pressure for each noise component.
    :param operatingPoint: Approach, Cutback, Sideline
    :param traj: A Trajectory object with the trajectory data
    :param modules: Fan, Lpc, Lpt, etc.
    :param noiseChoice: A noiseChoice object with the required input arguments for the noise prediction
    :param weightChoice: A weightChoice object with the engine size and architecture
    :param performanceChoice: A performanceChoice object with the engine performance data
    :param nop: Number of operating point
    :return: A Prms object with the rms acoustic pressure for each component, the directivity angle and the 1/3 octave
    band frequencies
    """

    theta = np.array([0.0 + float(i) * 5.0 for i in range(choice_data.nthet)])
    [fband, f, freq] = choice_aux.set_frequencies(choice_data.nb, choice_data.nfreq, float(choice_data.fmin),
                                                  float(choice_data.fmax))
    prms = Prms(traj.n_traj_pts)

    airframe = choice_physics.Airframe(operatingPoint, noise.N_wheels_airfrm, noise.N_struts_airfrm, noise.nlg_airfrm,
                                       noise.d_wheel_airfrm, noise.d_strut_airfrm, noise.d_wire_airfrm, noise.NoFlap,
                                       noise.S_flap_airfrm, noise.span_flap_airfrm, noise.Sw_airfrm, noise.span_airfrm,
                                       noise.ND, noise.span_hor_tail, noise.Sht, noise.span_ver_tail, noise.Svt,
                                       noise.flap_type, noise.slat_type, theta, fband)
    if 'cold_nozzle' in modules: jet = choice_physics.Jet(weight.A_core_caj, weight.A_bypass_caj, noise.type_nozzles,
                                                          theta, fband)
    if 'Comb' in modules:
        comb = choice_physics.Combustor(weight.type_comb, weight.Aec_comb, weight.De_comb, weight.Dh_comb,
                                        weight.Lc_comb, weight.h_comb, weight.Nfmax_comb, theta, fband)
    if 'Lpt' in modules:
        turb = choice_physics.Turbine(weight.N_rotors_lpt, weight.n_stages_lpt, weight.SRS_lpt, theta, fband, f)
    if 'Fan' in modules:
        fan = choice_physics.FanCompressor('Fan', weight.MtipD_fan, weight.N_rotors_fan, weight.N_stators_fan,
                                           weight.rss_fan, theta, fband, f)
    if 'Ipc' in modules:
        lpc = choice_physics.FanCompressor('Ipc', weight.MtipD_lpc, weight.N_rotors_lpc, weight.N_stators_lpc,
                                           weight.rss_lpc, theta, fband, f)
    if 'Lpc' in modules:
        lpc = choice_physics.FanCompressor('Lpc', weight.MtipD_lpc, weight.N_rotors_lpc, weight.N_stators_lpc,
                                           weight.rss_lpc, theta, fband, f)
    if 'fuselage_fan' in modules:
        ffan = choice_physics.FanCompressor('fuselage_fan', weight.MtipD_ff, weight.N_rotors_ff, weight.N_stators_ff,
                                            weight.rss_ff, theta, fband, f)
    if 'ff_nozzle' in modules: ffjet = choice_physics.Jet(weight.A_core_caj_ffn, weight.A_bypass_caj_ffn, 'mix')

    # evaluate component noise models
    for i in range(traj.n_traj_pts):  # for all points along trajectory - evaluate noise sources
        pa = choice_physics.AtmosphericEffects.get_p_ambient(traj.y[i])
        for module in modules:  # evaluate all noise sources in this engine model
            if module.rstrip() == 'Fan':
                temp = fan.calcFanandCompressor(operatingPoint, performance.Mtip_fan[i], performance.Mu_fan[i],
                                                performance.dt_fan[i], performance.xnl_fan[i], performance.g1_fan[i])
                prms.Fan_inlet_tone[:, :, i] = temp[0]
                prms.Fan_discharge_tone[:, :, i] = temp[1]
                prms.Fan_inlet_broadband[:, :, i] = temp[2]
                prms.Fan_discharge_broadband[:, :, i] = temp[3]
                prms.Fan_inlet_combination[:, :, i] = temp[4]

                if noise.use_suppression_factor:
                    prms.Fan_inlet[:, :, i] = choice_physics.calcSuppression(temp[5], noise.S_fan_inlet)
                    prms.Fan_discharge[:, :, i] = choice_physics.calcSuppression(temp[6], noise.S_fan_dis)
                else:
                    prms.Fan_inlet[:, :, i] = temp[5]
                    prms.Fan_discharge[:, :, i] = temp[6]
            elif module == 'Ipc' or module == 'Lpc':
                temp = lpc.calcFanandCompressor(operatingPoint, performance.Mtip_lpc[i], performance.Mu_lpc[i],
                                                performance.dt_lpc[i], performance.xnl_lpc[i], performance.g1_lpc[i])
                prms.Lpc_inlet_tone[:, :, i] = temp[0]
                prms.Lpc_inlet_broadband[:, :, i] = temp[2]
                prms.Lpc_inlet_combination[:, :, i] = temp[4]
                prms.Lpc_inlet[:, :, i] = temp[5]
            elif module == 'Lpt':
                prms.Lpt[:, :, i] = turb.calcTurbine(performance.Vtr_lpt[i], performance.Texit_lpt[i],
                                                     performance.xnl_lpt[i], performance.mcore_lpt[i],
                                                     performance.Cax_lpt[i])

                if noise.use_suppression_factor:
                    prms.Lpt[:, :, i] = choice_physics.calcSuppression(prms.Lpt[:, :, i], noise.S_lpt)
            elif module == 'Comb':
                prms.Comb[:, :, i] = comb.calcComb(min(noise.Nf_comb_ign[nop], weight.Nfmax_comb),
                                                   noise.Nf_comb_pattern, pa, performance.P3_comb[i],
                                                   performance.P4_comb[i], performance.P7_comb[i], traj.ta[i],
                                                   performance.T3_comb[i], performance.T4_comb[i],
                                                   performance.T5_comb[i], performance.W3_comb[i])
            elif module == 'cold_nozzle':
                prms.Caj[:, :, i] = jet.calcCaj(performance.dmdt_1_caj[i], performance.dmdt_2_caj[i],
                                                performance.v_1_caj[i], performance.v_2_caj[i], performance.T_1_caj[i],
                                                performance.T_2_caj[i], traj.ta[i], pa)
            elif module == 'fuselage_fan':
                temp = ffan.calcFanandCompressor(module, operatingPoint, performance.Mtip_ff[i], performance.Mu_ff[i],
                                                 performance.dt_ff[i], performance.xnl_ff[i], performance.g1_ff[i])
                prms.Ff_inlet_tone[:, :, i] = temp[0]
                prms.Ff_discharge_tone[:, :, i] = temp[1]
                prms.Ff_inlet_broadband[:, :, i] = temp[2]
                prms.Ff_discharge_broadband[:, :, i] = temp[3]
                prms.Ff_inlet_combination[:, :, i] = temp[4]
                prms.Ff_inlet[:, :, i] = temp[5]
                prms.Ff_discharge[:, :, i] = temp[6]
            elif module == 'ff_nozzle':
                prms.Caj_ffn[:, :, i] = ffjet.calcCaj(performance.dmdt_1_caj_ffn[i], 0.0, performance.v_1_caj_ffn[i],
                                                      0.0, performance.T_1_caj_ffn[i], 0.0, traj.ta[i], pa)
            else:
                pass

        prms.Airfrm[:, :, i] = airframe.calcAirfrm(traj.ta[i], traj.y[i], traj.Ma[i], traj.Va[i],
                                                   performance.defl_flap_airfrm[i], performance.defl_slat_airfrm[i],
                                                   performance.LandingGear[i])

    # include multiple engines
    for key in prms.__dict__:
        if key != 'Airfrm':
            prms.__dict__[key] = include_multiple_engines(noise.no_engines, prms.__dict__[key])

    if noise.gen_noise_source_matr:
        prms_int = Prms(traj.n_traj_pts)
        # for key in prms.__dict__:
        #     prms_int.__dict__[key] = directivity_interpolation(
        #         prms.__dict__[key], theta, np.degrees(performanceChoice.psi_airfrm), traj.n_traj_pts)
        choice_aux.gen_noise_source_matr_subr(operatingPoint, choice_data.nfreq, choice_data.nthet, traj.n_traj_pts,
                                              prms)

    return [prms, theta, fband]


def include_multiple_engines(no_engines, prms):
    """ Account for multiple engines on the aircraft. """
    return np.sqrt(float(no_engines) * prms ** 2)


def directivity_interpolation(prms, theta, psi, ntraj):
    """ Interpolate the rms acoustic pressure accounting for aircraft pitch. """
    prmsi = np.zeros((choice_data.nfreq, choice_data.nthet, ntraj))
    for ifr in range(choice_data.nfreq):
        for jtr in range(ntraj):
            prmsi[ifr, :, jtr] = interp1d(theta, prms[ifr, :, jtr], fill_value="extrapolate")(theta + psi[jtr])
            prmsi[ifr, prmsi[ifr, :, jtr] < 0, jtr] = 0
    return prmsi


class NoiseMatrices:
    """
    A class that defines the noise level for each aircraft noise component.
    :param modules: Fan, Lpc, Lpt, etc.
    """

    def __init__(self, modules = None):
        """ Initializes the noise level matrix for each component. """
        if modules is not None:
            if 'Fan' in modules:
                self.Fan_inlet = []
                self.Fan_discharge = []
            if 'Ipc' or 'Lpc' in modules: self.Lpc_inlet = []
            if 'Lpt' in modules: self.Lpt = []
            if 'Comb' in modules: self.Comb = []
            if 'cold_nozzle' in modules: self.Caj = []
            if 'fuselage_fan' in modules:
                self.Ff_inlet = []
                self.Ff_discharge = []
            if 'ff_nozzle' in modules:
                self.Caj_ffn = []
            self.Airfrm = []


def interpolate_to_t_source(traj, modules, prms):
    """
    Computes the Sound Pressure Level for the times that the sound reaches the microphone.
    :param traj: A Trajectory object with the trajectory data
    :param modules: Fan, Lpc, Lpt, etc.
    :param prms: A Prms object with the rms acoustic pressure for each component
    :return: Fot the times that the sound reaches the microphone, an SPL object with the sound pressure level for each
    component, the observation angle, the Mach number and the angle of attack
    """

    xsii = interp1d(traj.time, traj.xsi, fill_value="extrapolate")(traj.t_source)
    Mai = interp1d(traj.time, traj.Ma, fill_value="extrapolate")(traj.t_source)
    Tai = interp1d(traj.time, traj.ta, fill_value="extrapolate")(traj.t_source)
    alphai = interp1d(traj.time, traj.alpha, fill_value="extrapolate")(traj.t_source)

    SPLi = NoiseMatrices(modules)

    for key in SPLi.__dict__:
        SPLi.__dict__[key] = compute_SPLi(source_interpolation(prms.__dict__[key], traj))

    return [SPLi, xsii, Mai, Tai, alphai]


def compute_SPLi(prmsi):
    """ Computes Sound Pressure Level matrix from rms acoustic pressure. """
    return 20.0 * np.log10(prmsi / choice_data.p0)


def source_interpolation(prms, traj):
    """ Interpolate the rms acoustic pressure for the times that reach the microphone. """
    return np.array([[interp1d(traj.time, prms[ifr, jth, :], fill_value="extrapolate")(traj.t_source) for jth in
                      range(choice_data.nthet)] for ifr in range(choice_data.nfreq)])


def compute_flight_effects(use_ground_refl, spherical_spr, atm_atten, traj, ymic, SPLi, xsii, Mai, Tai, alphai, theta,
                           fband):
    """
    Computes the sound pressure level matrices along the trajectory accounting for propagation effects.
    :param use_ground_refl: True to account for ground reflection or False else
    :param spherical_spr: True to account for spherical spreading or False else
    :param atm_atten: True to account for atmospheric attenuation or False else
    :param traj: A Trajectory object with the trajectory data
    :param ymic: Microphone height (m)
    :param SPLi: Sound Pressure Level matrix
    :param xsii: Observation angle (rad)
    :param Mai: Mach number
    :param Tai: Atmospheric temperature (K)
    :param alphai: Angle of attack
    :param theta: Directivity angle (deg)
    :param fband: Octave band frequency (Hz)
    :return: Sound Pressure Level accounting for flight effects and shifted frequency due to aircraft motion
    """

    propagation = choice_physics.PropagationEffects(ymic, use_ground_refl, spherical_spr, atm_atten, fband, xsii, Mai,
                                                    alphai)
    SPLp = NoiseMatrices()
    prmsp = NoiseMatrices()
    for key in SPLi.__dict__:
        [SPLp.__dict__[key], prmsp.__dict__[key]] = \
            propagation.flightEffects(traj.n_times, theta, traj.x_source, traj.y_source, traj.r1, Tai, SPLi.__dict__[key])
    return [SPLp, propagation.fobs]


def CertificationData(n_times, fobs, fband, SPLp):
    """ Computes the Effective Perceived Noise Level (EPNL) for each aircraft noise source and for the total
    aircraft. """
    PNL = NoiseMatrices()
    PNLT = NoiseMatrices()
    EPNL = NoiseMatrices()
    for key in SPLp.__dict__:
        PNL.__dict__[key] = choice_physics.PerceivedNoiseMetrics().getPNL(n_times, fobs, SPLp.__dict__[key])
        PNLT.__dict__[key] = choice_physics.PerceivedNoiseMetrics().getPNLT(n_times, fband, PNL.__dict__[key],
                                                                            SPLp.__dict__[key])
        EPNL.__dict__[key] = choice_physics.PerceivedNoiseMetrics().getEPNL(PNLT.__dict__[key])

    all_sources = np.array([PNLT_value for PNLT_value in PNLT.__dict__.values()])
    PNLT.tot = choice_aux.getTotLevel(all_sources)
    EPNL.tot = choice_physics.PerceivedNoiseMetrics().getEPNL(PNLT.tot)

    return EPNL


def save_noise_points(fname, opPoint, fuselage_fan, EPNL):
    """ Saves the EPNL for each component and for the total in output file. """
    with open(fname, 'a') as fp:
        fp.write('\n' + '***Operating point is ' + opPoint + '\n')
        fp.write('Fan inlet EPNL is'.rjust(23) + str(format(round(EPNL.Fan_inlet, 4), '.4f')).rjust(17) + '\n')
        fp.write('Fan discharge EPNL is'.rjust(23) + str(format(round(EPNL.Fan_discharge, 4), '.4f')).rjust(17) + '\n')
        fp.write('Inlet LPC EPNL is'.rjust(23) + str(format(round(EPNL.Lpc_inlet, 4), '.4f')).rjust(17) + '\n')
        fp.write('LPT EPNL is'.rjust(23) + str(format(round(EPNL.Lpt, 4), '.4f')).rjust(17) + '\n')
        fp.write('Comb EPNL is'.rjust(23) + str(format(round(EPNL.Comb, 4), '.4f')).rjust(17) + '\n')
        fp.write('Caj EPNL is'.rjust(23) + str(format(round(EPNL.Caj, 4), '.4f')).rjust(17) + '\n')

        # Engine total noise
        kernel = (10.0 ** (EPNL.Fan_inlet / 10.0) + 10.0 ** (EPNL.Fan_discharge / 10.0) +
                  10.0 ** (EPNL.Lpc_inlet / 10) + 10.0 ** (EPNL.Lpt / 10.0) + 10.0 **
                  (EPNL.Comb / 10.0) + 10.0 ** (EPNL.Caj / 10.0))
        fp.write(
            'EPNL_tot_engine is'.rjust(23) + str(format(round(10.0 * math.log10(kernel), 4), '.4f')).rjust(17) + '\n')

        fp.write(
            'Airframe EPNL is'.rjust(23) + str(format(round(EPNL.Airfrm, 4), '.4f')).rjust(17) + '\n')

        if fuselage_fan:
            fp.write('EPNL_inlet_ff is' + str(format(round(EPNL.Ff_inlet, 4), '.4f')).rjust(17) + '\n')
            fp.write('EPNL_discharge_ff is' + str(format(round(EPNL.Ff_discharge, 4), '.4f')).rjust(17) + '\n')
            fp.write('EPNL_caj_ff is'.rjust(23) + str(format(round(EPNL.Caj_ffn, 4), '.4f')).rjust(17) + '\n')
            # EPNL_tot_ff is the total EPNL of the fuselage fan assembly which includes the fuselage fan and the nozzle
            fp.write('EPNL_tot_ff is'.rjust(23) +
                     str(format(round(10. * math.log10(10.0 ** (EPNL.Ff_inlet / 10.0) +
                                                       10.0 ** (EPNL.Ff_discharge / 10.0) +
                                                       10.0 ** (EPNL.Caj_ffn / 10.0)), 4), '.4f')).rjust(17) + '\n')

        fp.write('EPNL_tot with D is'.rjust(23) + str(format(round(EPNL.tot, 4), '.4f')).rjust(17))


def certificationLimits(noEngines, MTOW):
    """ Computes EPNL certification limits. """

    EPNL = choice_aux.chapter3(noEngines, MTOW)

    EPNL_cum = EPNL.lateral + EPNL.cutback + EPNL.approach
