# -*- coding: cp1252 -*-
"""
Module choice_physics

Physics and methods for the noise prediction broken down in modules.
"""

import sys
import math
import numpy as np
import cmath
from scipy import interpolate
import choice.choice_data as choice_data
import choice.choice_aux as choice_aux


class AtmosphericEffects:

    @staticmethod
    def get_t_ambient(alt, dTisa):
        """
        Computes the ambient temperature at a given altitude.

        :param ndarray alt: Altitude (m)
        :param ndarray dTisa: Deviation from ISA temperature

        :return ndarray: Ambient temperature at given altitude
        """
        temp = np.zeros_like(alt)
        alt = np.asarray(alt)
        temp[alt < 11000.0] = 288.15 - 6.5e-3 * alt[alt < 11000.0] + dTisa
        temp[(11000.0 <= alt) & (25000.0 > alt)] = 216.65 + dTisa

        return temp

    @staticmethod
    def get_p_ambient(alt):
        """
        Computes the ambient pressure at a given altitude.

        :param ndarray alt: Altitude (m)

        :return ndarray: Ambient pressure at given altitude
        """
        pa = np.zeros_like(alt)
        alt = np.asarray(alt)
        pa[alt < 11000.0] = 101325.0 * (1 - 2.2557E-5 * alt[alt < 11000.0]) ** 5.2561
        pa[(11000.0 <= alt) & (25000.0 > alt)] = 22632.0 / np.exp(
            1.5769e-4 * (alt[(11000.0 <= alt) & (25000.0 > alt)] - 11000.0))
        pa[alt >= 25000.0] = 22632.0 / np.exp(1.5769e-4 * (alt[alt >= 25000.0] - 11000.0))

        return pa

    @staticmethod
    def getVel(gam, T, M):
        """
        Computes speed from temperature and Mach number.

        :param float gam: Specific hear ratio
        :param float T: Temperature (K)
        :param float M: Mach number

        return float: Speed (m/s)
        """
        ts = T / (1.0 + ((gam - 1.0) / 2.0) * M ** 2)
        return math.sqrt(gam * choice_data.Risa * ts) * M

    @staticmethod
    def get_sound_speed(temp):
        """
        Computes speed of sound for the given temperature.

        :param ndarray temp: Ambient temperature (K)
        
        :return ndarray: Sound speed (m/s)
        """
        c = 331.4 * np.sqrt(temp / 273.15)
        return c


class NoiseSource:

    @staticmethod
    def suppression(prms, s_db):
        """ Calculates the suppressed acoustic pressure. """
        return np.sqrt((prms ** 2) * 10 ** (-s_db / 10))


class Airframe(NoiseSource):
    """
    Instantiate airframe source noise prediction.

    :param list opPoint: Approach, Cutback, Sideline
    :param int N_wheels: Array containing the number of wheels in each landing gear system
    :param int N_struts: Array containing the number of main struts in each landing gear system
    :param int nlg: Number of landing gear systems
    :param ndarray d_wheel: 1D array containing the diameters of landing gear wheels (in)
    :param ndarray d_strut: 1D array containing typical diameter of landing gear struts (in)
    :param ndarray d_wire: 1D array containing typical diameter of landing gear wire/small pipes (in)
    :param int NoFlap: Number of flap elements
    :param ndarray S_flap: 1D array containing the area of each flap (m**2)
    :param ndarray span_flap: 1D array containing the span of each flap (m)
    :param float Sw: Aircraft reference area (m**2)
    :param float span: Wing span (m)
    :param int ND: Coefficient for jet aircraft (1 => + 6dB in wing trailing edge OASPL)
    :param float span_hor_tail: Horizontal tail span (m)
    :param float Sht: Horizontal tail area (m)
    :param float span_ver_tail: Vertical tail span (m)
    :param float Svt: Vertical tail area (m)
    :param str flap_type: Flap type: 1slot, 2slot, 3slot
    :param str slat_type: Leading edge high-lift system type: slat, le_flap
    :param ndarray theta: 1D array containing the longitudinal directivity angles (deg)
    :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
    """

    def __init__(self, opPoint, N_wheels, N_struts, nlg, d_wheel, d_strut, d_wire, NoFlap, S_flap, span_flap, Sw, span,
                 ND, span_hor_tail, Sht, span_ver_tail, Svt, flap_type, slat_type, theta, fband):
        self.opPoint = opPoint
        self.N_wheels = N_wheels
        self.N_struts = N_struts
        self.NoLG = nlg
        self.d_wheel = d_wheel
        self.d_strut = d_strut
        self.d_wire = d_wire
        self.NoFlaps = NoFlap
        self.Sf = S_flap
        self.bf = span_flap
        self.Sw = Sw
        self.bw = span
        self.ND = ND
        self.bht = span_hor_tail
        self.Sht = Sht
        self.bvt = span_ver_tail
        self.Svt = Svt
        self.flap_type = flap_type
        self.slat_type = slat_type
        self.theta = theta
        self.fband = fband
        self.nthet = len(theta)
        self.nfreq = len(fband)
        self.R = 1
        self.Po = \
            np.array([[-9.877825, -9.675465, -9.771083, -9.700787, -9.791079, -1.001907e1, -9.975054, -1.008466e1],
                      [-7.632699, -6.857752, -5.604022, -5.892989, -6.587548, -6.9668, -6.810322, -9.393665],
                      [1.661801e-1, -1.117547, -2.498093, -6.764243e-1, -7.746056e-1, 2.283479, 2.421265, 6.13125],
                      [-1.444081, -1.222514, -7.675554e-1, -6.764243e-1, -1.553996, -3.257356, -3.296517, -4.564913]])
        self.Qo = \
            np.array([[-1.025003e1, -1.066351e1, -1.018087e1, -1.033656e1, -1.044686e1, -1.076274e1, -1.076015e1,
                       -1.059679],
                      [-7.112481, -8.831999, -6.14302, -1.058539e1, -7.7798, -1.142915e1, -1.086217e1, -9.508286],
                      [-1.52365e1, -1.315696e1, -1.525915e1, -1.234325e1, -1.48048e1, -1.585081e1, -1.434196e1,
                       -1.494926e1],
                      [1.0224451e1, 9.298285, 5.283273, 8.264513, 5.24673, 9.143245, 9.873318, 1.202191e1]])
        self.So = \
            np.array([[-3.379221, -4.897243, -5.820157, -6.295315, -6.8834, -7.390352, -5.588916, -3.84597],
                      [3.370469e1, 2.333926e1, 7.075035, -6.580083, -1.700693e1, 3.915408e1, 5.34791, 1.130759e1],
                      [-2.713406e2, -2.496018e2, -2.482546e2, -1.907594e2, -1.803207e2, -1.636399e2, -1.92612e2,
                       -2.409802e2]])
        self.X = \
            np.array([-9.992824, -7.587345, -1.474888e1, 3.307829e1, 1.141251e2, -3.080667e2, 2.104914e2, -4.519879e1])

    def calc(self, Ta, Ha, Ma, Va, phi, defl_flap, defl_slat, LG):
        """
        Airframe source noise model based on a combination of methods found in public literature. Trailing-edge and
        landing gear noise are modelled separately. The references for each method are found in the description of the
        corresponding functions.

        :param float Ta: Atmospheric static temperature (K)
        :param float Ha: Flight altitude (m)
        :param float Ma: Aircraft Mach number
        :param float Va: Aircraft speed (m/s)
        :param float phi: Lateral directivity angle (deg)
        :param float defl_flap: Flap deflection angle (rad)
        :param float defl_flap: Slat deflection angle (rad)
        :param int LG: Landing gear position (0 => retracted, 1 => extended)
        :return ndarray: 2D array of Rms or effective acoustic pressure for airframe component
        """

        Pa = AtmosphericEffects.get_p_ambient(Ha)
        rho0 = Pa / (choice_data.Risa * Ta)

        # Fink's model for trailing edge noise
        if Va > 0.001 and Ha > 0:

            if LG == 0 or LG == 1:
                if defl_flap > 0:
                    flaps = True
                else:
                    flaps = False
                if defl_slat > 0:
                    slats = True
                else:
                    slats = False
                if LG == 0:
                    lgs = False
                else:
                    lgs = True
            else:
                if self.opPoint.strip() == 'Approach' or self.opPoint.strip() == 'Sideline':
                    flaps = True
                    slats = True
                    lgs = True
                else:
                    flaps = False
                    slats = False
                    lgs = False

            # Noise from wing
            ny = DynamicViscocity.calculate(Ta) / rho0
            phi_sid = phi  # Sideline angle (lateral directivity) is calculated from the flyover plane->In flyover, phi=0
            horizontal_factor = np.cos(np.radians(phi_sid))
            wing = self.get_wing_and_tail(Va, ny, self.Sw, self.bw, horizontal_factor, self.ND)
            prms_wing = wing
            ps_trailing_edge = pow(wing, 2)

            # Noise from horizontal tail
            hor_tail = self.get_wing_and_tail(Va, ny, self.Sht, self.bht, horizontal_factor, 0)
            prms_hor_tail = hor_tail
            ps_trailing_edge = pow(hor_tail, 2) + ps_trailing_edge

            # Noise from vertical tail (0 when lateral directivity is fixed to 90 degrees)
            vertical_factor = np.sin(np.radians(phi_sid))
            ver_tail = self.get_wing_and_tail(Va, ny, self.Svt, self.bvt, vertical_factor, 0)
            ps_trailing_edge = pow(ver_tail, 2) + ps_trailing_edge
            ps_flap = np.zeros((self.nfreq, self.nthet))
            if flaps:
                for iflap in range(self.NoFlaps):
                    flap = self.get_te_flaps(Va, self.Sf[iflap], self.bf[iflap], defl_flap, self.flap_type, phi_sid)
                    ps_trailing_edge = ps_trailing_edge + pow(flap, 2)
                    ps_flap = ps_flap + pow(flap, 2)
            else:
                ps_flap = np.zeros(np.shape(ps_trailing_edge))
            prms_flap = np.sqrt(ps_flap)

            if slats:
                horizontal_factor = math.cos(np.radians(phi_sid))
                p_slat = self.get_le_slats(Va, ny, self.Sw, self.bw, horizontal_factor, self.ND, self.slat_type)
                ps_trailing_edge = ps_trailing_edge + pow(p_slat, 2)
            else:
                p_slat = np.zeros(np.shape(ps_trailing_edge))
            prms_slat = p_slat

            SPLte = choice_aux.prms2SPL(
                np.sqrt(ps_trailing_edge)) - 3  # Method predicts the level 3 dB above free field
            prms_trailing_edge = choice_aux.SPL2prms(SPLte)

            if lgs:
                ps_lg = np.zeros(np.shape(ps_trailing_edge))
                ps_lg_m = np.zeros(np.shape(ps_trailing_edge))
                for ilg in range(self.NoLG):
                    p_landinggear = self.getLandingGear(self.N_wheels[ilg], self.N_struts[ilg], Ma * 0.75,
                                                        np.sqrt(choice_data.gamma_air * choice_data.Risa * Ta),
                                                        self.d_strut[ilg], self.d_wire[ilg], self.d_wheel[ilg])
                    ps_lg = ps_lg + pow(p_landinggear, 2)
                    if ilg <= 1:
                        ps_lg_m = ps_lg_m + pow(p_landinggear, 2)
                        prms_main_lg = np.sqrt(ps_lg_m)
                    else:
                        prms_nose_lg = p_landinggear

                prms_lg = np.sqrt(ps_lg)

            else:
                prms_lg = np.zeros(np.shape(ps_trailing_edge))
                prms_nose_lg = np.zeros(np.shape(ps_trailing_edge))
                prms_main_lg = np.zeros(np.shape(ps_trailing_edge))

        else:
            prms_trailing_edge = np.zeros((self.nfreq, self.nthet))
            prms_wing = np.zeros((self.nfreq, self.nthet))
            prms_hor_tail = np.zeros((self.nfreq, self.nthet))
            prms_slat = np.zeros((self.nfreq, self.nthet))
            prms_flap = np.zeros((self.nfreq, self.nthet))
            prms_nose_lg = np.zeros((self.nfreq, self.nthet))
            prms_main_lg = np.zeros((self.nfreq, self.nthet))
            prms_lg = np.zeros((self.nfreq, self.nthet))

        return [np.sqrt(pow(prms_trailing_edge, 2) + pow(prms_lg, 2)), prms_wing, prms_hor_tail, prms_slat, prms_flap,
                prms_nose_lg, prms_main_lg, prms_lg]

    def get_wing_and_tail(self, Va, ny, S, b, dir_factor, ND):
        """
        Trailing edge source noise model for wing and tail surfaces developed by Fink (M.R. Fink, "Noise Component Method
        for Airframe Noise").

        :param float Va: Aircraft speed (m/s)
        :param float ny: Kinematic viscocity (m**2/s)
        :param float S: Wing or tail area (m**2)
        :param float b: Wing or tail span (m)
        :param float dir_factor: Equals cos(phi) where phi is lateral directivity calculated from the flyover plane
        :param int ND: Coefficient for jet aircraft (1 => + 6dB in wing trailing edge OASPL)

        :return ndarray: 2D array of Rms or effective acoustic pressure for wing and tail airframe component
        """
        fband = self.fband
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet
        R = self.R
        delta = 0.37 * (S / b) * ((Va * S / (b * ny)) ** (-0.2))
        H = R
        OASPL_theta = 50 * math.log10(Va / (100 * choice_data.kt2mps)) + 10 * np.log10(delta * b / (H ** 2)) + \
                      ND + 10 * np.log10((dir_factor ** 2) * (np.cos(np.radians(theta) / 2)) ** 2) + 101.3
        OASPL_theta[(theta == 180)] = OASPL_theta[(theta == 175)]
        OASPL = np.tile(np.reshape(OASPL_theta, (1, nthet)), (nfreq, 1))

        fmax = 0.1 * Va / delta  # Maximum amplitude frequency
        fb = np.tile(np.reshape(fband, (nfreq, 1)), (1, nthet))
        SPL = OASPL + 10 * np.log10(0.613 * ((fb / fmax) ** 4) * (((fb / fmax) ** (3 / 2) + 0.5) ** (-4))) - \
              0.03 * (R / (500 * choice_data.ft2m)) * np.abs((fb / fmax) - 1) ** (3 / 2)

        return choice_aux.SPL2prms(SPL)

    def get_te_flaps(self, Va, Sf, bf, defl_flap, flap_type, phi):
        """
        Trailing edge flap source noise model developed by Fink (M.R. Fink, "Airframe Noise Prediction Method",
        FAA-RD-77-29 and M.R. Fink, "Noise Component Method for Airframe Noise").

        :param float Va: Aircraft speed (m/s)
        :param float Sf: Flap area (m**2)
        :param float bf: Flap span (m)
        :param float defl_flap: Flap deflection angle (rad)
        :param float phi: Lateral directivity calculated from the flyover plane

        :return ndarray: 2D array of Rms or effective acoustic pressure for trailing edge flaps
        """
        fband = self.fband
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet
        R = self.R
        H = R
        cf = Sf / bf
        St = fband * cf / Va
        if "1slot" or "2slot" in flap_type:
            G67 = np.zeros(nfreq)
            G67[(St < 2)] = 99 + 10 * np.log10(St[(St < 2)])
            G67[(2 <= St) & (St < 20)] = 103.82 - 6 * np.log10(St[(2 <= St) & (St < 20)])
            G67[(20 <= St)] = 135.04 - 30 * np.log10(St[(20 <= St)])
            G67_dir = np.tile(np.reshape(G67, (nfreq, 1)), (1, nthet))
        elif "3slot" in flap_type:
            G67 = np.zeros(nfreq)
            G67[(St < 2)] = 99 + 10 * np.log10(St[(St < 2)])
            G67[(2 <= St) & (St < 75)] = 102.61 - 2 * np.log10(St[(2 <= St) & (St < 75)])
            G67[(75 <= St)] = 155.11 - 30 * np.log10(St[(75 <= St)])
            G67_dir = np.tile(np.reshape(G67, (nfreq, 1)), (1, nthet))
        dir_factor = np.sin(np.radians(theta) + defl_flap) * np.cos(np.radians(phi))
        dir_factor[dir_factor < 10 ** (-3)] = 10 ** (-3)
        dir_fac = np.tile(np.reshape(20 * np.log10(dir_factor), (1, nthet)), (nfreq, 1))
        SPL = G67_dir + 10 * np.log10(Sf * (np.sin(defl_flap)) ** 2 / (H ** 2)) + \
              60 * math.log10(Va / (100 * choice_data.kt2mps)) + dir_fac

        return choice_aux.SPL2prms(SPL)

    def get_le_slats(self, Va, ny, S, b, dir_factor, ND, slat_type):
        """
        Leading edge high-lift system source noise model developed by Fink (M.R. Fink, "Airframe Noise Prediction
        Method", FAA-RD-77-29 and M.R. Fink, "Noise Component Method for Airframe Noise").

        :param float Va: Aircraft speed (m/s)
        :param float ny: Kinematic viscocity (m**2/s)
        :param float S: Wing or tail area (m**2)
        :param float b: Wing or tail span (m)
        :param float dir_factor: Equals cos(phi) where phi is lateral directivity calculated from the flyover plane
        :param int ND: Coefficient for jet aircraft (1 => + 6dB in wing trailing edge OASPL)

        :return ndarray: 2D array of Rms or effective acoustic pressure for leading edge flaps
        """
        fband = self.fband
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet
        R = self.R
        if "slat" in slat_type:
            SPLi = choice_aux.prms2SPL(self.get_wing_and_tail(Va, ny, S, b, dir_factor, ND)) + 3  # spectrum 1
            prmsi = choice_aux.SPL2prms(SPLi)
            delta = 0.37 * (S / b) * ((Va * S / (b * ny)) ** (-0.2))
            H = R
            OASPL_theta = 50 * math.log10(Va / (100 * choice_data.kt2mps)) + 10 * np.log10(delta * b / (H ** 2)) + \
                          ND + 10 * np.log10((dir_factor ** 2) * (np.cos(np.radians(theta) / 2)) ** 2) + 101.3
            OASPL_theta[(theta == 180)] = OASPL_theta[(theta == 175)]
            OASPL = np.tile(np.reshape(OASPL_theta, (1, nthet)), (nfreq, 1))
            fmax = 0.4562 * Va / delta  # Maximum amplitude frequency
            fb = np.tile(np.reshape(fband, (nfreq, 1)), (1, nthet))
            SPLii = OASPL + 10 * np.log10(0.613 * ((fb / fmax) ** 4) * (((fb / fmax) ** (3 / 2) + 0.5) ** (-4))) - \
                    0.03 * (R / (500 * choice_data.ft2m)) * np.abs((fb / fmax) - 1) ** (3 / 2) + 3  # spectrum 2
            prmsii = choice_aux.SPL2prms(SPLii)
        elif "flap" in slat_type:
            SPLi = choice_aux.prms2SPL(self.get_wing_and_tail(Va, ny, S, b, dir_factor, 8))
            prmsi = choice_aux.SPL2prms(SPLi)
            prmsii = 0.0

        return np.sqrt(pow(prmsi, 2) + pow(prmsii, 2))

    def getLandingGear(self, Nt, Ns, M, c0, dl, dh, dtire):
        """
        Landing gear source noise model based on the method presented by Sen et al. (R. Sen, B. Hardy, K. Yamamoto,
        Y. Guo., G. Miller "Airframe Noise Sub-Component Definition and Model". NASA/CR-2004-213255.)

        :param int Nt: Number of wheels/tires
        :param int Ns: Number of main struts
        :param float M: Aircraft Mach number
        :param float c0: Speed of sound (m/s)
        :param float dl: Representative length scale for low frequency noise component (ft)
        :param float dh: Representative length scale for high frequency noise component (ft)
        :param float dtire: Representative length scale for tire noise component (ft)

        :return ndarray: 2D array of Rms or effective acoustic pressure for landing gear
        """
        fband = self.fband
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet
        R = self.R

        V = M * c0 / choice_data.ft2m
        dmid = (dl + dh) / 2
        R = R / choice_data.ft2m

        # Step 1: Calculate lossless, de-Dopplerized OASPL(theta) for all components

        # 1a) Low freq. component
        A = np.array([1.641991e2, 4.48107e-4])
        DELTA1 = A[0] * np.exp(A[1] * theta)
        OASPL1 = DELTA1 + 60 * np.log10(M) + 20 * np.log10(dl * np.sin(np.radians(theta)) / R) + 10 * np.log10(Ns * Nt)

        # 1b) High freq. component
        C = np.array([2.220403e2, -1.328178, 1.325498e-2, -4.2385e-5])
        DELTA2 = np.full(nthet, 180)
        DELTA2[theta > 140] = C[0] + C[1] * theta[theta > 140] + C[2] * pow(theta[theta > 140], 2) + \
                              C[3] * pow(theta[theta > 140], 3)
        OASPL2 = DELTA2 + 60 * np.log10(M) + 20 * np.log10(dh * np.sin(np.radians(theta)) / R) + 10 * np.log10(Ns * Nt)

        # 1c) Mid freq. component
        B = np.array([1.933593e2, -3.55895e-1, 1.795617e-3])
        DELTA3 = B[0] + B[1] * theta + B[2] * pow(theta, 2)
        OASPL3 = DELTA3 + 60 * np.log10(M) + 20 * np.log10(dmid * np.sin(np.radians(theta)) / R) + 10 * np.log10(
            Ns * Nt)

        # 1d) Tire noise component
        OASPL4 = 162 + 60 * np.log10(M) + 20 * np.log10(dtire * np.sin(np.radians(theta)) / R) + 10 * np.log10(Ns * Nt)

        # Step 2: Calculate Strouhal number

        St1 = np.log10(dl * fband / V)
        St2 = np.log10(dh * fband / V)
        St3 = np.log10(dmid * fband / V)
        St4 = np.log10(dtire * fband / V)

        # Step 3: Calculate 1/3 octave SPL

        # 3a) Low freq. component
        P = self.full_matrix(self.Po, 4)
        d1 = np.array([P[0, :] + P[1, :] * St1[i] + P[2, :] * pow(St1[i], 2) + P[3, :] * pow(St1[i], 3)
                       for i in range(nfreq)])
        SPL1 = OASPL1 + d1

        # 3b) High freq. component
        Q = self.full_matrix(self.Qo, 4)
        d2 = np.array([Q[0, :] + Q[1, :] * St2[i] + Q[2, :] * pow(St2[i], 2) + Q[3, :] * pow(St2[i], 3)
                       for i in range(nfreq)])
        SPL2 = OASPL2 + d2

        # 3c) Mid freq. component
        S = self.full_matrix(self.So, 3)
        d3 = np.array([S[0, :] + S[1, :] * St3[i] + S[2, :] * pow(St3[i], 2) for i in range(nfreq)])
        SPL3 = OASPL3 + d3

        # 3d) Tire noise component
        X = self.X
        d4 = np.repeat(np.reshape(X[0] + X[1] * St4 + X[2] * pow(St4, 2) + X[3] * pow(St4, 3) + X[4] * pow(St4, 4) +
                                  X[5] * pow(St4, 5) + X[6] * pow(St4, 6) + X[7] * pow(St4, 7), (nfreq, 1)), nthet,
                       axis=1)
        SPL4 = OASPL4 + d4

        # Step 5: Calculate total noise level
        prms1 = choice_aux.SPL2prms(SPL1)
        prms2 = choice_aux.SPL2prms(SPL2)
        prms3 = choice_aux.SPL2prms(SPL3)
        prms4 = choice_aux.SPL2prms(SPL4)
        prms_tot = np.sqrt(prms1 ** 2 + prms2 ** 2 + prms3 ** 2 + prms4 ** 2)

        return prms_tot

    def full_matrix(self, mato, nl):
        """
        Computes the coefficients that are required for the landing gear calculation for every directivity angle and
        returns a nc x nthet array where nc is the order of the equation of each noise component.
        """
        theta = self.theta
        mat = np.zeros((nl, len(theta)))
        for i, th in enumerate(theta):
            if th <= 60:
                mat[:, i] = mato[:, 0]
            elif 60 < th <= 120:
                k = th / 10 - 5
                for j in range(nl):
                    mat[j, i] = interpolate.interp1d(np.array([1, 2, 3, 4, 5, 6, 7, 8]), mato[j, :],
                                                     fill_value="extrapolate")(k)
            elif 120 < th < 140:
                for j in range(nl):
                    mat[j, i] = interpolate.interp1d(
                        np.array([120, 130, 140]), np.array([mato[j, 5], mato[j, 6], mato[j, 7]]),
                        fill_value="extrapolate")(th)
            elif th >= 140:
                mat[:, i] = mato[:, 7]
        return mat


class Jet(NoiseSource):
    """
    Instantiate jet source noise prediction.

    :param float A_core_caj: Nozzle exit flow area of inner stream or circular jet (m^2)
    :param float A_bypass_caj: Nozzle exit flow area of outer stream (m^2)
    :param str type: Nozzle type, mix or separate
    :param ndarray theta: 1D array containing the directivity angles (deg)
    :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
    """

    def __init__(self, A_core_caj, A_bypass_caj, type, theta, fband):
        self.A_1 = A_core_caj
        if type == 'separate':
            self.type_jet = 'coaxial'
            self.A_2 = A_bypass_caj
        elif type == 'mix':
            self.type_jet = 'circular'
            self.A_2 = 0
        self.fband = fband
        self.theta = theta
        self.nthet = len(theta)
        self.nfreq = len(fband)
        self.theta_c = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0])
        self.eta_c = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        self.r_const = 4
        self.n_thetas = 7
        self.table_iv = np.array(
            [[-41.64, 64.84, 5.85, 45.66, -45.95, 28.98, -67.28, 65.60, -3.78, -3.81, -12.14, -6.75, 14.77, 37.95, 0.75,
              -11.83, -6.53, 12.77, -17.63, 5.05, -0.12, 11.86, -41.33, 40.51, -37.63, 29.26, 33.74, -12.19, 68.07,
              -69.25, 23.03, 10.90, -1.15, 34.02, 8.42, 43.73],
             [-12.43, -17.35, 3.35, 5.25, -75.93, 142.80, -147.56, 180.53, 10.22, -2.33, 9.26, 15.17, -11.61, 0.47,
              -53.84, 3.31, 43.94, -43.15, 3.14, 44.33, -4.68, 37.70, -1.70, 18.73, -23.67, -0.44, -43.67, -13.22,
              19.20, -7.82, -17.62, 50.11, 23.04, -2.11, 3.65, -8.43],
             [-10.57, -16.34, 0.54, -6.42, -56.72, 107.84, -178.07, 168.22, 13.12, -3.30, 8.54, 14.35, -1.72, -1.10,
              -52.80, 3.71, 46.06, -34.62, 3.37, 25.17, -2.74, 13.31, -19.93, 18.16, -28.99, -2.44, -32.07, -27.97,
              20.33, 0.94, -12.62, 32.54, 11.21, -4.00, 19.64, -6.21],
             [-7.84, -16.71, 1.26, 11.35, -47.74, 76.07, -192.58, 119.55, 13.54, -3.39, 3.45, 5.40, 13.00, 0.91, -39.79,
              1.46, 42.20, -24.48, 4.10, 6.61, -4.62, 5.29, -59.88, 12.75, -27.16, 0.44, -13.20, -54.74, 23.72, -1.97,
              -5.29, 26.46, 4.20, -1.86, 27.43, 7.00],
             [-4.54, -12.48, 0.20, -28.69, -14.32, 28.13, -137.39, 43.89, 9.91, -0.83, -2.45, 15.72, 8.69, 3.07, -14.75,
              1.27, 33.04, -11.60, 9.30, -1.50, -2.88, -17.63, -16.95, 9.76, -35.00, -1.72, -29.24, -37.21, 23.68,
              -4.61, 3.08, 15.57, -4.62, -6.54, 10.65, -2.40],
             [0.44, -4.83, 2.24, -21.59, 2.43, 13.19, -83.21, 55.58, 5.39, -1.46, -1.91, 10.19, 9.09, 5.28, -20.67,
              2.11, 16.57, 0.30, 5.16, 1.27, -11.52, -30.21, -22.64, 2.25, -11.17, -8.14, -21.80, -48.71, 11.00, 6.05,
              1.40, 26.99, 9.05, -0.49, 3.77, 21.33],
             [6.29, 6.87, -0.20, -19.65, 8.96, -24.37, 21.51, -30.98, -4.83, -0.42, 2.45, -1.38, -14.63, -4.27, 17.42,
              -1.23, -18.83, 12.34, -4.12, -14.60, 9.39, 5.37, 30.05, 7.51, -2.11, 7.46, 6.99, 52.23, -20.08, 8.12,
              -0.34, -8.18, -14.19, -3.57, 10.43, -14.97],
             [6.38, 4.36, -12.57, -51.73, -14.87, -69.90, 102.77, 179.12, -12.40, 9.62, 60.26, 9.77, -1.29, -19.71,
              56.67, -19.54, -24.69, 16.01, 31.81, -44.19, -5.09, 148.29, 111.51, 105.11, -14.68, 69.81, 52.26, 111.33,
              -13.36, -8.01, 49.66, -20.66, -60.82, -25.51, -44.59, -11.41]])
        self.table_v = np.array(
            [[-27.59, -8.46, 19.20, 0.00, 0.00, -40.02, 0.00, 0.00, 13.65, -10.32, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00, 19.06, 4.30, -9.09, -36.86, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00, 0.00, 0.00],
             [-14.16, 8.44, -0.12, 25.92, -47.71, 15.36, -138.82, 103.69, 2.61, 2.16, 10.82, 8.90, -19.14, -15.81,
              -6.02, 2.97, 17.79, -11.70, 42.09, -27.74, -5.00, 0.00, 2.81, 0.00, 0.00, 0.00, 0.00, 0.00, -72.96,
              98.39, -18.80, -74.92, -6.34, -75.38, 13.14, -1.47],
             [-10.17, 1.49, -0.77, -3.57, -7.41, -3.01, 56.76, 46.69, -12.30, 1.81, -24.04, 12.25, -4.07, -17.73, 19.18,
              0.77, -35.24, 15.35, 1.49, -21.10, 6.32, -14.08, 57.68, -48.21, 9.70, -8.41, -6.49, 27.94, -64.83, 52.29,
              12.53,
              -8.12, -41.86, -5.52, 25.61, -13.69],
             [-13.97, -4.45, -1.52, -4.99, 27.91, -16.45, 113.51, -28.24, -1.04, 1.38, -5.18, 6.03, 5.56, -10.58, 12.39,
              -1.34, -10.08, 15.94, -22.60, -9.46, 0.04, -30.83, 11.72, -14.29, 39.73, -8.39, 5.86, -6.40, 7.96, 5.73,
              4.15,
              -8.41, -5.54, 32.27, 16.43, 12.21],
             [-17.56, -6.33, -8.44, 2.30, 21.08, -0.99, 148.84, 60.35, 11.00, 3.47, 38.23, -12.64, 8.09, -19.32, 5.69,
              2.86, 29.79, -8.78, -28.44, 24.30, 3.37, 11.15, 14.29, 40.54, 63.43, -1.51, -1.82, -35.42, 81.68, -32.15,
              12.02,
              -42.27, 15.64, 62.84, -34.37, -18.90],
             [-23.89, -20.25, -1.12, -24.48, -6.53, 81.36, -87.26, 308.24, 15.90, 1.82, 5.54, -22.06, 4.88, -6.16,
              18.48, 6.38, 83.56, -39.83, -9.27, 36.32, 3.23, -67.15, -48.23, 72.22, -13.79, 5.93, -40.51, -78.46,
              116.27, -35.47, -3.24, -33.64, 42.59, 44.41, -27.71, -17.02],
             [-29.75, -44.32, -20.97, 0.00, -51.27, 0.00, 0.00, 0.00, 10.24, 2.51, -25.51, 0.00, 60.14, 0.00, 0.00,
              -1.02, 189.29, -51.46, 25.99, 17.11, 35.87, 0.00, 0.00, 83.44, 0.00, 57.28, 0.00, -57.85, 187.73, 0.00,
              -42.41, 0.00, 0.00, -51.21, 211.60, 0.00]])
        table_vi_1 = np.array(
            [[3.70, 7.34, -10.40, 0.00, 0.00, -18.95, 0.00, 0.00, -14.34, 7.58, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00,
              -56.30, 23.01, 116.22, -153.51, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00,
              0.00, 0.00],
             [-2.47, -23.49, 12.67, -4.52, 79.23, -80.04, 129.39, -110.57, 5.46, -8.73, -1.96, -41.58, -3.23, 0.99,
              4.78, 11.20,
              -30.45, 72.74, -20.76, -68.87, -14.73, 0.00, -52.60, 0.00, 0.00, 0.00, 0.00, 0.00, -40.89, 28.94, -14.94,
              97.14,
              -28.07, -1.42, 52.79, 16.83],
             [-1.86, 1.68, 3.69, 45.78, 1.83, -26.44, -56.85, -133.33, 11.36, -6.40, -6.81, 29.32, 61.90, 46.61, -90.89,
              -14.99,
              13.60, 29.52, 28.84, -49.73, -52.93, -5.21, -32.86, -24.66, 30.99, -0.45, 8.72, -112.81, 3.99, 9.23,
              -6.12, 89.70,
              27.76, -37.32, 43.89, 114.16],
             [2.47, 3.78, -2.71, -58.04, 13.59, -13.19, -13.42, 50.04, -7.62, 7.93, 4.70, 13.68, -17.86, -18.29, 17.67,
              7.15,
              -24.54, 2.09, 3.56, -0.92, 5.43, -7.88, 34.90, 1.03, 4.62, 9.05, 3.72, 54.06, -18.37, -6.37, 16.18,
              -30.99,
              -38.71, -3.53, -10.51, -27.20],
             [3.43, -7.11, -1.73, -134.84, 30.92, -11.66, -148.64, 517.96, -27.22, 13.60, -10.47, -21.77, -75.31,
              -68.09,
              100.51, 22.38, -57.11, -20.12, -37.31, 34.58, 65.15, -27.75, 78.78, 11.79, -51.20, 5.80, 1.63, 155.31,
              43.95,
              11.03, 16.62, -50.66, -55.94, 54.04, -3.60, -133.30],
             [3.09, -3.09, 7.00, -144.66, 19.66, 48.07, -310.26, 444.11, -35.73, 15.86, -32.32, -34.43, -82.60, -46.64,
              145.35,
              19.47, -93.90, -23.78, -39.03, 75.72, 87.70, -91.66, 6.24, -24.29, -101.39, -38.76, 42.76, 144.63, -41.88,
              -57.54,
              25.73, -67.03, -78.03, 26.98, -27.94, -135.23],
             [0.79, -5.46, 10.52, 0.00, 145.06, 0.00, 0.00, 0.00, -16.11, 9.31, 21.79, 0.00, 55.93, 0.00, 0.00, 35.58,
              46.63,
              -42.10, 17.53, 6.62, 24.43, 0.00, 0.00, 11.27, 0.00, -27.01, 0.00, -58.70, 118.17, 0.00, 40.40, 0.00,
              0.00,
              -10.95, 253.16, 0.00]])
        table_vi_2 = np.array(
            [[1.22, 3.17, -0.48, 0.00, 0.00, -26.42, 0.00, 0.00, -9.00, 3.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              -40.64,
              21.42, 110.97, -134.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00,
              0.00],
             [-3.65, -12.44, 11.92, 16.82, 29.00, -24.89, -65.02, -61.09, -1.28, -3.79, -9.73, -35.24, -23.90, -10.10,
              26.77,
              16.09, -37.30, 52.12, -32.66, -36.49, -9.04, 0.00, 40.74, 0.00, 0.00, 0.00, 0.00, 0.00, -10.65, -6.73,
              7.83,
              79.16, -33.59, 34.85, 29.77, 38.85],
             [-2.37, 1.37, 5.55, 31.26, -6.71, -1.74, -98.78, -90.94, 9.15, -5.02, -9.05, 30.51, 42.28, 43.55, -77.65,
              -10.81,
              9.24, 24.29, 25.96, -37.40, -45.04, -15.31, -35.61, -27.28, 32.81, -2.58, 3.39, -82.89, 0.48, -15.65,
              -9.92,
              82.58, 27.01, -37.25, 28.51, 93.89],
             [2.62, 2.62, -1.35, -26.27, -2.08, -8.55, -21.59, 38.34, -6.66, 6.10, 5.80, 5.86, -9.11, -14.30, 11.22,
              2.55,
              -23.27, 2.35, 4.99, -5.28, 2.02, 15.41, 16.35, 1.25, 7.86, 15.43, 14.65, 44.36, -20.26, -7.14, 15.08,
              -16.82,
              -35.62, -4.91, -0.75, -14.88],
             [4.47, -2.10, -4.34, -60.59, 21.60, -50.63, 73.99, 266.55, -23.97, 7.71, 11.00, -25.82, -58.50, -60.97,
              54.51,
              18.98, -57.94, -6.73, -27.44, 8.84, 36.34, 45.05, 66.24, 22.95, -11.47, 24.57, 12.79, 134.04, -60.13,
              20.79, 9.30,
              19.56, -40.64, 41.64, 16.14, -82.56],
             [4.41, 0.79, 4.23, -78.79, 29.71, -18.07, -88.84, 177.25, -30.07, 8.90, -14.57, -35.86, -47.12, -41.78,
              92.39,
              9.40, -84.83, -7.98, -29.21, 41.15, 53.58, -46.19, 1.65, -11.08, -62.92, -17.51, 52.01, 103.78, -56.46,
              -30.57,
              14.24, 0.54, -55.15, 15.97, -2.06, -73.93],
             [1.94, -3.77, 1.59, 0.00, 48.83, 0.00, 0.00, 0.00, -14.78, 4.99, 7.99, 0.00, 25.90, 0.00, 0.00, 28.87,
              18.83,
              -35.89, -0.04, 1.35, 8.25, 0.00, 0.00, 26.79, 0.00, -3.62, 0.00, -25.88, 37.24, 0.00, 7.52, 0.00, 0.00,
              3.23,
              140.03, 0.00]])
        table_vi_3 = np.array(
            [[-1.04, 1.68, 8.40, 0.00, 0.00, -54.75, 0.00, 0.00, -8.24, 1.85, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              -34.67,
              15.08, 90.30, -94.68, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00],
             [-4.01, -10.40, 8.03, -15.22, 32.16, -22.01, 67.03, -63.85, -3.07, -4.67, -8.12, -21.02, -17.50, -6.67,
              3.70,
              11.84, -34.77, 43.27, -27.72, -25.73, -6.51, 0.00, 84.30, 0.00, 0.00, 0.00, 0.00, 0.00, -28.94, 8.84,
              -5.87,
              94.57, -18.78, 25.00, 22.52, 15.45],
             [-2.59, 0.82, 5.49, 15.47, -10.11, 11.27, -98.32, -73.38, 7.45, -3.65, -10.09, 26.33, 23.21, 33.87, -56.95,
              -7.34,
              9.51, 12.71, 19.30, -22.37, -28.63, -19.09, -17.47, -24.30, 21.20, -5.30, -2.59, -46.18, -0.50, -13.79,
              -12.63,
              56.22, 24.01, -31.07, 15.73, 59.52],
             [2.49, 3.27, -0.05, -10.56, -10.91, -3.47, -37.55, 15.96, -7.22, 5.41, 1.40, 3.39, -13.25, -14.76, 16.79,
              2.45,
              -22.56, -0.88, 4.08, -3.00, 6.53, 19.66, -0.33, -2.10, -0.69, 15.67, 14.39, 47.14, -25.94, -0.10, 12.37,
              -20.01,
              -33.64, -5.05, -0.36, -20.12],
             [4.71, 2.65, -2.26, -16.55, 4.78, -40.41, 78.35, 127.54, -21.48, 5.29, 8.36, -20.13, -46.79, -48.93, 36.10,
              16.31,
              -54.45, -1.44, -20.55, -0.65, 21.71, 52.86, 18.33, 19.39, -7.37, 29.17, 10.40, 97.73, -62.34, 23.46, 7.23,
              38.75,
              -31.31, 33.85, 18.51, -52.72],
             [5.31, 4.65, 0.86, -10.41, 24.80, -60.99, 104.97, -46.86, -21.52, 3.11, 3.94, -32.52, -11.60, -29.83,
              50.35, 0.34,
              -71.08, 8.36, -20.51, 10.03, 23.89, 14.42, 18.42, -7.09, -13.42, -2.62, 55.69, 59.63, -56.14, -14.60,
              7.76, 33.84,
              -36.27, 9.21, 8.29, -22.75],
             [5.70, 13.61, -1.43, 0.00, -18.04, 0.00, 0.00, 0.00, -3.63, -5.95, 15.90, 0.00, -13.52, 0.00, 0.00, 22.72,
              -30.83,
              -10.62, -27.21, -9.68, -5.39, 0.00, 0.00, 8.13, 0.00, -23.46, 0.00, -7.09, -30.64, 0.00, -3.71, 0.00,
              0.00, 24.65,
              -23.59, 0.00]])
        table_vi_4 = np.array(
            [[-1.64, 0.01, -10.22, 0.00, 0.00, 55.56, 0.00, 0.00, -9.26, 2.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              -24.61,
              2.26, 30.25, -13.68, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00],
             [-5.07, -5.84, 2.57, -3.41, 24.45, -4.72, 121.57, -115.10, -7.37, -2.83, -6.49, -7.52, 0.15, -6.25, -9.20,
              6.09,
              -34.43, 14.69, -29.04, 7.18, 7.44, 0.00, 52.85, 0.00, 0.00, 0.00, 0.00, 0.00, -14.76, -27.12, -7.97,
              97.36,
              -5.12, 32.30, 9.93, -17.83],
             [-2.41, -1.71, 4.45, -0.09, -5.45, 9.32, -71.18, -44.51, 4.64, -2.47, -3.32, 9.00, 2.08, 11.58, -28.32,
              -3.78,
              6.60, -1.76, 8.14, -8.14, -9.72, -20.17, -9.48, -16.28, 14.87, -6.62, 5.94, -0.69, -9.06, -3.05, -16.56,
              29.94,
              19.28, -20.86, 8.12, 26.20],
             [2.46, 4.05, 1.99, -5.26, -17.12, 1.67, -90.44, -4.80, -5.91, 2.04, -2.26, 3.32, -13.37, -10.70, 10.92,
              1.41,
              -17.84, -0.14, 9.82, -9.00, 4.43, 9.20, -16.29, -5.91, -15.15, 11.31, 9.25, 38.42, -40.22, 21.81, 7.49,
              -12.12,
              -27.67, -16.58, 8.77, -9.70],
             [4.07, 8.19, 2.55, 4.20, -3.89, -11.66, -28.79, -9.91, -11.96, 1.12, -11.98, -1.24, -14.47, -12.29, 16.60,
              -1.65,
              -29.55, 10.36, 5.10, -17.37, 0.43, 7.91, -19.67, -1.19, -35.69, 16.32, 1.63, 28.92, -56.62, 46.23, 7.59,
              6.42,
              -19.83, -5.19, 12.36, 1.61],
             [5.42, 7.07, 0.41, -16.48, 6.16, -35.36, 20.68, -70.58, -8.03, 0.34, 6.59, -6.76, 11.29, -2.58, 30.37,
              -4.28,
              -49.04, 23.52, -13.15, -10.68, 10.26, 17.34, 57.36, -30.38, 18.35, -13.90, 30.34, 19.82, -34.81, -11.04,
              9.20,
              -9.61, -30.84, 2.62, -0.88, -6.07],
             [5.12, 19.60, 8.46, 0.00, 29.34, 0.00, 0.00, 0.00, 8.45, -6.16, 23.63, 0.00, -8.09, 0.00, 0.00, -7.00,
              -55.14,
              28.75, -35.69, 3.12, -0.50, 0.00, 0.00, -48.93, 0.00, -56.75, 0.00, -6.66, -8.59, 0.00, 22.75, 0.00, 0.00,
              33.55,
              -64.11, 0.00]])
        table_vi_5 = np.array(
            [[-3.75, -10.99, 6.77, 0.00, 0.00, -27.09, 0.00, 0.00, -1.58, -0.66, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00, 3.96,
              -3.56, -28.10, 28.47, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00],
             [-5.07, -5.83, 6.36, -1.46, 8.00, 20.42, -17.17, -82.36, -4.94, -3.83, -0.15, 9.98, 18.91, -0.31, -9.64,
              -7.50,
              -23.61, 1.57, -14.05, 6.69, 3.23, 0.00, -13.92, 0.00, 0.00, 0.00, 0.00, 0.00, 2.05, -62.76, -10.84, 97.85,
              -2.13,
              -1.19, -2.44, 11.61],
             [-1.38, -0.71, 4.60, 5.42, -5.25, 4.04, -79.12, -4.86, 0.77, -1.68, -16.01, 9.60, -1.24, 2.47, -9.53, 4.34,
              3.52,
              -2.95, 6.19, -6.91, -0.63, -39.90, -18.90, -24.09, -1.02, -15.83, -8.10, -25.06, -5.22, -5.59, -8.15,
              43.33, 8.79,
              -14.11, 14.15, 10.29],
             [2.84, 5.74, -0.15, -1.37, -23.25, -3.05, -92.67, -1.39, -3.71, -0.59, 1.83, 0.45, -9.72, -4.45, -1.42,
              2.51,
              -12.23, 3.88, 14.59, -7.38, 0.65, 14.80, 2.66, -4.64, -16.51, 4.07, 4.35, 28.55, -37.38, 23.58, 5.74,
              10.87,
              -20.43, -24.01, 3.45, -6.60],
             [2.78, 8.91, -1.05, 9.15, -0.79, -23.47, 13.61, -116.57, -4.01, -2.15, -7.76, -2.78, 3.36, 9.74, 4.99,
              -7.77,
              -12.54, 19.95, 16.09, -26.65, -3.58, 35.21, 16.38, 5.10, -54.63, 16.82, 13.98, 31.90, -47.57, 54.21, 6.65,
              -15.19,
              -18.55, -25.52, 9.84, 10.36],
             [2.90, 8.99, -5.86, 9.47, 23.37, -60.23, 219.91, -264.53, 3.44, -3.59, 23.14, -4.94, 14.36, 11.03, 14.15,
              -12.63,
              -20.67, 25.73, -5.74, -26.72, 7.60, 74.22, 80.70, -2.84, 23.63, 6.56, 33.27, 45.37, -7.72, -7.63, 10.65,
              -46.29,
              -24.10, -0.60, -3.12, -6.70],
             [2.00, 22.10, 8.12, 0.00, 74.06, 0.00, 0.00, 0.00, 9.85, -7.35, 30.71, 0.00, -9.17, 0.00, 0.00, -12.49,
              -34.22,
              32.38, -12.25, -20.14, -2.19, 0.00, 0.00, -44.56, 0.00, -50.69, 0.00, 20.46, 32.20, 0.00, 35.61, 0.00,
              0.00, 6.02,
              -40.05, 0.00]])
        table_vi_6 = np.array(
            [[0.57, -3.13, -7.24, 0.00, 0.00, 68.97, 0.00, 0.00, 1.92, 2.55, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              7.66, 7.72, 1.75, -33.12, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00, 0.00],
             [1.48, -3.25, 3.72, -20.25, 11.92, -4.15, 12.63, -67.21, 1.97, -2.33, 2.84, -4.78, 15.12, 0.80, -11.26,
              0.62, -3.72, -0.22, -10.19, 14.22, 0.31, 0.00, -51.56, 0.00, 0.00, 0.00, 0.00, 0.00, 21.93, -53.50, -1.44,
              79.25, -1.08, 7.52, 3.99, 12.71],
             [1.16, -1.81, -0.78, -1.80, 11.21, -22.85, 31.88, -44.76, 1.92, -1.43, -2.80, 3.29, 2.02, -4.05, -9.27,
              2.58, 11.77, -4.35, 0.37, 2.96, -1.74, 23.56, 4.43, 2.85, -25.37, 7.39, -24.99, -6.14, 3.32, 17.20, 0.70,
              0.40, 6.44, 9.15, -6.95, -10.21],
             [-4.05, 3.84, -9.83, 43.37, 41.01, -30.81, 242.06, -99.58, 0.41, -0.21, 16.32, -5.57, 31.06, -2.26, 7.92,
              -13.42, -3.41, 11.95, -6.48, 9.16, -3.54, 14.44, 27.26, 7.28, 38.64, -0.14, 37.73, -24.67, 35.00, -31.01,
              17.49, 11.73, -5.40, 16.30, -32.98, 9.01],
             [-8.48, 11.45, -7.76, 69.04, 40.45, -25.27, 152.09, -224.10, 23.11, -6.60, -10.12, 10.57, 34.53, 53.96,
              -21.05, -25.25, 56.88, 9.86, 13.93, -6.88, -12.28, -3.67, 67.90, -22.91, 15.03, -25.53, 2.31, -30.80,
              93.37, -67.11, -3.57, -9.13, 23.62, -24.65, -36.57, 29.39],
             [-9.64, 8.21, -3.31, 48.36, 46.86, 18.50, 115.90, -68.30, 16.02, -0.98, -18.37, 7.82, 10.95, 20.53, 7.66,
              -22.41, 22.00, 19.70, 5.61, 0.10, -5.68, -79.52, 41.75, -68.32, 103.03, -26.80, 26.87, 21.39, 55.56,
              -66.72, 3.56, -13.06, -13.87, -26.73, -45.42, 17.86],
             [-6.38, -0.40, 10.60, 0.00, 45.50, 0.00, 0.00, 0.00, -2.13, -1.19, -19.07, 0.00, 10.60, 0.00, 0.00, -5.96,
              14.92, -21.98, -1.69, 10.68, 0.17, 0.00, 0.00, -43.88, 0.00, -18.93, 0.00, -23.25, -37.64, 0.00, -28.65,
              0.00, 0.00, -10.78, -23.67, 0.00]])
        table_vi_7 = np.array(
            [[6.13, 8.21, -27.22, 0.00, 0.00, 301.38, 0.00, 0.00, -4.75, -2.31, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00,
              -59.56, 12.66, 183.17, -144.23, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              0.00,
              0.00, 0.00],
             [3.28, -15.04, -3.27, -56.74, 14.15, 64.74, 29.51, 201.87, 8.79, 1.91, 17.85, -45.07, 12.86, -38.36,
              -30.24, 6.90,
              6.71, -2.31, -29.84, 57.93, 2.79, 0.00, 27.43, 0.00, 0.00, 0.00, 0.00, 0.00, -37.94, 102.07, -13.03,
              -134.85,
              -69.43, -21.05, 158.03, 61.37],
             [-1.66, -11.38, -4.87, -20.87, 63.07, -37.06, 268.47, -150.70, 7.06, 2.04, 16.31, 10.33, -16.91, 2.67,
              -18.53,
              -5.87, 16.54, -37.03, 1.55, 37.41, 11.86, 56.87, 35.98, -9.77, -5.69, 17.37, -60.67, 81.75, -26.52, 32.66,
              -37.87,
              -151.19, 29.27, -15.31, -1.66, -73.92],
             [-9.46, 36.90, -7.76, 165.10, 91.22, -141.02, 436.49, -887.25, 2.43, -10.21, -23.56, 12.97, 66.40, 28.36,
              15.12,
              -12.50, 1.82, 29.44, -31.55, -22.43, 11.33, -89.17, -54.51, 28.36, -8.15, -11.99, 60.97, -141.60, 105.37,
              -131.42,
              33.14, 137.04, -42.15, 47.86, 46.07, 40.43],
             [-13.73, 38.22, -9.06, 229.84, 54.84, -142.23, 335.42, -759.51, 36.60, -29.55, -97.40, 31.14, 111.36,
              129.73,
              -77.39, -12.68, 88.47, 49.66, -12.22, -85.53, -13.83, -95.05, 184.80, -145.77, 91.04, -112.43, -33.31,
              -218.76,
              322.06, -371.95, 14.40, 394.27, 50.64, 35.66, 48.81, 56.84],
             [-17.83, 25.34, 2.92, 206.89, 157.94, -83.90, 825.13, -510.85, 1.49, -4.60, -114.67, 18.73, 28.15, -1.39,
              -25.45,
              -29.86, -6.30, 65.14, 5.36, -53.26, -28.29, -228.34, 48.70, -222.84, 226.45, -59.14, 5.14, -11.09, 28.70,
              -28.07,
              -20.46, 213.23, 80.24, -22.57, 39.88, 84.53],
             [-5.16, 17.61, 47.67, 0.00, 132.35, 0.00, 0.00, 0.00, -29.70, 3.56, -96.11, 0.00, -13.06, 0.00, 0.00,
              11.73,
              -20.05, -31.08, 4.64, 5.13, -13.92, 0.00, 0.00, -122.21, 0.00, 21.94, 0.00, 45.86, -81.44, 0.00, -77.55,
              0.00,
              0.00, -50.13, -78.17, 0.00]])

        self.table_vi = np.array([table_vi_1, table_vi_2, table_vi_3, table_vi_4, table_vi_5, table_vi_6, table_vi_7])

    def calc(self, dmdt_1, dmdt_2, v_1, v_2, T_1, T_2, Ta, Pa):
        """
        Jet source noise model for circular and coaxial jets based on the methods presented by Russel (J. W. Russel.
        "An empirical Method for Predicting the mixing Noise Levels of Subsonic Circular and Coaxial Jets".
        NASA-CR-3786).

        :param float dmdt_1: Mass flow rate of inner stream or circular jet (kg/s)
        :param float dmdt_2: Mass flow rate of outer stream (kg/s)
        :param float v_1: Nozzle exit flow velocity of inner stream or circular jet (m/s)
        :param float v_2: Nozzle exit flow velocity of outer stream (m/s)
        :param float T_1: Nozzle exit flow total temperature of inner stream or circular jet (K)
        :param float T_2: Nozzle exit flow total temperature of outer stream (K)
        :param float Ta: Ambient static temperature (K)
        :param float Pa: Ambient pressure (Pa)

        :return ndarray: 2D array of Rms or effective acoustic pressure for jet component
        """
        fband = self.fband
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet

        gamma_1 = choice_data.gamma_gas
        gamma_2 = choice_data.gamma_air

        if self.type_jet == 'circular':
            dmdt_2 = 0
            v_2 = 0
            T_2 = 0

        dmdt_e = dmdt_1 + dmdt_2  # equivalent mass flow
        v_e = (dmdt_1 * v_1 + dmdt_2 * v_2) / dmdt_e  # equivalent velocity
        T_e = (dmdt_1 * (gamma_1 / (gamma_1 - 1)) * T_1 + dmdt_2 * (gamma_2 / (gamma_2 - 1)) * T_2) / (
                dmdt_1 * (gamma_1 / (gamma_1 - 1)) + dmdt_2 * (gamma_2 / (gamma_2 - 1)))  # equivalent temperature
        gamma_e_gamma_e_1 = (dmdt_1 * (gamma_1 / (gamma_1 - 1)) + dmdt_2 * (gamma_2 / (gamma_2 - 1))) / (
                dmdt_1 + dmdt_2)
        # equivalent ratio of spec. heat
        gamma_e = gamma_e_gamma_e_1 / (gamma_e_gamma_e_1 - 1)

        rho_amb = Pa / (choice_data.Risa * Ta)
        c_amb = math.sqrt(gamma_2 * choice_data.Risa * Ta)
        t_amb = Ta
        rho_ISA = 101325.0 / (choice_data.Risa * 288.15)
        c_ISA = math.sqrt(gamma_2 * choice_data.Risa * choice_data.Tisa)

        rho_e = rho_amb * (T_e / t_amb - ((gamma_e - 1) / 2) * (v_e / c_amb) ** 2) ** (-1)

        A_e = dmdt_e / (rho_e * v_e)  # equivalent jet area
        D_e = math.sqrt(4 * A_e / math.pi)  # equivalent diameter

        if self.type_jet.lower() == 'coaxial':
            # turbofan case
            x_1 = math.log10((v_e / c_amb) / 1)
            x_2 = math.log10((T_e / t_amb) / 2)
            x_3 = math.log10((v_2 / v_1) / 1)
            x_4 = math.log10((T_2 / T_1) / 1)
            x_5 = math.log10((self.A_2 / self.A_1) / 1)
            # v_2/v_1 operating range in method development is (0.02 - 2.5)
            # T_2/T_1 operating range in method development is (0.2 - 4.0)
            if self.A_2 / self.A_1 < 0.50 or self.A_2 / self.A_1 > 10.0:
                print('A_2/A_1 is not within the valid range (0.5 - 10.0)', 'call from calcCaj')
        elif self.type_jet.lower() == 'circular':
            # jet case
            v_e = v_1
            T_e = T_1
            x_1 = math.log10((v_e / c_amb) / 1)
            x_2 = math.log10((T_e / t_amb) / 2)
            x_3 = 0.0
            x_4 = 0.0
            x_5 = 0.0
        else:
            print('type_jet not known!')

        # v_e / c_amb operating range in method development is (0.3 - 2.0)
        # T_e/t_amb operating range in method development is (0.7 - 4.5)

        # STEP 3 - Compute Derivative Multipliers
        # Circular and coannular jet
        D1 = 1
        D2 = x_1
        D3 = x_2
        D4 = (x_1 ** 2) / 2
        D5 = x_1 * x_2
        D6 = (x_2 ** 2) / 2
        D7 = (x_1 ** 2 * x_2) / 2
        D8 = (x_1 * x_2 ** 2) / 2
        # Coannular jet only:
        D9 = x_3
        D10 = x_4
        D11 = x_1 * x_3
        D12 = x_1 * x_4
        D13 = x_1 * x_5
        D14 = x_2 * x_3
        D15 = x_2 * x_4
        D16 = x_2 * x_5
        D17 = (x_3 ** 2) / 2
        D18 = x_3 * x_4
        D19 = x_3 * x_5
        D20 = (x_4 ** 2) / 2
        D21 = x_4 * x_5
        D22 = (x_1 ** 2 * x_3) / 2
        D23 = (x_1 ** 2 * x_5) / 2
        D24 = (x_1 * x_3 ** 2) / 2
        D25 = x_1 * x_3 * x_4
        D26 = x_1 * x_3 * x_5
        D27 = x_1 * x_4 * x_5
        D28 = (x_1 * x_5 ** 2) / 2
        D29 = (x_3 ** 3) / 6
        D30 = (x_3 ** 2 * x_4) / 2
        D31 = (x_3 ** 2 * x_5) / 2
        D32 = (x_3 * x_4 ** 2) / 2
        D33 = x_3 * x_4 * x_5
        D34 = (x_3 * x_5 ** 2) / 2
        D35 = (x_4 ** 2 * x_5) / 2
        D36 = (x_4 * x_5 ** 2) / 2

        DerivMultipliers = np.array(
            [D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19,
             D20, D21, D22, D23, D24, D25, D26, D27, D28, D29, D30, D31, D32, D33, D34, D35, D36])

        OAPWL_norm = 0
        RSL = np.zeros((self.n_thetas, self.n_thetas))
        if self.type_jet == 'coaxial':
            OAPWL_norm = np.sum(self.table_iv[0, :] * DerivMultipliers)
            D = np.sum(self.table_iv[1:self.n_thetas + 1, :] * DerivMultipliers, axis=1)
            F = np.sum(self.table_v[0:self.n_thetas, :] * DerivMultipliers, axis=1)
            for k in range(self.n_thetas):
                RSL[:, k] = np.sum(self.table_vi[k, 0:self.n_thetas, :] * DerivMultipliers, axis=1)
        elif self.type_jet == 'circular':
            OAPWL_norm = np.sum(self.table_iv[0, 0:8] * DerivMultipliers[0:8])
            D = np.sum(self.table_iv[1:self.n_thetas + 1, :] * DerivMultipliers, axis=1)
            F = np.sum(self.table_v[0:self.n_thetas, :] * DerivMultipliers, axis=1)
            for k in range(self.n_thetas):
                RSL[:, k] = np.sum(self.table_vi[k, 0:self.n_thetas, :] * DerivMultipliers, axis=1)
        else:
            print('type_jet not known!')

        A_ref = dmdt_e / (rho_amb * c_amb)

        OASPL = OAPWL_norm + D + 20.0 * np.log10((rho_amb * c_amb ** 2) / (rho_ISA * c_ISA ** 2.0)) + \
                10.0 * np.log10(A_ref / (4.0 * math.pi * self.r_const ** 2)) + 197.0

        # STEP 9 - Compute the predicted OASPL

        OASPL_p = interpolate.PchipInterpolator(self.theta_c, OASPL)(theta)

        eta_p = np.log10((fband * D_e) / v_e)

        RSL_p = interpolate.interp2d(self.theta_c, self.eta_c, RSL)(theta, eta_p)

        #  STEP 12 - Compute the norm frequency params
        F_p = interpolate.CubicSpline(self.eta_c, F)(eta_p)

        # STEP 13 - Compute the norm frequency params
        SPL_temp = RSL_p + OASPL_p
        SPL = SPL_temp + np.repeat(np.reshape(F_p, (nfreq, 1)), nthet, axis=1)

        return choice_aux.SPL2prms(SPL)


class Combustor(NoiseSource):
    """
    Instantiate combustor source noise prediction.

    :param str type_comb: Combustor type, SAC or DAC
    :param float Aec: Combustor exit area (ft^2)
    :param float De: Exhaust nozzle exit plane effective diameter (ft)
    :param float Dh: Exhaust nozzle exit plane hydraulic diameter (ft)
    :param float Lc: Combustor nominal length (ft)
    :param float h: Annulus height at combustor exit (ft)
    :param int Nfmax: Total number of DAC fuel nozzles
    :param ndarray theta: 1D array containing the directivity angles (deg)
    :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
    """

    def __init__(self, type_comb, Aec, De, Dh, Lc, h, Nfmax, theta, fband):
        self.cmbtype = type_comb
        self.Aec = Aec
        self.De = De
        self.Dh = Dh
        self.Lc = Lc
        self.h = h
        self.Nfmax = Nfmax
        self.theta = theta
        self.fband = fband
        self.nthet = len(theta)
        self.nfreq = len(fband)

    def calc(self, Nf, pattern, pa, p3, p4, p7, ta, t3, t4, t5, w3):
        """
        Combustor source noise model based on the method for low-emissions combustors developed by Gliebe et al.
        (P. Gliebe, R. Mani, H. Shin, B. Mitchell, G. Ashford, S. Salamah and S. Connel. "Aeroacoustic Prediction
        Codes". NASA-CR-210244). Includes model for SAC (Single-Annular Combustor) and DAC (Dual-Annular Combustor).

        :param int Nf: Number of ignited fuel nozzles
        :param int/float pattern: Fuel nozzle firing pattern
        :param float pa: Atmospheric pressure (Pa)
        :param float p3: Combustor inlet pressure (Pa)
        :param float p4: Combustor exit pressure (Pa)
        :param float p7: Turbine last stage exit pressure (Pa)
        :param float ta: Atmospheric temperature (K)
        :param float t3: Combustor inlet temperature (K)
        :param float t4: Combustor exit temperature (K)
        :param float t5: Turbine last stage exit temperature (K)
        :param float w3: Combustor inlet flow (kg/s)

        :return ndarray: 2D array of Rms or effective acoustic pressure for combustor component
        """

        pa_psia = pa * choice_data.pa2psia
        p3_psia = p3 * choice_data.pa2psia
        p4_psia = p4 * choice_data.pa2psia
        p7_psia = p7 * choice_data.pa2psia
        t3_fahr = 1.8 * (t3 - 273) + 32.0
        t4_fahr = 1.8 * (t4 - 273) + 32.0
        t5_fahr = 1.8 * (t5 - 273) + 32.0
        w3_lbms = w3 * 2.20462
        self.R0 = 150  # ft

        AA = PropagationEffects.atmospheric_attenuation(ta, pa, choice_data.RH, self.R0 * choice_data.ft2m - 1,
                                                        self.fband, third_octave_band=True)  # dB for 150 ft - 1m

        if self.cmbtype == 'SAC':
            SPL = self.getSAC(Nf, pa_psia, p3_psia, p4_psia, p7_psia,
                              t3_fahr, t4_fahr, t5_fahr, w3_lbms, AA)
        elif self.cmbtype == 'DAC':
            SPL = self.getDAC(pattern, pa_psia, p3_psia, p4_psia, p7_psia, t3_fahr, t4_fahr, t5_fahr, w3_lbms, AA)
        else:
            sys.exit('combtype not recognised in calcComb')

        return choice_aux.SPL2prms(SPL)

    def getSAC(self, Nf, Pa, P3, P4, P7, T3, T4, T5, W3, AA):
        """ Calculate the noise level from a Single Annular Combustor. """
        fband = self.fband
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet

        # Step 1
        fp = np.array([63.0, 160.0, 630.0])
        ap = np.array([150.0, 130.0, 130.0])

        # Step 2 - normalized frequency and directivity angle
        fN = np.array([fband / fp[0], fband / fp[1], fband / fp[2]])
        aN = np.array([theta / ap[0], theta / ap[1], theta / ap[2]])

        # Step 3
        OASPLN = np.array([-67.8 * (aN[0, :]) ** 2 + 141.7 * aN[0, :] - 66.84,
                           -26.019 * (aN[1, :]) ** 3 - 5.2974 * aN[1, :] ** 2 + 93.43 * aN[1, :] - 61.75,
                           -156.5 * (aN[2, :]) ** 2 + 322.34 * aN[2, :] - 164.89])

        # Step 5
        CP = (W3 * math.sqrt(T3) / P3) * ((T4 - T3) / T4) * (P3 / Pa) * ((self.Dh / self.De) ** 0.5)
        Hcp = np.array([76.45 + 14.256 * math.log(CP), 108.5 + 3.31 * math.log(CP), 106.38 + 6.938 * math.log(CP)])

        OASPL = np.zeros((3, nthet))

        iptr1 = np.where(theta >= ap[0])[0][0]
        OASPL[0, iptr1] = -20.0 * math.log10(self.R0) + Hcp[0] * ((30.0 / Nf) ** (-0.225))

        iptr2 = np.where(theta >= ap[1])[0][0]
        OASPL[1, iptr2] = -20.0 * math.log10(self.R0) + Hcp[1] * ((30.0 / Nf) ** 0.05)

        iptr3 = np.where(theta >= ap[2])[0][0]
        OASPL[2, iptr3] = -20.0 * math.log10(self.R0) + Hcp[2] * ((30.0 / Nf) ** 0.02)

        # Step 4
        # Loss parameter
        Fc = (W3 * math.sqrt(T4 - T3) / (P3 * self.Aec ** 2 * Nf))
        Ft = (P4 / P7) * math.sqrt(T5 / T4)
        Tl = ((1 + Ft) ** 2) / (4.0 * self.Lc * Ft / (math.pi * self.h))

        SPL_Fc = 20.0 * math.log10(Fc)
        SPL_Tl = 20.0 * math.log10(Tl)

        OASPL = np.array([OASPLN[0, :] + OASPL[0, iptr1] + 0.4 * (SPL_Fc - SPL_Tl),
                          OASPLN[1, :] + OASPL[1, iptr2] + 0.1 * (SPL_Fc - SPL_Tl),
                          OASPLN[2, :] + OASPL[2, iptr3] + 0.3 * (SPL_Fc - SPL_Tl)])

        # Step 6
        SPLN = np.array([-152.70 + 295.46 * fN[0, :] - 145.61 * fN[0, :] ** 2,
                         -170.07 + 331.33 * fN[1, :] - 163.34 * fN[1, :] ** 2,
                         -147.50 + 286.40 * fN[2, :] - 142.31 * fN[2, :] ** 2])

        SD = 20.0 * math.log10(self.R0 / (1 / choice_data.ft2m))  # to back-propagate at the source (1 m)
        SPL = np.array([[OASPL[:, i] + SPLN[:, j] + AA[j] + SD for i in range(nthet)] for j in range(nfreq)])

        prms = choice_aux.SPL2prms(SPL)

        prms_total = np.sqrt(prms[:, :, 0] ** 2 + prms[:, :, 1] ** 2 + prms[:, :, 2] ** 2)

        prms_total[prms_total < choice_data.p0] = choice_data.p0

        return choice_aux.prms2SPL(prms_total)

    def getDAC(self, pattern, Pa, P3, P4, P7, T3, T4, T5, W3, AA):
        """ Calculate the noise level from a Dual Annular Combustor. """
        fband = self.fband
        theta = self.theta
        nfreq = choice_data.nfreq
        nthet = choice_data.nthet

        # Step 1
        ap = 130.0
        fp = np.array([160.0, 500.0])
        Mf = np.array([0.020, 0.180])

        if pattern == 40:
            Ninner = 20
            Xk = 0.250
            Knf = np.array([1.2, 1])
        elif pattern == 30:
            Ninner = 10
            Xk = 0.250
            Knf = np.array([0.98, 0.9])
        elif pattern == 22.5:
            Ninner = 10
            Xk = 0.200
            Knf = np.array([0.98, 0.9])
        elif pattern == 20:
            Ninner = 0
            Xk = 0.000
            Knf = np.array([1.1, 0.98])

        # Step 2 - normalized frequency and directivity angle
        fN = np.array([fband / fp[0], fband / fp[1]])
        aN = np.array(theta / ap)

        # Step 3
        OASPLN = np.array([-116.95 * aN ** 2 + 235.23 * aN - 120.65,
                           -137.59 * aN ** 2 + 283.40 * aN - 147.73])

        # Step 5
        CP = np.array([(W3 * math.sqrt(T3) / P3) * ((T4 - T3) / T4) * (P3 / Pa) * ((self.Dh / self.De) ** 2),
                       (W3 * math.sqrt(T3) / P3) * ((T4 - T3) / T4) * (P3 / Pa) * ((self.Dh / self.De) ** 1.2)])
        Hcp = np.array([110.44 + 2.6931 * math.log(CP[0]), 110.62 + 2.9917 * math.log(CP[1])])

        OASPL = np.zeros((2, nthet))

        iptr = np.where(theta >= ap)[0][0]
        OASPL[0, iptr] = Knf[0] * (-20.0 * math.log10(self.R0) + Hcp[0] * (((20.0 + Ninner) / self.Nfmax) ** (-Xk)) *
                                   ((30.0 / (20.0 + Ninner)) ** Mf[0]))  # eq 227
        OASPL[1, iptr] = Knf[1] * (-20.0 * math.log10(self.R0) + Hcp[1] * (((20.0 + Ninner) / self.Nfmax) ** (-Xk)) *
                                   ((30.0 / (20.0 + Ninner)) ** Mf[1]))  # eq 227

        # Step 4
        # Loss parameter
        Fc = (W3 * math.sqrt(T4 - T3) / (P3 * self.Aec ** 2 * math.sqrt(20.0 + Ninner)))
        Ft = (P4 / P7) * math.sqrt(T5 / T4)
        Tl = ((1.0 + Ft) ** 2) / (4.0 * self.Lc * Ft / (math.pi * self.h))

        SPL_Fc = 20.0 * math.log10(Fc)
        SPL_Tl = 20.0 * math.log10(Tl)

        OASPL = np.array([OASPLN[0, :] + OASPL[0, iptr] + 0.45 * (SPL_Fc - SPL_Tl),
                          OASPLN[1, :] + OASPL[1, iptr] - 0.10 * (SPL_Fc - SPL_Tl)])

        # STEP 6
        SPLN = np.array([-143.07 + 280.04 * fN[0, :] - 143.00 * fN[0, :] ** 2,
                         -137.21 + 268.99 * fN[1, :] - 135.81 * fN[1, :] ** 2])

        SD = 20.0 * math.log10(self.R0 / (1 / choice_data.ft2m))  # to back-propagate at the source (1 m)
        SPL = np.array([[OASPL[:, i] + SPLN[:, j] + AA[j] + SD for i in range(nthet)] for j in range(nfreq)])

        prms = choice_aux.SPL2prms(SPL)
        prms_total = np.sqrt(prms[:, :, 0] ** 2 + prms[:, :, 1] ** 2)

        prms_total[prms_total < choice_data.p0] = choice_data.p0

        return choice_aux.prms2SPL(prms_total)


class Turbine(NoiseSource):
    """
    Instantiate turbine source noise prediction.

    :param int N_rotors: Number of rotors
    :param int n_stages: Number of stages
    :param float SRS: Stator-rotor spacing
    :param ndarray theta: 1D array containing the directivity angles (deg)
    :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
    :param ndarray f: 1D array containing frequencies (Hz)
    """

    def __init__(self, N_rotors, n_stages, SRS, theta, fband, f):
        self.N_rotors = N_rotors
        self.n_stages = n_stages
        self.SRS = SRS
        self.theta = theta
        self.fband = fband
        self.nthet = len(theta)
        self.nfreq = len(fband)
        self.f = f
        self.broadband_corr = np.array([-37.0, -29.0, -21.0, -13.0, -7.4, -4.0, -1.2, 0.0, -1.2, -9.15, -19.0, -29.0])
        self.angle = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0, 180.0])
        self.tone_corr = np.array([-47.0, -37.0, -27.0, -18.2, -10.0, -6.0, -2.5, 0.0, -2.5, -14.8, -26.0, -37.0])
        self.K = -10.0
        self.Vref = 0.305
        self.cref = 340.3
        self.mref = 0.4536
        self.turb_atm_abs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.20, 0.20, 0.3, 0.4,
                                      0.5, 0.6, 0.8, 1.20, 1.4, 2.0, 2.9, 4.2])

    def calc(self, Vtr, Texit, xnl, mcore, Cax):
        """
        Turbine source noise model based on the method developed by Dunn and Peart (D. G. Dunn and N. A. Peart.
        "Aircraft Noise Source and Contour Estimation". NASA-CR-114649).

        :param float Vtr: Relative tip speed of turbine last rotor
        :param float Texit: Exhaust temperature (K)
        :param float xnl: Rotational speed (rps)
        :param float mcore: Mass flow (kg/s)
        :param float Cax: Axial velocity (m/s)

        :return ndarray: 2D array of Rms or effective acoustic pressure for turbine component
        """
        fband = self.fband
        theta = self.theta
        f = self.f
        nfreq = self.nfreq
        nthet = self.nthet

        t_static = Texit - Cax ** 2 / (2 * choice_data.cp_gas)
        c_exit = math.sqrt(choice_data.gamma_gas * choice_aux.get_R(0.0) * t_static)

        BPF = np.repeat(np.round(self.N_rotors * xnl * 60 / 60), nthet)
        # xnl is in rps but in NASA report, rpm is required

        # Broadband noise

        F1 = interpolate.CubicSpline(self.angle, self.broadband_corr)(theta)

        fac1 = ((Vtr * self.cref) / (self.Vref * c_exit)) ** 3
        fac2 = mcore / self.mref
        fac3 = np.ones(nthet) ** (-4)
        SPL_peak_b = 10.0 * np.log10(fac1 * fac2 * fac3) + F1 - 10.0

        SPL_46m_b = np.zeros((nthet, nfreq))

        for j, fb in enumerate(fband):
            SPL_46m_b[fb / BPF <= 1, j] = SPL_peak_b[fb / BPF <= 1] + 10.0 * np.log10(fb / BPF[fb / BPF <= 1])
            SPL_46m_b[fb / BPF > 1, j] = SPL_peak_b[fb / BPF > 1] - 20.0 * np.log10(fb / BPF[fb / BPF > 1])

        # Tone noise

        F3 = interpolate.CubicSpline(self.angle, self.tone_corr)(theta)

        fac1 = (Vtr / self.Vref) ** 0.6 * (self.cref / c_exit) ** 3
        fac2 = (mcore / self.mref) * (1 / self.SRS)
        fac3 = np.ones(nthet) ** (-4)
        SPL_peak_t = 10.0 * np.log10(fac1 * fac2 * fac3) + F3 + 56.0 + self.K
        BPF_max = fband[-1] / BPF

        SPL_46m_t = np.zeros(np.shape(SPL_46m_b))
        for i in range(nthet):
            for j in range(math.floor(BPF_max[i])):
                iptr = np.where(f > (j + 1) * BPF[i])[0][0] - 1
                if SPL_46m_t[i, iptr] == 0:
                    SPL_46m_t[i, iptr] = SPL_peak_t[i] - j * 10.0
                else:
                    prms_temp = choice_aux.SPL2prms(SPL_46m_t[i, iptr])
                    prms_46m_t = choice_aux.SPL2prms(SPL_peak_t[i] - j * 10.0)
                    prms_46m_t = math.sqrt(prms_temp ** 2 + prms_46m_t ** 2)
                    SPL_46m_t[i, iptr] = choice_aux.prms2SPL(prms_46m_t)
                if SPL_46m_t[i, iptr] < 0.0: SPL_46m_t[i, iptr] = 0.0

        prms_b = choice_aux.SPL2prms(SPL_46m_b)
        prms_t = choice_aux.SPL2prms(SPL_46m_t)
        prms_tot = np.sqrt(prms_b ** 2 + prms_t ** 2).T
        SPL_tot = choice_aux.prms2SPL(prms_tot)
        SPL_tot_46m = SPL_tot + 10.0 * math.log10(self.n_stages)

        SPL_1m = SPL_tot_46m + 33.2 + np.tile(np.reshape(self.turb_atm_abs, (nfreq, 1)), (1, nthet))

        return choice_aux.SPL2prms(SPL_1m)


class FanCompressor(NoiseSource):
    """
    Instantiate fan or compressor source noise prediction.

    :param str component: Compressor component, fan, LPC, IPC
    :param float MtipD: Rotor tip relative inlet Mach number at design point
    :param int N_rotors: Number of rotors
    :param int N_stators: Number of stators
    :param float rss: Rotor-stator spacing
    :param ndarray theta: 1D array containing the directivity angles (deg)
    :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
    :param ndarray f: 1D array containing frequencies (Hz)
    """

    def __init__(self, component, MtipD, N_rotors, N_stators, rss, theta, fband, f, distortion=None):
        self.comp = component
        self.MtrD = MtipD
        self.N_rotors = N_rotors
        self.N_stators = N_stators
        self.rss = rss
        self.theta = theta
        self.fband = fband
        self.nthet = len(theta)
        self.nfreq = len(fband)
        self.f = f
        self.f2_data = np.array([
            -9.50, -9.00, -8.50, -7.75, -7.00, -6.00, -5.00, -3.50, -2.00, -1.00, 0.00, 0.00, 0.00, -1.75, -3.50,
            -5.50, -7.50, -8.25, -9.00, -9.25, -9.50, -9.75, -10.00, -10.25, -10.50, -10.75, -11.00, -11.25, -11.50,
            -11.75, -12.00, -12.25, -12.50, -12.75, -13.00, -13.25, -13.50])
        self.dTref = 0.5550
        self.mref = 0.4536
        if distortion:
            self.distortion = distortion
        else:
            self.distortion = False
        self.F3a_data = np.array([
            -3.00, -2.25, -1.50, -0.75, 0.00, 0.00, 0.00, 0.00, 0.00, -0.60, -1.20, -2.35, -3.50, -5.15, -6.80, -8.65,
            -10.50, -12.50, -14.50, -16.75, -19.00, -21.25, -23.50, -25.75, -28.00, -30.25, -32.50, -34.75, -37.00,
            -39.25, -41.50, -43.75, -46.00, -48.25, -50.50, -52.75, -55.00])
        # figure 13(a) FOR INLET (extrapolated from 100 degrees to 180)
        self.F3b_data = np.array([
            -39.0, -37.0, -35.0, -33.0, -31.0, -29.0, -27.0, -25.0, -23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0,
            -9.5, -8.0, -6.5, -5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.0, 0.0, -1.0, -2.0, -3.75, -5.5, -7.25, -9.0,
            -11.0, -13.0, -15.5, -18.0])
        self.angles = np.array([
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0])
        # GE "Flight Cleanup" -- TCS Suppression - Table 4.3. Idea is to make sure that the installation effects in
        # take-off and approach are removed.
        self.approach_BPF = np.array([5.6, 5.8, 4.7, 4.6, 4.9, 5.1, 2.9, 3.2, 1.6, 1.6, 1.8, 2.1, 2.4, 2.2, 2.0, 2.8])
        self.approach_2BPF = np.array([5.4, 4.3, 3.4, 4.1, 2.0, 2.9, 1.6, 1.3, 1.5, 1.1, 1.4, 1.5, 1.0, 1.8, 1.6, 1.6])
        self.takeoff_BPF = np.array([4.8, 5.5, 5.5, 5.3, 5.3, 5.1, 4.4, 3.9, 2.6, 2.3, 1.8, 2.1, 1.7, 1.7, 2.6, 3.5])
        self.takeoff_2BPF = np.array([5.8, 3.8, 5.3, 6.4, 3.5, 3.0, 2.1, 2.1, 1.1, 1.4, 0.9, 0.7, 0.7, 0.4, 0.6, 0.8])
        self.f3a_data_broad = np.array([
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.25, -4.5, -6.0, -7.5, -9.25, -11.0, -13.0,
            -15.0, -17.0, -19.0, -22.0, -25.0, -28.0, -31.0, -34.0, -37.0, -40.0, -43.0, -46.0, -49.0, -52.0, -55.0,
            -58.0, -61.0, -64.0, -67.0])
        self.f3b_data_broad = np.array([
            -41.600, -39.450, -37.300, -35.150, -33.000, -30.850, -28.700, -26.550, -24.400, -22.250, -20.100, -17.950,
            -15.8000, -13.650, -11.500, -9.750, -8.000, -6.500, -5.000, -3.850, -2.700, -1.950, -1.200, -0.750, -0.300,
            -0.1500, 0.000, -1.000, -2.000, -4.000, -6.000, -8.000, -10.000, -12.500, -15.000, -17.500, -20.000])

    def calc(self, operatingPoint, Mtip, Mu, dT, xnl, g1):
        """
        Fan and compressor source noise model based on the methods developed by Heidman (M. F. Heidmann. "Interim
        Prediction Method for Fan and Compressor Source Noise". NASA-TM-X71763) and updated by Kontos et al.
        (K. B. Kontos, B. A. Janardan and P. R. Gliebe. "Improved NASA-ANOPP Noise Prediction Computer Code for Advanced
        Subsonic Propulsion Systems". NASA-CR-195480).

        :param str operatingPoint: Approach, Cutback, Sideline
        :param float Mtip: Rotor tip relative inlet Mach number at operating point
        :param float Mu: Blade Mach number
        :param float dT: Temperature rise across fan or compressor stage
        :param float xnl: Fan rotational speed (rps)
        :param float g1: Mass flow rate passing through fan or compressor (kg/s)

        :return ndarray: 2D array of Rms or effective acoustic pressure for tone, broadband and combination tone noise
        """

        f = self.f
        fb = xnl * self.N_rotors
        acc = int(round(f[-1] / fb))

        delta_cutoff = abs(Mu / (1 - self.N_stators / self.N_rotors))  # cut-off factor

        # get broadband noise
        [prms_broadband_inlet, prms_broadband_discharge] = self.fanCompr_broadband(operatingPoint, Mtip, fb, dT, g1)

        # get tone noise
        [prms_tone_inlet, prms_tone_discharge] = self.fanCompr_tone(operatingPoint, acc, Mtip, dT, g1, delta_cutoff, fb)

        # get combination noise
        if Mtip > 1:
            prms_inlet_combination = self.fanCompr_comb_tone(Mtip, dT, g1, fb)
        else:
            prms_inlet_combination = np.zeros(np.shape(prms_tone_inlet))

        prms_inlet = np.sqrt(prms_broadband_inlet ** 2 + prms_tone_inlet ** 2 + prms_inlet_combination ** 2)
        # create outlet combination (tone + broadband)
        prms_discharge = np.sqrt(prms_tone_discharge ** 2 + prms_broadband_discharge ** 2)

        return [prms_tone_inlet, prms_tone_discharge, prms_broadband_inlet, prms_broadband_discharge,
                prms_inlet_combination, prms_inlet, prms_discharge]

    def fanCompr_comb_tone(self, Mtr, dT, g1, fb):
        """ Fan and compressor combination tone noise estimation. """
        fband = self.fband
        f = self.f
        nfreq = self.nfreq
        nthet = self.nthet

        F1 = np.zeros(3)
        # Creating F1 according to fig 15(a)
        F1[0] = interpolate.interp1d(np.array([1.0, 1.14, 2.0]), np.array([30.0, 72.5, 4.4]))(Mtr)
        F1[1] = interpolate.interp1d(np.array([1.0, 1.25, 2.0]), np.array([30.0, 68.6, 10.5]))(Mtr)
        F1[2] = interpolate.interp1d(np.array([1.0, 1.61, 2.0]), np.array([36.0, 60.6, 56.5]))(Mtr)

        if self.comp == 'Fan' and choice_data.no_fan_stages == 1 and choice_data.fan_IGV:
            C = -5.0
        elif self.comp == 'Fuselage_fan' and choice_data.no_ff_stages == 1 and choice_data.ff_IGV:
            C = -5.0
        else:
            C = 0.0

        # eq 8
        Lc_combination = 20.0 * np.log10(dT / self.dTref) + 10.0 * np.log10(g1 / self.mref) + self.f2_data + C

        # set peak
        peak = np.array([Lc_combination + F1[i] for i in range(3)])

        peakf = [np.argmin(abs(f - 0.500 * fb)), np.argmin(abs(f - 0.250 * fb)),
                 np.argmin(abs(f - 0.125 * fb))]

        # Figure 14
        F3 = np.zeros((3, nfreq))
        F3[0, 0:peakf[0]] = 30.0 * np.log10(2 * fband[0:peakf[0]] / fb)
        F3[0, peakf[0]:] = -30.0 * np.log10(2 * fband[peakf[0]:] / fb)
        F3[1, 0:peakf[1]] = 50.0 * np.log10(4.0 * fband[0:peakf[1]] / fb)
        F3[1, peakf[1]:] = -50.0 * np.log10(4.0 * fband[peakf[1]:] / fb)
        F3[2, 0:peakf[2]] = 50.0 * np.log10(8.0 * fband[0:peakf[2]] / fb)
        F3[2, peakf[2]:] = -30.0 * np.log10(8.0 * fband[peakf[2]:] / fb)

        SPL_combination = np.array([[peak[:, j] + F3[:, i] for j in range(nthet)] for i in range(nfreq)])

        prms_combination = choice_aux.SPL2prms(SPL_combination)
        return np.sqrt(prms_combination[:, :, 0] ** 2 + prms_combination[:, :, 1] ** 2 + prms_combination[:, :, 2] ** 2)

    def fanCompr_tone(self, operatingPoint, acc, Mtr, dT, g1, delta_cutoff, fb):
        """ Fan and compressor discrete tone noise estimation. """
        theta = self.theta
        nfreq = self.nfreq
        nthet = self.nthet
        f = self.f

        F1a = self.get_Fig10a(Mtr)  # Creating F1a according to figure 10(a) FOR INLET
        F1b = self.get_Fig10b(Mtr)  # Creating F1b according to figure 10(b) FOR DISCHARGE

        # Creating F2a according to figure 12 FOR INLET
        F2a = 0.0

        # Creating F2b according to figure 12 FOR DISCHARGE
        F2b = -10.0 * math.log10(self.rss / 300.0)

        k = np.arange(1, acc + 2, 1)
        nk = len(k)

        # re-interpolate GE-data onto theta angles (GE "Flight Cleanup" -- TCS Suppression - Table 4.3.)
        if operatingPoint.strip() == 'Take-off' or operatingPoint.strip() == 'Cutback' or \
                operatingPoint.strip() == 'Sideline':
            BPF1 = interpolate.interp1d(self.angles, self.takeoff_BPF, fill_value="extrapolate")(theta)
            BPF2 = interpolate.interp1d(self.angles, self.takeoff_2BPF, fill_value="extrapolate")(theta)
        elif operatingPoint.strip() == 'Approach':
            BPF1 = interpolate.interp1d(self.angles, self.approach_BPF, fill_value="extrapolate")(theta)
            BPF2 = interpolate.interp1d(self.angles, self.approach_2BPF, fill_value="extrapolate")(theta)

        # set inlet data
        Lc_inlet = 20.0 * np.log10(dT / self.dTref) + 10.0 * np.log10(g1 / self.mref) + F1a + F2a + self.F3a_data  # eq6

        F4a = np.zeros(nk)
        if self.comp.strip() == 'Fan':
            F4a = self.get_Fig8(choice_data.fan_IGV, Mtr, k, delta_cutoff)
        elif self.comp == 'fuselage_fan':
            F4a = self.get_Fig8(choice_data.ff_IGV, Mtr, k, delta_cutoff)

        if self.distortion:
            if self.comp == 'fuselage_fan':
                F5 = self.get_Fig9(choice_data.no_ff_stages, k)
            else:
                F5 = self.get_Fig9(choice_data.no_fan_stages, k)
            F4F5 = 10.0 * np.log10(10 ** (0.10 * F4a) + 10 ** (0.1 * F5))
        else:
            F4F5 = F4a

        SPL_inlet = np.zeros((nfreq, nthet))
        prms_inlet = np.zeros((nfreq, nthet))

        for i in range(nthet):
            for j in range(acc):
                positions = np.where(f > float(fb * (j + 1)))
                if positions[0].size > 0:
                    pos = positions[0][0] - 1
                else:
                    pos = -2

                if pos == -2: continue

                if SPL_inlet[pos, i] == 0.0:
                    if j == 0:  # first tone
                        SPL_inlet[pos, i] = Lc_inlet[i] + F4F5[j] - BPF1[i]
                    elif j == 1:  # second tone
                        SPL_inlet[pos, i] = Lc_inlet[i] + F4F5[j] - BPF2[i]
                    else:
                        SPL_inlet[pos, i] = Lc_inlet[i] + F4F5[j]
                else:  # coinciding tones
                    prms_temp = choice_aux.SPL2prms(SPL_inlet[pos, i])
                    if j == 0:  # first tone
                        prms_inlet[pos, i] = choice_aux.SPL2prms((Lc_inlet[i] + F4F5[j] - BPF1[i]))
                    elif j == 1:  # second tone
                        prms_inlet[pos, i] = choice_aux.SPL2prms((Lc_inlet[i] + F4F5[j] - BPF2[i]))
                    else:
                        prms_inlet[pos, i] = choice_aux.SPL2prms((Lc_inlet[i] + F4F5[j]))

                    prms_tot = math.sqrt(prms_temp ** 2 + prms_inlet[pos, i] ** 2)
                    SPL_inlet[pos, i] = 20.0 * np.log10(prms_tot / choice_data.p0)

        prms_inlet = choice_aux.SPL2prms(SPL_inlet)

        if self.comp == 'Fan' or self.comp == 'Fuselage_fan':
            # set discharge data
            F4b = np.zeros(nk)
            if self.comp == 'Fan':
                F4b = self.get_Fig8(choice_data.fan_IGV, Mtr, k, delta_cutoff)
            elif self.comp == 'Fuselage_fan':
                F4b = self.get_Fig8(choice_data.ff_IGV, Mtr, k, delta_cutoff)

            Lc_discharge = 20.0 * np.log10(dT / self.dTref) + 10.0 * np.log10(g1 / self.mref) + F1b + F2b + \
                           self.F3b_data + self.get_C(nthet)  # eq 6

            SPL_discharge = np.zeros((nfreq, nthet))
            prms_discharge = np.zeros((nfreq, nthet))

            for i in range(nthet):
                for j in range(acc):
                    positions = np.where(f > float(fb * (j + 1)))
                    if positions[0].size > 0:
                        pos = positions[0][0] - 1
                    else:
                        pos = -2

                    if pos == -2: continue

                    if SPL_discharge[pos, i] == 0.0:
                        SPL_discharge[pos, i] = Lc_discharge[i] + F4b[j]

                    elif SPL_discharge[pos, i] != 0.0:
                        prms_temp = choice_aux.SPL2prms(SPL_discharge[pos, i])
                        prms_discharge[pos, i] = choice_aux.SPL2prms((Lc_discharge[i] + F4b[j]))
                        prms_tot = math.sqrt(prms_temp ** 2 + prms_discharge[pos, i] ** 2)
                        SPL_discharge[pos, i] = 20.0 * np.log10(prms_tot / choice_data.p0)

            prms_discharge = choice_aux.SPL2prms(SPL_discharge)
        else:
            prms_discharge = np.zeros((nfreq, nthet))

        return [prms_inlet, prms_discharge]

    def fanCompr_broadband(self, operatingPoint, Mtr, fb, dT, g1):
        """ Fan and compressor broadband noise estimation. """
        fband = self.fband
        nfreq = self.nfreq
        nthet = self.nthet

        F1a = self.get_Fig4a(Mtr)
        F1b = self.get_Fig4b(Mtr)

        # Creating F2a according to Fig 6(a) FOR INLET
        F2a = 0.0
        # Creating F2b according to Fig 6(b) FOR DISCHARGE
        F2b = -5.0 * math.log10(self.rss / 300.0)

        Lc_inlet = np.repeat([20.0 * np.log10(dT / self.dTref) + 10.0 * np.log10(g1 / self.mref) + F1a + F2a +
                              self.f3a_data_broad], nfreq, axis=0)  # eq 4
        SPL_inlet = Lc_inlet + self.get_Fig3a(fband.reshape((nfreq, 1)) / fb)  # eq 5
        prms_inlet = choice_aux.SPL2prms(SPL_inlet)

        if self.comp == 'Fan' or self.comp == 'Fuselage_fan':

            Lc_discharge = np.full((nfreq, nthet), 20.0 * np.log10(dT / self.dTref) + 10.0 * np.log10(g1 / self.mref) +
                                   F1b + F2b + self.f3b_data_broad)  # eq 4
            SPL_discharge = Lc_discharge + self.get_Fig3a(fband.reshape((nfreq, 1)) / fb)  # eq 5

            prms_discharge = choice_aux.SPL2prms(SPL_discharge)
        else:
            prms_discharge = np.zeros((nfreq, nthet))

        return [prms_inlet, prms_discharge]

    def get_C(self, nthet):
        """ C coefficient from eq. 12 in Heidmann"""
        if (choice_data.fan_IGV or choice_data.no_fan_stages == 2) or \
                (choice_data.ff_IGV or choice_data.no_ff_stages == 2):
            return np.full(nthet, 6.0)
        else:
            return np.zeros(nthet)

    def get_Fig3a(self, f_over_fb):
        """ Creating F4 according to eq. 2 in Heidmann"""
        sigmae = 2.2
        return 10.0 * np.log10(1.0 / (np.exp(0.5 * (np.log(f_over_fb / 2.5) / np.log(sigmae)) ** 2)))

    def get_Fig4a(self, M_tr):
        """ Get peak broadband sound pressure levels from Fig 4(a) for inlet duct according to Heidmann. """
        if self.MtrD <= 1.0 and M_tr <= 0.9:
            return 58.5
        elif self.MtrD <= 1.0 and M_tr > 0.9:
            return 58.5 - 20.0 * math.log10(M_tr / 0.9)
        elif self.MtrD > 1.0 and M_tr <= 0.9:
            return 58.5 + 20.0 * math.log10(self.MtrD)
        elif self.MtrD > 1.0 and M_tr > 0.9:
            return 58.5 + 20.0 * math.log10(self.MtrD) - 50.0 * math.log10(M_tr / 0.9)  # ANOPP update (1996 - Kontos)
        else:
            print('unexpected combination of tip Mach numbers in get_Fig4a in choice_physics')
            return 0.0

    def get_Fig4b(self, M_tr):
        """ Get peak broadband sound pressure levels from Fig 4(b) for discharge duct according to Heidmann and
        eq. 3a-3b in Kontos. """
        if self.MtrD <= 1.0 and M_tr <= 1.0:
            return 60.0
        elif self.MtrD > 1.0 and M_tr <= 1.0:
            return 63.0 + 20.0 * math.log10(self.MtrD)  # ANOPP update (1996 - Kontos)
        elif self.MtrD > 1.0 and M_tr > 1.0:
            return 63.0 + 20.0 * math.log10(self.MtrD) - 30.0 * math.log10(M_tr)  # ANOPP update (1996 - Kontos)
        else:
            print('unexpected combination of tip Mach numbers in get_Fig4b in choice_physics')
            return 0.0

    def get_Fig8(self, igv, Mtr, k, delta):
        """ Get rotor-stator interaction discrete tone harmonic levels from Fig 8 in Heidmann and eq. 7a-8c in
        Kontos. """
        if not igv:
            if Mtr < 1.15:
                if delta > 1.05:
                    Fig8 = 6.0 - 6.0 * k
                else:
                    Fig8 = 6.0 - 6.0 * k
                    Fig8[0] = -8.0
            else:
                if delta > 1.05:
                    Fig8 = 9.0 - 9.0 * k
                else:
                    Fig8 = 9.0 - 9.0 * k
                    Fig8[0] = -8.0
        else:
            if delta > 1.05:
                Fig8 = -3.0 - 3.0 * k
                Fig8[0] = 0.0
            else:
                Fig8 = -3.0 - 3.0 * k
                Fig8[0] = -8.0
        return Fig8

    def get_Fig9(self, no_fan_stages, k):
        """ Get inlet flow distortion discrete tone harmonic levels for inlet duct from Fig 9 in Heidmann. """
        if no_fan_stages == 1:
            return 10.0 - 10.0 * k
        elif no_fan_stages == 2:
            nk = len(k)
            return np.zeros(nk)
        else:
            print('Unexpected number of stages in get_Fig9 in choice_physics')
            return 0.0

    def get_Fig10a(self, M_tr):
        """ Get characteristic peak sound pressure level of fundamental discrete tone for inlet duct according to
        Fig. 10(a) in Heidmann and eq. 5 in Kontos. """
        if M_tr <= 0.72:
            if self.MtrD <= 1.0:
                return 60.5
            elif self.MtrD > 1.0:
                return 60.5 + 20.0 * math.log10(self.MtrD)
        else:  # Kontos update
            F1a1 = 60.5 + 20.0 * math.log10(self.MtrD) + 50.0 * math.log10(M_tr / 0.72)
            F1a2 = 64.5 + 80.0 * math.log10(self.MtrD / M_tr)
            return min(F1a1, F1a2)

    def get_Fig10b(self, M_tr):
        """ Get characteristic peak sound pressure level of fundamental discrete tone for discharge duct according to
        Fig. 10(b) in Heidmann. """
        if M_tr <= 1.0:
            if self.MtrD <= 1.0:
                return 63.0
            elif self.MtrD > 1.0:
                return 63.0 + 20.0 * math.log10(self.MtrD)
        else:
            return 63.0 + 20.0 * math.log10(self.MtrD) - 20.0 * math.log10(M_tr)


class PropagationEffects:
    """
    Instantiate noise estimation at microphone/observer.

    :param float ymic: Microphone height (m)
    :param bool use_ground_refl: True to account for ground reflection or False else
    :param bool spherical_spr: True to account for spherical spreading or False else
    :param bool atm_atten: True to account for atmospheric attenuation or False else
    :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
    :param ndarray xsii: 1D array containing observation angles (rad)
    :param ndarray Mai: 1D array containing Mach number
    :param ndarray xsi_alphai: 1D array containing dircetivity angle accounting for angle of attack (deg)
    :param float dTisa: Deviation from ISA temperature (K)
    :param float elevation: Ground elevation at microphone location (m)
    """

    def __init__(self, ymic, use_ground_refl, spherical_spr, atm_atten, fband, xsii, Mai, xsi_alphai, dTisa, elevation):
        self.ymic = ymic
        self.use_ground_refl = use_ground_refl
        self.spherical_spr = spherical_spr
        self.atm_atten = atm_atten
        self.Mach = Mai
        self.fband = fband
        if fband is not None:
            self.fds = self.getDopplerShift(fband, xsii, Mai)
            # calculate shifted band limits, needed for auralization
            self.fupper_ds = self.getDopplerShift(fband * (2 ** (1 / 6)), xsii, Mai)
            self.flower_ds = self.getDopplerShift(fband / (2 ** (1 / 6)), xsii, Mai)
            self.nfreq = len(self.fds)
            self.xsii_alpha = xsi_alphai
        self.N_b = 5
        self.max_n_freqs = 150
        self.dTisa = dTisa
        self.elevation = elevation

    def flightEffects(self, nti, theta, x, y, r1, tai, SPLi, comp, phi):
        """
        Computes the sound pressure level and the acoustic pressure matrices accounting for propagation effects.

        :param int nti: Number of trajectory points
        :param ndarray theta: 1D array containing the directivity angles (deg)
        :param ndarray x: 1D array containing aircraft horizontal distance relative to the microphone (m)
        :param ndarray y: 1D array containing aircraft altitude relative to the microphone (m)
        :param ndarray r1: 1D array containing aircraft distance relative to the microphone (m)
        :param ndarray tai: 1D array containing atmospheric temperature (K)
        :param ndarray SPLi: 3D array containing Sound pressure level at the source (dB)
        :param string comp: Noise component

        :return: An SPL and a prms array
        """

        # Convective amplification - source motion effect
        DF = np.array([1 / (1 - np.multiply(self.Mach, np.cos(np.radians(thet)))) for thet in theta])
        SPLi = np.array([SPLi[i_f, :, :] - 40 * np.log10(1 / DF) for i_f in range(self.nfreq)])

        # Only one directivity reaches the microphone. Here we choose the element in the directivity vector closest
        # to the computed angle.
        closest_value_ptr = np.array([np.abs(theta - math.degrees(xsii_a)).argmin() for xsii_a in self.xsii_alpha])

        # extract the directivity elements from the SPL matrix
        SPLp_ds = np.array([SPLi[:, closest_value_ptr[j], j] for j in range(nti)]).T
        DF_mic = np.array([DF[closest_value_ptr[j], j] for j in range(nti)]).T

        if 'tone' in comp or 'combination' in comp:
            SPLp = self.tones_to_3rd_octave_bands(SPLp_ds, DF_mic)
        else:
            SPLp = self.convert_to_3rd_octave_bands(SPLp_ds, self.fupper_ds, self.flower_ds)
        prmsp = choice_aux.SPL2prms(SPLp)

        pa = AtmosphericEffects.get_p_ambient(y + self.ymic + self.elevation)

        rho_ac = pa / (choice_data.Risa * tai)
        c_ac = AtmosphericEffects.get_sound_speed(tai)
        pa_mic = AtmosphericEffects.get_p_ambient(self.ymic + self.elevation)
        t_mic = AtmosphericEffects.get_t_ambient(self.ymic + self.elevation, self.dTisa)
        rho_mic = pa_mic / (choice_data.Risa * t_mic)
        c_mic = AtmosphericEffects.get_sound_speed(t_mic)
        impedance_factor = np.sqrt((rho_mic * c_mic) / (rho_ac * c_ac))
        # spherical spreading/ characteristic impedance
        if self.spherical_spr:
            SPLp = choice_aux.prms2SPL(prmsp / r1 * impedance_factor)
        else:
            SPLp = choice_aux.prms2SPL(prmsp * impedance_factor)

        prmsp = choice_aux.SPL2prms(SPLp)

        # Ground Reflection and atmospheric absorption
        if self.use_ground_refl or self.atm_atten:
            sub_band, prms_band = self.subband_division(prmsp)

            if self.atm_atten:
                if not hasattr(self, 'atm_absorption'):
                    # Coefficients remain the same for all components. Save to reuse in the next component
                    atm_absorption = np.array(
                        [self.atmospheric_attenuation(ta_i, pa[i], choice_data.RH, r1[i], sub_band,
                                                      third_octave_band=False) for i, ta_i in enumerate(tai)])
                    self.atm_absorption = atm_absorption.T

                prms_band = choice_aux.SPL2prms(choice_aux.prms2SPL(prms_band) - self.atm_absorption)

            if self.use_ground_refl:
                if not hasattr(self, 'ground_refl'):
                    G1mat = np.array([self.ground_reflection(y[i], r1[i], t_mic, sub_band) for i in range(nti)]).T
                    self.ground_refl = G1mat

                prms_band = np.sqrt(self.ground_refl * prms_band ** 2)

            prmsp = self.subband_combination(prms_band)

        if hasattr(phi, "__len__"):
            TL_lateral = self.lateral_attenuation(phi, 'wing')
            SPLp = choice_aux.prms2SPL(prmsp) + TL_lateral
        else:
            SPLp = choice_aux.prms2SPL(prmsp)

        self.fobs = np.array([self.fband for i in range(nti)]).T  # observer frequencies are now in 1/3 octave bands
        self.fupper_obs = np.array([self.fband * (2 ** (1 / 6)) for i in range(nti)]).T
        self.flower_obs = np.array([self.fband / (2 ** (1 / 6)) for i in range(nti)]).T

        return [SPLp, prmsp]

    def ground_reflection(self, y_plane, r1, ta, fband):
        """
        Computes the ground effects factor for a sound wave that does not travel directly from the source to the
        observer.

        :param float y_plane: Aircraft altitude relative to the microphone (m)
        :param float r1: Aircraft distance relative to the microphone (m)
        :param float ta: Atmospheric static temperature (K)
        :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)

        :return array: 1D array containing ground effects factor
        """
        ainc = 0.01  # incoherence coefficient
        sigma = 225000  # specific flow resistance of the ground
        nf = len(fband)

        y1 = y_plane + self.ymic  # aircraft altitude from the ground
        c = math.sqrt(choice_data.Risa * choice_data.gamma_air * ta)

        r_xz = np.sqrt(r1 ** 2 - y_plane ** 2)  # ground distance to microphone
        thet = math.pi / 2 - math.atan((y1 + self.ymic) / r_xz)  # incidence angle
        dr = 2.0 * self.ymic * math.cos(thet)  # path length difference
        r2 = r1 + dr  # source image to receiver distance

        k = 2 * math.pi * fband / c  # wave number k
        eta = (2 * math.pi * choice_data.rhoisa / sigma) * fband  # dimensionless frequency

        ny = np.array([1 / (1 + (6.86 * e) ** (-0.75) + 1j * (4.36 * e) ** (-0.73)) for e in eta])  # the complex

        # specific ground admittance
        tau = np.array([cmath.sqrt((k_i * r2) / 2j) * (math.cos(thet) + ny[i]) for i, k_i in enumerate(k)])

        t = np.array([i for i in range(-100, 101)])

        Fcap = np.full(nf, complex(0, 0))
        for i, taui in enumerate(tau):
            if abs(taui) > 10.0:
                if -np.real(taui) > 0:
                    U = 1  # U is a step function
                elif -np.real(taui) == 0:
                    U = 0.5
                elif -np.real(taui) < 0:
                    U = 0
                Fcap[i] = -2 * math.sqrt(math.pi) * U * taui * cmath.exp(taui ** 2) + 1 / (2 * taui ** 2) - 3 / (
                        (2 * taui ** 2) ** 2)
            else:
                vec = np.exp(-t ** 2) / (1j * taui - t)
                W = (1j / math.pi) * sum(vec)  # Imag(z)>0 complex error function
                Fcap[i] = 1 - cmath.sqrt(math.pi) * taui * W

        gam = (math.cos(thet) - ny) / (math.cos(thet) + ny)  # complex plane-wave reflection coefficient
        Amat = gam + (1 - gam) * Fcap  # a complex spherical-wave reflection coefficient
        Rvec = abs(Amat)  # R = magnitude of the complex spherical-wave reflection coefficient
        alfa = np.array([math.atan2(A.imag, A.real) for A in Amat])
        Ccap = np.exp(-(ainc * k * dr) ** 2)  # coherence coefficient

        Kcap = 2 ** (1 / (6.0 * self.N_b))

        G1 = 1 + Rvec ** 2 + 2 * Rvec * Ccap * np.cos(alfa + k * dr) * np.sin((Kcap - 1) * k * dr) / (
                (Kcap - 1) * k * dr)

        return G1

    def getDopplerShift(self, fband, xsii, Mai):
        """ Computes the frequency shift due to the movement of the aircraft. """
        return np.array([fb / (1.0 - Mai * np.cos(xsii)) for fb in fband])

    def subband_division(self, prms):
        """ Divide into subbands to be used in the ground effects application."""

        # Ratio of subband center frequencies
        w = 10 ** (1 / (10 * self.N_b))

        h = np.arange(1, self.N_b + 1)
        i = np.arange(1, self.nfreq + 1)
        m = int((self.N_b - 1) / 2)

        # Subband center frequency
        f = np.reshape(np.array([w ** (h - m - 1) * fb for fb in self.fband]), self.nfreq * self.N_b)

        # Calculate prms for each subband
        u = np.zeros_like(prms)
        v = np.zeros_like(prms)
        u[1:, :] = np.divide(prms[1:, :] ** 2, prms[:-1, :] ** 2)  # slope in lower half of band
        v[:-1, :] = u[1:, :]  # slope in upper half of band
        u[0, :] = v[0, :]
        v[-1, :] = u[-1, :]

        A = np.array([[1 + sum(ui ** ((h[:m] - m - 1) / self.N_b) + vi ** (h[:m] / self.N_b)) for ui, vi in zip(u[:, i], v[:, i])] for i in
                      range(u.shape[1])]).T

        prms_subband = np.zeros((len(f), prms.shape[1]))
        for i in range(self.nfreq):
            j = i * self.N_b + h - 1
            prms_subband[j[:m], :] = np.sqrt((np.tile(np.reshape(prms[i, :], (1, prms.shape[1])), (len(h[:m]), 1))
                                              ** 2 / A[i]) * u[i] ** ((h[:m].reshape(len(h[:m]), 1) - m - 1) / self.N_b))
            prms_subband[j[m], :] = np.sqrt(prms[i, :] ** 2 / A[i])
            prms_subband[j[m + 1:], :] = np.sqrt(
                (np.tile(np.reshape(prms[i, :], (1, prms.shape[1])), (len(h[m + 1:]), 1))
                 ** 2 / A[i]) * v[i] ** ((h[m + 1:].reshape(len(h[m + 1:]), 1) - m - 1) / self.N_b))

        return f, prms_subband

    def subband_combination(self, prms_subband):
        """ Combine into subband prms."""

        h = np.arange(1, self.N_b + 1) - 1
        prms = np.zeros((self.nfreq, prms_subband.shape[1]))
        for i in range(self.nfreq):
            j = i * self.N_b + h
            prms[i, :] = np.sqrt(sum(prms_subband[j, :] ** 2))

        return prms

    def tones_to_3rd_octave_bands(self, spl_dop, DF):
        """ Convert dopplerized frequency spectrum of tonal components to 1/3 octave bands"""

        dn = -3 * np.log2(1 / DF)
        nt = spl_dop.shape[1]
        spl = np.zeros_like(spl_dop)
        for i in range(nt):
            if abs(dn[i]) >= 0.5:
                if dn[i] > 0:
                    n = int(np.round(dn[i]))
                    spl[n:, i] = spl_dop[:-n, i]
                else:
                    n = int(np.round(dn[i]))
                    spl[:n, i] = spl_dop[-n:, i]
            else:
                spl[:, i] = spl_dop[:, i]

        return spl

    def convert_to_3rd_octave_bands(self, spl_dop, fdop_u, fdop_l):
        """ Convert dopplerized frequency spectrum to 1/3 octave bands"""

        fband = self.fband
        freq = np.arange(choice_data.fmin, choice_data.fmax)
        fband_upper = fband * (2 ** (1 / 6))
        fband_lower = fband / (2 ** (1 / 6))
        nt = spl_dop.shape[1]
        spl = np.zeros_like(spl_dop)
        for i in range(nt):
            spl_narrow = self.convert_to_narrowband(spl_dop[:, i], fdop_u[:, i], fdop_l[:, i], freq)
            # Compute mean square acoustic pressure in the narrowband frequencies
            psd_initial = choice_aux.SPL2prms(spl_narrow) ** 2
            # Sum mean square acoustic pressures falling in the same 1/3 octave band
            pressure_amplitude = np.zeros_like(fband)
            for fi in range(self.nfreq):
                freq_ind = np.where((freq >= fband_lower[fi]) & (freq <= fband_upper[fi]))[0]
                portion_lower = fband_lower[fi] - int(fband_lower[fi])
                portion_upper = fband_upper[fi] - int(fband_upper[fi])
                if fi > 0:
                    psd_band = sum(psd_initial[freq_ind[:-1]]) + psd_initial[freq_ind[0] - 1] * (1 - portion_lower) + \
                               psd_initial[freq_ind[-1]] * portion_upper
                else:
                    psd_band = sum(psd_initial[freq_ind[:-1]]) + psd_initial[freq_ind[-1]] * portion_upper
                # Compute pressure amplitude at desired frequency resolution
                pressure_amplitude[fi] = np.sqrt(psd_band)
            spl[:, i] = choice_aux.prms2SPL(pressure_amplitude)

        return spl

    def convert_to_narrowband(self, Lp, fupper, flower, fc_new):
        """ Compute the narrowband spectrum"""

        Lp_narrow = np.zeros(fc_new.shape)
        prms = np.zeros(fc_new.shape)
        jfend = np.where((fc_new >= flower[0]))[0][0]
        for fi, lp in enumerate(Lp):
            # Compute number of narrowband bins in the band
            nf = (fc_new[np.where((fc_new >= flower[fi]) & (fc_new <= fupper[fi]))]).size
            jfstart = jfend
            jfend = jfend + nf
            # Compute mean square acoustic pressure and divide with the number of bins within the band
            prms_sq = choice_aux.SPL2prms(lp) ** 2
            if nf > 0:
                prms[jfstart:jfend] = np.sqrt(prms_sq / nf)
            else:
                prms[jfstart:jfend] = choice_data.p0
            Lp_narrow[jfstart:jfend] = choice_aux.prms2SPL(prms[jfstart:jfend])

        return Lp_narrow

    @staticmethod
    def atmospheric_attenuation(t_k, pa, relHum, r, fband, third_octave_band=False):
        """
        Computes the absorption for a sound wave travelling through the atmosphere. Modelled according to the
        ISO 9613-1:1993 report.

        :param t_k: 1D array containing atmospheric temperature (K)
        :param pa: 1D array containing atmospheric pressure (Pa)
        :param float relHum: Relative humidity
        :param r: 1D array containing distance between source and observer (m)
        :param ndarray fband: 1D array containing the 1/3 octave band frequencies (Hz)
        :param bool third_octave_band: If True 1/3 octave bands are used and coefficients are estimated using the Volpe
        method (E. D. Rickley, G. G. Fleming, C. Roof. "Simplified procedure for computing the absorption of sound by
        the atmosphere")

        :return ndarray: 1D array containing the atmospheric absorption in dB.
        """
        pr = 101325  # Reference pressure (Pa)
        Tr = 293.15  # Reference temperature (K)
        T01 = 273.16  # Triple-point isotherm temperature

        # Molar concentration of water vapour
        V = 10.79586 * (1 - T01 / t_k) - 5.02808 * np.log10(t_k / T01) + 1.50474e-4 * (
                    1 - 10 ** (-8.29692 * (t_k / T01 - 1))) \
            + 0.42873e-3 * (-1 + 10 ** 4.76955 * (1 - T01 / t_k)) - 2.2195983
        psat = pr * 10 ** V
        h = relHum * (psat / pr) * (pa / pr) ** (-1)

        # Relaxation frequency of oxygen
        fro = pa / pr * (24.0 + 4.04 * 10.0 ** 4.0 * h * (0.02 + h) / (0.391 + h))

        # Relaxation frequency of nitrogen
        frn = pa / pr * (t_k / Tr) ** (-0.5) * (9.0 + 280.0 * h * np.exp(-4.170 * ((t_k / Tr) ** (-1.0 / 3.0) - 1.0)))

        # Mid-band attenuation rate in dB/m
        alpha = 8.686 * fband ** 2 * ((1.84 * 10.0 ** (-11.0) * (pr / pa) * (t_k / Tr) ** 0.5) + (t_k / Tr) ** (-2.5) *
                                      (0.01275 * np.exp(-2239.1 / t_k) * (fro / (fband ** 2.0 + fro ** 2)) +
                                       0.1068 * np.exp(-3352.0 / t_k) * (frn / (fband ** 2.0 + frn ** 2))))
        # Mid_band attenuation in dB
        delta_t = alpha * r

        # Volpe Method for calculating attenuation by atmospheric absorption on wideband sounds analyzed by 1/3
        # octave-band filters
        if third_octave_band:
            A = 0.867942
            B = 0.111761
            C = 0.95824
            D = 0.008191
            E = 1.6
            F = 9.2
            G = 0.765
            delta_b = np.zeros_like(delta_t)
            delta_b[delta_t < 150] = A * delta_t[delta_t < 150] * (1 + B * (C - D * delta_t[delta_t < 150])) ** E
            delta_b[delta_t >= 150] = F + G * delta_t[delta_t >= 150]
            return delta_b
        else:
            return delta_t

    def lateral_attenuation(self, phi, engine='wing'):
        """
        Calculation of engine-installation effects (lateral directional effects attributed to wing or fuselage mounted
        engines) for conventional aircraft. The method is based on the SAE AIR 5662 report, "Method for Predicting Lateral
        Attenuation of Airplane Noise".

        :param phi: depression angle (deg)
        :param engine: wing or fus depending on where the engine is mounted
        """
        # The method accounts for engine installation effects, ground effects and refraction and scattering due to wind
        # and meteorological conditions. Here only the installations effects are implemented as more accurate models are
        # used for the propagation effects in all directions (not just lateral). The correction for installation effects is
        # applied to the total aircraft source noise, directly under it

        phi[phi < 0] = 0
        phi[phi > 180] = 0

        phi_rad = np.radians(phi)

        Einst = np.zeros_like(phi_rad)
        if 'wing' in engine.lower():
            # Engine installation effects for wing mounted engines
            Einst = Einst - 1.49
            ind = np.where(phi <= 180)
            Einst[ind] = 10 * np.log10((0.0039 * np.cos(phi_rad[ind]) ** 2 + np.sin(phi_rad[ind]) ** 2) ** 0.062 /
                                       (0.8786 * np.sin(2 * phi_rad[ind]) ** 2 + np.cos(2 * phi_rad[ind]) ** 2))
        elif 'fus' in engine.lower():
            # Engine installation effects for fuselage mounted engines
            Einst = 10 * np.log10((0.1225 * np.cos(phi_rad) ** 2 + np.sin(phi_rad) ** 2) ** 0.329)
        else:
            print('Lateral attenuation method is not valid for this configuration and it will be ignored')

        return Einst


class PerceivedNoiseMetrics:

    @classmethod
    def getPNL(cls, nti, fdoppler, SPLp):
        """
        Computes the Perceived Noise Level.

        :param int nti: Number of times (for interpolated grid)
        :param ndarray fdoppler: 1D array containing the Doppler shifted frequency (Hz)
        :param ndarray SPLp: 2D array containing Sound Pressure Level (dB)

        :return ndarray: 1D array containing PNL
        """

        vec2 = np.zeros(nti)
        Ffactor = 0.15

        for i in range(nti):
            wrk = cls.getNoys(fdoppler[:, i], SPLp[:, i])
            vec1 = max(wrk)
            vec2[i] = (1 - Ffactor) * vec1 + Ffactor * sum(wrk)

        return 40.0 + (10.0 / np.log10(2)) * np.log10(vec2)

    @staticmethod
    def getPNLT(nti, fband, PNL, SPL):
        """
        Computes the Perceived Noise Level corrected for spectral irregularities

        :param int nti: Number of times / points
        :param ndarray fband: 1D array containing octave band frequencies (Hz)
        :param ndarray PNL: 1D array containing Perceived Noise Level (dB)
        :param ndarray SPL: 2D array containing Sound Pressure Level (dB)

        :return ndarray: 1D array containing PNLT
        """
        nfr = len(fband)
        s = np.zeros((nfr, nti))
        encircled = np.zeros((nfr, nti), dtype=int)
        snew = np.zeros((nfr + 1, nti))
        sbar = np.zeros((nfr, nti))
        SPLfinal = np.zeros((nfr, nti))

        s[3:nfr, :] = SPL[3:nfr, :] - SPL[2:nfr - 1, :]

        for i in range(1, nfr):
            for k in range(nti):
                if abs(s[i, k] - s[i - 1, k]) > 5.0:
                    if s[i, k] > 0.0 and s[i, k] > s[i - 1, k]:
                        encircled[i, k] = 1
                    elif s[i, k] <= 0.0 < s[i - 1, k]:
                        encircled[i - 1, k] = 1

        SPLnew = np.zeros((nfr, nti))

        SPLnew[0, :] = SPL[0, :]
        SPLnew[encircled == 0] = SPL[encircled == 0]
        for i in range(nfr):
            for k in range(nti):
                if i < 23 and encircled[i, k] != 0:
                    SPLnew[i, k] = 0.5 * (SPL[i - 1, k] + SPL[i + 1, k])
                elif i == 23 and encircled[i, k] != 0:
                    SPLnew[23, k] = SPL[22, k] + s[22, k]

        # new slopes
        snew[3:nfr, :] = SPLnew[3:nfr, :] - SPLnew[2:nfr - 1, :]
        snew[2, :] = snew[3, :]
        snew[nfr, :] = snew[nfr - 1, :]

        sbar[2:nfr - 1, :] = (1 / 3) * (snew[2:nfr - 1, :] + snew[3:nfr, :] + snew[4:nfr + 1, :])

        # SPLfinal
        SPLfinal[2, :] = SPL[2, :]
        for i in range(3, nfr):
            SPLfinal[i, :] = SPLfinal[i - 1, :] + sbar[i - 1, :]

        # difference
        F = SPL - SPLfinal

        F[F < 1.5] = 0.0
        F[0:2, :] = 0.0

        for i in range(2, nfr):
            for k in range(nti):
                if 500.0 > fband[i] >= 50.0 or fband[i] > 5000.0:
                    if 3.0 <= F[i, k] < 20.0:
                        F[i, k] = F[i, k] / 6.0
                    elif F[i, k] >= 20.0:
                        F[i, k] = 10.0 / 3.0
                    elif 1.5 <= F[i, k] < 3.0:
                        F[i, k] = F[i, k] / 3.0 - 0.5
                elif 500.0 <= fband[i] <= 5000.0:
                    if 3.0 <= F[i, k] < 20.0:
                        F[i, k] = F[i, k] / 3.0
                    elif F[i, k] >= 20.0:
                        F[i, k] = 20.0 / 3.0
                    elif 1.5 <= F[i, k] < 3.0:
                        F[i, k] = 2 * F[i, k] / 3.0 - 1

        C = np.amax(F, axis=0)

        return PNL + C

    @staticmethod
    def getEPNL(PNLT):
        """
        Computes the Effective Perceived Noise Level.

        :param ndarray PNLT: 1D array containing the corrected Perceived Noise Level (dB)

        :return float: EPNL
        """
        PNLTM = max(PNLT)
        # Duration correction D calculated based on the threshold PNLTM-10 for all the conditions
        istart = np.where(PNLTM - PNLT < 10.0)[0][0]
        istop = istart + np.where(PNLTM - PNLT[istart:-1] < 10.0)[0][-1]
        if istop > len(PNLT): istop = len(PNLT)

        PNLTsum = sum(10 ** (PNLT[istart:istop + 1] / 10.0))

        D = 10.0 * math.log10(PNLTsum) - PNLTM - 13.0

        return D + PNLTM

    @staticmethod
    def getNoys(fband, SPL_vec):
        """ Computes Noys metric from SPL and frequency. Source:
        https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-36/appendix-Appendix%20A%20to%20Part%2036"""
        Noys = np.zeros(len(fband))

        Mb = np.array([0.043478, 0.040570, 0.036831, 0.036831, 0.035336, 0.033333, 0.033333, 0.032051, 0.030675,
                       0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.029960, 0.029960, 0.029960,
                       0.029960, 0.029960, 0.029960, 0.029960, 0.042285, 0.042285])
        Mc = np.array([0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103, 0.030103,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.029960, 0.029960])
        Md = np.array([0.079520, 0.068160, 0.068160, 0.059640, 0.053013, 0.053013, 0.053013, 0.053013, 0.053013,
                       0.053013, 0.053013, 0.053013, 0.053013, 0.053013, 0.059640, 0.053013, 0.053013, 0.047712,
                       0.047712, 0.053013, 0.053013, 0.068160, 0.079520, 0.059640])
        Me = np.array([0.058098, 0.058098, 0.052288, 0.047534, 0.043573, 0.043573, 0.040221, 0.037349, 0.034859,
                       0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.034859, 0.040221, 0.037349, 0.034859,
                       0.034859, 0.034859, 0.034859, 0.037349, 0.037349, 0.043573])
        SPLa = np.array([91.0, 85.9, 87.3, 79.9, 79.8, 76.0, 74.0, 74.9, 94.6, np.inf, np.inf, np.inf, np.inf, np.inf,
                         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 44.3, 50.7])
        SPLb = np.array([64.0, 60.0, 56.0, 53.0, 51.0, 48.0, 46.0, 44.0, 42.0, 40.0, 40.0, 40.0, 40.0, 40.0, 38.0, 34.0,
                         32.0, 30.0, 29.0, 29.0, 30.0, 30.0, 37.0, 41.0])
        SPLc = np.array([52.0, 51.0, 49.0, 47.0, 46.0, 45.0, 43.0, 42.0, 41.0, 40.0, 40.0, 40.0, 40.0, 40.0, 38.0, 34.0,
                         32.0, 30.0, 29.0, 29.0, 30.0, 31.0, 34.0, 37.0])
        SPLd = np.array([49, 44, 39, 34, 30, 27, 24, 21, 18, 16, 16, 16, 16, 16, 15, 12, 9, 5, 4, 5, 6, 10, 17, 21])
        SPLe = np.array([
            55, 51, 46, 42, 39, 36, 33, 30, 27, 25, 25, 25, 25, 25, 23, 21, 18, 15, 14, 14, 15, 17, 23, 29])

        for i, fb in enumerate(fband):
            if SPL_vec[i] >= SPLa[i]:
                Noys[i] = 10 ** (Mc[i] * (SPL_vec[i] - SPLc[i]))
            elif SPLb[i] <= SPL_vec[i] < SPLa[i]:
                Noys[i] = 10 ** (Mb[i] * (SPL_vec[i] - SPLb[i]))
            elif SPLe[i] <= SPL_vec[i] < SPLb[i]:
                Noys[i] = 0.3 * 10 ** (Me[i] * (SPL_vec[i] - SPLe[i]))
            elif SPLd[i] <= SPL_vec[i] < SPLe[i]:
                Noys[i] = 0.1 * 10 ** (Md[i] * (SPL_vec[i] - SPLd[i]))

        return Noys


class DynamicViscocity:

    @staticmethod
    def calculate(t):
        """ Computes the dynamic viscocity for a given temperature. """
        mu_0 = 17.1e-6
        if 273.15 <= t < 373.15:
            beta = np.array([0.4743742289, 0.0445219791, 0.1651857290])
        elif 373.15 <= t < 973.15:
            beta = np.array([0.5688972460, 0.0414204258, 0.1192514508])
        elif t < 273.15:
            return 17.1e-6
        elif t > 973.15:
            return 41.5e-6
        return mu_0 * ((t / 288.15) ** beta[0] + beta[1] * (t / 288.15) + beta[2] * math.log((t / 288.15) ** 2))
