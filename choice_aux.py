"""
Module choice_aux
Choice auxiliary routines such as error handling and file editing and writing.
"""
import os.path
import sys
import math
import numpy as np
import choice_data
from scipy.optimize import brentq

release_version = True
error_file_open = False


def loadStorageMat(fname, ndata, nvars):
    """
    Reads data from file and returns a 2D array.
    :param fname: File name
    :param ndata: Number of data loaded along the trajectory
    :param nvars: Number of variables stored for particular module
    :return: A 2D array with the data
    """
    mat = np.zeros((ndata, nvars))
    with open(fname) as fp:
        i = 0
        for line in fp:
            if line and not line.startswith('!'):  # data line
                string = line.split()
                for j, st in enumerate(string):
                    mat[i, j] = float(st)
                i = i + 1
    if i != ndata:
        sys.exit('confusion in number of data - ndata different from number of lines in file for file ')

    return mat


def preProcessFanFile(fname, d1, a2):
    """ Computes missing data from fan performance file and saves new file. (This routine is only called if the file
    has stars) """

    def parseFanLine(s, d1, a2):
        """Computes missing data from a line in the fan performance file."""
        maxNchars = 18
        xnl = float(s[2])
        mdot = float(s[4])
        t = float(s[5])
        p = float(s[7])
        [Mtip, Mu, Umid] = setMachNumbers(p, t, mdot, a2, d1, xnl, choice_data.gamma_air)
        l = []
        varStr = str(Mtip)
        l.append(varStr + '  ')
        varStr = str(Mu)
        l.append(varStr + '  ')
        for i in range(2, 5):
            l.append(s[i][0:maxNchars - 1] + ' ')
        fanLine = ''.join(l)
        return fanLine

    # maxNCols = 8  # valid for fan files with stars
    newFile = []
    with open(fname) as fp:
        for i, line in enumerate(fp):
            if line.startswith('!'):
                newFile.append(line)
                continue
            string = line.split()
            nvars = len(string)
            if nvars == -1:
                continue
            elif nvars != 8:
                sys.exit('failure in parsing logic - expected 8 number of columns in fan file')
                # consistency check to not forget that this logic is specific - avoid future simulation errors
            newFile.append(parseFanLine(string, d1, a2))

    # dump newFile onto file
    with open(fname, 'w') as fp:
        for i in range(len(newFile)):
            fp.write(newFile[i] + '\n')

        newFile[i] = parseFanLine(string, d1, a2)


def preProcessLptFile(fname, de, ae):
    """ Computes missing data from lpt performance file and saves new file.
    (This routine is only called if the file has stars) """

    def parseLptLine(s, de, ae):
        """Computes missing data from a line in the lpt performance file."""
        maxNchars = 18
        xnl = float(s[2])
        mdot = float(s[3])
        t = float(s[5])
        p = float(s[6])
        Cax = get_Cax(choice_data.gamma_gas, p, t, mdot, ae)
        Vtr = (math.pi * de) * xnl
        l = []
        varStr = str(Vtr)
        l.append(varStr + '  ')
        for i in range(1, 4):
            l.append(s[i][0:maxNchars - 1] + ' ')
        # i = 4
        varStr = str(Cax)
        l.append(varStr + '  ')
        lptLine = ''.join(l) + '\n'
        return lptLine

    new_file = []
    with open(fname) as fp:
        for line in fp:
            if line.startswith('!'):
                new_file.append(line)
                continue
            # data line
            string = line.split()
            nvars = len(string)
            if nvars == -1:
                continue
            elif nvars != 8:
                sys.exit('failure in parsing logic - expected 8 number of columns in Lpt file')
                # consistency check to not forget that this logic is specific - avoid future simulation errors
            new_file.append(parseLptLine(string, de, ae))

    # dump new_file onto file
    with open(fname, 'w') as fp:
        for i in range(len(new_file)):
            fp.write(new_file[i])

            new_file[i] = parseLptLine(string, de, ae)


def containsStars(fname):
    """ Checks if the file contains stars (****) instead of data. """
    stars = False
    with open(fname) as fp:
        for line in fp:
            if line.startswith('!'): continue
            string = line.split()
            for s in string:
                if ('*****') in s:
                    stars = True
                    break

            if stars: break

    return stars


def set_frequencies(nb, nfreq, fmin, fmax):
    """
    Computes the 1/3 octave band frequency, the mid-frequency of each band and all the frequencies from the given
    minimum to the maximum value.
    :param nb: Number of bands.
    :param nfreq: Number of frequencies.
    :param fmin: The minimum frequency value (Hz).
    :param fmax: The maximum frequency value (Hz).
    :return: The computed frequencies in array form.
    """
    fband = np.zeros(nfreq)
    f = np.zeros(nfreq + 1)
    f[0] = 0.0
    fband[0] = fmin
    pot = 1.0 / (3.0 * nb)

    for i in range(1, nfreq):
        fband[i] = fband[i - 1] * 2 ** pot
        f[i] = 0.5 * (fband[i - 1] + fband[i])

    f[nfreq] = fband[nfreq - 1] + 0.5 * (fband[nfreq - 1] - fband[nfreq - 2])
    freq = np.linspace(int(fmin), int(fmax), math.ceil(fmax) - math.ceil(fmin) + 1)

    return [fband, f, freq]


def get_R(fa):
    """ Computes gas constant. """
    return choice_data.Risa / (fa + 1.0)


def get_M(gam, xfunc0):
    """ Computes the Mach number in the fan and compressor components. """
    rpar = [gam, xfunc0]
    if xfunc_err(0, rpar) * xfunc_err(1, rpar) < 0:
        [M, r] = brentq(xfunc_err, 0.0, 1.0, args=rpar, full_output=True)
        if not r.converged:
            report_error('failed to bracket M', 'get_M', 'physics')
    else:
        M = 0.999

    return M


def xfunc_err(M, rpar):
    return xfunc(M, rpar[0]) - rpar[1]


def xfunc(M, gam):
    return math.sqrt(gam) * M / ((1 + (gam - 1) * M * M / 2.0) ** ((gam + 1) / (2.0 * (gam - 1))))


def report_error(mess, rout, mod):
    """
    Reports error in file.
    :param mess: Error message
    :param rout: Module where error occurred
    :param mod: Routine where error occurred
    """
    global error_file_open, error_file
    if release_version: return

    if error_file_open:
        with open(error_file, 'a') as f:
            f.write(mess)
            f.write('occurred in module ' + mod)
            f.write('in routine ' + rout)
            f.close()
    else:
        error_file_open = True
        error_file = 'weicoErrorReport.txt'
        with open(error_file, 'w') as f:
            f.write(mess)
            f.write(' occurred in module ' + mod)
            f.write(' in routine ' + rout)
            f.close()

    return


def gen_noise_source_matr_subr(operatingPoint, nfreq, nthet, n_traj_pts, prms):
    """
    Save component SPL matrices to files.
    :param operatingPoint: Approach, Cutback, Sideline
    :param nfreq: Number of frequencies
    :param nthet: Number of directivity angles
    :param n_traj_pts: Number of trajectory points
    :param prms: Rms or effective acoustic pressure for all components
    """
    directory = 'Output/'
    if not os.path.isdir(directory):
        os.mkdir(directory)

    temp3DMatrix_Fan_inlet = prms2SPL(prms.Fan_inlet)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanInlet', nfreq, nthet, n_traj_pts, temp3DMatrix_Fan_inlet)

    temp3DMatrix_Fan_discharge = prms2SPL(prms.Fan_discharge)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanDischarge', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Fan_discharge)

    temp3DMatrix_Lpc_inlet = prms2SPL(prms.Lpc_inlet)
    save_3D_matrix(directory + operatingPoint.strip() + '_lpcInlet', nfreq, nthet, n_traj_pts, temp3DMatrix_Lpc_inlet)

    temp3DMatrix_Lpt = prms2SPL(prms.Lpt)
    save_3D_matrix(directory + operatingPoint.strip() + '_Lpt', nfreq, nthet, n_traj_pts, temp3DMatrix_Lpt)

    temp3DMatrix_Comb = prms2SPL(prms.Comb)
    save_3D_matrix(directory + operatingPoint.strip() + '_Comb', nfreq, nthet, n_traj_pts, temp3DMatrix_Comb)

    temp3DMatrix_Caj = prms2SPL(prms.Caj)
    save_3D_matrix(directory + operatingPoint.strip() + '_Caj', nfreq, nthet, n_traj_pts, temp3DMatrix_Caj)

    temp3DMatrix_Airfrm = prms2SPL(prms.Airfrm)
    save_3D_matrix(directory + operatingPoint.strip() + '_Airfrm', nfreq, nthet, n_traj_pts, temp3DMatrix_Airfrm)

    # implementation for URT
    temp3DMatrix_Fan_inlet_tone = prms2SPL(prms.Fan_inlet_tone)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanInletTone', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Fan_inlet_tone)

    temp3DMatrix_Fan_discharge_tone = prms2SPL(prms.Fan_discharge_tone)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanDischargeTone', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Fan_discharge_tone)

    temp3DMatrix_Fan_inlet_broadband = prms2SPL(prms.Fan_inlet_broadband)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanInletBroadband', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Fan_inlet_broadband)

    temp3DMatrix_Fan_discharge_broadband = prms2SPL(prms.Fan_discharge_broadband)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanDischargeBroadband', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Fan_discharge_broadband)

    temp3DMatrix_Fan_inlet_combination = prms2SPL(prms.Fan_inlet_combination)
    save_3D_matrix(directory + operatingPoint.strip() + '_fanInletCombination', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Fan_inlet_combination)

    temp3DMatrix_Lpc_inlet_tone = prms2SPL(prms.Lpc_inlet_tone)
    save_3D_matrix(directory + operatingPoint.strip() + '_LpcInletTone', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Lpc_inlet_tone)

    temp3DMatrix_Lpc_inlet_broadband = prms2SPL(prms.Lpc_inlet_broadband)
    save_3D_matrix(directory + operatingPoint.strip() + '_LpcInletBroadband', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Lpc_inlet_broadband)

    temp3DMatrix_Lpc_inlet_broadband = prms2SPL(prms.Lpc_inlet_combination)
    save_3D_matrix(directory + operatingPoint.strip() + '_LpcInletCombination', nfreq, nthet, n_traj_pts,
                   temp3DMatrix_Lpc_inlet_broadband)

    temp3DMatrix_Total = getTotLevel(np.array([temp3DMatrix_Fan_inlet, temp3DMatrix_Fan_discharge,
                                               temp3DMatrix_Lpc_inlet, temp3DMatrix_Lpt, temp3DMatrix_Comb,
                                               temp3DMatrix_Caj, temp3DMatrix_Airfrm]))
    save_3D_matrix(directory + operatingPoint.strip() + '_Total', nfreq, nthet, n_traj_pts, temp3DMatrix_Total)


def prms2SPL(prms):
    """ Converts rms acoustic pressure to Sound Pressure Level. """
    castSPL = np.zeros(prms.shape)
    castSPL[np.nonzero(prms)] = 20.0 * np.log10(prms[np.nonzero(prms)] / choice_data.p0)

    return castSPL


def SPL2prms(SPL):
    """ For a given SPL matrix computes the acoustic mean square pressure."""
    return choice_data.p0 * 10.0 ** (SPL / 20.0)


def getTotLevel(L):
    """ Computes the total PNLT for all noise sources. """
    s = np.sum(10.00 ** (L / 10), axis=0)
    return 10.0 * np.log10(s)


def save_3D_matrix(fname, n_rows, n_cols, n_2d_mat, mat, ext=None):
    """ Saves a 3D matrix for in output file. """
    if ext is not None:
        file = fname + 'choice3DMatrix' + ext
        print('Creating file ' + fname + 'choice3DMatrix' + ext)
    else:
        file = fname + 'choice3DMatrix.m'
        print('Creating file ' + fname + 'choice3DMatrix.m')

    with open(file, 'w') as fp:
        for imat in range(n_2d_mat):
            fp.write('chMat(:,:,' + str(imat + 1).strip() + ') = [')
            for j in range(n_rows):
                for k in range(n_cols):
                    fp.write(str(format(mat[j, k, imat], '20.10f')))
                    fp.write(' ')
                fp.write('\n')
            fp.write('];\n')


def setMachNumbers(p1, t1, g1, Ain, D1, xnl, gamma):
    """
    Computes the blade Mach numbers for a component.
    :param p1: Pressure
    :param t1: Temperature
    :param g1: Mass flow
    :param Ain: Area
    :param D1: Diameter
    :param xnl: Rotational speed
    :param gamma: Heat capacity ratio
    :return: Blade and relative tip Mach number and mid point speed
    """
    rt = D1 / 2
    rh = math.sqrt(rt ** 2 - Ain / math.pi)
    R = get_R(0)
    xfunc0 = g1 * math.sqrt(R * t1) / (p1 * Ain)
    Max = get_M(gamma, xfunc0)
    ts = t1 / (1 + 0.5 * (gamma - 1) * Max ** 2)
    Utip = 2 * math.pi * rt * xnl
    Umid = 2 * math.pi * ((rt + rh) / 2) * xnl
    Mu = Utip / math.sqrt(gamma * R * ts)  # blade Mach number
    Mtip = math.sqrt(Mu ** 2 + Max ** 2)  # use Mach number triangle, assume zero swirl
    return [Mtip, Mu, Umid]


def get_Cax(gam, p, t, w, A):
    """ Computes the axial flow speed from pressure, temperature, mass flow and area. """
    R = get_R(0)
    xfunc0 = w * math.sqrt(R * t) / (p * A)
    Max = get_M(gam, xfunc0)
    ts = t / (1 + 0.5 * (gam - 1) * Max ** 2)
    Cax = Max * math.sqrt(gam * R * ts)
    return Cax


class EPNL_point:
    def __init__(self, lateral, cutback, approach):
        self.lateral = lateral
        self.cutback = cutback
        self.approach = approach


class chapter3():
    """
    A class that defined the EPNL certification limits for every operating point.
    :param noEngines: Number of engines
    :param MTOW: Maximum take-off weight (tn)
    """

    def __init__(self, noEngines, MTOW):
        self.noEng = noEngines
        self.MTOW = MTOW
        self.lateral = self.get_EPNL_lateral()
        self.cutback = self.get_EPNL_cutback()
        self.approach = self.get_EPNL_approach()

    def get_EPNL_approach(self):
        """ Computes EPNL limit for Approach. """
        if self.MTOW < 35.0:
            return 86.03 + 7.75 * math.log10(35.0)  # 97.9965
        elif 35.0 <= self.MTOW < 400.0:
            return 86.03 + 7.75 * math.log10(self.MTOW)
        else:
            return 86.03 + 7.75 * math.log10(400.0)  # 104.995

    def get_EPNL_cutback(self):
        """ Computes EPNL limit for Cutback. """
        if self.noEng == 2:
            if self.MTOW < 48.1:
                return 66.65 + 13.29 * math.log10(48.1)  # 89.0057
            elif 48.1 <= self.MTOW < 385.0:
                return 66.65 + 13.29 * math.log10(self.MTOW)
            else:
                return 66.65 + 13.29 * math.log10(385.0)  # 101.01077
        elif self.noEng == 3:
            if self.MTOW < 28.6:
                return 69.65 + 13.29 * math.log10(28.6)  # 89.0051
            elif 48.1 <= self.MTOW < 385.0:
                return 69.65 + 13.29 * math.log10(self.MTOW)
            else:
                return 69.65 + 13.29 * math.log10(385.0)  # 104.01077
        elif self.noEng == 4:
            if self.MTOW < 20.2:
                return 71.65 + 13.29 * math.log10(20.2)  # 88.998
            elif 48.1 <= self.MTOW < 385.0:
                return 71.65 + 13.29 * math.log10(self.MTOW)
            else:
                return 71.65 + 13.29 * math.log10(385.0)  # 106.01077

    def get_EPNL_lateral(self):
        """ Computes EPNL limit for Sideline. """
        if self.MTOW < 35.0:
            return 80.57 + 8.51 * math.log10(35.0)  # 93.71
        elif 35.0 <= self.MTOW < 400.0:
            return 80.57 + 8.51 * math.log10(self.MTOW)
        else:
            return 80.57 + 8.51 * math.log10(400.0)  # 102.71


def isfloat(num):
    """ Checks if the provided string is a float number. """
    try:
        float(num)
        return True
    except ValueError:
        return False
