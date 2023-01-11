"""
Module choice_read_and write

Used to read and edit external / input files.
"""

import sys
import choice.choice_aux as choice_aux
from os import path
import math


class ReadFiles:
    """
    Instantiate files to be read.

    :param str dim_file: File with the modules and engine architecture details
    :param str weight_file: File with data on engine sizing
    :param str noise_file: File to define the cases to be run and required inputs for noise prediction
    :param str perf_file: File containing the engine performance data for every point
    """

    def __init__(self, dim_file, weight_file, noise_file, perf_file):
        self.dim_file = dim_file
        self.weight_file = weight_file
        self.noise_file = noise_file
        self.perf_file = perf_file
        self.modules = []
        self.perfFile = []

        self.weightFile = self.retrieve_file(self.weight_file)
        self.noiseFile = self.retrieve_file(self.noise_file)

        if path.exists(self.perf_file):  # Check if multipoint performance data exist
            self.open_performance_file()
            self.mpd = MultptPerfData(self.perfFile)
        else:
            print('Multipoint perf. file not found. Case must be traj. perf. type.')

        # parse dimensions file and set modules
        self.set_modules()

    @staticmethod
    def retrieve_file(file):
        """ Reads the data from the given file and creates a dictionary. """
        file_data = {}
        with open(file) as fp:
            for line in fp:
                if line and not line.startswith('!'):
                    li = line.split(':')
                    if isfloat(li[1]):
                        file_data[li[0].strip()] = float(li[1])
                    else:
                        file_data[li[0].strip()] = li[1].strip()
                else:
                    continue
        return file_data

    def set_modules(self):
        """ Returns a list of the modules in the dimensionsWeight.txt file"""
        with open(self.dim_file) as df:
            for line in df:
                if line and not line.startswith('!'):
                    if 'end module' in line:
                        self.modules.append(line.split()[2].strip())

    def open_performance_file(self):
        """ Reads a given performance file into a list. """
        with open(self.perf_file) as fp:
            for line in fp:
                if line and not line.startswith('!'):
                    self.perfFile.append(line.strip())
                else:
                    continue


class MultptPerfData:
    """
    Instantiate the multipoint performance data for each point.

    :param list pf: A list of data from the performance file
    """

    def __init__(self, pf):
        for i, p in enumerate(pf):
            if p.find('Point Name:') > -1 or p.find('Point name:') > -1:
                tag_found = False
                if p.find('ISA') > -1 or p.find('SLS Hot Day') > -1 or p.find('ICAO') > -1 or p.find('design') > -1:
                    continue
                if p.find('Cruise') > -1 or p.find('MID-CRUISE') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_cr = self.load_dict(spos, pf)
                if p.find('Approach') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_approach = self.load_dict(spos, pf)
                if p.find('Cutback') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_cutback = self.load_dict(spos, pf)
                if p.find('Cruise') > -1 or p.find('MID-CRUISE') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_cr = self.load_dict(spos, pf)
                if p.find('Top of Climb') > -1 or p.find('TOP_OF_CLIMB') > -1 or p.find('Top-of-climb') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_toc = self.load_dict(spos, pf)
                if p.find('Sideline') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_sl = self.load_dict(spos, pf)
                if p.find('Take-Off') > -1 or p.find('Take-off') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_to = self.load_dict(spos, pf)
                if not tag_found:
                    choice_aux.report_error(
                        'Point name specifier likely to have been misspelt. You wrote ' + pf[i].strip() +
                        '. Try Take-off, Top-of-climb or Cruise', 'split_performance_file', 'read_and_write_files')
            else:
                continue

    @staticmethod
    def load_dict(spos, pf):
        """ Returns a dictionary with all the data for a given operating point. """
        farr = {}
        for i in range(spos, len(pf)):
            if pf[i].find(':') > -1:
                li = pf[i].split(':')
                if isfloat(li[1]):
                    farr[li[0].strip()] = float(li[1])
                else:
                    farr[li[0].strip()] = li[1].strip()
            else:
                break
        return farr


def preparse_trajectories(traj_perf, opPnt, modules, input_folder):
    """
    Preparses the trajectory and component performance files by removing data corresponding to stationary points.
    """

    def parseTrajectoryFile(opPnt, input_folder):
        """ Preparses the trajectory files by removing stationary points. """
        filename = input_folder + opPnt.rstrip() + '.txt'
        lines = []
        with open(filename) as fp:
            for line in fp:
                if line:
                    li = line.strip()
                    lines.append(li)

        # if the file starts with a comment line try to figure out the position of x and Va (method is based on units)
        if lines[0].startswith('!'):
            string = lines[0].split()[1:]
            xpos = -1
            j = -1
            for st in string:
                str1 = string[0].lstrip()
                if str1[0] == '!': continue
                if '(m)' in st: continue
                if '(m/s)' in st: continue
                if '(s)' in st: continue
                if '(deg)' in st: continue

                j = j + 1
                if 'xpos' in st: xpos = j
                if 'Va' in st: vapos = j

        else:  # if no clues from comments on first line put default positions
            xpos = 0
            vapos = 2

        nBeg = -1
        for i, li in enumerate(lines):
            if li.startswith('!'): continue
            string = li.split()
            if float(string[xpos]) == 0.0 or float(string[vapos]) == 0.0:
                nBeg = nBeg + 1
                print('Detected standstill point in line ' + str(i + 1) + '=> removing line. CHOICE will remove '
                                                                          'corresponding lines in performance files '
                                                                          'if trajectory performance is true')
            else:
                break

        nEnd = -1
        j = -1
        with open(filename, 'w') as f:
            for li in lines:
                if li.startswith('!'):
                    f.write(li.strip() + '\n')
                else:
                    j = j + 1
                    if j <= nBeg:
                        continue
                    else:
                        f.write(li.strip() + '\n')

        return [nBeg, nEnd]

    def parsePerformanceFiles(opPnt, nBeg, modules, input_folder):
        """ Preparses all the performance files for a given operating point. """
        if 'Fan' in modules:
            filename = opPnt.rstrip() + '_fan_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        if 'Ipc' or 'Lpc' in modules:
            filename = opPnt.rstrip() + '_lpc_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        if 'Lpt' in modules:
            filename = opPnt.rstrip() + '_lpt_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        if 'Comb' in modules:
            filename = opPnt.rstrip() + '_comb_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        if 'cold_nozzle' in modules:
            filename = opPnt.rstrip() + '_coAxialJet_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        if 'fuselage_fan' in modules:
            filename = opPnt.rstrip() + '_fuselagefan_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        if 'ff_nozzle' in modules:
            filename = opPnt.rstrip() + '_ffnJet_performance.txt'
            parsePerfFile(filename, nBeg, input_folder)
        filename = opPnt.rstrip() + '_airfrm_performance.txt'
        parsePerfFile(filename, nBeg, input_folder)

    def parsePerfFile(filename, nBeg, input_folder):
        """ Preparses the performance file for a given operating point and component by removing the lines corresponding
        to stationary points. """

        print(' opening file', filename.strip())
        filename = input_folder + filename
        lines = []
        with open(filename) as fp:
            for line in fp:
                if line:
                    li = line.strip()
                    lines.append(li)

        # re-open unit and dump parsed file
        j = -1
        with open(filename, 'w') as fp:
            for i, li in enumerate(lines):
                if li.startswith('!'):
                    fp.write(li.strip() + '\n')
                else:
                    j = j + 1
                    if j <= nBeg:
                        print(' skipping line number ' + str(i + 1), ' to avoid standstill operation')
                        continue
                    else:
                        fp.write(li.strip() + '\n')

    print('Preparsing routines was requested...')

    for op in opPnt:
        print(' parsing', op.rstrip(), '.txt')
        n = parseTrajectoryFile(op.rstrip(), input_folder)

        if traj_perf:
            print('Trajectory performance file being parsed')
            parsePerformanceFiles(op.rstrip(), n[0], modules, input_folder)
    sys.exit('stopped after preparsing - turn preparsing false to continue to computations')


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


def isfloat(num):
    """ Checks if the provided string is a float number. """
    try:
        float(num)
        return True
    except ValueError:
        return False
