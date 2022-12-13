"""
Module choice_read_and write
Used to read and edit external / input files.
"""

import sys
import choice_aux
from os import path, chdir


def read_external_files(dim_file, weight_file, noise_file, perf_file):
    """
    Reads data from files and creates and returns them in lists or dictionaries.
    :param dim_file: File with the modules and engine architecture details
    :param weight_file: File with data on engine sizing
    :param noise_file: File to define the cases to be run and required inputs for noise prediction
    :param perf_file: File containing the engine performance data for every point
    :return: the data in list or dictionary format
    """
    weightFile = retrieve_file(weight_file)
    noiseFile = retrieve_file(noise_file)

    if path.exists(perf_file):  # Check if multipoint performance data exist
        perfFile = open_performance_file(perf_file)
        mpd = MultptPerfData(perfFile)
    else:
        print('Multipoint perf. file not found. Case must be traj. perf. type.')

    # parse dimensions file and set modules
    modules = set_modules(dim_file)

    return [modules, weightFile, noiseFile, mpd]


def retrieve_file(file):
    """ Reads the data from the given file and creates a dictionary. """
    file_data = {}
    with open(file) as fp:
        for line in fp:
            if line and not line.startswith('!'):
                li = line.split(':')
                if choice_aux.isfloat(li[1]):
                    file_data[li[0].strip()] = float(li[1])
                else:
                    file_data[li[0].strip()] = li[1].strip()
            else:
                continue
    return file_data


def open_performance_file(file):
    """ Reads a given performance file into a list. """
    perfFile = []
    with open(file) as fp:
        for line in fp:
            if line and not line.startswith('!'):
                perfFile.append(line.strip())
            else:
                continue
    return perfFile


class MultptPerfData:
    """
    A class to split and store, in dictionaries, the multipoint performance data for each point.
    :param pf: A list of data from the performance file
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
                    self.perf_file_cr = load_dict(spos, pf)
                if p.find('Approach') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_approach = load_dict(spos, pf)
                if p.find('Cutback') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_cutback = load_dict(spos, pf)
                if p.find('Cruise') > -1 or p.find('MID-CRUISE') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_cr = load_dict(spos, pf)
                if p.find('Top of Climb') > -1 or p.find('TOP_OF_CLIMB') > -1 or p.find('Top-of-climb') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_toc = load_dict(spos, pf)
                if p.find('Sideline') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_sl = load_dict(spos, pf)
                if p.find('Take-Off') > -1 or p.find('Take-off') > -1:
                    spos = i + 3
                    tag_found = True
                    self.perf_file_to = load_dict(spos, pf)
                if not tag_found:
                    choice_aux.report_error('Point name specifier likely to have been misspelt. You wrote ' + pf[
                        i].strip() + '. Try Take-off, Top-of-climb or Cruise', 'split_performance_file',
                                            'read_and_write_files')
            else:
                continue


def load_dict(spos, pf):
    """ Returns a dictionary with all the data for a given operating point. """
    farr = {}
    for i in range(spos, len(pf)):
        if pf[i].find(':') > -1:
            li = pf[i].split(':')
            if choice_aux.isfloat(li[1]):
                farr[li[0].strip()] = float(li[1])
            else:
                farr[li[0].strip()] = li[1].strip()
        else:
            break
    return farr


def set_modules(dimF):
    """ Returns a list of the modules in the dimensionsWeight.txt file"""
    modules = []
    with open(dimF) as df:
        for line in df:
            if line and not line.startswith('!'):
                if 'end module' in line:
                    modules.append(line.split()[2].strip())
    return modules


def preparse_trajectories(traj_perf, opPnt, modules):
    """
    Preparses the trajectory and component performance files by removing data corresponding to stationary points.
    """

    def parseTrajectoryFile(opPnt):
        """ Preparses the trajectory files by removing stationary points. """
        filename = 'Input/' + opPnt.rstrip() + '.txt'
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

    def parsePerformanceFiles(opPnt, nBeg, modules):
        """ Preparses all the performance files for a given operating point. """
        if 'Fan' in modules:
            filename = opPnt.rstrip() + '_fan_performance.txt'
            parsePerfFile(filename, nBeg)
        if 'Ipc' or 'Lpc' in modules:
            filename = opPnt.rstrip() + '_lpc_performance.txt'
            parsePerfFile(filename, nBeg)
        if 'Lpt' in modules:
            filename = opPnt.rstrip() + '_lpt_performance.txt'
            parsePerfFile(filename, nBeg)
        if 'Comb' in modules:
            filename = opPnt.rstrip() + '_comb_performance.txt'
            parsePerfFile(filename, nBeg)
        if 'cold_nozzle' in modules:
            filename = opPnt.rstrip() + '_coAxialJet_performance.txt'
            parsePerfFile(filename, nBeg)
        if 'fuselage_fan' in modules:
            filename = opPnt.rstrip() + '_fuselagefan_performance.txt'
            parsePerfFile(filename, nBeg)
        if 'ff_nozzle' in modules:
            filename = opPnt.rstrip() + '_ffnJet_performance.txt'
            parsePerfFile(filename, nBeg)
        filename = opPnt.rstrip() + '_airfrm_performance.txt'
        parsePerfFile(filename, nBeg)

    def parsePerfFile(filename, nBeg):
        """ Preparses the performance file for a given operating point and component by removing the lines corresponding
        to stationary points. """

        print(' opening file', filename.strip())
        filename = 'Input/' + filename
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
        n = parseTrajectoryFile(op.rstrip())

        if traj_perf:
            print('Trajectory performance file being parsed')
            parsePerformanceFiles(op.rstrip(), n[0], modules)
    sys.exit('stopped after preparsing - turn preparsing false to continue to computations')
