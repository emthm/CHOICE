from choice import CHOICE

if __name__ == "__main__":

    #  Input files folder path
    input_folder = 'Input/'

    #  Output folder path and file type for noise source matrices
    output_folder = 'Output/'
    file_type = 'csv'

    #  Specify input files
    perf_file = 'performanceResults.txt'
    weight_file = 'weightAircraft.txt'
    noise_file = 'inputNoise.txt'

    #  Call CHOICE
    noise = CHOICE(input_folder, output_folder, perf_file, weight_file, noise_file, file_type)
    noise.run_choice()
