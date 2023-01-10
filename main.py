from choice import CHOICE

if __name__ == "__main__":

    #  Input files folder path
    input_folder = 'Input/'

    #  Output folder path
    output_folder = 'Output/'

    #  Specify input files
    perf_file = 'performanceResults.txt'
    dim_file = 'dimensionsWeight.txt'
    weight_file = 'weightAircraft.txt'
    noise_file = 'inputNoise.txt'

    #  Call CHOICE
    noise = CHOICE()
    noise.run_choice()
