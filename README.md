# CHOICE
An aircraft noise prediction framework based on empirical and semi-empirical methods found in public literature. 
The code is able to predict the noise at the source (aircraft) and at the certification points on the ground (so far for 2D propagation only).  

The following files are used as input:

inputNoise.txt --> The noise study cases are defined
point: can be Approach, Cutback or Sideline
trajectory_performance: true/false. If set to true, files for trjectory data should be provided e.g. Approach_airframe_performnce.txt. 
				If set to false a single point performance study is perfomed using the performanceResults.txt
use_trajectory_preparser: true/false. Some could be zero velocity of aircraft at early or late points, empty lines in input files or input files with multiple empty lines after data records end. You can then apply this command. It will work on the input files and report any changes. When you set this true choice will work on the files but stop after this process. This procedure makes several sanity checks on the input files so it may be valuable to make this process any time a new set of data is being checked.
generate_noise_sources:true
use_ground_reflection: true
use_spherical_spreading: true
use_atmospheric_attenuation: true
use_suppression_factor: true
no_engines: 2
type_nozzles: separate
