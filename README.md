# CHOICE
An aircraft noise prediction framework based on empirical and semi-empirical methods found in public literature. 
The code is able to predict the noise at the source (aircraft) and at the certification points on the ground (so far for 2D propagation only).  

-------------------------------------------------------------------------------
The following files are used as input:

inputNoise.txt --> The noise study cases are defined
- point: specify the noise certification point e.g. Approach, Cutback or Sideline
- trajectory_performance: true/false. If set to true, files for trjectory data should be provided e.g. Approach_airframe_performnce.txt. If set to false a single point performance study is perfomed using the performanceResults.txt
- use_trajectory_preparser: true/false. If set to true, the input trajectory performance files are processed to remove empty lines or zero velocity of aircraft at early or late points
- generate_noise_sources: true/false. If set to true, a 3D matrix (frequencies x directivities x trajectory points) is generated for every noise component.
- use_ground_reflection: true/false. If set to true, ground reflection is included in the propagation effects.
- use_spherical_spreading: true/false. If set to true, spherical spreading is included in the propagation effects.
- use_atmospheric_attenuation: true/false. If set to true, atmospheric absorption is included in the propagation effects.
- use_suppression_factor: true/false. If set to true, noise suppression is applied on certain components. Values should be provided on a later instance.
- no_engines: number of engines on the aircraft
- type_nozzles: separate/mix. To define separate or mix exhaust nozzles.
- xmic: distance of microphone from the landing point, along the runway path 
- ymic: microphone height from ground 
- zmic: lateral distance from runway. 
- dtIsa: deviation from ISA temperature.
- aircraft_mass: mass of aircraft in kg
- wing_area: wing area in m**2
- wing_span: wing span in m
- horizontal_tail_area: horizontal tail area in m**2
- horizontal_tail_span: horizontal tail span in m
- vertical_tail_area: vertical tail area in m**2
- vertical_tail_span: vertical tail span m
- jet_aircraft_coef: constant value (in dB) that should be added to the clean wing noise prediction. According to the litterature, it shoud be 0 for aerodynamically clean sailplanes, 6 for most jet aircraft and 8 for conventional low-speed aircraft.
- noFlaps: number of flap elements.
- flap_area: flap area of each flap element. If noFlaps = 1 the total flap area should be used.
- flap_span: flap span of each flap element. If noFlaps = 1 the total flap span should be used.
- flap_type: 1slot/2slot/3slot for single, double anf tripple slotted flaps respectively.
- leading_edge_type: slat/flap for slats or leading edge flaps
- LandingGear_vec: 0/1 for retracted or extended
- noLG: number of landing gears
- d_wheel: wheel diameter for each landing gear in in
- d_strut: typical strut diameter for each landing gear in in
- d_wire:  typical wire or hydraulic pipe diameter for each landing gear in in
- Nwheels: number of wheels on each landing gear
- Nstruts: number of main strust on each landing gear
- comb_ign_nozzles: ignitted nozzles for SAC (Single Annular Combustor)
- dac_nozzle_pattern: nozzle firing pattern for DAC (Dual Annular Combustor). Can be 40/30/22.5/20, see relevant literature for more information.
- fan_distortion: 0/1 to not account or account for distortion at the fan
- ff_distortion: 0/1 to not account or account for distortion at the fan
- fuselage_fan: false/true if fuselage fan exists or doesn't exist
- psi: airctaft pitch angle in rad. Only used whe trajectory_performance is false
- defl_flap: flap deflection angle in rad
- fan_inlet_suppression: noise suppression constant for fan inlet
- fan_dis_suppression: noise suppression constant for fan inlet
- lpt_suppression: noise suppression constant for fan inlet
