# CHOICE - CHalmers nOIse CodE
An aircraft noise prediction framework based on empirical and semi-empirical methods found in public literature. 
The code is able to predict the noise at the source (aircraft) and at the certification points on the ground (so far for 2D propagation only).  

-------------------------------------------------------------------------------
Required libraries: numpy, scipy

Python version: 3

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

dimensionsWeight.txt --> The modules of the engine architecture are provided, e.g. Fan, Lpc, Comb, etc. The rest of the parameters are not used.

weightAircraft.txt --> Engine sizing data. Metric system is used
- GBX_ratio: gear box ratio. If the engine is direct driven, set this parameter to 1.0,
- FanR1BDiaOuter: fan outer diameter
- FanA2: fan annulus area
- FanR1BNb: number of fan rotor blades.
- FanRss: rotor stator spacing parameter. See the relevant literature for definition.
- MtipFan: relative tip Mach number of fan in design point (typically the top-of-climb point during a flight mission). 
- xnl: fan rotation speed in design point in rps
- FanVsOgvNb: number of fan stator blades of
- stages_LPC: number of LPC stages
- r_compr: LPC radius
- rh_compr: LPC hydraulic radius
- RSS_compr: rotor stator spacing parameter
- MtipLpc: relative tip Mach number of lpc in design point (typically the top-of-climb point during a flight mission). 
- N_compr: number of rotor blades for first stage rotor.
- S_compr: number of stator blades for first stage stator.
- CombType: SAC/DAC, combustor type
- maxNo_nozzles_dac: total number of fuel nozzles in DAC
- A_comb_exit: combustor exit area
- Deff_comb: exhaust nozzle exit plane effective diameter
- Dhyd_comb: exhaust nozzle exit plane hydraulic diameter
- Lc_comb: combustor nominal length
- h_annulus_comb: annulus height at combustor exit
- LptStgLastBNb: number of rotor at the last stage of the LPT---------------------------------
- stages_LPT: number of LPT stages
- LptStgLastDiaOuter: outer diameter of the last stage of the LPT
- LptStgLastAExit: exit area of the last stage of the LPT
- SRS: rotor stator spacing
- A_core: core nozzle area
- A_bypass: bypass nozzle are

performanceResults.txt or trajectory performance files --> engine and aircraft performance data for every point

Fan performance:
-	Mtip_fan: relative tip Mach number. In single point mode it is computed from inlet temperature, pressure mass flow area, diameter and rotational speed.
-	Mu_fan: blade Mach number. In single point mode it is computed from inlet temperature, pressure mass flow area, diameter and rotational speed.
-	xnl_fan: rotational speed in rps. 
-	dt_fan: stage temperature rise over fan rotor.
- g1_fan: mass flow at fan face in kg/s.

IPC/LPC performance:
-	Mtip_lpc: relative tip Mach number. In single point mode it is computed from inlet temperature, pressure mass flow area, diameter and rotational speed.
-	Mu_lpc: blade Mach number. In single point mode it is computed from inlet temperature, pressure mass flow area, diameter and rotational speed.
-	xnl_fan: rotational speed in rps.
-	dt_lpc: stage temperature rise over LPC rotor. In single point mode it is computed from a stage loading parameter (0.70 for non-geared and 0.45 for geared).
-	g1_lpc: mass flow at LPC in kg/s.

Combustor performance:
- P3: combustor inlet pressure in Pa
- P4: combustor exit pressure in Pa
- P7: turbine exit pressure in Pa
- T3: combustor inlet temperature in K
- T4: combustor exit temperature in K
- T5: turbine exit temperature in K
- W3: combustor inlet flow in kg/s

LPT performance:
-	V_TR: relative tip speed of turbine last rotor in m/s
-	T_LPT_exit: turbine exhaust temperature in K
-	n_LPS: rotational speed in rps
-	m_core: mass flow in kg/s
-	Cax: axial velocity in m/s. In single point mode it is computed from the previous parameters and the LPT exit area.

Jet performance:
- dmdt_1_hot: mass flow rate of inner stream or circular jet in kg/s
- dmdt_2_cold: mass flow rate of outer stream  in kg/s
- v_1: nozzle exit flow velocity of inner stream or circular jet in m/s
- v_2: nozzle exit flow velocity of outer stream in m/s
- T_1: nozzle exit flow total temperature of inner stream or circular jet in K
- T_2: nozzle exit flow total temperature of outer stream in K

Airframe performance:
- psi: aircraft pitch angle in rad
- defl_flap: flap deflection angle in rad
- defl_slat: slat deflection angle in rad
- LandingGear: landing gear position, 0 or 1
