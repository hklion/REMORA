# ------------------  INPUTS TO MAIN PROGRAM  -------------------
# Tracer particles carried by a uniform initial flow over the Seamount
# bathymetry. The particles start at cell centres over the flanks of the
# seamount, where the depth varies between 500 m and 5000 m. The flow carries
# them along those slopes, so the water column under each particle changes
# depth as it moves, and the flow over the bump has a vertical velocity. The
# seeding fills the whole water column, so the stencil's bottom and top half
# cells are exercised.
remora.prob_name = Seamount

remora.max_step = 40
remora.stop_time = 30000.0

amrex.fpe_trap_invalid = 1

# PROBLEM SIZE & GEOMETRY
remora.prob_lo     =   0.       0.  -5000.
remora.prob_hi     =   320000.  320000.       0.

remora.n_cell           =  49   48   13

remora.is_periodic = 1 1 0

# TIME STEP CONTROL
remora.fixed_dt       = 300.0 # Timestep size (seconds)
remora.ndtfast  = 30

# DIAGNOSTICS & VERBOSITY
remora.sum_interval   = 1       # timesteps between computing mass
remora.v              = 0       # verbosity in REMORA.cpp (0: none, 1: print boxes, etc, 2: print values)
amr.v              = 1       # verbosity in Amr.cpp

# CHECKPOINT FILES
remora.check_file      = chk        # root name of checkpoint file
remora.check_int       = -57600      # number of timesteps between checkpoints

# PLOTFILES
remora.plot_file     = plt        # prefix of plotfile name
remora.plot_int      = 40         # number of timesteps between plotfiles
remora.plot_vars_3d  = salt temp x_velocity y_velocity z_velocity tracer_particles_count
remora.plotfile_type = amrex

# PARTICLES
# The seamount is a Gaussian of half-width 40 km centred at (160 km, 160 km);
# the box covers its flanks out to two half-widths. The z range takes every cell.
remora.use_tracer_particles = 1
remora.tracer_particles.initial_distribution_type = box
remora.tracer_particles.particle_box_lo = 80000.   80000.  -100000.
remora.tracer_particles.particle_box_hi = 240000.  240000.  100000.
remora.tracer_particles.place_randomly_in_cells = false

# SOLVER CHOICE
remora.use_coriolis = true
remora.tracer_horizontal_advection_scheme = "upstream3" # upstream3 or centered4

remora.coriolis_f0 = 1.0e-4 #-8.26e-5

remora.theta_s = 6.5
remora.theta_b = 2.0
remora.tcline = 100.

# PROBLEM PARAMETERS (optional)
# Uniform initial flow, diagonal so both horizontal components carry particles
remora.prob.u_0 = 0.5
remora.prob.v_0 = 0.25

remora.R0    = 1027.0  # background density value (Kg/m3) used in Linear Equation of State
remora.S0    = 32.0    # background salinity (nondimensional) constant
remora.T0    = 10.0    # background potential temperature (Celsius) constant
remora.Tcoef = 1.7e-4  # linear equation of state parameter (1/Celsius)
remora.Scoef = 0.0     # linear equation of state parameter (nondimensional)
remora.rho0  = 1025.0  # Mean density (Kg/m3) used when Boussinesq approx is inferred
