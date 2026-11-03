This problem tests advection of tracer particles over non-flat bathymetry. It
uses the :ref:`Seamount<seamount-desc>` grid, bathymetry and stratification,
but starts the fluid with a uniform velocity of 0.5 m/s in x and 0.25 m/s in y
(``remora.prob.u_0`` and ``remora.prob.v_0``) instead of at rest. Particles
start at cell centres over the flanks of the seamount, through the whole water
column, and the flow carries them up and down the slopes. The depth of the water
column beneath a particle therefore changes as it moves, so this case exercises
the vertical part of the particle advection, which the flat bottom of
:ref:`Particle Advection Flat<particleadvectionflat>` does not. Build with
particles enabled to run it.

The ``Seamount_particles`` regression test is a shorter version of this case.
