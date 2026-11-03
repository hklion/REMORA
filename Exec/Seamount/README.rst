The Seamount problem is a standard test in
`ROMS <https://www.myroms.org/wiki/SEAMOUNT_CASE>`_.
The problem involves an (analytically) stably stratified fluid at rest over a
seamount. In the absence of numerical errors, the fluid will remain at rest.
However, this may not occur due to numerical errors in the calculation of the
horizontal pressure gradient when the vertical coordinates are misaligned with
the geopotential surfaces, as is the case in problems with spatially-varying
bathymetry in ROMS/REMORA.

Setting ``remora.prob.u_0`` and/or ``remora.prob.v_0`` (m/s, both 0 by
default) starts the fluid with that uniform velocity instead, so that it flows
over the seamount. The :ref:`Particle Advection Seamount<particleadvectionseamount>`
case and the ``Seamount_particles`` regression test use this to carry tracer
particles over the changing bathymetry.
