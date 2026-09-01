    // Coordinates come from the Geometry rather than remora.vec_xr/vec_yr so that this hook
    // also works on the full-domain MultiFab used for the hires_grid_level lane, which is
    // not built on grids[lev] and so has no per-level coordinate arrays to index. This
    // matches set_grid_coords_from_grid_scale, which puts the rho point of cell i at
    // (i + 1/2) * dx measured from index 0.
    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);

    for (MFIter mfi(mf_mskr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Every grow cell, not just NGROW of them: the full-domain array carries as many as
        // the cumulative refinement ratio needs, and any it left unwritten would coarsen
        // down as water.
        const Box &bx = mfi.growntilebox();

        Array4<      Real> const& mskr = mf_mskr.array(mfi);

        // mskr is 1 everywhere already
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int ) noexcept
        {
            const Real x_r = (i + Real(0.5)) * dx;
            const Real y_r = (j + Real(0.5)) * dy;
            if ((x_r > Real(2800.0) && x_r < Real(5600.0)) &&
                (y_r < Real(250.0) || y_r > Real(500.0))) {
                mskr(i,j,0) = zero;
            }
        });
    }
