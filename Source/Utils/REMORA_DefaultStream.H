#ifndef REMORA_DEFAULTSTREAM_H_
#define REMORA_DEFAULTSTREAM_H_

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

/**
 * \brief MFIter options for a loop whose kernels all run on the default stream, with no
 *        synchronization before or after the loop.
 *
 * By default an MFIter loop on the GPU spreads its boxes over several streams. It then
 * synchronizes before it starts, so that it sees what was queued on the default stream,
 * and once per stream when it ends, so that what follows sees its results. With these
 * options every kernel of the loop goes to the default stream instead, so both orderings
 * hold without any synchronization:
 * - kernels already queued on the default stream finish before the loop's kernels start;
 * - later work on the default stream starts after them, and a later loop over several
 *   streams synchronizes the default stream before it starts (outside a Gpu::NoSyncRegion).
 *
 * The loop body may only queue work on the default stream: kernels, and FABs on
 * The_Async_Arena, which are freed in stream order. It must not read results back on the
 * host. A MultiFab the loop's kernels use may be destroyed once the loop has been queued:
 * its memory goes back to the arena, and the work that reuses it is queued after the loop
 * on the default stream, or behind a loop's synchronization.
 *
 * Without a GPU launch region the options are those of info, so a host loop keeps its
 * tiling and threading.
 */
inline amrex::MFItInfo OnDefaultStream (amrex::MFItInfo info)
{
    if (amrex::Gpu::inLaunchRegion()) { info.UseDefaultStream().DisableDeviceSync(); }
    return info;
}

/** \brief OnDefaultStream for a loop tiled when not on the GPU, or never tiled */
inline amrex::MFItInfo OnDefaultStream (bool tiling = false)
{
    amrex::MFItInfo info;
    if (tiling) { info.EnableTiling(); }
    return OnDefaultStream(info);
}

/**
 * \brief mf.setVal(val, comp, ncomp, nghost) as one kernel for all boxes on the default
 *        stream, without synchronizing (see OnDefaultStream). On the host, mf.setVal.
 */
inline void SetValOnDefaultStream (amrex::MultiFab& mf, amrex::Real val, int comp, int ncomp,
                                   amrex::IntVect const& nghost)
{
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion()) {
        auto const& fa = mf.arrays();
        amrex::ParallelFor(mf, nghost, ncomp,
        [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n) noexcept
        {
            fa[b](i,j,k,n+comp) = val;
        });
        return;
    }
#endif
    mf.setVal(val, comp, ncomp, nghost);
}

/** \brief SetValOnDefaultStream on every component and ghost cell */
inline void SetValOnDefaultStream (amrex::MultiFab& mf, amrex::Real val)
{
    SetValOnDefaultStream(mf, val, 0, mf.nComp(), mf.nGrowVect());
}

/**
 * \brief MultiFab::Copy(dst, src, srccomp, dstcomp, ncomp, nghost) as one kernel for all
 *        boxes on the default stream, without synchronizing (see OnDefaultStream). On the
 *        host, MultiFab::Copy.
 */
inline void CopyOnDefaultStream (amrex::MultiFab& dst, amrex::MultiFab const& src,
                                 int srccomp, int dstcomp, int ncomp,
                                 amrex::IntVect const& nghost)
{
#ifdef AMREX_USE_GPU
    if (amrex::Gpu::inLaunchRegion()) {
        auto const& d = dst.arrays();
        auto const& s = src.const_arrays();
        amrex::ParallelFor(dst, nghost, ncomp,
        [=] AMREX_GPU_DEVICE (int b, int i, int j, int k, int n) noexcept
        {
            d[b](i,j,k,dstcomp+n) = s[b](i,j,k,srccomp+n);
        });
        return;
    }
#endif
    amrex::MultiFab::Copy(dst, src, srccomp, dstcomp, ncomp, nghost);
}

#endif
