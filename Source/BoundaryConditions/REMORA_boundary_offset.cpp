#include "REMORA.H"

#include "AMReX_Geometry.H"

using namespace amrex;

void REMORA::boundary_offset (MultiFab* mf, int lev, int icomp, Real offset_EW, Real offset_NS) {
    auto domain = Geom(lev).Domain();
    const auto& dom_lo = lbound(domain);
    const auto& dom_hi = ubound(domain);

    for (MFIter mfi(*mf,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        Box bx = mfi.growntilebox(mf->nGrowVect());
        const auto& bx_lo = lbound(bx);
        const auto& bx_hi = ubound(bx);

        const Array4<Real> arr = mf->array(mfi, icomp);

        if (bx_lo.x <= dom_lo.x) {
            Box bx_crop_xlo = bx; bx_crop_xlo.setBig(0,dom_lo.x-1);
            ParallelFor(bx_crop_xlo, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                arr(i,j,k) -= offset_EW;
            });
        }
        if (bx_hi.x >= dom_hi.x) {
            Box bx_crop_xhi = bx; bx_crop_xhi.setSmall(0,dom_hi.x+1);
            ParallelFor(bx_crop_xhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                arr(i,j,k) += offset_EW;
            });
        }
        if (bx_lo.y <= dom_lo.y) {
            Box bx_crop_ylo = bx; bx_crop_ylo.setBig(1,dom_lo.y-1);
            ParallelFor(bx_crop_ylo, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                arr(i,j,k) -= offset_NS;
            });
        }
        if (bx_hi.y >= dom_hi.y) {
            Box bx_crop_yhi = bx; bx_crop_yhi.setSmall(1,dom_hi.y+1);
            ParallelFor(bx_crop_yhi, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                arr(i,j,k) += offset_NS;
            });
        }
    }
}

