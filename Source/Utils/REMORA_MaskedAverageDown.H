#ifndef REMORA_MASKED_AVERAGE_DOWN_H_
#define REMORA_MASKED_AVERAGE_DOWN_H_

#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

/**
 * Two-way nesting's fine-to-coarse update, weighted by the land/sea mask.
 *
 * Ported from ROMS fine2coarse2d/fine2coarse3d in Nonlinear/nesting.F, the AreaAvg = .FALSE.
 * branch that every call site there uses:
 *
 *     num  = sum over the block of (fine value * fine mask)
 *     den  = sum over the block of min(1, fine mask)
 *     crse = num/den * coarse mask
 *
 * The divisor is the count of WET fine points, not the number of points in the block, so a
 * coarse point that is only partly covered by water takes the mean of the water under it
 * rather than a value diluted toward zero by land. That is not conservative; ROMS makes the
 * same trade, on the grounds that a coastal tracer value pulled toward zero is worse than a
 * small loss of conservation. The coarse mask has the final say on wet or dry.
 *
 * den == 0 under a wet coarse point would leave nothing to divide by. That cannot happen
 * once check_mask_consistency passes, so the kernels write zero and leave the diagnosis to
 * it rather than trying to report from a device kernel.
 *
 * A fully wet block has to reproduce amrex_avgdown and amrex_avgdown_faces bit for bit, or
 * every multilevel answer moves in its last bit. That is why the block is walked in the same
 * order AMReX walks it, why the mask multiplies the value before it is accumulated (m is
 * exactly 1.0 there, and x * 1.0 == x), and why the division is written as a multiplication
 * by the reciprocal (den is then exactly the point count, so 1.0/den is exactly AMReX's
 * volfrac). Writing num/den instead would differ in the last bit.
 */
namespace REMORAMaskedAvgDown {

/** Cell-centered fields: the whole ratio[0] x ratio[1] x ratio[2] block. The mask is 2D and
 *  is read on its z-slab. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void avgdown_masked (int i, int j, int k, int n,
                     amrex::Array4<amrex::Real>       const& crse,
                     amrex::Array4<amrex::Real const> const& fine,
                     amrex::Array4<amrex::Real const> const& fmsk,
                     amrex::Array4<amrex::Real const> const& cmsk,
                     int ccomp, int fcomp, amrex::IntVect const& ratio) noexcept
{
    using amrex::Real;
    const int facx = ratio[0];
    const int facy = ratio[1];
    const int facz = ratio[2];
    const int ii = i*facx;
    const int jj = j*facy;
    const int kk = k*facz;

    Real num = Real(0.0);
    Real den = Real(0.0);
    for (int kref = 0; kref < facz; ++kref) {
    for (int jref = 0; jref < facy; ++jref) {
    for (int iref = 0; iref < facx; ++iref) {
        const Real m = amrex::min(Real(1.0), fmsk(ii+iref,jj+jref,0));
        num += fine(ii+iref,jj+jref,kk+kref,n+fcomp) * m;
        den += m;
    }}}

    crse(i,j,k,n+ccomp) = (den > Real(0.0))
                        ? num * (Real(1.0)/den) * cmsk(i,j,0)
                        : Real(0.0);
}

/** Face-centered fields: only the fine faces lying in the coarse face's own plane, which is
 *  the stencil amrex_avgdown_faces uses. "Wet" then means the flow really does cross here. */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void avgdown_faces_masked (int i, int j, int k, int n,
                           amrex::Array4<amrex::Real>       const& crse,
                           amrex::Array4<amrex::Real const> const& fine,
                           amrex::Array4<amrex::Real const> const& fmsk,
                           amrex::Array4<amrex::Real const> const& cmsk,
                           int ccomp, int fcomp, amrex::IntVect const& ratio, int idir) noexcept
{
    using amrex::Real;
    const int facx = ratio[0];
    const int facy = ratio[1];
    const int facz = ratio[2];
    const int ii = i*facx;
    const int jj = j*facy;
    const int kk = k*facz;

    Real num = Real(0.0);
    Real den = Real(0.0);

    if (idir == 0) {
        for (int kref = 0; kref < facz; ++kref) {
        for (int jref = 0; jref < facy; ++jref) {
            const Real m = amrex::min(Real(1.0), fmsk(ii,jj+jref,0));
            num += fine(ii,jj+jref,kk+kref,n+fcomp) * m;
            den += m;
        }}
    } else if (idir == 1) {
        for (int kref = 0; kref < facz; ++kref) {
        for (int iref = 0; iref < facx; ++iref) {
            const Real m = amrex::min(Real(1.0), fmsk(ii+iref,jj,0));
            num += fine(ii+iref,jj,kk+kref,n+fcomp) * m;
            den += m;
        }}
    } else {
        for (int jref = 0; jref < facy; ++jref) {
        for (int iref = 0; iref < facx; ++iref) {
            const Real m = amrex::min(Real(1.0), fmsk(ii+iref,jj+jref,0));
            num += fine(ii+iref,jj+jref,kk,n+fcomp) * m;
            den += m;
        }}
    }

    crse(i,j,k,n+ccomp) = (den > Real(0.0))
                        ? num * (Real(1.0)/den) * cmsk(i,j,0)
                        : Real(0.0);
}

} // namespace REMORAMaskedAvgDown

#endif
