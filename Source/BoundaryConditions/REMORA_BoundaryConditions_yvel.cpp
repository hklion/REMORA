#include "AMReX_PhysBCFunct.H"
#include <REMORA_PhysBCFunct.H>
#include <REMORA_BoxBatch.H>

using namespace amrex;

/**
 * @param[inout] batch         records the kernels for one launch over all boxes
 * @param[inout] dest_arr      data on which to apply BCs
 * @param[in   ] bx            box to update on
 * @param[in   ] domain        domain box
 * @param[in   ] dxInv         pm or pn
 * @param[in   ] mskv          land-sea mask on v-points
 * @param[in   ] calc_arr_in   data to use in the RHS of calculations
 * @param[in   ] time          current time
 * @param[in   ] bccomp        index into both domain_bcs_type_bcr and bc_extdir_vals for icomp=0
 */
void REMORAPhysBCFunct::impose_yvel_bcs (BoxBatch& batch, const Array4<Real>& dest_arr, const Box& bx, const Box& domain,
                                        const GpuArray<Real,AMREX_SPACEDIM> /*dxInv*/, const Array4<const Real>& mskv,
                                        const Array4<const Real>& calc_arr_in,
                                        Real /*time*/, int bccomp)
{
    BL_PROFILE_VAR("impose_yvel_bcs()",impose_yvel_bcs);
    const auto& dom_lo = amrex::lbound(domain);
    const auto& dom_hi = amrex::ubound(domain);

    // Based on BCRec for the domain, we need to make BCRec for this Box
    // bccomp is used as starting index for m_domain_bcs_type
    //      0 is used as starting index for bcrs
    int ncomp = 1;
    Vector<BCRec> bcrs(ncomp);
    amrex::setBC(bx, domain, bccomp, 0, ncomp, m_domain_bcs_type, bcrs);

    // xlo: ori = 0
    // ylo: ori = 1
    // zlo: ori = 2
    // xhi: ori = 3
    // yhi: ori = 4
    // zhi: ori = 5

    // A single component, so the kernels take its BCRec by value
    const BCRec bcr = bcrs[0];
    const auto* bc_extdir_vals_ptr = m_bc_extdir_vals_d.data();

    GeometryData const& geomdata = m_geom.data();
    bool is_periodic_in_x = geomdata.isPeriodic(0);
    bool is_periodic_in_y = geomdata.isPeriodic(1);
    const Real eps= Real(1.0e-20);
    // If calc_arr isn't passed in, we actually just want to use dest_arr values instead.
    // That matches ROMS.
    Array4<const Real> calc_arr = calc_arr_in.ok() ? calc_arr_in : Array4<const Real>(dest_arr);

    Box dest_arr_box = growHi(convert(Box(dest_arr),IntVect(0,1,0)),1,-1);
    // First do all ext_dir bcs
    if (!is_periodic_in_x or bccomp == BCVars::foextrap_bc(m_ncons))
    {
        // Populate ghost cells on lo-x and hi-x domain boundaries
        Box bx_xlo(bx);  bx_xlo.setBig  (0,dom_lo.x-1);
        Box bx_xhi(bx);  bx_xhi.setSmall(0,dom_hi.x+1);
        batch.ParallelFor(
            grow(bx_xlo,IntVect(0,-1,0)) & dest_arr_box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                int iflip = dom_lo.x - 1- i;
                if (bcr.lo(0) == REMORABCType::ext_dir) {
                    dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][0]*mskv(i,j,0);
                } else if (bcr.lo(0) == REMORABCType::foextrap || bcr.lo(0) == REMORABCType::clamped) {
                    dest_arr(i,j,k) =  dest_arr(dom_lo.x,j,k)*mskv(i,j,0);
                } else if (bcr.lo(0) == REMORABCType::orlanski_rad) {
                    Real grad_lo      = calc_arr(dom_lo.x  ,j+1,k) - calc_arr(dom_lo.x  ,j  ,k);
                    Real grad_lo_jm1  = calc_arr(dom_lo.x  ,j  ,k) - calc_arr(dom_lo.x  ,j-1,k);
                    Real dVdt = calc_arr(dom_lo.x,j,k) - dest_arr(dom_lo.x  ,j,k);
                    Real dVdx = dest_arr(dom_lo.x,j,k) - dest_arr(dom_lo.x+1,j,k);
                    if (dVdt * dVdx < zero) dVdt = zero;
                    Real dVde = (dVdt * (grad_lo_jm1 + grad_lo) > zero) ? grad_lo_jm1 : grad_lo;
                    Real cff = std::max(dVdx*dVdx + dVde*dVde,eps);
                    Real Cx = dVdt * dVdx;
                    dest_arr(i,j,k) = (cff * calc_arr(dom_lo.x-1,j,k) + Cx * dest_arr(dom_lo.x,j,k)) * mskv(i,j,0) / (cff + Cx);
                } else if (bcr.lo(0) == REMORABCType::reflect_even) {
                    dest_arr(i,j,k) =  dest_arr(iflip,j,k)*mskv(i,j,0);
                } else if (bcr.lo(0) == REMORABCType::reflect_odd) {
                    dest_arr(i,j,k) = -dest_arr(iflip,j,k)*mskv(i,j,0);
                }
            },
            grow(bx_xhi,IntVect(0,-1,0)) & dest_arr_box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                int iflip =  2*dom_hi.x + 1 - i;
                if (bcr.hi(0) == REMORABCType::ext_dir) {
                    dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][3]*mskv(i,j,0);
                } else if (bcr.hi(0) == REMORABCType::foextrap || bcr.hi(0) == REMORABCType::clamped) {
                    dest_arr(i,j,k) =  dest_arr(dom_hi.x,j,k)*mskv(i,j,0);
                } else if (bcr.hi(0) == REMORABCType::orlanski_rad) {
                    Real grad_hi        = calc_arr(dom_hi.x  ,j+1,k) - calc_arr(dom_hi.x  ,j  ,k);
                    Real grad_hi_jm1    = calc_arr(dom_hi.x  ,j  ,k) - calc_arr(dom_hi.x  ,j-1,k);
                    Real dVdt = calc_arr(dom_hi.x,j,k) - dest_arr(dom_hi.x  ,j,k);
                    Real dVdx = dest_arr(dom_hi.x,j,k) - dest_arr(dom_hi.x-1,j,k);
                    if (dVdt*dVdx < zero) dVdt = zero;
                    Real dVde = (dVdt * (grad_hi_jm1 + grad_hi) > zero) ? grad_hi_jm1 : grad_hi;
                    Real cff = std::max(dVdx*dVdx+dVde*dVde,eps);
                    Real Cx = dVdt * dVdx;
                    dest_arr(i,j,k) = (cff * calc_arr(dom_hi.x+1,j,k) + Cx * dest_arr(dom_hi.x,j,k)) * mskv(i,j,0) / (cff + Cx);
                } else if (bcr.hi(0) == REMORABCType::reflect_even) {
                    dest_arr(i,j,k) =  dest_arr(iflip,j,k)*mskv(i,j,0);
                } else if (bcr.hi(0) == REMORABCType::reflect_odd) {
                    dest_arr(i,j,k) = -dest_arr(iflip,j,k)*mskv(i,j,0);
                }
            }
        );
    }

    if (!is_periodic_in_y or bccomp == BCVars::foextrap_bc(m_ncons))
    {
        // Populate ghost cells on lo-y and hi-y domain boundaries
        Box bx_ylo(bx);  bx_ylo.setBig  (1,dom_lo.y-1);
        Box bx_yhi(bx);  bx_yhi.setSmall(1,dom_hi.y+2);
        Box bx_ylo_face(bx); bx_ylo_face.setSmall(1,dom_lo.y  ); bx_ylo_face.setBig(1,dom_lo.y  );
        Box bx_yhi_face(bx); bx_yhi_face.setSmall(1,dom_hi.y+1); bx_yhi_face.setBig(1,dom_hi.y+1);

        // We only set the values on the domain faces themselves if EXT_DIR or outflow
        auto f_lo_face = [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            if (bcr.lo(1) == REMORABCType::ext_dir) {
                dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][1]*mskv(i,j,0);
            } else if (bcr.lo(1) == REMORABCType::foextrap) {
                dest_arr(i,j,k) =  dest_arr(i,dom_lo.y+1,k)*mskv(i,j,0);
            } else if (bcr.lo(1) == REMORABCType::orlanski_rad) {
                Real grad_lo_jp1  = calc_arr(i  ,dom_lo.y+1,k) - calc_arr(i-1,dom_lo.y+1,k);
                Real grad_lo_ijp1 = calc_arr(i+1,dom_lo.y+1,k) - calc_arr(i  ,dom_lo.y+1,k);
                Real dVdt = calc_arr(i,dom_lo.y+1,k) - dest_arr(i,dom_lo.y+1,k);
                Real dVde = dest_arr(i,dom_lo.y+1,k) - dest_arr(i,dom_lo.y+2,k);
                if (dVdt*dVde < zero) dVdt = zero;
                Real dVdx = (dVdt * (grad_lo_jp1 + grad_lo_ijp1) > zero) ? grad_lo_jp1 : grad_lo_ijp1;
                Real cff = std::max(dVdx*dVdx + dVde*dVde, eps);
                Real Ce = dVdt * dVde;
                dest_arr(i,j,k) = (cff * calc_arr(i,dom_lo.y,k) + Ce * dest_arr(i,dom_lo.y+1,k)) * mskv(i,j,0) / (cff + Ce);
            }
        };
        auto f_lo = [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            int jflip = dom_lo.y-j;
            int inner = (bcr.lo(1) == REMORABCType::foextrap) ? 1 : 0;
            if (bcr.lo(1) == REMORABCType::ext_dir) {
                dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][1]*mskv(i,j,0);
            } else if (bcr.lo(1) == REMORABCType::foextrap || bcr.lo(1) == REMORABCType::clamped ||
                    bcr.lo(1) == REMORABCType::orlanski_rad) {
                dest_arr(i,j,k) =  dest_arr(i,dom_lo.y+inner,k)*mskv(i,j,0);
            } else if (bcr.lo(1) == REMORABCType::reflect_even) {
                dest_arr(i,j,k) =  dest_arr(i,jflip,k)*mskv(i,j,0);
            } else if (bcr.lo(1) == REMORABCType::reflect_odd) {
                dest_arr(i,j,k) = -dest_arr(i,jflip,k)*mskv(i,j,0);
            }
        };
        // We only set the values on the domain faces themselves if EXT_DIR or outflow
        auto f_hi_face = [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
            if (bcr.hi(1) == REMORABCType::ext_dir) {
                dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][4]*mskv(i,j,0);
            } else if (bcr.hi(1) == REMORABCType::foextrap) {
                dest_arr(i,j,k) =  dest_arr(i,dom_hi.y,k)*mskv(i,j,0);
             } else if (bcr.hi(1) == REMORABCType::orlanski_rad) {
                Real grad_hi      = calc_arr(i  ,dom_hi.y  ,k) - calc_arr(i-1,dom_hi.y  ,k);
                Real grad_hi_ip1  = calc_arr(i+1,dom_hi.y  ,k) - calc_arr(i  ,dom_hi.y  ,k);
                Real dVdt = calc_arr(i,dom_hi.y,k) - dest_arr(i,dom_hi.y  ,k);
                Real dVde = dest_arr(i,dom_hi.y,k) - dest_arr(i,dom_hi.y-1,k);
                if (dVdt*dVde < zero) dVdt = zero;
                Real dVdx = (dVdt * (grad_hi + grad_hi_ip1) > zero) ? grad_hi : grad_hi_ip1;
                Real cff = std::max(dVdx*dVdx + dVde*dVde, eps);
                Real Ce = dVdt * dVde;
                dest_arr(i,j,k) = (cff * calc_arr(i,dom_hi.y+1,k) + Ce * dest_arr(i,dom_hi.y,k)) * mskv(i,j,0) / (cff + Ce);
            }
        };
        auto f_hi = [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
             int jflip =  2*(dom_hi.y + 1) - j;
             int inner = (bcr.hi(1) == REMORABCType::foextrap) ? 1 : 0;
             if (bcr.hi(1) == REMORABCType::ext_dir) {
                 dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][4]*mskv(i,j,0);
             } else if (bcr.hi(1) == REMORABCType::foextrap || bcr.hi(1) == REMORABCType::clamped ||
                     bcr.hi(1) == REMORABCType::orlanski_rad) {
                 dest_arr(i,j,k) =  dest_arr(i,dom_hi.y+1-inner,k)*mskv(i,j,0);
             } else if (bcr.hi(1) == REMORABCType::reflect_even) {
                 dest_arr(i,j,k) =  dest_arr(i,jflip,k)*mskv(i,j,0);
             } else if (bcr.hi(1) == REMORABCType::reflect_odd) {
                 dest_arr(i,j,k) = -dest_arr(i,jflip,k)*mskv(i,j,0);
            }
        };
        // Fill the ylo and yhi faces in one loop and then their ghost cells in one. A box
        // reaching both sides may do so when the domain is wide enough that neither side
        // reads what the other writes (reflection reads cells gy from index 0); otherwise
        // it fills the lo side first, then the hi side, in the last two loops.
        const int gy = std::max(std::max(dom_lo.y - bx.smallEnd(1), bx.bigEnd(1) - (dom_hi.y+1)), 0);
        const bool pair_sides = dom_lo.y == 0 && dom_hi.y + 1 > gy + 3;
        const Box lo_face = grow(bx_ylo_face,IntVect(-1,0,0)) & dest_arr_box;
        const Box lo_ghost = grow(bx_ylo,IntVect(-1,0,0)) & dest_arr_box;
        const Box hi_face = grow(bx_yhi_face,IntVect(-1,0,0)) & dest_arr_box;
        const Box hi_ghost = grow(bx_yhi,IntVect(-1,0,0)) & dest_arr_box;
        batch.ParallelFor(lo_face, ncomp, f_lo_face, pair_sides ? hi_face : Box(), ncomp, f_hi_face);
        batch.ParallelFor(lo_ghost, ncomp, f_lo, pair_sides ? hi_ghost : Box(), ncomp, f_hi);
        batch.ParallelFor(Box(), ncomp, f_lo_face, pair_sides ? Box() : hi_face, ncomp, f_hi_face);
        batch.ParallelFor(Box(), ncomp, f_lo, pair_sides ? Box() : hi_ghost, ncomp, f_hi);
    }

    {
        // Populate ghost cells on lo-z and hi-z domain boundaries
        Box bx_zlo(bx);  bx_zlo.setBig  (2,dom_lo.z-1);
        Box bx_zhi(bx);  bx_zhi.setSmall(2,dom_hi.z+1);
        batch.ParallelFor(
            bx_zlo & dest_arr_box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                int kflip = dom_lo.z - 1 - k;
                if (bcr.lo(2) == REMORABCType::ext_dir) {
                    dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][2]*mskv(i,j,0);
                } else if (bcr.lo(2) == REMORABCType::foextrap) {
                    dest_arr(i,j,k) =  dest_arr(i,j,dom_lo.z)*mskv(i,j,0);
                } else if (bcr.lo(2) == REMORABCType::reflect_even) {
                    dest_arr(i,j,k) =  dest_arr(i,j,kflip)*mskv(i,j,0);
                } else if (bcr.lo(2) == REMORABCType::reflect_odd) {
                    dest_arr(i,j,k) = -dest_arr(i,j,kflip)*mskv(i,j,0);
                }
            },
            bx_zhi & dest_arr_box, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) {
                int kflip =  2*dom_hi.z + 1 - k;
                if (bcr.hi(2) == REMORABCType::ext_dir) {
                    dest_arr(i,j,k) = bc_extdir_vals_ptr[bccomp+n][5]*mskv(i,j,0);
                } else if (bcr.hi(2) == REMORABCType::foextrap) {
                    dest_arr(i,j,k) =  dest_arr(i,j,dom_hi.z)*mskv(i,j,0);
                } else if (bcr.hi(2) == REMORABCType::reflect_even) {
                    dest_arr(i,j,k) =  dest_arr(i,j,kflip)*mskv(i,j,0);
                } else if (bcr.hi(2) == REMORABCType::reflect_odd) {
                    dest_arr(i,j,k) = -dest_arr(i,j,kflip)*mskv(i,j,0);
                }
            }
        );
    }

    if ((!is_periodic_in_x or bccomp == BCVars::foextrap_bc(m_ncons)) and
        (!is_periodic_in_y or bccomp == BCVars::foextrap_bc(m_ncons))) {
        Box xlo(bx);  xlo.setBig  (0,dom_lo.x-1);
        Box xhi(bx);  xhi.setSmall(0,dom_hi.x+1);
        Box ylo(bx);  ylo.setBig  (1,dom_lo.y  );
        Box yhi(bx);  yhi.setSmall(1,dom_hi.y+1);
        Box xlo_ylo = xlo & ylo;
        Box xlo_yhi = xlo & yhi;
        Box xhi_ylo = xhi & ylo;
        Box xhi_yhi = xhi & yhi;
        const bool clamp_west = m_domain_bcs_type[bccomp].lo(0) == REMORABCType::clamped;
        const bool clamp_east = m_domain_bcs_type[bccomp].hi(0) == REMORABCType::clamped;
        const bool clamp_south = m_domain_bcs_type[bccomp].lo(1) == REMORABCType::clamped;
        const bool clamp_north = m_domain_bcs_type[bccomp].hi(1) == REMORABCType::clamped;

        if (!clamp_west && !clamp_south) {
            batch.ParallelFor(xlo_ylo & dest_arr_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dest_arr(i,j,k) = Real(0.5) * (dest_arr(i,dom_lo.y+1,k) + dest_arr(dom_lo.x,j,k));
            });
        }
        if (!clamp_west && !clamp_north) {
            batch.ParallelFor(xlo_yhi & dest_arr_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dest_arr(i,j,k) = Real(0.5) * (dest_arr(i,dom_hi.y,k) + dest_arr(dom_lo.x,j,k));
            });
        }
        if (!clamp_east && !clamp_south) {
            batch.ParallelFor(xhi_ylo & dest_arr_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dest_arr(i,j,k) = Real(0.5) * (dest_arr(i,dom_lo.y+1,k) + dest_arr(dom_hi.x,j,k));
            });
        }
        if (!clamp_east && !clamp_north) {
            batch.ParallelFor(xhi_yhi & dest_arr_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dest_arr(i,j,k) = Real(0.5) * (dest_arr(i,dom_hi.y,k) + dest_arr(dom_hi.x,j,k));
            });
        }
    }
}
