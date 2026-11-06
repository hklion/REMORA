#ifndef REMORA_BOXBATCH_H_
#define REMORA_BOXBATCH_H_

#include <AMReX_Arena.H>
#include <AMReX_Box.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

/**
 * \brief Runs the box loops of an MFIter loop as one kernel per loop for all boxes.
 *
 * In an MFIter loop on the GPU every ParallelFor is one kernel launch per box. Called
 * through a BoxBatch instead, the calls are recorded: the n-th call a box makes goes to
 * site n, and launch() runs each site as a single kernel over the boxes of all its calls,
 * site after site, on the current stream. Each cell is computed by the same function as in
 * the per-box launch, and every box finishes a site before any box starts the next, so the
 * result is that of the per-box loops whenever each box reads and writes only its own
 * FABs, as in an MFIter loop whose boxes run on different streams.
 *
 * Rules for the recorded loop:
 * - call next_box() at the top of every iteration;
 * - every box makes the same sequence of calls (same functions, boxes may be empty), or
 *   none at all; a box that makes a different number of calls aborts;
 * - work the loop body launches itself must be on the default stream, so that it runs
 *   before the recorded calls, and everything the functions capture stays valid until
 *   launch(): pass temporaries to keep();
 * - the functions are copied byte for byte to the device, so they must be trivially
 *   copyable, as any lambda capturing Array4s and scalars is;
 * - iterate with mfiter_info(), which keeps the loop on the default stream without
 *   synchronizing, so the recorded calls and anything the loop body stages on the device
 *   are in stream order with the launch.
 *
 * Without a GPU launch region (CPU builds, or a GPU build running on the host) every call
 * runs at once as amrex::ParallelFor, so the loop keeps its tiling and threading.
 */
class BoxBatch
{
public:
    explicit BoxBatch (bool allow = true)
        : m_defer(allow && amrex::Gpu::inLaunchRegion()) {}

#ifdef AMREX_USE_GPU
    ~BoxBatch () {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_sites.empty(), "BoxBatch: recorded calls never launched");
    }
#else
    ~BoxBatch () = default;
#endif

    BoxBatch (BoxBatch const&) = delete;
    BoxBatch& operator= (BoxBatch const&) = delete;
    BoxBatch (BoxBatch&&) = delete;
    BoxBatch& operator= (BoxBatch&&) = delete;

    /** True when calls are recorded for launch() rather than run at once */
    [[nodiscard]] bool deferred () const noexcept { return m_defer; }

    /** MFIter options for the recorded loop: tiling on the host if asked, the default
     *  stream and no synchronization on the GPU when recording */
    [[nodiscard]] amrex::MFItInfo mfiter_info (bool tiling = true) const
    {
        amrex::MFItInfo info;
        if (tiling && amrex::TilingIfNotGPU()) { info.EnableTiling(); }
        if (m_defer) { info.UseDefaultStream().DisableDeviceSync(); }
        return info;
    }

    /** Start recording the calls of the next box at the first site. Without recording
     *  this does nothing, so threads of a host loop may share the BoxBatch. */
    void next_box ()
    {
        if (m_defer) {
#ifdef AMREX_USE_GPU
            end_box();
#endif
            m_cur = 0;
        }
    }

    /** Keep obj alive until launch() has queued the kernels that use it */
    template <class T>
    void keep (T&& obj)
    {
        if (m_defer) {
            m_keep.push_back(std::make_shared<std::decay_t<T>>(std::forward<T>(obj)));
        }
    }

    template <class F>
    void ParallelFor (amrex::Box const& bx, F const& f)
    {
#ifdef AMREX_USE_GPU
        if (m_defer) { record(Entry<false,F,false,NoFunc>{bx, amrex::Box(), 1, 1, f, NoFunc{}}); return; }
#endif
        amrex::ParallelFor(bx, f);
    }

    template <class F>
    void ParallelFor (amrex::Box const& bx, int ncomp, F const& f)
    {
#ifdef AMREX_USE_GPU
        if (m_defer) { record(Entry<true,F,false,NoFunc>{bx, amrex::Box(), ncomp, 1, f, NoFunc{}}); return; }
#endif
        amrex::ParallelFor(bx, ncomp, f);
    }

    template <class F1, class F2>
    void ParallelFor (amrex::Box const& bx1, amrex::Box const& bx2, F1 const& f1, F2 const& f2)
    {
#ifdef AMREX_USE_GPU
        if (m_defer) { record(Entry<false,F1,false,F2>{bx1, bx2, 1, 1, f1, f2}); return; }
#endif
        amrex::ParallelFor(bx1, bx2, f1, f2);
    }

    template <class F1, class F2>
    void ParallelFor (amrex::Box const& bx1, int ncomp1, F1 const& f1,
                      amrex::Box const& bx2, int ncomp2, F2 const& f2)
    {
#ifdef AMREX_USE_GPU
        if (m_defer) {
            record(Entry<true,F1,true,F2>{bx1, bx2, ncomp1, ncomp2, f1, f2});
            return;
        }
#endif
        amrex::ParallelFor(bx1, ncomp1, f1, bx2, ncomp2, f2);
    }

    /** Launch every site recorded since the last launch, in order, then release what
     *  keep() holds in stream order */
    void launch ()
    {
        if (!m_defer) { return; }
#ifdef AMREX_USE_GPU
        end_box();
        // A site with one box to work on is launched as the per-box loop would be. The
        // entries of the sites with more go to the device in one copy.
        std::size_t nbytes = 0;
        for (auto& s : m_sites) {
            if (s.nlive > 1) {
                s.offset = nbytes;
                nbytes = amrex::Arena::align(nbytes + s.bytes.size());
            }
        }
        char* hp = nullptr;
        char* dp = nullptr;
        if (nbytes > 0) {
            hp = static_cast<char*>(amrex::The_Pinned_Arena()->alloc(nbytes));
            dp = static_cast<char*>(amrex::The_Arena()->alloc(nbytes));
            for (auto const& s : m_sites) {
                if (s.nlive > 1) { std::memcpy(hp + s.offset, s.bytes.data(), s.bytes.size()); }
            }
            amrex::Gpu::htod_memcpy_async(dp, hp, nbytes);
        }
        for (auto const& s : m_sites) {
            if (s.nlive == 1) {
                s.direct(s.bytes.data() + s.live*s.entry_size);
            } else if (s.nlive > 1) {
                s.launcher(dp + s.offset, s.nentries, s.maxcells);
            }
        }
        if (nbytes > 0) {
            amrex::Gpu::freeAsync(amrex::The_Pinned_Arena(), hp);
            amrex::Gpu::freeAsync(amrex::The_Arena(), dp);
        }
        m_sites.clear();
#endif
        m_keep.clear();
        m_cur = 0;
        m_nboxes = 0;
    }

#ifdef AMREX_USE_GPU
    struct NoFunc {
        AMREX_GPU_HOST_DEVICE void operator() (int, int, int) const noexcept {}
    };

    /** Cells of a box rounded up to whole warps, so that no warp spans two boxes */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static int warp_cells (amrex::Box const& b) noexcept {
        constexpr int w = amrex::Gpu::Device::warp_size;
        if (!b.ok()) { return 0; }
        const auto len = amrex::length(b);
        return ((len.x*len.y*len.z + w - 1) / w) * w;
    }

    /** One recorded call: up to two boxes, each with its function. Its cells are those
     *  of the first box, then those of the second, each rounded up to whole warps. */
    template <bool C1, class F1, bool C2, class F2>
    struct Entry {
        amrex::Box b1, b2;
        int n1, n2;
        F1 f1;
        F2 f2;

        template <bool C, class F>
        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        static void call (amrex::Box const& b, int nc, F const& f, int icell) noexcept
        {
            const auto len = amrex::length(b);
            if (!b.ok() || icell >= len.x*len.y*len.z) { return; }
            const auto lo  = amrex::lbound(b);
            const int k =  icell /   (len.x*len.y);
            const int j = (icell - k*(len.x*len.y)) /   len.x;
            const int i = (icell - k*(len.x*len.y)) - j*len.x;
            if constexpr (C) {
                for (int n = 0; n < nc; ++n) { f(i+lo.x, j+lo.y, k+lo.z, n); }
            } else {
                amrex::ignore_unused(nc);
                f(i+lo.x, j+lo.y, k+lo.z);
            }
        }

        /** Launch this call alone, as amrex::ParallelFor */
        void launch () const
        {
            if constexpr (std::is_same_v<F2, NoFunc>) {
                if constexpr (C1) { amrex::ParallelFor(b1, n1, f1); }
                else              { amrex::ParallelFor(b1, f1); }
            } else {
                static_assert(C1 == C2);
                if constexpr (C1) { amrex::ParallelFor(b1, n1, f1, b2, n2, f2); }
                else              { amrex::ParallelFor(b1, b2, f1, f2); }
            }
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        void run (int icell) const noexcept
        {
            const int nw1 = warp_cells(b1);
            if (icell < nw1) {
                call<C1>(b1, n1, f1, icell);
            } else {
                call<C2>(b2, n2, f2, icell - nw1);
            }
        }
    };

    /** One kernel over the entries of a site, each given the cells of the largest */
    template <class E>
    static void launch_site (char const* d_entries, int nentries, int maxcells)
    {
        auto const* entries = reinterpret_cast<E const*>(d_entries);
        amrex::ParallelFor(nentries*maxcells, [=] AMREX_GPU_DEVICE (int idx) noexcept
        {
            const int e = idx / maxcells;
            entries[e].run(idx - e*maxcells);
        });
    }

    template <class E>
    static void launch_one (char const* h_entry)
    {
        reinterpret_cast<E const*>(h_entry)->launch();
    }

private:
    using Launcher = void (*) (char const*, int, int);
    using Direct = void (*) (char const*);

    struct Site {
        Launcher launcher = nullptr;
        Direct direct = nullptr;
        std::size_t entry_size = 0;
        amrex::Vector<char> bytes;
        int nentries = 0;
        int nlive = 0;     // entries with cells to work on
        int live = -1;     // the last of them
        int maxcells = 0;
        std::size_t offset = 0;
    };

    /** Cells of a box rounded up to whole warps, counted without overflow */
    static amrex::Long warp_cells_long (amrex::Box const& b) noexcept {
        constexpr amrex::Long w = amrex::Gpu::Device::warp_size;
        if (!b.ok()) { return 0; }
        return ((b.numPts() + w - 1) / w) * w;
    }

    /** A box that made calls must have made one at every site */
    void end_box () const
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_cur == 0 || m_cur == int(m_sites.size()),
            "BoxBatch: the boxes of a recorded loop made different numbers of calls");
    }

    template <class E>
    void record (E const& e)
    {
        static_assert(std::is_trivially_copyable<E>::value,
                      "BoxBatch: the recorded functions must be trivially copyable");
        if (m_cur == 0) { ++m_nboxes; }
        if (m_cur == int(m_sites.size())) {
            m_sites.emplace_back();
            m_sites.back().launcher = &launch_site<E>;
            m_sites.back().direct = &launch_one<E>;
            m_sites.back().entry_size = sizeof(E);
        }
        Site& s = m_sites[m_cur++];
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(s.launcher == &launch_site<E> && s.nentries == m_nboxes-1,
            "BoxBatch: the boxes of a recorded loop made different calls");
        const amrex::Long ncells = warp_cells_long(e.b1) + warp_cells_long(e.b2);
        const amrex::Long maxcells = std::max(amrex::Long(s.maxcells), ncells);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(amrex::Long(s.nentries+1)*maxcells
                                         < amrex::Long(std::numeric_limits<int>::max()),
            "BoxBatch: too many cells for one launch");
        const auto n = s.bytes.size();
        s.bytes.resize(n + sizeof(E));
        std::memcpy(s.bytes.data() + n, &e, sizeof(E));
        s.maxcells = int(maxcells);
        if (ncells > 0) { ++s.nlive; s.live = s.nentries; }
        ++s.nentries;
    }

    amrex::Vector<Site> m_sites;
#else
private:
#endif

    bool m_defer;
    int m_cur = 0;      // calls the current box has made
    int m_nboxes = 0;   // boxes that have made calls since the last launch
    amrex::Vector<std::shared_ptr<void>> m_keep;
};

#endif
