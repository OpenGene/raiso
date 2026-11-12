use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

pub(crate) struct SendPtr<T>(pub(crate) *mut T);

impl<T> SendPtr<T> {
    #[inline]
    pub(crate) unsafe fn add(&self, off: usize) -> *mut T {
        self.0.add(off)
    }
}

unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

#[link(name = "blas")]
unsafe extern "C" {
    fn openblas_set_num_threads(n: std::os::raw::c_int);
}

static THREADS: std::sync::Mutex<usize> = std::sync::Mutex::new(0);

static GLOBAL: AtomicUsize = AtomicUsize::new(0);

const STEP_CAPS: &[(&str, usize)] = &[("leiden", 4), ("marker", 8)];

fn cap_for(step: &str) -> Option<usize> {
    STEP_CAPS
        .iter()
        .find_map(|&(s, c)| (s == step).then_some(c))
}

pub(crate) fn global_threads() -> usize {
    GLOBAL.load(Ordering::Relaxed).max(1)
}

#[allow(dead_code)]
pub(crate) fn set_rayon_threads(n: usize) {
    *THREADS.lock().unwrap() = n;
    if n > 0 {
        GLOBAL.store(n, Ordering::Relaxed);
        unsafe { openblas_set_num_threads(n as std::os::raw::c_int) };
    }
}

pub(crate) fn init_rayon(threads: usize) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let n = *THREADS.lock().unwrap();
        let n = if n > 0 {
            n
        } else if threads > 0 {
            threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(32)
        };
        GLOBAL.store(n, Ordering::Relaxed);
        unsafe { openblas_set_num_threads(n as std::os::raw::c_int) };
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    });
}

static POOLS: OnceLock<std::sync::Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();

fn pool_for(n: usize) -> Arc<ThreadPool> {
    let pools = POOLS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut map = pools.lock().unwrap();
    if let Some(p) = map.get(&n) {
        return p.clone();
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("per-step thread pool"),
    );
    map.insert(n, pool.clone());
    pool
}

pub(crate) fn with_thread_cap<R: Send>(step: &'static str, f: impl FnOnce() -> R + Send) -> R {
    let g = global_threads();
    let eff = match cap_for(step) {
        Some(cap) => g.min(cap),
        None => g,
    };
    if eff >= g {
        return f();
    }
    unsafe { openblas_set_num_threads(eff as std::os::raw::c_int) };
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            unsafe { openblas_set_num_threads(self.0 as std::os::raw::c_int) };
        }
    }
    let _restore = Restore(g);
    pool_for(eff).install(f)
}

pub(crate) fn csr_to_dense<T>(csr: &nalgebra_sparse::csr::CsrMatrix<T>) -> ndarray::Array2<T>
where
    T: num_traits::Float + nalgebra::Scalar + Send + Sync,
{
    use rayon::prelude::*;
    let (nrows, ncols) = (csr.nrows(), csr.ncols());
    let mut dense = ndarray::Array2::<T>::zeros((nrows, ncols));
    let offsets = csr.row_offsets();
    let cols = csr.col_indices();
    let vals = csr.values();
    let ptr = SendPtr(dense.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let row_ptr = unsafe { ptr.add(row * ncols) };
        for idx in offsets[row]..offsets[row + 1] {
            unsafe { *row_ptr.add(cols[idx]) = vals[idx] };
        }
    });
    dense
}

pub(crate) fn csr_f32_to_f32(
    csr: &nalgebra_sparse::csr::CsrMatrix<f32>,
) -> nalgebra_sparse::csr::CsrMatrix<f32> {
    let (nrows, ncols) = (csr.nrows(), csr.ncols());
    let offsets = csr.row_offsets().to_vec();
    let cols = csr.col_indices().to_vec();
    let vals: Vec<f32> = csr.values().to_vec();
    nalgebra_sparse::csr::CsrMatrix::try_from_csr_data(nrows, ncols, offsets, cols, vals)
        .expect("csr_f32_to_f32: valid CSR")
}

pub(crate) fn par_min_max(vals: &[f32]) -> (f32, f32) {
    use rayon::prelude::*;
    vals.par_iter()
        .fold(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(lo, hi), &v| (if v < lo { v } else { lo }, if v > hi { v } else { hi }),
        )
        .reduce(
            || (f32::INFINITY, f32::NEG_INFINITY),
            |(l0, h0), (l1, h1)| (l0.min(l1), h0.max(h1)),
        )
}

pub(crate) fn par_bbox_opt(xs: &[f32], ys: &[f32], skip_non_finite: bool) -> (f32, f32, f32, f32) {
    use rayon::prelude::*;
    const ID: (f32, f32, f32, f32) = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    xs.par_iter()
        .zip(ys.par_iter())
        .fold(
            || ID,
            |(xl, xh, yl, yh), (&x, &y)| {
                if skip_non_finite && !(x.is_finite() && y.is_finite()) {
                    return (xl, xh, yl, yh);
                }
                (
                    if x < xl { x } else { xl },
                    if x > xh { x } else { xh },
                    if y < yl { y } else { yl },
                    if y > yh { y } else { yh },
                )
            },
        )
        .reduce(
            || ID,
            |(a, b, c, d), (e, f, g, h)| (a.min(e), b.max(f), c.min(g), d.max(h)),
        )
}

pub(crate) fn par_bbox(xs: &[f32], ys: &[f32]) -> (f32, f32, f32, f32) {
    par_bbox_opt(xs, ys, false)
}

pub(crate) fn par_histogram_counts(
    values: &[f32],
    lo: f32,
    bin_w: f32,
    n_bins: usize,
) -> Vec<usize> {
    use rayon::prelude::*;
    values
        .par_iter()
        .fold(
            || vec![0usize; n_bins],
            |mut acc, &v| {
                let mut idx = ((v - lo) / bin_w).floor() as isize;
                if idx < 0 {
                    idx = 0;
                }
                if idx >= n_bins as isize {
                    idx = n_bins as isize - 1;
                }
                acc[idx as usize] += 1;
                acc
            },
        )
        .reduce(
            || vec![0usize; n_bins],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}
