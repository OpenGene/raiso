pub(crate) struct SendPtr<T>(pub(crate) *mut T);

impl<T> SendPtr<T> {
    #[inline]
    pub(crate) unsafe fn add(&self, off: usize) -> *mut T {
        self.0.add(off)
    }
}

unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

static THREADS: std::sync::Mutex<usize> = std::sync::Mutex::new(0);

#[allow(dead_code)]
pub(crate) fn set_rayon_threads(n: usize) {
    *THREADS.lock().unwrap() = n;
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
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    });
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

pub(crate) fn csr_f32_to_f64(
    csr: &nalgebra_sparse::csr::CsrMatrix<f32>,
) -> nalgebra_sparse::csr::CsrMatrix<f64> {
    let (nrows, ncols) = (csr.nrows(), csr.ncols());
    let offsets = csr.row_offsets().to_vec();
    let cols = csr.col_indices().to_vec();
    let vals: Vec<f64> = csr.values().iter().map(|&v| v as f64).collect();
    nalgebra_sparse::csr::CsrMatrix::try_from_csr_data(nrows, ncols, offsets, cols, vals)
        .expect("csr_f32_to_f64: valid CSR")
}

pub(crate) fn par_min_max(vals: &[f64]) -> (f64, f64) {
    use rayon::prelude::*;
    vals.par_iter()
        .fold(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(lo, hi), &v| (if v < lo { v } else { lo }, if v > hi { v } else { hi }),
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |(l0, h0), (l1, h1)| (l0.min(l1), h0.max(h1)),
        )
}

pub(crate) fn par_bbox_opt(xs: &[f64], ys: &[f64], skip_non_finite: bool) -> (f64, f64, f64, f64) {
    use rayon::prelude::*;
    const ID: (f64, f64, f64, f64) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
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

pub(crate) fn par_bbox(xs: &[f64], ys: &[f64]) -> (f64, f64, f64, f64) {
    par_bbox_opt(xs, ys, false)
}

pub(crate) fn par_histogram_counts(
    values: &[f64],
    lo: f64,
    bin_w: f64,
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
