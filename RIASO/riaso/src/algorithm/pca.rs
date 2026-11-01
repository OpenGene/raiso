use anyhow::{anyhow, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array1, Array2, Axis};

use arpack::Options;
use arpack_sys::{dsaupd_c, dseupd_c};
use rayon::prelude::*;
use std::os::raw::c_int;
use std::sync::Mutex;

use crate::core::{ExpMatrix, PcaResult, SoData};
use crate::parallel::{csr_to_dense, SendPtr};

use crate::algorithm::rng::StdNormal;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdSolver {
    Auto,

    Full,

    Arpack,

    Randomized,
}

impl Default for SvdSolver {
    fn default() -> Self {
        SvdSolver::Auto
    }
}

#[derive(Debug, Clone)]
pub struct PcaParams {
    pub n_components: usize,

    pub random_state: u64,

    pub svd_solver: SvdSolver,
}

impl Default for PcaParams {
    fn default() -> Self {
        Self {
            n_components: 30,
            random_state: 0,
            svd_solver: SvdSolver::Auto,
        }
    }
}

pub fn pca(x: &ExpMatrix, params: &PcaParams) -> Result<PcaResult> {
    match params.svd_solver {
        SvdSolver::Auto => match x {
            ExpMatrix::Sparse(csr) => pca_sparse(&crate::parallel::csr_f32_to_f64(csr), params),
            ExpMatrix::Dense(arr) => {
                let arr = arr.mapv(|v| v as f64);
                if auto_picks_randomized(&arr, params.n_components) {
                    pca_randomized(&arr, params)
                } else {
                    pca_dense(&arr, params)
                }
            }
        },
        SvdSolver::Full => match x {
            ExpMatrix::Dense(arr) => pca_dense(&arr.mapv(|v| v as f64), params),
            ExpMatrix::Sparse(csr) => {
                pca_dense(&csr_to_dense(&crate::parallel::csr_f32_to_f64(csr)), params)
            }
        },
        SvdSolver::Arpack => match x {
            ExpMatrix::Dense(arr) => pca_sparse(&dense_to_csr(&arr.mapv(|v| v as f64)), params),
            ExpMatrix::Sparse(csr) => pca_sparse(&crate::parallel::csr_f32_to_f64(csr), params),
        },

        SvdSolver::Randomized => match x {
            ExpMatrix::Dense(arr) => pca_randomized(&arr.mapv(|v| v as f64), params),
            ExpMatrix::Sparse(csr) => pca_sparse(&crate::parallel::csr_f32_to_f64(csr), params),
        },
    }
}

fn auto_picks_randomized(arr: &Array2<f64>, n_components: usize) -> bool {
    let (n, d) = arr.dim();
    let k = n_components.min(n).min(d);
    if k == 0 || n.max(d) <= 500 {
        return false;
    }
    (k as f64) < 0.8 * (n.min(d) as f64)
}

pub fn run_pca(data: &mut SoData, params: &PcaParams, use_hvg: bool) {
    data.result.pca = None;

    let x = match select_pca_input(data, use_hvg) {
        Some(x) => x,
        None => return,
    };

    if let Ok(result) = pca(&x, params) {
        data.result.pca = Some(result);
    }
}

fn hvg_selected(data: &SoData) -> Option<Vec<usize>> {
    let hv = data.result.hvg.as_ref()?;
    if hv.highly_variable.len() != data.n_vars() {
        return None;
    }
    Some(
        (0..hv.highly_variable.len())
            .filter(|&j| hv.highly_variable[j])
            .collect(),
    )
}

fn select_pca_input(data: &SoData, use_hvg: bool) -> Option<ExpMatrix> {
    if use_hvg {
        if let Some(selected) = hvg_selected(data) {
            if !selected.is_empty() && selected.len() < data.n_vars() {
                if let Some(exp) = &data.exp_matrix {
                    return Some(match exp {
                        ExpMatrix::Sparse(csr) => {
                            ExpMatrix::Sparse(sparse_select_columns(csr, &selected))
                        }
                        ExpMatrix::Dense(arr) => ExpMatrix::Dense(arr.select(Axis(1), &selected)),
                    });
                }
            }
        }
    }

    data.exp_matrix.clone()
}

fn dense_to_csr(arr: &Array2<f64>) -> CsrMatrix<f64> {
    let (nrows, ncols) = arr.dim();

    let mut row_nnz = vec![0usize; nrows];
    let nnz_ptr = SendPtr(row_nnz.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut cnt = 0usize;
        for col in 0..ncols {
            if arr[[row, col]] != 0.0 {
                cnt += 1;
            }
        }
        unsafe {
            *nnz_ptr.add(row) = cnt;
        }
    });

    let mut offsets = vec![0usize; nrows + 1];
    let mut acc = 0usize;
    for (r, &c) in row_nnz.iter().enumerate() {
        acc += c;
        offsets[r + 1] = acc;
    }
    let mut cols = vec![0usize; acc];
    let mut vals = vec![0f64; acc];

    let cols_ptr = SendPtr(cols.as_mut_ptr());
    let vals_ptr = SendPtr(vals.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut k = offsets[row];
        for col in 0..ncols {
            let v = arr[[row, col]];
            if v != 0.0 {
                unsafe {
                    *cols_ptr.add(k) = col;
                    *vals_ptr.add(k) = v;
                }
                k += 1;
            }
        }
    });
    CsrMatrix::try_from_csr_data(nrows, ncols, offsets, cols, vals)
        .expect("dense_to_csr: valid CSR")
}

pub(crate) fn sparse_select_columns<T>(csr: &CsrMatrix<T>, selected: &[usize]) -> CsrMatrix<T>
where
    T: nalgebra::Scalar + Copy + Default + Send + Sync,
{
    let nrows = csr.nrows();
    let offsets = csr.row_offsets();
    let cols = csr.col_indices();
    let vals = csr.values();

    let mut pos = vec![usize::MAX; csr.ncols()];
    for (out_col, &orig) in selected.iter().enumerate() {
        if orig < pos.len() {
            pos[orig] = out_col;
        }
    }
    let keep = |c: usize| pos[c] != usize::MAX;

    let mut row_nnz = vec![0usize; nrows];
    let nnz_ptr = SendPtr(row_nnz.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut cnt = 0usize;
        for &c in &cols[offsets[row]..offsets[row + 1]] {
            if keep(c) {
                cnt += 1;
            }
        }
        unsafe {
            *nnz_ptr.add(row) = cnt;
        }
    });

    let mut out_offsets = vec![0usize; nrows + 1];
    let mut acc = 0usize;
    for (r, &c) in row_nnz.iter().enumerate() {
        acc += c;
        out_offsets[r + 1] = acc;
    }
    let mut out_cols = vec![0usize; acc];
    let mut out_vals = vec![T::default(); acc];

    let cols_ptr = SendPtr(out_cols.as_mut_ptr());
    let vals_ptr = SendPtr(out_vals.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut k = out_offsets[row];
        for idx in offsets[row]..offsets[row + 1] {
            let c = cols[idx];
            if keep(c) {
                unsafe {
                    *cols_ptr.add(k) = pos[c];
                    *vals_ptr.add(k) = vals[idx];
                }
                k += 1;
            }
        }
    });
    CsrMatrix::try_from_csr_data(nrows, selected.len(), out_offsets, out_cols, out_vals)
        .expect("sparse_select_columns: valid CSR")
}

pub fn pca_dense(x: &Array2<f64>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = (x.nrows(), x.ncols());

    let k = params.n_components.min(n).min(d);

    let mean = x.mean_axis(Axis(0)).unwrap();

    let centered: Array2<f64> = (x - &mean).as_standard_layout().to_owned();

    let m = nalgebra::DMatrix::from_row_slice(n, d, centered.as_slice().unwrap());

    let svd = nalgebra::SVD::new(m, true, true);

    let v_t = svd.v_t.unwrap();
    let singular_values = svd.singular_values;

    let sv_vec: Vec<f64> = singular_values.iter().copied().collect();
    let n_total = sv_vec.len().min(v_t.nrows());
    let use_k = k.min(n_total);

    let mut components = Array2::<f64>::zeros((use_k, d));
    for i in 0..use_k {
        for j in 0..d {
            components[[i, j]] = v_t[(i, j)];
        }
    }

    let sv_sq: Vec<f64> = sv_vec
        .iter()
        .take(use_k)
        .map(|s| s * s / (n.max(1) - 1) as f64)
        .collect();

    let total_var: f64 = sv_vec
        .iter()
        .map(|s| s * s / (n.max(1) - 1) as f64)
        .sum::<f64>();

    let explained_variance = Array1::from_vec(sv_sq);
    let explained_variance_ratio = Array1::from_vec(
        explained_variance
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect(),
    );

    let full_proj = centered.dot(&components.t());
    let mut projected = Array2::<f64>::zeros((n, use_k));
    for i in 0..n {
        for j in 0..use_k {
            projected[[i, j]] = full_proj[[i, j]];
        }
    }

    let result = PcaResult {
        projected,
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
        total_variance: total_var,
    };

    Ok(result)
}

const RANDOMIZED_OVERSAMPLES: usize = 10;

pub fn pca_randomized(x: &Array2<f64>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = (x.nrows(), x.ncols());
    if n == 0 || d == 0 {
        return Err(anyhow!("pca_randomized: expression matrix is empty"));
    }
    let k = params.n_components.min(n).min(d);
    if k == 0 {
        return Err(anyhow!("pca_randomized: n_components is 0"));
    }

    let mean = x.mean_axis(Axis(0)).unwrap();
    let centered: Array2<f64> = (x - &mean).as_standard_layout().to_owned();

    let l = (k + RANDOMIZED_OVERSAMPLES).min(d);
    let n_iter = if (k as f64) < 0.1 * (n.min(d) as f64) {
        7
    } else {
        4
    };

    let xm = nalgebra::DMatrix::from_row_slice(n, d, centered.as_slice().unwrap());

    let mut rng = StdNormal::new(params.random_state);
    let mut omega = nalgebra::DMatrix::<f64>::zeros(d, l);
    for j in 0..l {
        for i in 0..d {
            omega[(i, j)] = rng.next_normal();
        }
    }

    let mut q = thin_q(&(xm.clone() * omega))?;
    for _ in 0..n_iter {
        let z = thin_q(&(xm.transpose() * &q))?;
        q = thin_q(&(&xm * z))?;
    }

    let b = q.transpose() * &xm;
    let svd = nalgebra::SVD::new(b, true, true);
    let s = svd.singular_values;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("pca_randomized: SVD did not return Vt"))?;

    let use_k = k.min(s.len()).min(v_t.nrows());
    let mut components = Array2::<f64>::zeros((use_k, d));
    for i in 0..use_k {
        for j in 0..d {
            components[[i, j]] = v_t[(i, j)];
        }
    }

    let sv_sq: Vec<f64> = (0..use_k)
        .map(|i| s[i] * s[i] / (n.max(1) - 1) as f64)
        .collect();
    let explained_variance = Array1::from_vec(sv_sq);

    let total_var = column_variance_sum(x, &mean, n);
    let explained_variance_ratio = Array1::from_vec(
        explained_variance
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect(),
    );

    let full_proj = centered.dot(&components.t());
    let mut projected = Array2::<f64>::zeros((n, use_k));
    for i in 0..n {
        for j in 0..use_k {
            projected[[i, j]] = full_proj[[i, j]];
        }
    }

    Ok(PcaResult {
        projected,
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
        total_variance: total_var,
    })
}

fn thin_q(m: &nalgebra::DMatrix<f64>) -> Result<nalgebra::DMatrix<f64>> {
    if m.nrows() == 0 || m.ncols() == 0 {
        return Err(anyhow!("pca_randomized: QR input is empty"));
    }
    Ok(nalgebra::QR::new(m.clone()).q())
}

fn column_variance_sum(x: &Array2<f64>, mean: &Array1<f64>, n: usize) -> f64 {
    let denom = (n.max(1) - 1) as f64;
    if denom <= 0.0 {
        return 0.0;
    }
    let d = x.ncols();
    (0..d)
        .into_par_iter()
        .map(|j| {
            let m = mean[j];
            let mut s = 0.0f64;
            for i in 0..n {
                let dv = x[[i, j]] - m;
                s += dv * dv;
            }
            s / denom
        })
        .sum()
}

pub fn pca_sparse(x: &CsrMatrix<f64>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = (x.nrows(), x.ncols());
    let k = params.n_components.min(n).min(d);

    if k > 0 && n.min(d) < k + 2 {
        return pca_dense(&csr_to_dense(x), params);
    }

    let mean = csr_column_means(x, n);

    let total_var = csr_total_variance(x, &mean, n);
    if total_var == 0.0 {
        return pca_dense(&csr_to_dense(x), params);
    }

    let (projected, components, eigenvalues) =
        arpack_pca_csr(x, &mean, k, n, d, params.random_state)?;

    let explained_variance = eigenvalues.mapv(|v| v / (n.max(1) - 1) as f64);

    let explained_variance_ratio = if total_var > 0.0 {
        explained_variance.mapv(|v| v / total_var)
    } else {
        Array1::zeros(eigenvalues.len())
    };

    Ok(PcaResult {
        projected,
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
        total_variance: total_var,
    })
}

struct MultiEigResult {
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Vec<f64>>,
    nconv: usize,
}

static ARPACK_MUTEX: Mutex<()> = Mutex::new(());

fn multi_eigenpairs_f64<F>(
    n: usize,
    nev: usize,
    matvec: &mut F,
    options: &Options,
) -> Result<MultiEigResult, String>
where
    F: FnMut(&[f64], &mut [f64]),
{
    if n < nev + 2 {
        return Err(format!(
            "n={n} too small for nev={nev} (require n >= nev + 2)"
        ));
    }

    let ncv = options
        .ncv
        .unwrap_or_else(|| (2 * nev + 4).min(n - 1).max(nev + 1));

    let n_i32 = c_int::try_from(n).map_err(|_| "n does not fit in c_int")?;
    let nev_i32 = c_int::try_from(nev).map_err(|_| "nev does not fit in c_int")?;
    let ncv_i32 = c_int::try_from(ncv).map_err(|_| "ncv does not fit in c_int")?;
    let max_iter_i32 =
        c_int::try_from(options.max_iter).map_err(|_| "max_iter does not fit in c_int")?;

    if !(nev_i32 > 0 && nev_i32 < ncv_i32 && ncv_i32 < n_i32) {
        return Err(format!("require 0 < nev={nev} < ncv={ncv} < n={n}"));
    }
    if max_iter_i32 <= 0 {
        return Err("max_iter must be positive".to_string());
    }

    let v_len = n.checked_mul(ncv).ok_or("n * ncv overflows usize")?;
    let workd_len = n.checked_mul(3).ok_or("3 * n overflows usize")?;
    let lworkl = ncv
        .checked_add(8)
        .and_then(|s| ncv.checked_mul(s))
        .ok_or("ncv * (ncv + 8) overflows usize")?;
    let lworkl_i32 = c_int::try_from(lworkl).map_err(|_| "lworkl does not fit in c_int")?;

    let mut resid = vec![0.0f64; n];
    let mut v = vec![0.0f64; v_len];
    let ldv = n_i32;
    let mut iparam = [0i32; 11];
    iparam[0] = 1;
    iparam[2] = max_iter_i32;
    iparam[3] = 1;
    iparam[6] = 1;
    let mut ipntr = [0i32; 11];
    let mut workd = vec![0.0f64; workd_len];
    let mut workl = vec![0.0f64; lworkl];

    let bmat = c"I".as_ptr();
    let which = c"LM".as_ptr();

    let _guard = ARPACK_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut x_buf = vec![0.0f64; n];

    loop {
        unsafe {
            dsaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which,
                nev_i32,
                options.tol,
                resid.as_mut_ptr(),
                ncv_i32,
                v.as_mut_ptr(),
                ldv,
                iparam.as_mut_ptr(),
                ipntr.as_mut_ptr(),
                workd.as_mut_ptr(),
                workl.as_mut_ptr(),
                lworkl_i32,
                &mut info,
            );
        }

        match ido {
            -1 | 1 => {
                let x_off = (ipntr[0] - 1) as usize;
                let y_off = (ipntr[1] - 1) as usize;
                debug_assert!(x_off + n <= workd.len() && y_off + n <= workd.len());
                x_buf.copy_from_slice(&workd[x_off..x_off + n]);
                matvec(&x_buf, &mut workd[y_off..y_off + n]);
            }
            99 => break,
            other => return Err(format!("ARPACK unexpected ido={other}")),
        }
    }

    let nconv = iparam[4] as usize;

    if nconv == 0 {
        let reason = if info == 1 {
            format!("max_iter={} reached", options.max_iter)
        } else if info != 0 {
            format!("dsaupd info={info}")
        } else {
            "unknown reason".to_string()
        };
        return Err(format!("ARPACK: 0/{nev} eigenvalues converged ({reason})"));
    }

    let rvec: c_int = 1;
    let howmny = c"A".as_ptr();
    let mut select = vec![0i32; ncv];
    let mut d = vec![0.0f64; nev];
    let sigma = 0.0f64;
    let mut info_eup: c_int = 0;

    unsafe {
        dseupd_c(
            rvec,
            howmny,
            select.as_mut_ptr(),
            d.as_mut_ptr(),
            v.as_mut_ptr(),
            ldv,
            sigma,
            bmat,
            n_i32,
            which,
            nev_i32,
            options.tol,
            resid.as_mut_ptr(),
            ncv_i32,
            v.as_mut_ptr(),
            ldv,
            iparam.as_mut_ptr(),
            ipntr.as_mut_ptr(),
            workd.as_mut_ptr(),
            workl.as_mut_ptr(),
            lworkl_i32,
            &mut info_eup,
        );
    }

    if info_eup != 0 {
        return Err(format!("ARPACK dseupd failed: info={info_eup}"));
    }

    let actual = nconv.min(nev);
    let mut eigenvalues = vec![0.0f64; actual];
    eigenvalues.copy_from_slice(&d[..actual]);

    let mut eigenvectors = Vec::with_capacity(actual);
    for i in 0..actual {
        eigenvectors.push(v[i * n..(i + 1) * n].to_vec());
    }

    Ok(MultiEigResult {
        eigenvalues,
        eigenvectors,
        nconv: actual,
    })
}

fn arpack_pca_csr(
    csr: &CsrMatrix<f64>,
    mean: &Array1<f64>,
    k: usize,
    n: usize,
    d: usize,
    _seed: u64,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    if k == 0 {
        return Ok((
            Array2::zeros((n, 0)),
            Array2::zeros((0, d)),
            Array1::zeros(0),
        ));
    }

    let mean_slice = mean.as_slice().unwrap();

    let use_ntimesn = n < d && n >= k + 2;
    let op_dim = if use_ntimesn { n } else { d };

    let n_threads = rayon::current_num_threads();
    let mut tbufs: Vec<Vec<f64>> = (0..n_threads).map(|_| vec![0.0f64; d]).collect();

    let mut temp_d = vec![0.0f64; d];
    let mut temp_n = vec![0.0f64; n];
    let mut result_buf = vec![0.0f64; op_dim];

    let mut matvec = |x: &[f64], y: &mut [f64]| {
        if use_ntimesn {
            csr_transpose_matmul_centered_col_par(csr, mean_slice, x, &mut temp_d, &mut tbufs);

            csr_matmul_centered_col_par(csr, mean_slice, &temp_d, y);
        } else {
            csr_matmul_centered_col_par(csr, mean_slice, x, &mut temp_n);

            csr_transpose_matmul_centered_col_par(
                csr,
                mean_slice,
                &temp_n,
                &mut result_buf,
                &mut tbufs,
            );
            y.copy_from_slice(&result_buf);
        }
    };

    let ncv = (k * 3).max(k + 2).min(op_dim - 1);
    let opts = Options {
        ncv: Some(ncv),
        ..Options::default()
    };

    let solution = multi_eigenpairs_f64(op_dim, k, &mut matvec, &opts)
        .map_err(|e| anyhow!("ARPACK error: {e}"))?;

    let nconv = solution.nconv.min(k);

    let mut eigenvalues = Array1::zeros(nconv);
    for i in 0..nconv {
        eigenvalues[i] = solution.eigenvalues[nconv - 1 - i];
    }

    let (projected, components) = if use_ntimesn {
        let mut proj = Array2::<f64>::zeros((n, nconv));
        let mut comps = Array2::<f64>::zeros((nconv, d));

        for i in 0..nconv {
            let src = nconv - 1 - i;
            let lambda = solution.eigenvalues[src];
            let sigma = lambda.sqrt();
            let ui = &solution.eigenvectors[src];

            for row in 0..n {
                proj[[row, i]] = ui[row] * sigma;
            }

            csr_transpose_matmul_centered_col(csr, mean_slice, ui, &mut temp_d);
            let inv_sigma = 1.0 / sigma;
            for col in 0..d {
                comps[[i, col]] = temp_d[col] * inv_sigma;
            }
        }
        (proj, comps)
    } else {
        let mut components = Array2::zeros((nconv, d));
        for i in 0..nconv {
            let src = nconv - 1 - i;
            for j in 0..d {
                components[[i, j]] = solution.eigenvectors[src][j];
            }
        }

        let mut mean_terms = vec![0.0f64; nconv];

        #[allow(clippy::needless_range_loop)]
        for j in 0..nconv {
            let src = nconv - 1 - j;
            let ev = &solution.eigenvectors[src];
            let mut s = 0.0;
            for col in 0..d {
                s += mean_slice[col] * ev[col];
            }
            mean_terms[j] = s;
        }

        let offsets = csr.row_offsets();
        let col_idx = csr.col_indices();
        let values = csr.values();
        let mut projected = Array2::<f64>::zeros((n, nconv));
        for row in 0..n {
            for j in 0..nconv {
                projected[[row, j]] = -mean_terms[j];
            }
            for idx in offsets[row]..offsets[row + 1] {
                let col = col_idx[idx];
                let val = values[idx];
                for j in 0..nconv {
                    let src = nconv - 1 - j;
                    projected[[row, j]] += val * solution.eigenvectors[src][col];
                }
            }
        }
        (projected, components)
    };

    Ok((projected, components, eigenvalues))
}

fn csr_column_means(csr: &CsrMatrix<f64>, n_rows: usize) -> Array1<f64> {
    let n_cols = csr.ncols();
    let n_threads = rayon::current_num_threads();
    let mut tbufs: Vec<Vec<f64>> = (0..n_threads).map(|_| vec![0.0f64; n_cols]).collect();

    let offsets = csr.row_offsets();
    let col_idx = csr.col_indices();
    let values = csr.values();

    let n = csr.nrows();
    let rows_per_thread = n.div_ceil(n_threads);
    tbufs.par_iter_mut().enumerate().for_each(|(tid, tbuf)| {
        let r0 = tid * rows_per_thread;
        let r1 = (r0 + rows_per_thread).min(n);
        for row in r0..r1 {
            for i in offsets[row]..offsets[row + 1] {
                tbuf[col_idx[i]] += values[i];
            }
        }
    });

    let mut col_sums = vec![0.0f64; n_cols];
    for tb in &tbufs {
        for (cs, &v) in col_sums.iter_mut().zip(tb.iter()) {
            *cs += v;
        }
    }

    let inv_n = 1.0 / (n_rows as f64);
    let means: Vec<f64> = col_sums.iter().map(|&s| s * inv_n).collect();
    Array1::from_vec(means)
}

fn csr_total_variance(csr: &CsrMatrix<f64>, mean: &Array1<f64>, n_rows: usize) -> f64 {
    let sum_sq_nnz: f64 = csr.values().par_iter().map(|&v| v * v).sum();
    let sum_sq_mean: f64 = mean.as_slice().unwrap().par_iter().map(|&m| m * m).sum();
    let frob_sq = sum_sq_nnz - (n_rows as f64) * sum_sq_mean;

    if frob_sq <= 0.0 {
        return 0.0;
    }
    frob_sq / (n_rows.max(1) - 1) as f64
}

fn csr_transpose_matmul_centered_col(csr: &CsrMatrix<f64>, mean: &[f64], x: &[f64], y: &mut [f64]) {
    let d = csr.ncols();

    let row_sum: f64 = x.iter().sum();

    for j in 0..d {
        y[j] = -mean[j] * row_sum;
    }

    let offsets = csr.row_offsets();
    let col_idx = csr.col_indices();
    let values = csr.values();

    for row in 0..csr.nrows() {
        let x_row = x[row];
        for idx in offsets[row]..offsets[row + 1] {
            y[col_idx[idx]] += values[idx] * x_row;
        }
    }
}

fn csr_matmul_centered_col_par(csr: &CsrMatrix<f64>, mean: &[f64], x: &[f64], y: &mut [f64]) {
    let mean_term: f64 = mean.iter().zip(x.iter()).map(|(m, xi)| m * xi).sum();
    let offsets = csr.row_offsets();
    let col_idx = csr.col_indices();
    let values = csr.values();

    y.par_iter_mut().enumerate().for_each(|(row, y_row)| {
        *y_row = -mean_term;
        let start = offsets[row];
        let end = offsets[row + 1];
        for idx in start..end {
            *y_row += values[idx] * x[col_idx[idx]];
        }
    });
}

fn csr_transpose_matmul_centered_col_par(
    csr: &CsrMatrix<f64>,
    mean: &[f64],
    x: &[f64],
    y: &mut [f64],
    tbufs: &mut [Vec<f64>],
) {
    let _d = csr.ncols();
    let n = csr.nrows();
    let n_threads = tbufs.len();
    let row_sum: f64 = x.iter().sum();
    let offsets = csr.row_offsets();
    let col_idx = csr.col_indices();
    let values = csr.values();

    let rows_per_thread = n.div_ceil(n_threads);
    tbufs.par_iter_mut().enumerate().for_each(|(tid, tbuf)| {
        let r0 = tid * rows_per_thread;
        let r1 = (r0 + rows_per_thread).min(n);
        for row in r0..r1 {
            let xr = x[row];
            for idx in offsets[row]..offsets[row + 1] {
                tbuf[col_idx[idx]] += values[idx] * xr;
            }
        }
    });

    y.par_iter_mut().enumerate().for_each(|(j, y_j)| {
        let mut s = -mean[j] * row_sum;

        #[allow(clippy::needless_range_loop)]
        for t in 0..n_threads {
            s += tbufs[t][j];
        }
        *y_j = s;
    });

    tbufs.par_iter_mut().for_each(|tb| tb.fill(0.0));
}
