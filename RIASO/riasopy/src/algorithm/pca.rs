use std::borrow::Cow;

use anyhow::{anyhow, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array1, Array2, ArrayView2, Axis};

use arpack::Options;
use arpack_sys::{ssaupd_c, sseupd_c};
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
            ExpMatrix::Sparse(csr) => pca_sparse(&crate::parallel::csr_f32_to_f32(csr), params),
            ExpMatrix::Dense(arr) => {
                let (n, d) = arr.dim();
                let k = params.n_components.min(n).min(d);
                if k > 0 && n.min(d) >= k + 2 && gram_viable(n, d, k) {
                    pca_dense_gram(arr, params)
                } else if auto_picks_randomized(arr, params.n_components) {
                    pca_randomized(arr, params)
                } else {
                    pca_dense(arr, params)
                }
            }
        },
        SvdSolver::Full => match x {
            ExpMatrix::Dense(arr) => pca_dense(arr, params),
            ExpMatrix::Sparse(csr) => {
                pca_dense(&csr_to_dense(&crate::parallel::csr_f32_to_f32(csr)), params)
            }
        },
        SvdSolver::Arpack => match x {
            ExpMatrix::Dense(arr) => pca_dense_arpack(arr, params),
            ExpMatrix::Sparse(csr) => pca_sparse(&crate::parallel::csr_f32_to_f32(csr), params),
        },

        SvdSolver::Randomized => match x {
            ExpMatrix::Dense(arr) => pca_randomized(arr, params),
            ExpMatrix::Sparse(csr) => pca_sparse(&crate::parallel::csr_f32_to_f32(csr), params),
        },
    }
}

fn auto_picks_randomized(arr: &Array2<f32>, n_components: usize) -> bool {
    let (n, d) = arr.dim();
    let k = n_components.min(n).min(d);
    if k == 0 || n.max(d) <= 500 {
        return false;
    }
    (k as f32) < 0.8 * (n.min(d) as f32)
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

fn select_pca_input<'a>(data: &'a SoData, use_hvg: bool) -> Option<Cow<'a, ExpMatrix>> {
    if use_hvg {
        if let Some(selected) = hvg_selected(data) {
            if !selected.is_empty() && selected.len() < data.n_vars() {
                if let Some(exp) = &data.exp_matrix {
                    return Some(match exp {
                        ExpMatrix::Sparse(csr) => {
                            Cow::Owned(ExpMatrix::Sparse(sparse_select_columns(csr, &selected)))
                        }
                        ExpMatrix::Dense(arr) => {
                            Cow::Owned(ExpMatrix::Dense(arr.select(Axis(1), &selected)))
                        }
                    });
                }
            }
        }
    }

    data.exp_matrix.as_ref().map(Cow::Borrowed)
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

pub fn pca_dense(x: &Array2<f32>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = (x.nrows(), x.ncols());

    let k = params.n_components.min(n).min(d);

    let mean = x.mean_axis(Axis(0)).unwrap();

    let centered: Array2<f32> = (x - &mean).as_standard_layout().to_owned();

    let m = nalgebra::DMatrix::from_row_slice(n, d, centered.as_slice().unwrap());

    let svd = nalgebra::SVD::new(m, true, true);

    let v_t = svd.v_t.unwrap();
    let singular_values = svd.singular_values;

    let sv_vec: Vec<f32> = singular_values.iter().copied().collect();
    let n_total = sv_vec.len().min(v_t.nrows());
    let use_k = k.min(n_total);

    let mut components = Array2::<f32>::zeros((use_k, d));
    for i in 0..use_k {
        for j in 0..d {
            components[[i, j]] = v_t[(i, j)];
        }
    }

    let sv_sq: Vec<f32> = sv_vec
        .iter()
        .take(use_k)
        .map(|s| s * s / (n.max(1) - 1) as f32)
        .collect();

    let total_var: f32 = sv_vec
        .iter()
        .map(|s| s * s / (n.max(1) - 1) as f32)
        .sum::<f32>();

    let explained_variance = Array1::from_vec(sv_sq);
    let explained_variance_ratio = Array1::from_vec(
        explained_variance
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect(),
    );

    let full_proj = centered.dot(&components.t());
    let mut projected = Array2::<f32>::zeros((n, use_k));
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

pub fn pca_randomized(x: &Array2<f32>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = (x.nrows(), x.ncols());
    if n == 0 || d == 0 {
        return Err(anyhow!("pca_randomized: expression matrix is empty"));
    }
    let k = params.n_components.min(n).min(d);
    if k == 0 {
        return Err(anyhow!("pca_randomized: n_components is 0"));
    }

    let mean = x.mean_axis(Axis(0)).unwrap();

    let x_std = x.as_standard_layout();
    let xv = x_std.view();
    let xs = xv
        .as_slice()
        .expect("pca_randomized: dense matrix must be C-contiguous");

    let l = (k + RANDOMIZED_OVERSAMPLES).min(d);
    let n_iter = if (k as f32) < 0.1 * (n.min(d) as f32) {
        7
    } else {
        4
    };

    let mut rng = StdNormal::new(params.random_state);
    let mut omega = Array2::<f32>::zeros((d, l));
    for j in 0..l {
        for i in 0..d {
            omega[[i, j]] = rng.next_normal();
        }
    }

    let mut q = Array2::<f32>::zeros((n, l));
    let mut m_omega = Array1::<f32>::zeros(l);
    for j in 0..l {
        for i in 0..d {
            m_omega[j] += mean[i] * omega[[i, j]];
        }
    }
    gemm_nn(
        n,
        d,
        l,
        xs,
        omega.as_slice().unwrap(),
        q.as_slice_mut().unwrap(),
    );
    for r in 0..n {
        for j in 0..l {
            q[[r, j]] -= m_omega[j];
        }
    }
    q = thin_q_array(&q)?;

    let mut z = Array2::<f32>::zeros((d, l));
    for _ in 0..n_iter {
        gemm_tn(
            d,
            n,
            l,
            xs,
            q.as_slice().unwrap(),
            z.as_slice_mut().unwrap(),
        );
        let mut colsum_q = Array1::<f32>::zeros(l);
        for r in 0..n {
            for j in 0..l {
                colsum_q[j] += q[[r, j]];
            }
        }
        for i in 0..d {
            for j in 0..l {
                z[[i, j]] -= mean[i] * colsum_q[j];
            }
        }
        z = thin_q_array(&z)?;

        gemm_nn(
            n,
            d,
            l,
            xs,
            z.as_slice().unwrap(),
            q.as_slice_mut().unwrap(),
        );
        let mut m_z = Array1::<f32>::zeros(l);
        for j in 0..l {
            for i in 0..d {
                m_z[j] += mean[i] * z[[i, j]];
            }
        }
        for r in 0..n {
            for j in 0..l {
                q[[r, j]] -= m_z[j];
            }
        }
        q = thin_q_array(&q)?;
    }

    let mut b = Array2::<f32>::zeros((l, d));
    gemm_tn(
        l,
        n,
        d,
        q.as_slice().unwrap(),
        xs,
        b.as_slice_mut().unwrap(),
    );
    let mut colsum_q = Array1::<f32>::zeros(l);
    for r in 0..n {
        for j in 0..l {
            colsum_q[j] += q[[r, j]];
        }
    }
    for j in 0..l {
        for i in 0..d {
            b[[j, i]] -= colsum_q[j] * mean[i];
        }
    }

    let bm = nalgebra::DMatrix::from_row_slice(l, d, b.as_slice().unwrap());
    let svd = nalgebra::SVD::new(bm, true, true);
    let s = svd.singular_values;
    let v_t = svd
        .v_t
        .ok_or_else(|| anyhow!("pca_randomized: SVD did not return Vt"))?;

    let use_k = k.min(s.len()).min(v_t.nrows());
    let mut components = Array2::<f32>::zeros((use_k, d));
    for i in 0..use_k {
        for j in 0..d {
            components[[i, j]] = v_t[(i, j)];
        }
    }

    let sv_sq: Vec<f32> = (0..use_k)
        .map(|i| s[i] * s[i] / (n.max(1) - 1) as f32)
        .collect();
    let explained_variance = Array1::from_vec(sv_sq);

    let total_var = column_variance_sum(x, &mean, n);
    let explained_variance_ratio = Array1::from_vec(
        explained_variance
            .iter()
            .map(|&v| if total_var > 0.0 { v / total_var } else { 0.0 })
            .collect(),
    );

    let mut full_proj = Array2::<f32>::zeros((n, use_k));
    gemm_nt(
        n,
        d,
        use_k,
        xs,
        components.as_slice().unwrap(),
        full_proj.as_slice_mut().unwrap(),
    );
    let m_comp = components.dot(&mean);
    for r in 0..n {
        for j in 0..use_k {
            full_proj[[r, j]] -= m_comp[j];
        }
    }
    let mut projected = Array2::<f32>::zeros((n, use_k));
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

fn thin_q_array(m: &Array2<f32>) -> Result<Array2<f32>> {
    let (n, l) = m.dim();
    if n == 0 || l == 0 {
        return Err(anyhow!("pca_randomized: QR input is empty"));
    }
    let mm = nalgebra::DMatrix::from_row_slice(n, l, m.as_slice().unwrap());
    let q = nalgebra::QR::new(mm).q();
    let mut out = Array2::<f32>::zeros((q.nrows(), q.ncols()));
    for i in 0..q.nrows() {
        for j in 0..q.ncols() {
            out[[i, j]] = q[(i, j)];
        }
    }
    Ok(out)
}

fn column_variance_sum(x: &Array2<f32>, mean: &Array1<f32>, n: usize) -> f32 {
    let denom = (n.max(1) - 1) as f32;
    if denom <= 0.0 {
        return 0.0;
    }
    let d = x.ncols();
    (0..d)
        .into_par_iter()
        .map(|j| {
            let m = mean[j];
            let mut s = 0.0f32;
            for i in 0..n {
                let dv = x[[i, j]] - m;
                s += dv * dv;
            }
            s / denom
        })
        .sum()
}

pub fn pca_sparse(x: &CsrMatrix<f32>, params: &PcaParams) -> Result<PcaResult> {
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

    let explained_variance = eigenvalues.mapv(|v| v / (n.max(1) - 1) as f32);

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
    eigenvalues: Vec<f32>,
    eigenvectors: Vec<Vec<f32>>,
    nconv: usize,
}

static ARPACK_MUTEX: Mutex<()> = Mutex::new(());

fn multi_eigenpairs_f32<F>(
    n: usize,
    nev: usize,
    matvec: &mut F,
    options: &Options,
) -> Result<MultiEigResult, String>
where
    F: FnMut(&[f32], &mut [f32]),
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

    let mut resid = vec![0.0f32; n];
    let mut v = vec![0.0f32; v_len];
    let ldv = n_i32;
    let mut iparam = [0i32; 11];
    iparam[0] = 1;
    iparam[2] = max_iter_i32;
    iparam[3] = 1;
    iparam[6] = 1;
    let mut ipntr = [0i32; 11];
    let mut workd = vec![0.0f32; workd_len];
    let mut workl = vec![0.0f32; lworkl];

    let bmat = c"I".as_ptr();
    let which = c"LM".as_ptr();

    let _guard = ARPACK_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let mut ido: c_int = 0;
    let mut info: c_int = 0;
    let mut x_buf = vec![0.0f32; n];

    loop {
        unsafe {
            ssaupd_c(
                &mut ido,
                bmat,
                n_i32,
                which,
                nev_i32,
                options.tol as f32,
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
    let mut d = vec![0.0f32; nev];
    let sigma = 0.0f32;
    let mut info_eup: c_int = 0;

    unsafe {
        sseupd_c(
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
            options.tol as f32,
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
    let mut eigenvalues = vec![0.0f32; actual];
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
    csr: &CsrMatrix<f32>,
    mean: &Array1<f32>,
    k: usize,
    n: usize,
    d: usize,
    _seed: u64,
) -> Result<(Array2<f32>, Array2<f32>, Array1<f32>)> {
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
    let mut tbufs: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; d]).collect();

    let mut temp_d = vec![0.0f32; d];
    let mut temp_n = vec![0.0f32; n];
    let mut result_buf = vec![0.0f32; op_dim];

    let mut matvec = |x: &[f32], y: &mut [f32]| {
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

    let solution = multi_eigenpairs_f32(op_dim, k, &mut matvec, &opts)
        .map_err(|e| anyhow!("ARPACK error: {e}"))?;

    let nconv = solution.nconv.min(k);

    let mut eigenvalues = Array1::zeros(nconv);
    for i in 0..nconv {
        eigenvalues[i] = solution.eigenvalues[nconv - 1 - i];
    }

    let (projected, components) = if use_ntimesn {
        let mut proj = Array2::<f32>::zeros((n, nconv));
        let mut comps = Array2::<f32>::zeros((nconv, d));

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

        let mut mean_terms = vec![0.0f32; nconv];

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
        let mut projected = Array2::<f32>::zeros((n, nconv));
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

fn csr_column_means(csr: &CsrMatrix<f32>, n_rows: usize) -> Array1<f32> {
    let n_cols = csr.ncols();
    let n_threads = rayon::current_num_threads();
    let mut tbufs: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; n_cols]).collect();

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

    let mut col_sums = vec![0.0f32; n_cols];
    for tb in &tbufs {
        for (cs, &v) in col_sums.iter_mut().zip(tb.iter()) {
            *cs += v;
        }
    }

    let inv_n = 1.0 / (n_rows as f32);
    let means: Vec<f32> = col_sums.iter().map(|&s| s * inv_n).collect();
    Array1::from_vec(means)
}

fn csr_total_variance(csr: &CsrMatrix<f32>, mean: &Array1<f32>, n_rows: usize) -> f32 {
    let sum_sq_nnz: f32 = csr.values().par_iter().map(|&v| v * v).sum();
    let sum_sq_mean: f32 = mean.as_slice().unwrap().par_iter().map(|&m| m * m).sum();
    let frob_sq = sum_sq_nnz - (n_rows as f32) * sum_sq_mean;

    if frob_sq <= 0.0 {
        return 0.0;
    }
    frob_sq / (n_rows.max(1) - 1) as f32
}

fn csr_transpose_matmul_centered_col(csr: &CsrMatrix<f32>, mean: &[f32], x: &[f32], y: &mut [f32]) {
    let d = csr.ncols();

    let row_sum: f32 = x.iter().sum();

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

fn csr_matmul_centered_col_par(csr: &CsrMatrix<f32>, mean: &[f32], x: &[f32], y: &mut [f32]) {
    let mean_term: f32 = mean.iter().zip(x.iter()).map(|(m, xi)| m * xi).sum();
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
    csr: &CsrMatrix<f32>,
    mean: &[f32],
    x: &[f32],
    y: &mut [f32],
    tbufs: &mut [Vec<f32>],
) {
    let _d = csr.ncols();
    let n = csr.nrows();
    let n_threads = tbufs.len();
    let row_sum: f32 = x.iter().sum();
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

fn gram_viable(n: usize, d: usize, k: usize) -> bool {
    if d < k + 2 {
        return false;
    }
    match std::env::var("RIASO_PCA_GRAM").ok().as_deref() {
        Some("0") => false,
        Some("1") => d <= 16384 && n >= 2 * d,
        _ => d <= 4096 && n >= 4 * d,
    }
}

fn mirror_upper(c: &mut Array2<f32>) {
    let m = c.nrows();
    for i in 0..m {
        for j in 0..i {
            c[[i, j]] = c[[j, i]];
        }
    }
}

fn gram_group_count(n: usize, m: usize, elem: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let per_group = (m * m).saturating_mul(elem).max(1);
    let by_mem = (512 * 1024 * 1024 / per_group).max(1);
    threads.min(by_mem).min(n.max(1)).max(1)
}

fn gram_nblk(m: usize) -> usize {
    if m >= 1024 {
        4
    } else {
        1
    }
}

fn gram_upper_accumulate(a: &ArrayView2<f32>, c: &mut Array2<f32>, nblk: usize) {
    let m = a.ncols();
    if m == 0 {
        return;
    }
    let bs = m.div_ceil(nblk.max(1)).max(1);
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + bs).min(m);
        let mut j0 = i0;
        while j0 < m {
            let j1 = (j0 + bs).min(m);
            let mut blk = Array2::<f32>::zeros((i1 - i0, j1 - j0));
            ndarray::linalg::general_mat_mul(
                1.0,
                &a.slice(ndarray::s![.., i0..i1]).t(),
                &a.slice(ndarray::s![.., j0..j1]),
                0.0,
                &mut blk,
            );
            if i0 == j0 {
                for r in 0..blk.nrows() {
                    for cc in r..blk.ncols() {
                        let idx = (i0 + r, i0 + cc);
                        c[idx] += blk[[r, cc]];
                    }
                }
            } else {
                for r in 0..blk.nrows() {
                    for cc in 0..blk.ncols() {
                        let idx = (i0 + r, j0 + cc);
                        c[idx] += blk[[r, cc]];
                    }
                }
            }
            j0 = j1;
        }
        i0 = i1;
    }
}

fn dense_gram_centered(x: &ArrayView2<f32>, mean: &Array1<f32>, n: usize) -> Array2<f32> {
    let d = x.ncols();
    if n == 0 || d == 0 {
        return Array2::<f32>::zeros((d, d));
    }
    let elem = std::mem::size_of::<f32>().max(1);
    let ngroups = gram_group_count(n, d, elem);
    let per = n.div_ceil(ngroups).max(1);
    let nblk = gram_nblk(d);
    let parts: Vec<Array2<f32>> = (0..ngroups)
        .into_par_iter()
        .map(|g| {
            let r0 = g * per;
            let r1 = (r0 + per).min(n);
            let mut cg = Array2::<f32>::zeros((d, d));
            gram_upper_accumulate(&x.slice(ndarray::s![r0..r1, ..]), &mut cg, nblk);
            cg
        })
        .collect();
    let mut iter = parts.into_iter();
    let mut c = iter.next().expect("dense_gram_centered");
    for cg in iter {
        c.zip_mut_with(&cg, |a, &b| *a += b);
    }
    mirror_upper(&mut c);
    let n_t = n as f32;
    let ms = mean.as_slice().expect("dense_gram_centered");
    for i in 0..d {
        let nmi = n_t * ms[i];
        for j in 0..d {
            c[[i, j]] -= nmi * ms[j];
        }
    }
    c
}

fn pca_dense_arpack(arr: &Array2<f32>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = arr.dim();
    let k = params.n_components.min(n).min(d);
    if k == 0 || n.min(d) < k + 2 {
        return pca_dense(arr, params);
    }
    if gram_viable(n, d, k) {
        return pca_dense_gram(arr, params);
    }
    let x_std = arr.as_standard_layout();
    let x = x_std.view();
    let mean = x.mean_axis(Axis(0)).unwrap();
    let total_var = column_variance_sum(arr, &mean, n);
    if total_var == 0.0 {
        return pca_dense(arr, params);
    }
    let (projected, components, eigenvalues) =
        arpack_pca_dense(x.as_slice().expect("pca_dense_arpack"), &mean, k, n, d)?;
    let explained_variance = eigenvalues.mapv(|v| v / (n.max(1) - 1) as f32);
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

fn pca_dense_gram(arr: &Array2<f32>, params: &PcaParams) -> Result<PcaResult> {
    let (n, d) = arr.dim();
    let k = params.n_components.min(n).min(d);
    if k == 0 || d < k + 2 {
        return pca_dense_arpack(arr, params);
    }
    let x_std = arr.as_standard_layout();
    let x = x_std.view();
    let mean = x.mean_axis(Axis(0)).unwrap();
    let c = dense_gram_centered(&x, &mean, n);
    match gram_finish(&x, mean, c, n, d, k)? {
        Some(r) => Ok(r),
        None => pca_dense(arr, params),
    }
}

fn gram_finish(
    x: &ArrayView2<f32>,
    mean: Array1<f32>,
    c: Array2<f32>,
    n: usize,
    d: usize,
    k: usize,
) -> Result<Option<PcaResult>> {
    let mean_slice = mean.as_slice().expect("gram_finish");
    let mut trace = 0.0;
    for i in 0..d {
        trace += c[[i, i]];
    }
    let denom = (n.max(1) - 1) as f32;
    let total_var = trace / denom;
    if total_var <= 0.0 {
        return Ok(None);
    }
    let tol = std::env::var("RIASO_PCA_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1e-5);
    let ncv = std::env::var("RIASO_PCA_NCV")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (2 * k + 1).max(20))
        .min(d - 1);
    let opts = Options {
        ncv: Some(ncv),
        tol,
        ..Options::default()
    };
    let solution = {
        let cs = c.as_slice().expect("gram_finish");
        let row_chunk = (d / (rayon::current_num_threads().max(1) * 4)).max(64);
        let mut matvec = |v: &[f32], y: &mut [f32]| {
            y.par_chunks_mut(row_chunk)
                .enumerate()
                .for_each(|(ci, yc)| {
                    let base = ci * row_chunk;
                    for (j, yi) in yc.iter_mut().enumerate() {
                        let i = base + j;
                        *yi = dot_4acc(&cs[i * d..(i + 1) * d], v);
                    }
                });
        };
        multi_eigenpairs_f32(d, k, &mut matvec, &opts).map_err(|e| anyhow!("ARPACK error: {e}"))?
    };
    drop(c);
    let nconv = solution.nconv.min(k);
    if nconv == 0 {
        return Err(anyhow!("pca_dense_gram: ARPACK converged no eigenpairs"));
    }
    let mut v_mat = Array2::<f32>::zeros((d, nconv));
    for i in 0..nconv {
        let src = nconv - 1 - i;
        for r in 0..d {
            v_mat[[r, i]] = solution.eigenvectors[src][r];
        }
    }
    let p = dense_x_dot_v(x.as_slice().expect("gram_finish"), n, d, mean_slice, &v_mat);
    let mut g = Array2::<f32>::zeros((nconv, nconv));
    for r in 0..n {
        let prow = p.row(r);
        for i in 0..nconv {
            let pi = prow[i];
            if pi == 0.0 {
                continue;
            }
            for j in i..nconv {
                g[[i, j]] += pi * prow[j];
            }
        }
    }
    for i in 0..nconv {
        for j in 0..i {
            g[[i, j]] = g[[j, i]];
        }
    }
    let gm = nalgebra::DMatrix::from_row_slice(nconv, nconv, g.as_slice().unwrap());
    let eig = nalgebra::linalg::SymmetricEigen::new(gm);
    let mut order: Vec<usize> = (0..nconv).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut components = Array2::<f32>::zeros((nconv, d));
    for (out_i, &o) in order.iter().enumerate() {
        for r in 0..d {
            let mut s = 0.0;
            for m in 0..nconv {
                s += v_mat[[r, m]] * eig.eigenvectors[(m, o)];
            }
            components[[out_i, r]] = s;
        }
    }
    let mut projected = Array2::<f32>::zeros((n, nconv));
    for r in 0..n {
        for (out_i, &o) in order.iter().enumerate() {
            let mut s = 0.0;
            for m in 0..nconv {
                s += p[[r, m]] * eig.eigenvectors[(m, o)];
            }
            projected[[r, out_i]] = s;
        }
    }
    let mut eigenvalues = Array1::<f32>::zeros(nconv);
    for (out_i, &o) in order.iter().enumerate() {
        eigenvalues[out_i] = eig.eigenvalues[o].max(0.0);
    }
    let explained_variance = eigenvalues.mapv(|v| v / denom);
    let explained_variance_ratio = explained_variance.mapv(|v| v / total_var);
    Ok(Some(PcaResult {
        projected,
        components,
        explained_variance,
        explained_variance_ratio,
        mean,
        total_variance: total_var,
    }))
}

fn dense_x_dot_v(xs: &[f32], n: usize, d: usize, mean: &[f32], v: &Array2<f32>) -> Array2<f32> {
    let l = v.ncols();
    let x = ArrayView2::from_shape((n, d), xs).expect("dense_x_dot_v");
    let mut out = Array2::<f32>::zeros((n, l));
    if n == 0 || l == 0 {
        return out;
    }
    let per = n.div_ceil(rayon::current_num_threads().max(1)).max(1);
    let views: Vec<_> = out.axis_chunks_iter_mut(Axis(0), per).collect();
    views.into_par_iter().enumerate().for_each(|(ci, mut o)| {
        let r0 = ci * per;
        let r1 = r0 + o.nrows();
        ndarray::linalg::general_mat_mul(1.0, &x.slice(ndarray::s![r0..r1, ..]), v, 0.0, &mut o);
    });
    let mv = Array1::from_vec(mean.to_vec()).dot(v);
    for r in 0..n {
        let mut row = out.row_mut(r);
        for c in 0..l {
            row[c] -= mv[c];
        }
    }
    out
}

fn dot_4acc(a: &[f32], b: &[f32]) -> f32 {
    let mut ai = a.chunks_exact(4);
    let mut bi = b.chunks_exact(4);
    let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
    for (x, y) in ai.by_ref().zip(bi.by_ref()) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for (x, y) in ai.remainder().iter().zip(bi.remainder()) {
        s += *x * *y;
    }
    s
}

#[link(name = "blas")]
unsafe extern "C" {
    fn sgemv_(
        trans: *const u8,
        m: *const c_int,
        n: *const c_int,
        alpha: *const f32,
        a: *const f32,
        lda: *const c_int,
        x: *const f32,
        incx: *const c_int,
        beta: *const f32,
        y: *mut f32,
        incy: *const c_int,
    );

    fn sgemm_(
        transa: *const u8,
        transb: *const u8,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f32,
        a: *const f32,
        lda: *const c_int,
        b: *const f32,
        ldb: *const c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: *const c_int,
    );
}

fn gemm_nn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    unsafe {
        let (nt, alpha, beta) = (b'N', 1.0f32, 0.0f32);
        let (mm, nn, kk) = (m as c_int, n as c_int, k as c_int);
        let (lda, ldb, ldc) = (nn, kk, nn);
        sgemm_(
            &nt,
            &nt,
            &nn,
            &mm,
            &kk,
            &alpha,
            b.as_ptr(),
            &lda,
            a.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
}

fn gemm_tn(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), k * m);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    unsafe {
        let (nt, tt, alpha, beta) = (b'N', b'T', 1.0f32, 0.0f32);
        let (mm, nn, kk) = (m as c_int, n as c_int, k as c_int);
        let (lda, ldb, ldc) = (nn, mm, nn);
        sgemm_(
            &nt,
            &tt,
            &nn,
            &mm,
            &kk,
            &alpha,
            b.as_ptr(),
            &lda,
            a.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
}

fn gemm_nt(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);
    if m == 0 || k == 0 || n == 0 {
        return;
    }
    unsafe {
        let (nt, tt, alpha, beta) = (b'N', b'T', 1.0f32, 0.0f32);
        let (mm, nn, kk) = (m as c_int, n as c_int, k as c_int);
        let (lda, ldb, ldc) = (kk, kk, nn);
        sgemm_(
            &tt,
            &nt,
            &nn,
            &mm,
            &kk,
            &alpha,
            b.as_ptr(),
            &lda,
            a.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        );
    }
}

fn dense_forward_centered(xs: &[f32], d: usize, mean: &[f32], v: &[f32], y: &mut [f32]) {
    let mean_term = mean
        .iter()
        .zip(v.iter())
        .fold(0.0, |a, (&m, &xi)| a + m * xi);
    unsafe {
        let m = d as c_int;
        let n = y.len() as c_int;
        let lda = d as c_int;
        let inc = 1;
        let (one, zero, t) = (1.0f32, 0.0f32, b'T');
        sgemv_(
            &t,
            &m,
            &n,
            &one,
            xs.as_ptr(),
            &lda,
            v.as_ptr(),
            &inc,
            &zero,
            y.as_mut_ptr(),
            &inc,
        );
    }
    for yi in y.iter_mut() {
        *yi -= mean_term;
    }
}

fn dense_transpose_centered(
    xs: &[f32],
    n: usize,
    d: usize,
    mean: &[f32],
    v: &[f32],
    y: &mut [f32],
) {
    let row_sum = v.iter().fold(0.0, |a, &b| a + b);
    unsafe {
        let m = d as c_int;
        let nn = n as c_int;
        let lda = d as c_int;
        let inc = 1;
        let (one, zero, t) = (1.0f32, 0.0f32, b'N');
        sgemv_(
            &t,
            &m,
            &nn,
            &one,
            xs.as_ptr(),
            &lda,
            v.as_ptr(),
            &inc,
            &zero,
            y.as_mut_ptr(),
            &inc,
        );
    }
    for c in 0..d {
        y[c] -= mean[c] * row_sum;
    }
}

fn dense_xt_dot_v(xs: &[f32], n: usize, d: usize, mean: &[f32], u: &Array2<f32>) -> Array2<f32> {
    let l = u.ncols();
    let x = ArrayView2::from_shape((n, d), xs).expect("dense_xt_dot_v");
    let mut out = x.t().dot(u);
    let mut ones = Array1::<f32>::zeros(n);
    ones.fill(1.0);
    let ones_u = ones.dot(u);
    for c in 0..d {
        let mc = mean[c];
        for i in 0..l {
            out[[c, i]] -= mc * ones_u[i];
        }
    }
    out
}

fn arpack_pca_dense(
    xs: &[f32],
    mean: &Array1<f32>,
    k: usize,
    n: usize,
    d: usize,
) -> Result<(Array2<f32>, Array2<f32>, Array1<f32>)> {
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
    let mut temp_d = vec![0.0; d];
    let mut temp_n = vec![0.0; n];
    let mut matvec = |v: &[f32], y: &mut [f32]| {
        if use_ntimesn {
            dense_transpose_centered(xs, n, d, mean_slice, v, &mut temp_d);
            dense_forward_centered(xs, d, mean_slice, &temp_d, y);
        } else {
            dense_forward_centered(xs, d, mean_slice, v, &mut temp_n);
            dense_transpose_centered(xs, n, d, mean_slice, &temp_n, &mut temp_d);
            y.copy_from_slice(&temp_d);
        }
    };
    let tol = std::env::var("RIASO_PCA_TOL")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1e-5);
    let ncv = std::env::var("RIASO_PCA_NCV")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| (2 * k + 1).max(20))
        .min(op_dim - 1);
    let opts = Options {
        ncv: Some(ncv),
        tol,
        ..Options::default()
    };
    let solution = multi_eigenpairs_f32(op_dim, k, &mut matvec, &opts)
        .map_err(|e| anyhow!("ARPACK error: {e}"))?;
    let nconv = solution.nconv.min(k);
    let mut eigenvalues = Array1::zeros(nconv);
    for i in 0..nconv {
        eigenvalues[i] = solution.eigenvalues[nconv - 1 - i];
    }
    let mut evec = Array2::<f32>::zeros((op_dim, nconv));
    for i in 0..nconv {
        let src = nconv - 1 - i;
        for r in 0..op_dim {
            evec[[r, i]] = solution.eigenvectors[src][r];
        }
    }
    let mut projected = Array2::<f32>::zeros((n, nconv));
    let mut components = Array2::<f32>::zeros((nconv, d));
    if use_ntimesn {
        for i in 0..nconv {
            let sigma = solution.eigenvalues[nconv - 1 - i].sqrt();
            for r in 0..n {
                projected[[r, i]] = evec[[r, i]] * sigma;
            }
        }
        let xt_u = dense_xt_dot_v(xs, n, d, mean_slice, &evec);
        for i in 0..nconv {
            let sigma = solution.eigenvalues[nconv - 1 - i].sqrt();
            let inv_sigma = 1.0 / sigma;
            for c in 0..d {
                components[[i, c]] = xt_u[[c, i]] * inv_sigma;
            }
        }
    } else {
        for i in 0..nconv {
            for c in 0..d {
                components[[i, c]] = evec[[c, i]];
            }
        }
        let x_v = dense_x_dot_v(xs, n, d, mean_slice, &evec);
        for r in 0..n {
            for i in 0..nconv {
                projected[[r, i]] = x_v[[r, i]];
            }
        }
    }
    Ok((projected, components, eigenvalues))
}
