use anyhow::{anyhow, Result};
use ndarray::s;
use ndarray::Array2;
use rayon::prelude::*;

use crate::algorithm::rng::Mt19937;
use crate::core::{BatchCorrectionResult, SoData};
use crate::parallel::SendPtr;

#[derive(Debug, Clone)]
pub struct HarmonyParams {
    pub nclust: Option<usize>,

    pub sigma: f32,

    pub theta: Option<f32>,

    pub lamb: Option<f32>,

    pub tau: f32,

    pub block_size: f32,

    pub max_iter_harmony: usize,

    pub max_iter_kmeans: usize,

    pub epsilon_cluster: f32,

    pub epsilon_harmony: f32,

    pub random_state: u64,
}

impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            nclust: None,
            sigma: 0.1,
            theta: Some(1.0),
            lamb: Some(1.0),
            tau: 0.0,
            block_size: 0.05,
            max_iter_harmony: 10,
            max_iter_kmeans: 20,
            epsilon_cluster: 1e-5,
            epsilon_harmony: 1e-4,
            random_state: 0,
        }
    }
}

pub fn run_harmony(data: &mut SoData, params: &HarmonyParams) {
    data.result.batch = None;

    let z = match data.result.pca.as_ref() {
        Some(pca) => &pca.projected,
        None => return,
    };

    let obs = match data.obs.as_ref() {
        Some(obs) => obs,
        None => return,
    };
    let batch_labels = match extract_batch_labels(obs) {
        Ok(b) => b,
        Err(_) => return,
    };

    if let Ok(result) = harmonize_matrix(z, &batch_labels, params) {
        data.result.batch = Some(result);
    }
}

pub fn harmonize_matrix(
    z: &Array2<f32>,
    batch_labels: &[usize],
    params: &HarmonyParams,
) -> Result<BatchCorrectionResult> {
    let z_in = Array2::from_shape_vec((z.ncols(), z.nrows()), z.t().iter().cloned().collect())
        .expect("transposed array dimensions are valid");
    let mut h = Harmony::new(&z_in, batch_labels, params)?;
    let (converged, n_iter) = h.harmonize();

    Ok(BatchCorrectionResult {
        corrected: h.z_corr.t().to_owned(),
        r: h.r.t().to_owned(),
        n_iter,
        converged,
    })
}

fn extract_batch_labels(obs: &polars::prelude::DataFrame) -> Result<Vec<usize>> {
    let batch_col = obs
        .column("batch")
        .map_err(|_| anyhow!("obs missing 'batch' column"))?;
    let batch_i64 = batch_col
        .i64()
        .map_err(|_| anyhow!("obs 'batch' column is not i64"))?;
    let labels: Vec<usize> = (0..batch_i64.len())
        .map(|i| batch_i64.get(i).unwrap_or(0) as usize)
        .collect();
    Ok(labels)
}

fn numpy_round(x: f32) -> f32 {
    let floor = x.floor();
    let frac = x - floor;
    if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

struct Harmony {
    z_corr: Array2<f32>,
    z_orig: Array2<f32>,
    z_cos: Array2<f32>,
    phi: Array2<f32>,
    phi_moe: Array2<f32>,
    pr_b: Vec<f32>,
    b: usize,
    d: usize,
    n: usize,
    k: usize,
    sigma: Vec<f32>,
    theta: Vec<f32>,
    lamb: Array2<f32>,
    block_size: f32,
    max_iter_harmony: usize,
    max_iter_kmeans: usize,
    epsilon_kmeans: f32,
    epsilon_harmony: f32,
    window_size: usize,

    scale_dist: Array2<f32>,
    dist_mat: Array2<f32>,
    o: Array2<f32>,
    e: Array2<f32>,
    w: Array2<f32>,
    phi_rk: Array2<f32>,
    y: Array2<f32>,
    r: Array2<f32>,
    batch_of_cell: Vec<usize>,
    objective_harmony: Vec<f32>,
    objective_kmeans: Vec<f32>,
    rng: Mt19937,
}

impl Harmony {
    fn new(z: &Array2<f32>, batch_labels: &[usize], params: &HarmonyParams) -> Result<Self> {
        let d = z.nrows();
        let n = z.ncols();
        let n_batches = batch_labels.iter().max().map(|&x| x + 1).unwrap_or(1);
        let k = match params.nclust {
            Some(k) => k,
            None => (numpy_round(n as f32 / 30.0) as usize).min(100),
        };
        let k = k.max(1);

        let mut phi = Array2::<f32>::zeros((n_batches, n));
        for (i, &b) in batch_labels.iter().enumerate() {
            if b < n_batches {
                phi[[b, i]] = 1.0;
            }
        }

        let n_b: Vec<f32> = (0..n_batches).map(|b| phi.row(b).sum()).collect();
        let pr_b: Vec<f32> = n_b.iter().map(|&nb| nb / n as f32).collect();

        let sigma: Vec<f32> = vec![params.sigma; k];

        let base_theta = params.theta.unwrap_or(1.0);
        let mut theta: Vec<f32> = vec![base_theta; n_batches];
        if params.tau > 0.0 {
            for b in 0..n_batches {
                let factor = 1.0 - (-((n_b[b] / (k as f32 * params.tau)).powi(2))).exp();
                theta[b] *= factor;
            }
        }

        let base_lamb = params.lamb.unwrap_or(1.0);
        let mut lamb = Array2::<f32>::zeros((n_batches + 1, n_batches + 1));
        for i in 1..=n_batches {
            lamb[[i, i]] = base_lamb;
        }

        let mut phi_moe = Array2::<f32>::zeros((n_batches + 1, n));
        phi_moe.row_mut(0).fill(1.0);
        for b in 0..n_batches {
            for j in 0..n {
                phi_moe[[b + 1, j]] = phi[[b, j]];
            }
        }

        let mut h = Harmony {
            z_corr: z.clone(),
            z_orig: z.clone(),
            z_cos: Array2::zeros((d, n)),
            phi,
            phi_moe,
            pr_b,
            b: n_batches,
            d,
            n,
            k,
            sigma,
            theta,
            lamb,
            block_size: params.block_size,
            max_iter_harmony: params.max_iter_harmony,
            max_iter_kmeans: params.max_iter_kmeans,
            epsilon_kmeans: params.epsilon_cluster,
            epsilon_harmony: params.epsilon_harmony,
            window_size: 3,
            scale_dist: Array2::zeros((k, n)),
            dist_mat: Array2::zeros((k, n)),
            o: Array2::zeros((k, n_batches)),
            e: Array2::zeros((k, n_batches)),
            w: Array2::zeros((n_batches + 1, d)),
            phi_rk: Array2::zeros((n_batches + 1, n)),
            y: Array2::zeros((d, k)),
            r: Array2::zeros((k, n)),
            batch_of_cell: batch_labels.to_vec(),
            objective_harmony: Vec::new(),
            objective_kmeans: Vec::new(),
            rng: Mt19937::new(params.random_state),
        };

        h.z_cos = h.z_orig.clone();
        let col_max = col_wise_max(&h.z_orig);

        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            for i in 0..d {
                h.z_cos[[i, j]] /= col_max[j];
            }
        }
        h.z_cos = l2_norm_cols(&h.z_cos);

        h.init_cluster();
        Ok(h)
    }

    fn init_cluster(&mut self) {
        let z_cos_t = Array2::from_shape_vec(
            (self.z_cos.ncols(), self.z_cos.nrows()),
            self.z_cos.t().iter().cloned().collect(),
        )
        .expect("transposed array dimensions are valid");
        let km_centroids = kmeans2(&z_cos_t, self.k, &mut self.rng);
        self.y = Array2::from_shape_vec(
            (km_centroids.ncols(), km_centroids.nrows()),
            km_centroids.t().iter().cloned().collect(),
        )
        .expect("transposed array dimensions are valid");

        self.y = l2_norm_cols(&self.y);

        self.dist_mat = cluster_dist(&self.y, &self.z_cos);
        self.r = -&self.dist_mat;
        self.r = divide_rows(&self.r, &self.sigma);
        self.r = sub_col_max(&self.r);
        self.r = self.r.mapv(f32::exp);
        self.r = normalize_cols_l1(&self.r);

        let r_row_sum = row_sums(&self.r);
        self.e = outer(&r_row_sum, &self.pr_b);
        self.o = gemm_abt(&self.r, &self.phi);
        self.compute_objective();
        let last = *self.objective_kmeans.last().unwrap();
        self.objective_harmony.push(last);
    }

    fn compute_objective(&mut self) {
        let k = self.k;
        let n = self.n;
        let b = self.b;
        let r = self.r.as_slice().unwrap();
        let dist = self.dist_mat.as_slice().unwrap();
        let sigma = &self.sigma;
        let batch = &self.batch_of_cell;
        let theta = &self.theta;
        let o = self.o.as_slice().unwrap();
        let e = self.e.as_slice().unwrap();

        let mut z = vec![0.0f32; k * b];
        for kk in 0..k {
            for bb in 0..b {
                z[kk * b + bb] = ((o[kk * b + bb] + 1.0) / (e[kk * b + bb] + 1.0)).ln();
            }
        }

        let (kmeans_error, entropy, cross_entropy): (f32, f32, f32) = (0..k)
            .into_par_iter()
            .map(|kk| {
                let s = sigma[kk];
                let mut ke = 0.0;
                let mut ent = 0.0;
                let mut ce = 0.0;
                for i in 0..n {
                    let rv = r[kk * n + i];
                    ke += rv * dist[kk * n + i];
                    if rv > 0.0 {
                        ent += rv * rv.ln() * s;
                    }

                    let bi = batch[i];
                    let w = theta[bi] * z[kk * b + bi];
                    ce += rv * s * w;
                }
                (ke, ent, ce)
            })
            .reduce(
                || (0.0, 0.0, 0.0),
                |a, bb| (a.0 + bb.0, a.1 + bb.1, a.2 + bb.2),
            );

        self.objective_kmeans
            .push(kmeans_error + entropy + cross_entropy);
    }

    fn harmonize(&mut self) -> (bool, usize) {
        let mut converged = false;
        let mut n_iter = 0;
        for i in 1..=self.max_iter_harmony {
            n_iter = i;
            self.cluster();
            self.moe_correct_ridge();
            converged = self.check_convergence(1);
            if converged {
                break;
            }
        }
        (converged, n_iter)
    }

    fn cluster(&mut self) {
        self.dist_mat = cluster_dist(&self.y, &self.z_cos);
        for i in 0..self.max_iter_kmeans {
            self.y = gemm_abt(&self.z_cos, &self.r);
            self.y = l2_norm_cols(&self.y);

            self.dist_mat = cluster_dist(&self.y, &self.z_cos);

            self.update_r();

            self.compute_objective();
            if i > self.window_size && self.check_convergence(0) {
                break;
            }
        }
        let last = *self.objective_kmeans.last().unwrap();
        self.objective_harmony.push(last);
    }

    fn update_r(&mut self) {
        self.scale_dist = -&self.dist_mat;
        self.scale_dist = divide_rows(&self.scale_dist, &self.sigma);
        self.scale_dist = sub_col_max(&self.scale_dist);
        self.scale_dist = self.scale_dist.mapv(f32::exp);

        let mut update_order: Vec<usize> = (0..self.n).collect();
        self.rng.shuffle(&mut update_order);
        let n_blocks = (1.0 / self.block_size).ceil() as usize;
        let blocks = array_split(&update_order, n_blocks);

        for block in blocks {
            for kk in 0..self.k {
                let mut r_sum = 0.0;
                for &i in block {
                    r_sum += self.r[[kk, i]];
                }
                for b in 0..self.b {
                    self.e[[kk, b]] -= r_sum * self.pr_b[b];
                    let mut o_sum = 0.0;
                    for &i in block {
                        o_sum += self.r[[kk, i]] * self.phi[[b, i]];
                    }
                    self.o[[kk, b]] -= o_sum;
                }
            }

            for &i in block {
                let bi = self.batch_of_cell[i];
                for kk in 0..self.k {
                    let ratio = (self.e[[kk, bi]] + 1.0) / (self.o[[kk, bi]] + 1.0);
                    let factor = ratio.powf(self.theta[bi]);
                    self.r[[kk, i]] = self.scale_dist[[kk, i]] * factor;
                }
            }

            for &i in block {
                let mut s = 0.0;
                for kk in 0..self.k {
                    s += self.r[[kk, i]];
                }
                for kk in 0..self.k {
                    self.r[[kk, i]] /= s;
                }
            }

            for kk in 0..self.k {
                let mut r_sum = 0.0;
                for &i in block {
                    r_sum += self.r[[kk, i]];
                }
                for b in 0..self.b {
                    self.e[[kk, b]] += r_sum * self.pr_b[b];
                    let mut o_sum = 0.0;
                    for &i in block {
                        o_sum += self.r[[kk, i]] * self.phi[[b, i]];
                    }
                    self.o[[kk, b]] += o_sum;
                }
            }
        }
    }

    fn moe_correct_ridge(&mut self) {
        self.z_corr = self.z_orig.clone();
        let b1 = self.b + 1;
        for kk in 0..self.k {
            {
                let n = self.n;
                let phi_moe_s = self.phi_moe.as_slice().unwrap();
                let r_s = self.r.as_slice().unwrap();
                let phi_rk_slice = self.phi_rk.as_slice_mut().unwrap();
                let ptr = SendPtr(phi_rk_slice.as_mut_ptr());
                (0..n).into_par_iter().for_each(move |i| {
                    let rv = r_s[kk * n + i];
                    for r in 0..b1 {
                        unsafe {
                            *ptr.add(r * n + i) = phi_moe_s[r * n + i] * rv;
                        }
                    }
                });
            }

            let x = &gemm_abt(&self.phi_rk, &self.phi_moe) + &self.lamb;

            let rhs = gemm_abt(&self.phi_rk, &self.z_orig);

            let x_nalg = nalgebra::DMatrix::from_row_slice(b1, b1, x.as_slice().unwrap());
            let rhs_nalg = nalgebra::DMatrix::from_row_slice(b1, self.d, rhs.as_slice().unwrap());
            if let Some(w_nalg) = x_nalg.lu().solve(&rhs_nalg) {
                for r in 0..b1 {
                    for j in 0..self.d {
                        self.w[[r, j]] = w_nalg[(r, j)];
                    }
                }
            } else {
                for r in 0..b1 {
                    for j in 0..self.d {
                        self.w[[r, j]] = 0.0;
                    }
                }
            }

            for j in 0..self.d {
                self.w[[0, j]] = 0.0;
            }

            let correction = gemm_atb(&self.w, &self.phi_rk);
            self.z_corr = &self.z_corr - &correction;
        }

        self.z_cos = l2_norm_cols(&self.z_corr);
    }

    fn check_convergence(&self, i_type: usize) -> bool {
        if i_type == 0 {
            let okl = self.objective_kmeans.len();
            let mut obj_old = 0.0;
            let mut obj_new = 0.0;
            for i in 0..self.window_size {
                obj_old += self.objective_kmeans[okl - 2 - i];
                obj_new += self.objective_kmeans[okl - 1 - i];
            }
            return (obj_old - obj_new).abs() / obj_old.abs() < self.epsilon_kmeans;
        }

        let obj_old = self.objective_harmony[self.objective_harmony.len() - 2];
        let obj_new = self.objective_harmony[self.objective_harmony.len() - 1];
        (obj_old - obj_new) / obj_old.abs() < self.epsilon_harmony
    }
}

fn gemm_atb(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (p, m) = (a.nrows(), a.ncols());
    let n = b.ncols();
    debug_assert_eq!(p, b.nrows());
    let a_s = a.as_slice().unwrap();
    let b_s = b.as_slice().unwrap();
    let mut c = Array2::<f32>::zeros((m, n));
    let c_slice = c.as_slice_mut().unwrap();
    c_slice.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        for q in 0..p {
            let av = a_s[q * m + i];
            let brow = &b_s[q * n..(q + 1) * n];
            for j in 0..n {
                crow[j] += av * brow[j];
            }
        }
    });
    c
}

fn gemm_abt(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, p) = (a.nrows(), a.ncols());
    let n = b.nrows();
    debug_assert_eq!(p, b.ncols());
    let a_s = a.as_slice().unwrap();
    let b_s = b.as_slice().unwrap();
    let mut c = Array2::<f32>::zeros((m, n));
    let c_slice = c.as_slice_mut().unwrap();
    let ptr = SendPtr(c_slice.as_mut_ptr());
    (0..m).into_par_iter().for_each(move |i| {
        let arow = &a_s[i * p..(i + 1) * p];
        for j in 0..n {
            let brow = &b_s[j * p..(j + 1) * p];
            let mut s = 0.0;
            for q in 0..p {
                s += arow[q] * brow[q];
            }
            unsafe {
                *ptr.add(i * n + j) = s;
            }
        }
    });
    c
}

fn cluster_dist(y: &Array2<f32>, z_cos: &Array2<f32>) -> Array2<f32> {
    let mut dm = gemm_atb(y, z_cos);
    dm.as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .for_each(|v| *v = 2.0 * (1.0 - *v));
    dm
}

fn l2_norm_cols(x: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = x.clone();
    let xs = x.as_slice().unwrap();
    let out_slice = out.as_slice_mut().unwrap();
    let ptr = SendPtr(out_slice.as_mut_ptr());
    (0..cols).into_par_iter().for_each(move |j| {
        let mut s = 0.0;
        for i in 0..rows {
            let v = xs[i * cols + j];
            s += v * v;
        }
        let norm = s.sqrt();
        for i in 0..rows {
            unsafe {
                *ptr.add(i * cols + j) = xs[i * cols + j] / norm;
            }
        }
    });
    out
}

fn normalize_cols_l1(x: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = x.clone();
    let xs = x.as_slice().unwrap();
    let out_slice = out.as_slice_mut().unwrap();
    let ptr = SendPtr(out_slice.as_mut_ptr());
    (0..cols).into_par_iter().for_each(move |j| {
        let mut s = 0.0;
        for i in 0..rows {
            s += xs[i * cols + j];
        }
        for i in 0..rows {
            unsafe {
                *ptr.add(i * cols + j) = xs[i * cols + j] / s;
            }
        }
    });
    out
}

fn col_wise_max(x: &Array2<f32>) -> Vec<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let xs = x.as_slice().unwrap();
    (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut m = f32::NEG_INFINITY;
            for i in 0..rows {
                let v = xs[i * cols + j];
                if v > m {
                    m = v;
                }
            }
            m
        })
        .collect()
}

fn divide_rows(x: &Array2<f32>, sigma: &[f32]) -> Array2<f32> {
    let mut out = x.clone();
    let cols = x.ncols();
    let out_slice = out.as_slice_mut().unwrap();
    out_slice
        .par_chunks_mut(cols)
        .zip(sigma)
        .for_each(|(row, &s)| {
            for v in row.iter_mut() {
                *v /= s;
            }
        });
    out
}

fn sub_col_max(x: &Array2<f32>) -> Array2<f32> {
    let (rows, cols) = (x.nrows(), x.ncols());
    let mut out = x.clone();
    let xs = x.as_slice().unwrap();
    let cm = col_wise_max(x);
    let out_slice = out.as_slice_mut().unwrap();
    let ptr = SendPtr(out_slice.as_mut_ptr());
    (0..cols).into_par_iter().for_each(move |j| {
        for i in 0..rows {
            unsafe {
                *ptr.add(i * cols + j) = xs[i * cols + j] - cm[j];
            }
        }
    });
    out
}

fn row_sums(x: &Array2<f32>) -> Vec<f32> {
    let cols = x.ncols();
    let xs = x.as_slice().unwrap();
    (0..x.nrows())
        .into_par_iter()
        .map(|i| xs[i * cols..(i + 1) * cols].iter().sum())
        .collect()
}

fn outer(a: &[f32], b: &[f32]) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            out[[i, j]] = a[i] * b[j];
        }
    }
    out
}

fn array_split(arr: &[usize], n: usize) -> Vec<&[usize]> {
    if n == 0 {
        return vec![arr];
    }
    let n = n.min(arr.len()).max(1);
    let mut out = Vec::with_capacity(n);
    let base = arr.len() / n;
    let rem = arr.len() % n;
    let mut start = 0;
    for i in 0..n {
        let size = base + if i < rem { 1 } else { 0 };
        let end = start + size;
        out.push(&arr[start..end]);
        start = end;
    }
    out
}

fn kmeans2(data: &Array2<f32>, k: usize, rng: &mut Mt19937) -> Array2<f32> {
    let mut code_book = kpp(data, k, rng);
    for _ in 0..10 {
        let label = vq(data, &code_book);
        let (mut new_code_book, has_members) = update_cluster_means(data, &label, k);
        for j in 0..k {
            if !has_members[j] {
                for dim in 0..data.ncols() {
                    new_code_book[[j, dim]] = code_book[[j, dim]];
                }
            }
        }
        code_book = new_code_book;
    }
    code_book
}

fn kpp(data: &Array2<f32>, k: usize, rng: &mut Mt19937) -> Array2<f32> {
    let n = data.nrows();
    let dims = data.ncols();
    let mut init = Array2::<f32>::zeros((k, dims));
    for i in 0..k {
        if i == 0 {
            let idx = rng.bounded_below(0, n as u64) as usize;
            init.row_mut(i).assign(&data.row(idx));
        } else {
            let d2 = min_sqeuclidean(&init.slice(s![..i, ..]), data);
            let total: f32 = d2.iter().sum();
            let mut cumprobs = Vec::with_capacity(n);
            let mut acc = 0.0;
            for &d in &d2 {
                acc += d / total;
                cumprobs.push(acc);
            }
            let r = rng.rk_double();
            let idx = searchsorted(&cumprobs, r).min(n - 1);
            init.row_mut(i).assign(&data.row(idx));
        }
    }
    init
}

fn min_sqeuclidean(init: &ndarray::ArrayView2<f32>, data: &Array2<f32>) -> Vec<f32> {
    let (i, d) = (init.nrows(), init.ncols());
    let n = data.nrows();
    let data_s = data.as_slice().unwrap();
    (0..n)
        .into_par_iter()
        .map(|j| {
            let mut out = f32::INFINITY;
            for c in 0..i {
                let mut s = 0.0;
                for dim in 0..d {
                    let diff = init[[c, dim]] - data_s[j * d + dim];
                    s += diff * diff;
                }
                if s < out {
                    out = s;
                }
            }
            out
        })
        .collect()
}

fn searchsorted(cumprobs: &[f32], r: f32) -> usize {
    cumprobs
        .iter()
        .position(|&c| c >= r)
        .unwrap_or(cumprobs.len())
}

fn vq(obs: &Array2<f32>, code_book: &Array2<f32>) -> Vec<usize> {
    let m = obs.nrows();
    let k = code_book.nrows();
    let d = obs.ncols();
    let obs_s = obs.as_slice().unwrap();
    let cb_s = code_book.as_slice().unwrap();
    (0..m)
        .into_par_iter()
        .map(|i| {
            let mut min_dist = f32::INFINITY;
            let mut best = 0usize;
            for j in 0..k {
                let mut s = 0.0;
                for dim in 0..d {
                    let diff = obs_s[i * d + dim] - cb_s[j * d + dim];
                    s += diff * diff;
                }
                if s < min_dist {
                    min_dist = s;
                    best = j;
                }
            }
            best
        })
        .collect()
}

fn update_cluster_means(obs: &Array2<f32>, code: &[usize], k: usize) -> (Array2<f32>, Vec<bool>) {
    let m = obs.nrows();
    let d = obs.ncols();
    let obs_s = obs.as_slice().unwrap();
    let n_threads = rayon::current_num_threads();
    let mut tbuf: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; k * d]).collect();
    let mut tcounts: Vec<Vec<usize>> = (0..n_threads).map(|_| vec![0usize; k]).collect();

    let rows_per_thread = m.div_ceil(n_threads);
    tbuf.par_iter_mut()
        .zip(tcounts.par_iter_mut())
        .enumerate()
        .for_each(|(tid, (buf, cnt))| {
            let r0 = tid * rows_per_thread;
            let r1 = (r0 + rows_per_thread).min(m);
            for i in r0..r1 {
                let j = code[i];
                cnt[j] += 1;
                for dim in 0..d {
                    buf[j * d + dim] += obs_s[i * d + dim];
                }
            }
        });

    let mut new_code_book = Array2::<f32>::zeros((k, d));
    let mut counts = vec![0usize; k];
    for t in 0..n_threads {
        for j in 0..k {
            counts[j] += tcounts[t][j];
            for dim in 0..d {
                new_code_book[[j, dim]] += tbuf[t][j * d + dim];
            }
        }
    }
    let has_members: Vec<bool> = counts.iter().map(|&c| c > 0).collect();
    for j in 0..k {
        if counts[j] > 0 {
            for dim in 0..d {
                new_code_book[[j, dim]] /= counts[j] as f32;
            }
        }
    }
    (new_code_book, has_members)
}
