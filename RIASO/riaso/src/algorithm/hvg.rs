use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use rayon::prelude::*;

use crate::core::{ExpMatrix, HvgResult, SoData};

const MAD_C: f32 = 0.674_489_74;

fn csr_col_sum_sq_par<F>(
    csr: &CsrMatrix<f32>,
    n_rows: usize,
    n_cols: usize,
    transform: F,
) -> (Vec<f32>, Vec<f32>)
where
    F: Fn(f32) -> f32 + Sync + Send,
{
    let n_threads = rayon::current_num_threads();
    let mut sum_buf: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; n_cols]).collect();
    let mut sq_buf: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; n_cols]).collect();

    let offsets = csr.row_offsets();
    let col_idx = csr.col_indices();
    let values = csr.values();

    let rows_per_thread = n_rows.div_ceil(n_threads);
    sum_buf
        .par_iter_mut()
        .zip(sq_buf.par_iter_mut())
        .enumerate()
        .for_each(|(tid, (sb, sqb))| {
            let r0 = tid * rows_per_thread;
            let r1 = (r0 + rows_per_thread).min(n_rows);
            for row in r0..r1 {
                for p in offsets[row]..offsets[row + 1] {
                    let u = transform(values[p]);
                    let col = col_idx[p];
                    sb[col] += u;
                    sqb[col] += u * u;
                }
            }
        });

    let mut col_sum = vec![0.0f32; n_cols];
    let mut col_sum_sq = vec![0.0f32; n_cols];
    for t in 0..n_threads {
        for j in 0..n_cols {
            col_sum[j] += sum_buf[t][j];
            col_sum_sq[j] += sq_buf[t][j];
        }
    }
    (col_sum, col_sum_sq)
}

fn dense_col_sum_sq_par<F>(arr: &Array2<f32>, n_cols: usize, transform: F) -> (Vec<f32>, Vec<f32>)
where
    F: Fn(f32) -> f32 + Sync + Send,
{
    (0..n_cols)
        .into_par_iter()
        .map(|j| {
            let mut s = 0.0;
            let mut sq = 0.0;
            for v in arr.column(j) {
                let u = transform(*v);
                s += u;
                sq += u * u;
            }
            (s, sq)
        })
        .unzip()
}

pub fn select_top_variable_genes(data: &mut SoData, n_top: usize, n_bins: usize, method: &str) {
    data.result.hvg = None;

    let is_seurat = match method {
        "seurat" => true,
        "cell_ranger" => false,
        _ => return,
    };

    let exp = match &data.exp_matrix {
        Some(e) => e,
        None => return,
    };

    let n_cells = data.n_obs();
    let n_genes = data.n_vars();

    if n_genes <= n_top || n_cells < 2 {
        return;
    }
    let n_bins = n_bins.max(1);

    let (col_sum, col_sum_sq) = if is_seurat {
        match exp {
            ExpMatrix::Sparse(csr) => csr_col_sum_sq_par(csr, n_cells, n_genes, |v| v.exp_m1()),
            ExpMatrix::Dense(arr) => dense_col_sum_sq_par(arr, n_genes, |v| v.exp_m1()),
        }
    } else {
        match exp {
            ExpMatrix::Sparse(csr) => csr_col_sum_sq_par(csr, n_cells, n_genes, |v| v),
            ExpMatrix::Dense(arr) => dense_col_sum_sq_par(arr, n_genes, |v| v),
        }
    };

    let n = n_cells as f32;
    let (means, disp, disp_norm) = if is_seurat {
        seurat_dispersion(&col_sum, &col_sum_sq, n, n_genes, n_bins)
    } else {
        cell_ranger_dispersion(&col_sum, &col_sum_sq, n, n_genes)
    };

    let mut vals: Vec<f32> = disp_norm.iter().copied().filter(|v| !v.is_nan()).collect();

    if vals.len() < n_top {
        return;
    }
    vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = vals[n_top - 1];

    let genes: Vec<String> = data.var_names().into_iter().map(String::from).collect();
    let hv: Vec<bool> = (0..n_genes)
        .map(|j| !disp_norm[j].is_nan() && disp_norm[j] >= cutoff)
        .collect();

    data.result.hvg = Some(HvgResult {
        genes,
        means,
        dispersions: disp,
        dispersions_norm: disp_norm,
        highly_variable: hv,
    });
}

fn seurat_dispersion(
    col_sum: &[f32],
    col_sum_sq: &[f32],
    n: f32,
    n_genes: usize,
    n_bins: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut disp = vec![f32::NAN; n_genes];
    let mut m_log = vec![0.0f32; n_genes];
    for j in 0..n_genes {
        let mean = col_sum[j] / n;
        let var = (col_sum_sq[j] / n - mean * mean) * n / (n - 1.0);
        let mean = if mean == 0.0 { 1e-12 } else { mean };
        let d = var / mean;
        if d != 0.0 {
            disp[j] = d.ln();
        }
        m_log[j] = mean.ln_1p();
    }

    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &m_log {
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let mut bin_of = vec![0usize; n_genes];
    if mx > mn {
        let adj = (mx - mn) * 0.001;
        let step = (mx - mn) / n_bins as f32;

        let edges: Vec<f32> = (0..=n_bins)
            .map(|i| {
                if i == 0 {
                    mn - adj
                } else {
                    mn + i as f32 * step
                }
            })
            .collect();
        for j in 0..n_genes {
            let b = edges.partition_point(|&e| e < m_log[j]).saturating_sub(1);
            bin_of[j] = b.min(n_bins - 1);
        }
    }

    let mut bin_sum = vec![0.0f32; n_bins];
    let mut bin_sum_sq = vec![0.0f32; n_bins];
    let mut bin_cnt = vec![0usize; n_bins];
    for j in 0..n_genes {
        let d = disp[j];
        if d.is_nan() {
            continue;
        }
        let b = bin_of[j];
        bin_sum[b] += d;
        bin_sum_sq[b] += d * d;
        bin_cnt[b] += 1;
    }
    let mut bin_mean = vec![f32::NAN; n_bins];
    let mut bin_std = vec![f32::NAN; n_bins];
    for b in 0..n_bins {
        if bin_cnt[b] == 0 {
            continue;
        }
        let m = bin_sum[b] / bin_cnt[b] as f32;
        if bin_cnt[b] < 2 {
            bin_mean[b] = 0.0;
            bin_std[b] = m;
        } else {
            let var = (bin_sum_sq[b] / bin_cnt[b] as f32 - m * m) * bin_cnt[b] as f32
                / (bin_cnt[b] as f32 - 1.0);
            bin_mean[b] = m;
            bin_std[b] = var.sqrt();
        }
    }

    let mut disp_norm = vec![f32::NAN; n_genes];
    for j in 0..n_genes {
        let b = bin_of[j];
        disp_norm[j] = (disp[j] - bin_mean[b]) / bin_std[b];
    }
    (m_log, disp, disp_norm)
}

fn cell_ranger_dispersion(
    col_sum: &[f32],
    col_sum_sq: &[f32],
    n: f32,
    n_genes: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut means = vec![0.0f32; n_genes];
    let mut disp = vec![0.0f32; n_genes];
    for j in 0..n_genes {
        let m0 = col_sum[j] / n;
        let var = (col_sum_sq[j] / n - m0 * m0) * n / (n - 1.0);
        let m = if m0 == 0.0 { 1e-12 } else { m0 };
        means[j] = m;
        disp[j] = var / m;
    }

    let mut sorted_mean = means.clone();
    sorted_mean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let edges: Vec<f32> = (10..=100)
        .step_by(5)
        .map(|p| percentile_linear_sorted(&sorted_mean, p as f32))
        .collect();
    let n_used = edges.len() + 1;

    let mut bin_of = vec![0usize; n_genes];
    for j in 0..n_genes {
        bin_of[j] = edges.partition_point(|&e| e < means[j]);
    }

    let mut bin_vals: Vec<Vec<f32>> = vec![Vec::new(); n_used];
    for j in 0..n_genes {
        bin_vals[bin_of[j]].push(disp[j]);
    }
    let mut bin_median = vec![f32::NAN; n_used];
    let mut bin_mad = vec![f32::NAN; n_used];
    for b in 0..n_used {
        if bin_vals[b].is_empty() {
            continue;
        }
        bin_vals[b].sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        let med = median_sorted(&bin_vals[b]);
        bin_median[b] = med;
        let mut devs: Vec<f32> = bin_vals[b].iter().map(|v| (v - med).abs()).collect();
        devs.sort_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
        bin_mad[b] = median_sorted(&devs) / MAD_C;
    }

    let mut disp_norm = vec![f32::NAN; n_genes];
    for j in 0..n_genes {
        let b = bin_of[j];
        disp_norm[j] = (disp[j] - bin_median[b]) / bin_mad[b];
    }
    (means, disp, disp_norm)
}

fn percentile_linear_sorted(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = (n as f32 - 1.0) * q / 100.0;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w_hi = idx - lo as f32;
        let w_lo = hi as f32 - idx;
        sorted[lo] * w_lo + sorted[hi] * w_hi
    }
}

fn median_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}
