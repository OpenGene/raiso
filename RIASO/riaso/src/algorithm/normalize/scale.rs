use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use rayon::prelude::*;

use crate::core::{ExpMatrix, SoData};

pub const SCALE_LAYER: &str = "scale";

pub fn scale(data: &mut SoData, zero_center: bool, max_value: Option<f64>, write_layer: bool) {
    if data.n_obs() < 2 {
        return;
    }

    super::save_layer(data, super::LOG1P_LAYER);

    if let Some(exp) = &mut data.exp_matrix {
        match exp {
            ExpMatrix::Dense(arr) => scale_dense(arr, zero_center, max_value),
            ExpMatrix::Sparse(csr) if zero_center => {
                let mut d = crate::parallel::csr_to_dense::<f32>(csr);
                scale_dense(&mut d, true, max_value);
                *exp = ExpMatrix::Dense(d);
            }
            ExpMatrix::Sparse(csr) => scale_sparse_inplace(csr, max_value),
        }
    }
    if write_layer {
        super::save_layer(data, SCALE_LAYER);
    }
}

fn scale_dense(arr: &mut Array2<f32>, zero_center: bool, max_value: Option<f64>) {
    let (n_rows, n_cols) = arr.dim();
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    if !arr.is_standard_layout() {
        *arr = arr.as_standard_layout().into_owned();
    }
    let slice = arr.as_slice_mut().expect("dense matrix must be contiguous");

    let (col_sum, col_sum_sq) = dense_col_stats(slice, n_rows, n_cols);
    let nn = n_rows as f32;
    let mut mean = vec![0.0f32; n_cols];
    let mut std = vec![0.0f32; n_cols];
    for j in 0..n_cols {
        let m = col_sum[j] / nn;
        let var = (col_sum_sq[j] / nn - m * m) * nn / (nn - 1.0f32);
        let var = if var < 0.0f32 { 0.0f32 } else { var };
        let s = var.sqrt();
        mean[j] = m;
        std[j] = if s == 0.0f32 { 1.0f32 } else { s };
    }

    slice.par_chunks_mut(n_cols).for_each(|row| {
        if zero_center {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (*v - mean[j]) / std[j];
            }
        } else {
            for (j, v) in row.iter_mut().enumerate() {
                *v /= std[j];
            }
        }
    });

    if let Some(mv) = max_value {
        let mv = mv as f32;
        slice.par_iter_mut().for_each(|v| {
            if *v > mv {
                *v = mv;
            }
        });
    }
}

fn scale_sparse_inplace(csr: &mut CsrMatrix<f32>, max_value: Option<f64>) {
    let n_rows = csr.nrows();
    let n_cols = csr.ncols();
    if n_rows == 0 || n_cols == 0 {
        return;
    }
    let (col_sum, col_sum_sq) = csr_col_stats(csr);
    let nn = n_rows as f32;
    let mut std = vec![0.0f32; n_cols];
    for j in 0..n_cols {
        let m = col_sum[j] / nn;
        let var = (col_sum_sq[j] / nn - m * m) * nn / (nn - 1.0f32);
        let var = if var < 0.0f32 { 0.0f32 } else { var };
        let s = var.sqrt();
        std[j] = if s == 0.0f32 { 1.0f32 } else { s };
    }

    let col_idx = csr.col_indices().to_vec();
    let values = csr.values_mut();
    let mv = max_value.map(|v| v as f32);
    values.par_iter_mut().enumerate().for_each(|(k, v)| {
        let j = col_idx[k];
        *v /= std[j];
        if let Some(mv) = mv {
            if *v > mv {
                *v = mv;
            }
        }
    });
}

fn dense_col_stats(slice: &[f32], n_rows: usize, n_cols: usize) -> (Vec<f32>, Vec<f32>) {
    let n_threads = rayon::current_num_threads();
    let mut sum_buf: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; n_cols]).collect();
    let mut sq_buf: Vec<Vec<f32>> = (0..n_threads).map(|_| vec![0.0f32; n_cols]).collect();
    let rows_per_thread = n_rows.div_ceil(n_threads);
    sum_buf
        .par_iter_mut()
        .zip(sq_buf.par_iter_mut())
        .enumerate()
        .for_each(|(tid, (sb, sqb))| {
            let r0 = tid * rows_per_thread;
            let r1 = (r0 + rows_per_thread).min(n_rows);
            for row in r0..r1 {
                let base = row * n_cols;
                for j in 0..n_cols {
                    let v = slice[base + j];
                    sb[j] += v;
                    sqb[j] += v * v;
                }
            }
        });
    reduce_thread_bufs(&sum_buf, &sq_buf, n_cols, n_threads)
}

fn csr_col_stats(csr: &CsrMatrix<f32>) -> (Vec<f32>, Vec<f32>) {
    let n_rows = csr.nrows();
    let n_cols = csr.ncols();
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
                    let v = values[p];
                    let col = col_idx[p];
                    sb[col] += v;
                    sqb[col] += v * v;
                }
            }
        });
    reduce_thread_bufs(&sum_buf, &sq_buf, n_cols, n_threads)
}

fn reduce_thread_bufs(
    sum_buf: &[Vec<f32>],
    sq_buf: &[Vec<f32>],
    n_cols: usize,
    n_threads: usize,
) -> (Vec<f32>, Vec<f32>) {
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
