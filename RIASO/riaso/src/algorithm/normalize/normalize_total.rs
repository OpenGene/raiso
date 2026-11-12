use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use rayon::prelude::*;

use crate::core::{ExpMatrix, SoData};
use crate::parallel::SendPtr;

pub const NORMALIZE_TOTAL_LAYER: &str = "normalize_total";

pub fn normalize_total(data: &mut SoData, target_sum: Option<f32>, write_layer: bool) {
    if let Some(exp) = &mut data.exp_matrix {
        match exp {
            ExpMatrix::Dense(arr) => normalize_total_dense(arr, target_sum),
            ExpMatrix::Sparse(csr) => normalize_total_sparse(csr, target_sum),
        }
    }
    if write_layer {
        super::save_layer(data, NORMALIZE_TOTAL_LAYER);
    }
}

fn normalize_total_dense(arr: &mut Array2<f32>, target_sum: Option<f32>) {
    if !arr.is_standard_layout() {
        *arr = arr.as_standard_layout().into_owned();
    }
    let m = arr.ncols();
    let slice = arr.as_slice_mut().expect("dense matrix must be contiguous");

    let row_sums: Vec<f32> = slice.par_chunks(m).map(|row| row.iter().sum()).collect();

    let default_target: f32 = match target_sum {
        Some(t) => t,
        None => median_nonzero(&row_sums).unwrap_or(10000.0f32),
    };

    slice
        .par_chunks_mut(m)
        .zip(&row_sums)
        .for_each(|(row, &sum)| {
            let scale = if sum > 0.0f32 {
                default_target / sum
            } else {
                0.0f32
            };
            if scale != 1.0f32 {
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
        });
}

fn normalize_total_sparse(csr: &mut CsrMatrix<f32>, target_sum: Option<f32>) {
    let n = csr.nrows();

    let offsets = csr.row_offsets().to_vec();

    let row_sums: Vec<f32> = {
        let values = csr.values();
        (0..n)
            .into_par_iter()
            .map(|i| values[offsets[i]..offsets[i + 1]].iter().sum())
            .collect()
    };

    let default_target: f32 = match target_sum {
        Some(t) => t,
        None => median_nonzero(&row_sums).unwrap_or(10000.0f32),
    };

    let values = csr.values_mut();
    let ptr = SendPtr(values.as_mut_ptr());
    (0..n).into_par_iter().for_each(move |i| {
        let scale = if row_sums[i] > 0.0f32 {
            default_target / row_sums[i]
        } else {
            0.0f32
        };
        for idx in offsets[i]..offsets[i + 1] {
            unsafe {
                *ptr.add(idx) *= scale;
            }
        }
    });
}

fn median_nonzero(vals: &[f32]) -> Option<f32> {
    let mut filtered: Vec<f32> = vals.iter().copied().filter(|&v| v > 0.0f32).collect();
    if filtered.is_empty() {
        return None;
    }
    filtered.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = filtered.len();
    Some(if n.is_multiple_of(2) {
        (filtered[n / 2 - 1] + filtered[n / 2]) / 2.0f32
    } else {
        filtered[n / 2]
    })
}
