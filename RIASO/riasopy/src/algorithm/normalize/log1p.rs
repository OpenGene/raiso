use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use rayon::prelude::*;

use crate::core::{ExpMatrix, SoData};

pub const LOG1P_LAYER: &str = "log1p";

pub fn log1p(data: &mut SoData, write_layer: bool) {
    if let Some(exp) = &mut data.exp_matrix {
        match exp {
            ExpMatrix::Dense(arr) => log1p_dense(arr),
            ExpMatrix::Sparse(csr) => log1p_sparse(csr),
        }
    }
    if write_layer {
        super::save_layer(data, LOG1P_LAYER);
    }
}

fn log1p_dense(arr: &mut Array2<f32>) {
    if !arr.is_standard_layout() {
        *arr = arr.as_standard_layout().into_owned();
    }
    let slice = arr.as_slice_mut().expect("dense matrix must be contiguous");
    slice.par_iter_mut().for_each(|v| *v = (*v + 1.0f32).ln());
}

fn log1p_sparse(csr: &mut CsrMatrix<f32>) {
    csr.values_mut()
        .par_iter_mut()
        .for_each(|v| *v = (*v + 1.0f32).ln());
}
