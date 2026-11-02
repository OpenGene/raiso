use std::path::PathBuf;

use anyhow::{Context, Result};
use hdf5::types::VarLenUnicode;
use hdf5::{Dataset, File, Group};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array1, Array2};
use polars::prelude::*;

use crate::core::{ExpMatrix, SoData};

pub fn read_h5ad(path: &str, spatial_key: &str, save_counts_layer: bool) -> Result<SoData> {
    crate::parallel::init_rayon(0);
    let file = File::open(path)?;
    let mut data = SoData::new();
    data.file_path = Some(PathBuf::from(path));
    data.file_format = Some("h5ad".into());

    if file.link_exists("X") {
        if let Ok(ds) = file.dataset("X") {
            let arr: Array2<f64> = read_dense_x(&ds)?;
            data.exp_matrix = Some(ExpMatrix::Dense(arr.mapv(|v| v as f32)));
        } else if let Ok(group) = file.group("X") {
            let csr = read_sparse_matrix(&group)?;
            let (nrows, ncols) = (csr.nrows(), csr.ncols());
            let offsets = csr.row_offsets().to_vec();
            let cols = csr.col_indices().to_vec();
            let vals: Vec<f32> = csr.values().iter().map(|&v| v as f32).collect();
            data.exp_matrix = Some(ExpMatrix::Sparse(
                CsrMatrix::try_from_csr_data(nrows, ncols, offsets, cols, vals)
                    .map_err(|e| anyhow::anyhow!("read_h5ad: failed to build sparse X: {e:?}"))?,
            ));
        }
    }

    if file.link_exists("obs") {
        if let Ok(group) = file.group("obs") {
            if let Ok(df) = read_index_only(&group, "cell") {
                data.obs = Some(df);
            }
        }
    }

    if file.link_exists("var") {
        if let Ok(group) = file.group("var") {
            if let Ok(df) = read_index_only(&group, "gene") {
                data.var = Some(df);
            }
        }
    }

    if file.link_exists("obsm") {
        if let Ok(obsm_group) = file.group("obsm") {
            if obsm_group.link_exists(spatial_key) {
                if let Ok(ds) = obsm_group.dataset(spatial_key) {
                    if let Ok(arr) = read_dense_x(&ds) {
                        data.position = Some(arr);
                    }
                }
            }
        }
    }

    if save_counts_layer {
        if let Some(ref exp) = data.exp_matrix {
            data.layers.insert("counts".into(), exp.clone());
        }
    }

    Ok(data)
}

pub(crate) fn read_sparse_matrix(group: &Group) -> Result<CsrMatrix<f64>> {
    let enc_type =
        read_attr_string(group, "encoding-type").unwrap_or_else(|_| "csr_matrix".to_string());

    let (n_rows, n_cols) = read_sparse_shape(group)?;

    if enc_type == "csc_matrix" {
        return read_csc_as_csr(group, n_rows, n_cols);
    }

    let data: Array1<f64> = read_data_as_f64(&group.dataset("data")?)?;
    let (indices, indptr) = read_sparse_indices_usize(group)?;

    let data_slice = data.as_slice().unwrap_or(&[]);
    let idx_slice = indices.as_slice().unwrap_or(&[]);
    let ptr_slice = indptr.as_slice().unwrap_or(&[]);

    let is_sorted = csr_is_sorted(idx_slice, ptr_slice, n_rows);

    if is_sorted {
        let sorted_indptr = indptr.into_raw_vec_and_offset().0;
        let sorted_indices = indices.into_raw_vec_and_offset().0;
        let sorted_data = data.into_raw_vec_and_offset().0;
        return CsrMatrix::try_from_csr_data(
            n_rows,
            n_cols,
            sorted_indptr,
            sorted_indices,
            sorted_data,
        )
        .map_err(|e| anyhow::anyhow!("Failed to construct CSR matrix: {:?}", e));
    }

    let mut sorted_data = Vec::with_capacity(data_slice.len());
    let mut sorted_indices = Vec::with_capacity(idx_slice.len());
    let mut sorted_indptr = Vec::with_capacity(ptr_slice.len());
    sorted_indptr.push(0);

    for row in 0..n_rows {
        let start = ptr_slice[row];
        let end = ptr_slice[row + 1];

        let mut pairs: Vec<(usize, f64)> = (start..end)
            .map(|i| (idx_slice[i], data_slice[i]))
            .collect();
        pairs.sort_by_key(|(col, _)| *col);

        for (col, val) in &pairs {
            sorted_indices.push(*col);
            sorted_data.push(*val);
        }
        sorted_indptr.push(sorted_indices.len());
    }

    CsrMatrix::try_from_csr_data(n_rows, n_cols, sorted_indptr, sorted_indices, sorted_data)
        .map_err(|e| anyhow::anyhow!("Failed to construct CSR matrix: {:?}", e))
}

fn read_csc_as_csr(group: &Group, n_rows: usize, n_cols: usize) -> Result<CsrMatrix<f64>> {
    use nalgebra_sparse::csc::CscMatrix;
    let data: Array1<f64> = read_data_as_f64(&group.dataset("data")?)?;
    let (indices, indptr) = read_sparse_indices_usize(group)?;

    let csc = CscMatrix::try_from_csc_data(
        n_rows,
        n_cols,
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to construct CSC matrix: {:?}", e))?;

    Ok(CsrMatrix::from(&csc))
}

fn csr_is_sorted(indices: &[usize], indptr: &[usize], n_rows: usize) -> bool {
    for row in 0..n_rows {
        let start = indptr[row];
        let end = indptr[row + 1];
        if start < end {
            let row_slice = &indices[start..end];
            if row_slice.windows(2).any(|w| w[0] > w[1]) {
                return false;
            }
        }
    }
    true
}

fn read_sparse_shape(group: &Group) -> Result<(usize, usize)> {
    if let Ok(attr) = group.attr("shape") {
        if let Ok(arr) = attr.read_1d::<u64>() {
            let s = arr.to_vec();
            if s.len() >= 2 {
                return Ok((s[0] as usize, s[1] as usize));
            }
        }
    }

    let indptr: Array1<usize> = group.dataset("indptr")?.read_1d::<usize>()?;
    let indices: Array1<usize> = group.dataset("indices")?.read_1d::<usize>()?;
    let n_rows = indptr.len().saturating_sub(1);
    let n_cols = indices.iter().max().map(|&m| m + 1).unwrap_or(0);
    Ok((n_rows, n_cols))
}

fn read_sparse_indices_usize(group: &Group) -> Result<(Array1<usize>, Array1<usize>)> {
    let map_to_usize = |arr: Array1<i64>| -> Array1<usize> {
        Array1::from_vec(arr.iter().map(|&x| x as usize).collect())
    };
    let map_u32 = |arr: Array1<u32>| -> Array1<usize> {
        Array1::from_vec(arr.iter().map(|&x| x as usize).collect())
    };

    let indices = if let Ok(arr) = group.dataset("indices")?.read_1d::<i64>() {
        map_to_usize(arr)
    } else if let Ok(arr) = group.dataset("indices")?.read_1d::<i32>() {
        Array1::from_vec(arr.iter().map(|&x| x as usize).collect())
    } else if let Ok(arr) = group.dataset("indices")?.read_1d::<u32>() {
        map_u32(arr)
    } else {
        group.dataset("indices")?.read_1d::<usize>()?
    };

    let indptr = if let Ok(arr) = group.dataset("indptr")?.read_1d::<i64>() {
        map_to_usize(arr)
    } else if let Ok(arr) = group.dataset("indptr")?.read_1d::<i32>() {
        Array1::from_vec(arr.iter().map(|&x| x as usize).collect())
    } else if let Ok(arr) = group.dataset("indptr")?.read_1d::<u32>() {
        map_u32(arr)
    } else {
        group.dataset("indptr")?.read_1d::<usize>()?
    };

    Ok((indices, indptr))
}

fn read_index_only(group: &Group, col_name: &str) -> Result<DataFrame> {
    let idx_name: String = read_attr_string(group, "_index").unwrap_or_else(|_| "_index".into());
    let ds = group.dataset(&idx_name)?;

    let column = if let Ok(arr) = ds.read_1d::<VarLenUnicode>() {
        let strings: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
        Column::new(col_name.into(), strings)
    } else if let Ok(arr) = ds.read_1d::<i64>() {
        let strings: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
        Column::new(col_name.into(), strings)
    } else if let Ok(arr) = ds.read_1d::<f64>() {
        let strings: Vec<String> = arr.iter().map(|v| format!("{}", v)).collect();
        Column::new(col_name.into(), strings)
    } else {
        anyhow::bail!("unsupported index dtype in {}", idx_name);
    };

    DataFrame::new(vec![column]).context("Failed to construct index DataFrame")
}

pub(crate) fn read_index_strings(group: &Group) -> Result<Vec<String>> {
    let idx_name: String = read_attr_string(group, "_index").unwrap_or_else(|_| "_index".into());
    let ds = group.dataset(&idx_name)?;
    read_string_array(&ds)
}

pub(crate) fn read_obs_column_strings(group: &Group, col_name: &str) -> Result<Vec<String>> {
    if let Ok(sub) = group.group(col_name) {
        let categories = read_string_array(&sub.dataset("categories")?)?;
        let codes = read_int_codes(&sub.dataset("codes")?)?;
        let mut out = Vec::with_capacity(codes.len());
        for c in codes {
            if c < 0 {
                out.push(String::new());
            } else {
                out.push(categories.get(c as usize).cloned().unwrap_or_default());
            }
        }
        return Ok(out);
    }
    let ds = group.dataset(col_name)?;
    read_string_array(&ds)
}

fn read_string_array(ds: &Dataset) -> Result<Vec<String>> {
    if let Ok(arr) = ds.read_1d::<VarLenUnicode>() {
        return Ok(arr.iter().map(|v| v.to_string()).collect());
    }
    anyhow::bail!("unsupported string-array dtype")
}

fn read_int_codes(ds: &Dataset) -> Result<Vec<i64>> {
    macro_rules! try_read {
        ($t:ty) => {
            if let Ok(arr) = ds.read_1d::<$t>() {
                return Ok(arr.iter().map(|&x| x as i64).collect());
            }
        };
    }
    try_read!(i64);
    try_read!(i32);
    try_read!(i16);
    try_read!(i8);
    try_read!(u64);
    try_read!(u32);
    try_read!(u16);
    try_read!(u8);
    anyhow::bail!("unsupported codes dtype")
}

fn read_attr_string(loc: &hdf5::Location, name: &str) -> Result<String> {
    let attr = loc.attr(name)?;
    if let Ok(v) = attr.read_scalar::<VarLenUnicode>() {
        Ok(v.to_string())
    } else if let Ok(v) = attr.read_scalar::<hdf5::types::FixedAscii<256>>() {
        Ok(v.to_string())
    } else {
        anyhow::bail!("unsupported attribute type for {}", name)
    }
}

pub(crate) fn read_dense_x(ds: &Dataset) -> Result<Array2<f64>> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;
    use hdf5::types::TypeDescriptor;
    match desc {
        TypeDescriptor::Float(_) => {
            if let Ok(arr) = ds.read_2d::<f64>() {
                return Ok(arr);
            }
            let arr: Array2<f32> = ds.read_2d::<f32>()?;
            Ok(arr.mapv(|v| v as f64))
        }
        TypeDescriptor::Integer(_) => {
            if let Ok(arr) = ds.read_2d::<i64>() {
                return Ok(arr.mapv(|v| v as f64));
            }
            let arr: Array2<i32> = ds.read_2d::<i32>()?;
            Ok(arr.mapv(|v| v as f64))
        }
        TypeDescriptor::Unsigned(_) => {
            let arr: Array2<u32> = ds.read_2d::<u32>()?;
            Ok(arr.mapv(|v| v as f64))
        }
        _ => ds.read_2d::<f64>().map_err(|e| anyhow::anyhow!("{}", e)),
    }
}

fn read_data_as_f64(ds: &Dataset) -> Result<Array1<f64>> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;
    use hdf5::types::TypeDescriptor;
    match desc {
        TypeDescriptor::Float(_) => {
            if let Ok(arr) = ds.read_1d::<f64>() {
                return Ok(arr);
            }
            let arr: Array1<f32> = ds.read_1d::<f32>()?;
            Ok(arr.mapv(|v| v as f64))
        }
        TypeDescriptor::Integer(_) => {
            if let Ok(arr) = ds.read_1d::<i64>() {
                return Ok(arr.mapv(|v| v as f64));
            }
            let arr: Array1<i32> = ds.read_1d::<i32>()?;
            Ok(arr.mapv(|v| v as f64))
        }
        TypeDescriptor::Unsigned(_) => {
            let arr: Array1<u32> = ds.read_1d::<u32>()?;
            Ok(arr.mapv(|v| v as f64))
        }
        _ => ds.read_1d::<f64>().map_err(|e| anyhow::anyhow!("{}", e)),
    }
}
