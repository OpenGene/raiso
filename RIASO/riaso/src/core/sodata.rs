use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use serde_json::Value;

use crate::core::AnalysisResults;
use crate::parallel::SendPtr;

#[derive(Debug, Clone)]
pub enum ExpMatrix {
    Dense(Array2<f32>),
    Sparse(CsrMatrix<f32>),
}

impl ExpMatrix {
    pub fn n_rows(&self) -> usize {
        match self {
            Self::Dense(m) => m.nrows(),
            Self::Sparse(m) => m.nrows(),
        }
    }

    pub fn n_cols(&self) -> usize {
        match self {
            Self::Dense(m) => m.ncols(),
            Self::Sparse(m) => m.ncols(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows(), self.n_cols())
    }

    pub fn to_dense(&self) -> Array2<f32> {
        match self {
            Self::Dense(arr) => arr.mapv(|v| v),
            Self::Sparse(csr) => {
                let (nrows, ncols) = (csr.nrows(), csr.ncols());
                let mut dense = Array2::<f32>::zeros((nrows, ncols));
                let offsets = csr.row_offsets();
                let cols = csr.col_indices();
                let vals = csr.values();
                let ptr = SendPtr(dense.as_mut_ptr());

                (0..nrows).into_par_iter().for_each(|r| {
                    let row_ptr = unsafe { ptr.add(r * ncols) };
                    for i in offsets[r]..offsets[r + 1] {
                        unsafe {
                            *row_ptr.add(cols[i]) = vals[i];
                        }
                    }
                });
                dense
            }
        }
    }
}

#[derive(Clone)]
pub struct SoData {
    pub exp_matrix: Option<ExpMatrix>,

    pub layers: HashMap<String, ExpMatrix>,

    pub obs: Option<DataFrame>,

    pub var: Option<DataFrame>,

    pub position: Option<Array2<f32>>,

    pub raw: Option<Box<SoData>>,

    pub samples_names: Vec<String>,

    pub file_path: Option<PathBuf>,

    pub uns: HashMap<String, Value>,

    pub file_format: Option<String>,

    pub result: AnalysisResults,
}

impl SoData {
    pub fn new() -> Self {
        Self {
            exp_matrix: None,
            layers: HashMap::new(),
            obs: None,
            var: None,
            position: None,
            raw: None,
            samples_names: Vec::new(),
            file_path: None,
            uns: HashMap::new(),
            file_format: None,
            result: AnalysisResults::default(),
        }
    }

    pub fn n_obs(&self) -> usize {
        if let Some(ref obs) = self.obs {
            obs.height()
        } else if let Some(ref x) = self.exp_matrix {
            x.n_rows()
        } else if let Some(ref pos) = self.position {
            pos.nrows()
        } else {
            0
        }
    }

    pub fn n_vars(&self) -> usize {
        if let Some(ref var) = self.var {
            var.height()
        } else if let Some(ref x) = self.exp_matrix {
            x.n_cols()
        } else {
            0
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    pub fn var_names(&self) -> Vec<&str> {
        self.var
            .as_ref()
            .and_then(|df| df.column("gene").ok())
            .and_then(|col| col.str().ok())
            .map(|ca| (0..ca.len()).filter_map(|i| ca.get(i)).collect())
            .unwrap_or_default()
    }

    pub fn gene_names(&self) -> Vec<&str> {
        self.var_names()
    }

    pub fn symbol_names(&self) -> Vec<String> {
        let symbol_names = self
            .var
            .as_ref()
            .and_then(|df| df.column("gene_name").ok())
            .and_then(|col| col.str().ok())
            .map(|ca| {
                (0..ca.len())
                    .filter_map(|i| ca.get(i))
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
            });
        match symbol_names {
            Some(v) if v.len() == self.n_vars() && !v.is_empty() => v,
            _ => self
                .var_names()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    pub fn mt_gene_names(&self) -> Vec<String> {
        self.symbol_names()
    }

    pub fn obs_names(&self) -> Vec<&str> {
        self.obs
            .as_ref()
            .and_then(|df| df.column("cell").ok())
            .and_then(|col| col.str().ok())
            .map(|ca| (0..ca.len()).filter_map(|i| ca.get(i)).collect())
            .unwrap_or_default()
    }

    pub fn cell_names(&self) -> Vec<&str> {
        self.obs_names()
    }

    pub fn sn(&self) -> Option<String> {
        if let Some(n) = self.samples_names.first() {
            Some(n.clone())
        } else {
            self.file_path.as_ref().and_then(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
        }
    }

    pub fn add_layer(&mut self, name: &str, data: ExpMatrix) {
        self.layers.insert(name.to_string(), data);
    }

    pub fn get_layer(&self, name: &str) -> Option<&ExpMatrix> {
        self.layers.get(name)
    }

    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.keys().map(|s| s.as_str()).collect()
    }

    pub fn slice_x(
        &self,
        obs_indices: Option<&[usize]>,
        var_indices: Option<&[usize]>,
    ) -> Option<ExpMatrix> {
        let exp = self.exp_matrix.as_ref()?;

        match exp {
            ExpMatrix::Dense(arr) => {
                let rows: Array2<f32> = match obs_indices {
                    Some(ix) => arr.select(Axis(0), ix).to_owned(),
                    None => arr.clone(),
                };
                let result = match var_indices {
                    Some(ix) => rows.select(Axis(1), ix).as_standard_layout().into_owned(),
                    None => rows,
                };
                Some(ExpMatrix::Dense(result))
            }
            ExpMatrix::Sparse(csr) => {
                let rows: CsrMatrix<f32> = match obs_indices {
                    Some(ix) => csr_select_rows(csr, ix),
                    None => csr.clone(),
                };
                let result = match var_indices {
                    Some(ix) => csr_select_cols(&rows, ix),
                    None => rows,
                };
                Some(ExpMatrix::Sparse(result))
            }
        }
    }

    pub fn remove_obs_indices(&mut self, indices: &[usize]) {
        let n = self.n_obs();
        if indices.is_empty() || n == 0 {
            return;
        }
        let keep = complement_indices(n, indices);

        if let Some(ref mut exp) = self.exp_matrix {
            match exp {
                ExpMatrix::Dense(ref mut arr) => {
                    *arr = arr.select(Axis(0), &keep);
                }
                ExpMatrix::Sparse(ref mut csr) => {
                    *csr = csr_select_rows(csr, &keep);
                }
            }
        }

        if let Some(ref mut obs) = self.obs {
            *obs = filter_dataframe_rows(obs, indices);
        }

        if let Some(ref mut pos) = self.position {
            *pos = pos.select(Axis(0), &keep);
        }
    }

    pub fn remove_var_indices(&mut self, indices: &[usize]) {
        let n = self.n_vars();
        if indices.is_empty() || n == 0 {
            return;
        }
        let keep = complement_indices(n, indices);

        if let Some(ref mut exp) = self.exp_matrix {
            match exp {
                ExpMatrix::Dense(ref mut arr) => {
                    *arr = arr.select(Axis(1), &keep).as_standard_layout().into_owned();
                }
                ExpMatrix::Sparse(ref mut csr) => {
                    *csr = csr_select_cols(csr, &keep);
                }
            }
        }

        if let Some(ref mut var) = self.var {
            *var = filter_dataframe_rows(var, indices);
        }
    }

    pub fn keep_obs_indices(&mut self, indices: &[usize]) {
        let n = self.n_obs();
        let remove: Vec<usize> = complement_to_remove(n, indices);
        self.remove_obs_indices(&remove);
    }

    pub fn keep_var_indices(&mut self, indices: &[usize]) {
        let n = self.n_vars();
        let remove: Vec<usize> = complement_to_remove(n, indices);
        self.remove_var_indices(&remove);
    }

    pub fn remove_obs_by_name(&mut self, names: &HashSet<String>) {
        let indices: Vec<usize> = self
            .obs_names()
            .iter()
            .enumerate()
            .filter_map(
                |(i, name)| {
                    if names.contains(*name) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect();
        self.remove_obs_indices(&indices);
    }

    pub fn remove_var_by_name(&mut self, names: &HashSet<String>) {
        let indices: Vec<usize> = self
            .var_names()
            .iter()
            .enumerate()
            .filter_map(
                |(i, name)| {
                    if names.contains(*name) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect();
        self.remove_var_indices(&indices);
    }

    pub fn keep_obs_by_name(&mut self, names: &HashSet<String>) {
        let indices: Vec<usize> = self
            .obs_names()
            .iter()
            .enumerate()
            .filter_map(
                |(i, name)| {
                    if names.contains(*name) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect();
        self.keep_obs_indices(&indices);
    }

    pub fn keep_var_by_name(&mut self, names: &HashSet<String>) {
        let indices: Vec<usize> = self
            .var_names()
            .iter()
            .enumerate()
            .filter_map(
                |(i, name)| {
                    if names.contains(*name) {
                        Some(i)
                    } else {
                        None
                    }
                },
            )
            .collect();
        self.keep_var_indices(&indices);
    }
}

impl Default for SoData {
    fn default() -> Self {
        Self::new()
    }
}

fn complement_indices(total: usize, remove: &[usize]) -> Vec<usize> {
    let remove_set: HashSet<usize> = remove.iter().copied().collect();
    (0..total).filter(|i| !remove_set.contains(i)).collect()
}

fn complement_to_remove(total: usize, keep: &[usize]) -> Vec<usize> {
    let keep_set: HashSet<usize> = keep.iter().copied().collect();
    (0..total).filter(|i| !keep_set.contains(i)).collect()
}

fn csr_select_rows(csr: &CsrMatrix<f32>, keep: &[usize]) -> CsrMatrix<f32> {
    let offsets = csr.row_offsets();
    let cols = csr.col_indices();
    let vals = csr.values();
    let old_ncols = csr.ncols();

    let mut new_offsets = vec![0usize; keep.len() + 1];
    let mut new_cols = Vec::new();
    let mut new_vals = Vec::new();

    for (new_idx, &old_row) in keep.iter().enumerate() {
        if old_row < offsets.len().saturating_sub(1) {
            let start = offsets[old_row];
            let end = offsets[old_row + 1];
            new_cols.extend_from_slice(&cols[start..end]);
            new_vals.extend_from_slice(&vals[start..end]);
        }
        new_offsets[new_idx + 1] = new_cols.len();
    }

    CsrMatrix::try_from_csr_data(keep.len(), old_ncols, new_offsets, new_cols, new_vals)
        .expect("csr_select_rows: valid CSR")
}

fn csr_select_cols(csr: &CsrMatrix<f32>, keep: &[usize]) -> CsrMatrix<f32> {
    let offsets = csr.row_offsets();
    let cols = csr.col_indices();
    let vals = csr.values();
    let nrows = csr.nrows();
    let old_ncols = csr.ncols();

    let mut col_map = vec![-1i32; old_ncols];
    for (new, &old) in keep.iter().enumerate() {
        col_map[old] = new as i32;
    }

    let row_kept: Vec<usize> = (0..nrows)
        .into_par_iter()
        .map(|row| {
            let mut c = 0usize;
            for i in offsets[row]..offsets[row + 1] {
                if col_map[cols[i]] >= 0 {
                    c += 1;
                }
            }
            c
        })
        .collect();

    let mut new_offsets = vec![0usize; nrows + 1];
    for row in 0..nrows {
        new_offsets[row + 1] = new_offsets[row] + row_kept[row];
    }
    let nnz_new = new_offsets[nrows];

    let mut new_cols = vec![0usize; nnz_new];
    let mut new_vals = vec![0f32; nnz_new];
    let new_cols_ptr = SendPtr(new_cols.as_mut_ptr());
    let new_vals_ptr = SendPtr(new_vals.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut pos = new_offsets[row];
        for i in offsets[row]..offsets[row + 1] {
            let c = col_map[cols[i]];
            if c >= 0 {
                unsafe {
                    *new_cols_ptr.add(pos) = c as usize;
                    *new_vals_ptr.add(pos) = vals[i];
                }
                pos += 1;
            }
        }
    });

    CsrMatrix::try_from_csr_data(nrows, keep.len(), new_offsets, new_cols, new_vals)
        .expect("csr_select_cols: valid CSR")
}

fn filter_dataframe_rows(df: &DataFrame, remove_indices: &[usize]) -> DataFrame {
    let n = df.height();
    let remove_set: HashSet<usize> = remove_indices.iter().copied().collect();
    let mask: Vec<bool> = (0..n).map(|i| !remove_set.contains(&i)).collect();
    let mask_chunked = BooleanChunked::new(PlSmallStr::EMPTY, &mask);
    df.filter(&mask_chunked)
        .expect("DataFrame row filter should succeed")
}
