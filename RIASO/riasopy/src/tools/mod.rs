use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{concatenate, s, Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;

use crate::parallel::SendPtr;

use crate::core::{ExpMatrix, SoData};

pub fn integrate(samples: &[SoData], save_counts_layer: bool) -> Result<SoData> {
    integrate_impl(samples, "intersect", save_counts_layer)
}

pub fn integrate_union(samples: &[SoData], save_counts_layer: bool) -> Result<SoData> {
    integrate_impl(samples, "union", save_counts_layer)
}

fn integrate_impl(samples: &[SoData], var_type: &str, save_counts_layer: bool) -> Result<SoData> {
    if samples.is_empty() {
        bail!("integrate: no sample");
    }
    if samples.len() == 1 {
        let mut merged = samples[0].clone();

        if let Some(mut obs) = merged.obs.clone() {
            if obs.column("batch").is_err() {
                let batch = Column::new("batch".into(), vec![0i64; obs.height()]);
                if obs.with_column(batch).is_ok() {
                    merged.obs = Some(obs);
                }
            }
        }

        merged.layers.clear();
        if save_counts_layer {
            if let Some(ref exp) = merged.exp_matrix {
                merged.layers.insert("counts".into(), exp.clone());
            }
        }

        if merged.samples_names.is_empty() {
            merged.samples_names = vec!["sample0".to_string()];
        }
        return Ok(merged);
    }

    let target_genes = if var_type == "union" {
        all_genes(samples)
    } else {
        common_genes(samples)
    };
    if target_genes.is_empty() {
        bail!("integrate: the target gene set is empty (no shared genes)");
    }

    let gene_order: HashMap<String, usize> = target_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    let n_vars = target_genes.len();

    let all_dense = samples
        .iter()
        .all(|s| matches!(&s.exp_matrix, Some(ExpMatrix::Dense(_))));

    let mut total_n_obs = 0;
    let mut cell_names: Vec<String> = Vec::new();
    let mut batch_ids: Vec<i64> = Vec::new();
    let mut positions: Vec<Array2<f64>> = Vec::new();

    let mut offsets: Vec<usize> = Vec::new();
    let mut cols: Vec<usize> = Vec::new();
    let mut vals: Vec<f32> = Vec::new();

    let mut dense_slices: Vec<(usize, usize, Array2<f32>)> = Vec::new();

    if !all_dense {
        offsets.push(0usize);
        let est_nnz: usize = samples
            .iter()
            .map(|sd| match &sd.exp_matrix {
                Some(ExpMatrix::Sparse(csr)) => csr.values().len(),
                Some(ExpMatrix::Dense(arr)) => arr.len(),
                None => 0,
            })
            .sum();
        cols.reserve(est_nnz);
        vals.reserve(est_nnz);
    }

    const NCOLS: usize = 3;
    const GAP: f64 = 100.0;

    let (cell_w, cell_h) = {
        let (mut cw, mut ch) = (0.0f64, 0.0f64);
        for s in samples {
            if let Some(pos) = &s.position {
                if pos.nrows() == 0 {
                    continue;
                }
                let min_x = pos.column(0).iter().copied().fold(f64::INFINITY, f64::min);
                let max_x = pos
                    .column(0)
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let min_y = pos.column(1).iter().copied().fold(f64::INFINITY, f64::min);
                let max_y = pos
                    .column(1)
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                cw = cw.max(max_x - min_x);
                ch = ch.max(max_y - min_y);
            }
        }
        (cw, ch)
    };

    for (batch, sample) in samples.iter().enumerate() {
        let n_obs = sample.n_obs();
        if n_obs == 0 {
            continue;
        }

        if all_dense {
            if let Some(ExpMatrix::Dense(arr)) = &sample.exp_matrix {
                let sub = align_dense_block(arr, &sample.var_names(), &gene_order, n_vars);
                let start = total_n_obs;
                dense_slices.push((start, start + n_obs, sub));
            }
        } else {
            match &sample.exp_matrix {
                Some(ExpMatrix::Sparse(csr)) => {
                    let mapping = build_mapping(&sample.var_names(), &gene_order);
                    let mono = is_monotonic(&mapping);
                    csr_align_append(csr, &mapping, mono, &mut offsets, &mut cols, &mut vals);
                }
                Some(ExpMatrix::Dense(arr)) => {
                    let sub = align_dense_block(arr, &sample.var_names(), &gene_order, n_vars);
                    dense_emit_nonzeros(&sub, &mut offsets, &mut cols, &mut vals);
                }
                None => continue,
            }
        }

        let cells = sample.obs_names();
        if cells.len() == n_obs {
            cell_names.extend(cells.iter().map(|s| s.to_string()));
        } else {
            for i in 0..n_obs {
                cell_names.push(format!("batch{}_{}", batch, i));
            }
        }

        total_n_obs += n_obs;

        batch_ids.extend((0..n_obs).map(|_| batch as i64));

        let pos_data = sample
            .position
            .clone()
            .unwrap_or_else(|| Array2::<f64>::zeros((n_obs, 2)));
        let min_x = pos_data
            .column(0)
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let min_y = pos_data
            .column(1)
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let row = batch / NCOLS;
        let col = batch % NCOLS;
        let x_off = col as f64 * (cell_w + GAP);
        let y_off = row as f64 * (cell_h + GAP);
        let mut p = pos_data;
        p.column_mut(0).mapv_inplace(|v| v - min_x + x_off);
        p.column_mut(1).mapv_inplace(|v| v - min_y + y_off);
        positions.push(p);
    }

    if total_n_obs == 0 {
        bail!("integrate: no valid observation");
    }

    let merged_exp = if all_dense {
        let mut merged_x = Array2::<f32>::zeros((total_n_obs, n_vars));
        for (start, end, sub) in dense_slices {
            merged_x.slice_mut(s![start..end, ..]).assign(&sub);
        }
        ExpMatrix::Dense(merged_x)
    } else {
        let csr = CsrMatrix::try_from_csr_data(total_n_obs, n_vars, offsets, cols, vals)
            .map_err(|e| anyhow!("integrate: failed to build the merged CSR: {e:?}"))?;
        ExpMatrix::Sparse(csr)
    };

    let merged_pos = if positions.is_empty() {
        None
    } else {
        Some(
            concatenate(
                Axis(0),
                &positions.iter().map(|p| p.view()).collect::<Vec<_>>(),
            )
            .map_err(|e| anyhow!("integrate: failed to concatenate position: {e:?}"))?,
        )
    };

    let obs_df = DataFrame::new(vec![
        Column::new("cell".into(), cell_names),
        Column::new("batch".into(), batch_ids),
    ])
    .map_err(|e| anyhow!("integrate: failed to build obs: {e:?}"))?;

    let mut gene_symbol: HashMap<String, String> = HashMap::new();
    for sample in samples {
        if let Some(var) = sample.var.as_ref() {
            if let (Ok(gene_col), Ok(name_col)) = (var.column("gene"), var.column("gene_name")) {
                if let (Ok(gca), Ok(nca)) = (gene_col.str(), name_col.str()) {
                    for i in 0..gca.len().min(nca.len()) {
                        if let (Some(gid), Some(sym)) = (gca.get(i), nca.get(i)) {
                            if !sym.is_empty() {
                                gene_symbol.insert(gid.to_string(), sym.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    let symbol_col: Vec<String> = target_genes
        .iter()
        .map(|g| gene_symbol.get(g).cloned().unwrap_or_else(|| g.clone()))
        .collect();
    let var_df = DataFrame::new(vec![
        Column::new("gene".into(), target_genes),
        Column::new("gene_name".into(), symbol_col),
    ])
    .map_err(|e| anyhow!("integrate: failed to build var: {e:?}"))?;

    let mut merged = SoData::new();

    if save_counts_layer {
        merged.layers.insert("counts".into(), merged_exp.clone());
    }
    merged.exp_matrix = Some(merged_exp);
    merged.obs = Some(obs_df);
    merged.var = Some(var_df);
    merged.position = merged_pos;

    let mut merged_names: Vec<String> = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        if s.samples_names.is_empty() {
            merged_names.push(format!("sample{i}"));
        } else {
            merged_names.extend(s.samples_names.iter().cloned());
        }
    }
    merged.samples_names = merged_names;

    Ok(merged)
}

fn common_genes(samples: &[SoData]) -> Vec<String> {
    let mut iter = samples.iter().map(|sd| {
        let genes: HashSet<&str> = sd.var_names().into_iter().collect();
        genes
    });
    let first = match iter.next() {
        Some(f) => f,
        None => return Vec::new(),
    };
    let common: HashSet<&str> = iter.fold(first, |acc, g| acc.intersection(&g).copied().collect());
    let mut v: Vec<String> = common.into_iter().map(|x| x.to_string()).collect();
    v.sort();
    v
}

fn all_genes(samples: &[SoData]) -> Vec<String> {
    let union: HashSet<&str> = samples.iter().flat_map(|sd| sd.var_names()).collect();
    let mut v: Vec<String> = union.into_iter().map(|x| x.to_string()).collect();
    v.sort();
    v
}

fn build_mapping(var_names: &[&str], gene_order: &HashMap<String, usize>) -> Vec<Option<usize>> {
    var_names
        .iter()
        .map(|g| gene_order.get(*g).copied())
        .collect()
}

fn is_monotonic(map: &[Option<usize>]) -> bool {
    let mut prev: Option<usize> = None;
    for &m in map {
        if let Some(c) = m {
            if let Some(p) = prev {
                if c < p {
                    return false;
                }
            }
            prev = Some(c);
        }
    }
    true
}

fn csr_align_append(
    csr: &CsrMatrix<f32>,
    old_to_new: &[Option<usize>],
    monotonic: bool,
    offsets: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<f32>,
) {
    let src_offsets = csr.row_offsets();
    let src_cols = csr.col_indices();
    let src_vals = csr.values();
    let nrows = csr.nrows();
    let start_len = cols.len();

    let mut counts = vec![0usize; nrows];
    let cnt_ptr = SendPtr(counts.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut cnt = 0usize;
        for i in src_offsets[row]..src_offsets[row + 1] {
            if old_to_new[src_cols[i]].is_some() {
                cnt += 1;
            }
        }
        unsafe {
            *cnt_ptr.add(row) = cnt;
        }
    });

    let mut pref = vec![0usize; nrows + 1];
    pref[0] = start_len;
    let mut acc = start_len;
    for (r, &c) in counts.iter().enumerate() {
        acc += c;
        pref[r + 1] = acc;
    }
    cols.resize(acc, 0);
    vals.resize(acc, 0.0);
    offsets.extend_from_slice(&pref[1..]);

    let cols_ptr = SendPtr(cols.as_mut_ptr());
    let vals_ptr = SendPtr(vals.as_mut_ptr());
    (0..nrows).into_par_iter().for_each(|row| {
        let mut k = pref[row];
        if monotonic {
            for i in src_offsets[row]..src_offsets[row + 1] {
                if let Some(nc) = old_to_new[src_cols[i]] {
                    unsafe {
                        *cols_ptr.add(k) = nc;
                        *vals_ptr.add(k) = src_vals[i];
                    }
                    k += 1;
                }
            }
        } else {
            let mut buf = Vec::with_capacity(src_offsets[row + 1] - src_offsets[row]);
            for i in src_offsets[row]..src_offsets[row + 1] {
                if let Some(nc) = old_to_new[src_cols[i]] {
                    buf.push((nc, src_vals[i]));
                }
            }
            buf.sort_unstable_by_key(|&(c, _)| c);
            for &(c, v) in buf.iter() {
                unsafe {
                    *cols_ptr.add(k) = c;
                    *vals_ptr.add(k) = v;
                }
                k += 1;
            }
        }
        debug_assert_eq!(k, pref[row + 1]);
    });
}

fn align_dense_block(
    arr: &Array2<f32>,
    var_names: &[&str],
    gene_order: &HashMap<String, usize>,
    n_vars: usize,
) -> Array2<f32> {
    let n_obs = arr.nrows();
    let mut sub = Array2::<f32>::zeros((n_obs, n_vars));
    for old_col in 0..arr.ncols() {
        if let Some(&gene) = var_names.get(old_col) {
            if let Some(&new_col) = gene_order.get(gene) {
                sub.column_mut(new_col).assign(&arr.column(old_col));
            }
        }
    }
    sub
}

fn dense_emit_nonzeros(
    arr: &Array2<f32>,
    offsets: &mut Vec<usize>,
    cols: &mut Vec<usize>,
    vals: &mut Vec<f32>,
) {
    for row in arr.rows() {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                cols.push(j);
                vals.push(v);
            }
        }
        offsets.push(cols.len());
    }
}
