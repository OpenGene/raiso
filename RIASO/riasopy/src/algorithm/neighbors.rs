use std::collections::HashMap;

use anyhow::{anyhow, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::{s, Array1, Array2};
use rayon::prelude::*;

use crate::core::{NeighborsResult, SoData};

fn pairwise_euclidean(x: &Array2<f32>) -> Array2<f32> {
    let n = x.nrows();
    let d = x.ncols();

    let norms: Vec<f32> = (0..n)
        .map(|i| {
            let mut s = 0.0f32;
            for c in 0..d {
                let a = x[[i, c]];
                s += a * a;
            }
            s
        })
        .collect();

    let dists_flat: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let norms = &norms;
            (0..n).map(move |j| {
                let mut dot = 0.0f32;
                for c in 0..d {
                    dot += x[[i, c]] * x[[j, c]];
                }
                let d2 = (-2.0 * dot + norms[i] + norms[j]).max(0.0);
                d2.sqrt()
            })
        })
        .collect();

    let mut dists = Array2::from_shape_vec((n, n), dists_flat).expect("pairwise_euclidean: shape");
    for i in 0..n {
        dists[[i, i]] = 0.0;
    }
    dists
}

fn knn_including_self(dists: &Array2<f32>, n_neighbors: usize) -> (Array2<i64>, Array2<f32>) {
    let n = dists.nrows();
    let rows: Vec<(Vec<i64>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut pairs: Vec<(i64, f32)> = (0..n).map(|j| (j as i64, dists[[i, j]])).collect();
            pairs.select_nth_unstable_by(n_neighbors - 1, |a, b| {
                a.1.total_cmp(&b.1).then(a.0.cmp(&b.0))
            });
            let mut sel = pairs[..n_neighbors].to_vec();
            sel.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            (
                sel.iter().map(|&(idx, _)| idx).collect(),
                sel.iter().map(|&(_, v)| v).collect(),
            )
        })
        .collect();

    let mut idx = Array2::<i64>::zeros((n, n_neighbors));
    let mut dvals = Array2::<f32>::zeros((n, n_neighbors));
    for (i, (ri, rv)) in rows.into_iter().enumerate() {
        for k in 0..n_neighbors {
            idx[[i, k]] = ri[k];
            dvals[[i, k]] = rv[k];
        }
    }
    (idx, dvals)
}

fn smooth_knn_dist(knn_dists: &Array2<f32>, n_neighbors: usize) -> (Array1<f32>, Array1<f32>) {
    let n = knn_dists.nrows();

    let target = (n_neighbors as f32).log2();

    let mut rho = Array1::<f32>::zeros(n);
    let mut result = Array1::<f32>::zeros(n);

    let mut mean_distances: f32 = knn_dists.iter().sum::<f32>() / (n * n_neighbors) as f32;

    for i in 0..n {
        let mut min_nonzero = f32::INFINITY;
        for j in 1..n_neighbors {
            let v = knn_dists[[i, j]];
            if v > 0.0 && v < min_nonzero {
                min_nonzero = v;
            }
        }
        rho[i] = if min_nonzero == f32::INFINITY {
            0.0
        } else {
            min_nonzero
        };

        let mut lo = 0.0f32;
        let mut hi = f32::INFINITY;
        let mut mid = 1.0f32;
        for _ in 0..64 {
            let mut psum = 0.0f32;
            for j in 1..n_neighbors {
                let d = knn_dists[[i, j]] - rho[i];
                psum += if d > 0.0 { (-(d / mid)).exp() } else { 1.0 };
            }
            if (psum - target).abs() < 1e-5 {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi == f32::INFINITY {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }
        result[i] = mid;

        if rho[i] > 0.0 {
            mean_distances -= rho[i] / (n as f32);
        }
    }

    if mean_distances > 0.0 {
        result.mapv_inplace(|v| v + mean_distances);
    }

    (result, rho)
}

fn compute_membership_strengths(
    knn_indices: &Array2<i64>,
    knn_dists: &Array2<f32>,
    sigmas: &Array1<f32>,
    rhos: &Array1<f32>,
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let n = knn_indices.nrows();
    let n_neighbors = knn_indices.ncols();
    let mut rows = Vec::with_capacity(n * n_neighbors);
    let mut cols = Vec::with_capacity(n * n_neighbors);
    let mut vals = Vec::with_capacity(n * n_neighbors);
    for i in 0..n {
        for j in 0..n_neighbors {
            let idx = knn_indices[[i, j]];
            if idx < 0 {
                continue;
            }
            let val = if idx == i as i64 {
                0.0
            } else {
                let d_minus_rho = knn_dists[[i, j]] - rhos[i];
                if d_minus_rho <= 0.0 || sigmas[i] == 0.0 {
                    1.0
                } else {
                    (-(d_minus_rho / sigmas[i])).exp()
                }
            };
            rows.push(i);
            cols.push(idx as usize);
            vals.push(val);
        }
    }
    (rows, cols, vals)
}

fn entries_to_csr(n: usize, mut entries: Vec<(usize, usize, f32)>) -> CsrMatrix<f32> {
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut offsets = vec![0usize; n + 1];
    let mut cols = Vec::with_capacity(entries.len());
    let mut vals = Vec::with_capacity(entries.len());
    for &(r, c, v) in &entries {
        cols.push(c);
        vals.push(v);
        offsets[r + 1] += 1;
    }
    for r in 0..n {
        offsets[r + 1] += offsets[r];
    }
    CsrMatrix::try_from_csr_data(n, n, offsets, cols, vals).expect("entries_to_csr: valid CSR")
}

fn fuzzy_simplicial_set(n: usize, rows: &[usize], cols: &[usize], vals: &[f32]) -> CsrMatrix<f32> {
    let mut a: HashMap<(usize, usize), f32> = HashMap::new();
    for k in 0..rows.len() {
        if vals[k] != 0.0 {
            a.insert((rows[k], cols[k]), vals[k]);
        }
    }

    let mut union: HashMap<(usize, usize), f32> = HashMap::new();
    for (&(i, j), &a_ij) in &a {
        let a_ji = a.get(&(j, i)).copied().unwrap_or(0.0);
        let c = a_ij + a_ji - a_ij * a_ji;
        union.insert((i, j), c);
        union.insert((j, i), c);
    }

    let entries: Vec<(usize, usize, f32)> =
        union.into_iter().map(|((i, j), v)| (i, j, v)).collect();
    entries_to_csr(n, entries)
}

fn nn_dist_csr(knn_indices: &Array2<i64>, knn_dists: &Array2<f32>) -> CsrMatrix<f32> {
    let n = knn_indices.nrows();
    let n_neighbors = knn_indices.ncols();
    let mut entries: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..n {
        for j in 0..n_neighbors {
            let idx = knn_indices[[i, j]];
            if idx < 0 || idx == i as i64 {
                continue;
            }
            entries.push((i, idx as usize, knn_dists[[i, j]]));
        }
    }
    entries_to_csr(n, entries)
}

pub fn neighbors(x: &Array2<f32>, n_neighbors: usize) -> Result<NeighborsResult> {
    let n = x.nrows();
    if n == 0 {
        return Err(anyhow!("neighbors: empty input (0 samples)"));
    }
    if n_neighbors == 0 || n_neighbors > n {
        return Err(anyhow!(
            "neighbors: n_neighbors ({}) must be in 1..=n_samples ({})",
            n_neighbors,
            n
        ));
    }

    let (knn_indices, knn_dists) = if n > super::knn::DENSE_KNN_THRESHOLD {
        super::knn::nn_descent_knn(x, n_neighbors)
    } else {
        let dists = pairwise_euclidean(x);
        knn_including_self(&dists, n_neighbors)
    };
    let (sigmas, rhos) = smooth_knn_dist(&knn_dists, n_neighbors);
    let (rows, cols, vals) = compute_membership_strengths(&knn_indices, &knn_dists, &sigmas, &rhos);
    let connectivities = fuzzy_simplicial_set(n, &rows, &cols, &vals);
    let distances = nn_dist_csr(&knn_indices, &knn_dists);

    Ok(NeighborsResult {
        knn_indices,
        connectivities,
        distances,
    })
}

pub fn run_neighbors(data: &mut SoData, n_neighbors: usize, n_pcs: Option<usize>) {
    data.result.neighbors = None;

    let z: &Array2<f32> = if let Some(b) = &data.result.batch {
        &b.corrected
    } else if let Some(p) = &data.result.pca {
        &p.projected
    } else {
        return;
    };
    let k = z.ncols();
    let n_pcs = n_pcs.unwrap_or(k);

    if n_pcs > k {
        return;
    }

    let x: Array2<f32> = z.slice(s![.., ..n_pcs]).to_owned();

    if let Ok(result) = neighbors(&x, n_neighbors) {
        data.result.neighbors = Some(result);
    }
}
