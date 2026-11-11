use std::collections::HashMap;

use anyhow::{anyhow, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use polars::prelude::*;

use crate::algorithm::qc::upsert_column;
use crate::algorithm::rng::SplitMix64;
use crate::core::SoData;

fn build_adj_dense(arr: &Array2<f32>, n: usize) -> (Vec<Vec<(usize, f32)>>, Vec<f32>) {
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    let mut deg = vec![0.0f32; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let w = arr[[i, j]];
            if w > 0.0 {
                adj[i].push((j, w));
                adj[j].push((i, w));
                deg[i] += w;
                deg[j] += w;
            }
        }
    }
    (adj, deg)
}

fn build_adj_csr(csr: &CsrMatrix<f32>, n: usize) -> (Vec<Vec<(usize, f32)>>, Vec<f32>) {
    let offsets = csr.row_offsets();
    let cols = csr.col_indices();
    let vals = csr.values();
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    let mut deg = vec![0.0f32; n];
    for i in 0..n {
        for p in offsets[i]..offsets[i + 1] {
            let j = cols[p];
            let w = vals[p];
            if j > i && w > 0.0 {
                adj[i].push((j, w));
                adj[j].push((i, w));
                deg[i] += w;
                deg[j] += w;
            }
        }
    }
    (adj, deg)
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);

    for i in (1..n).rev() {
        let r = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, r);
    }
    order
}

fn local_moving(
    adj: &[Vec<(usize, f32)>],
    deg: &[f32],
    gamma: f32,
    m: f32,
    seed: u64,
) -> (Vec<usize>, bool) {
    let n = deg.len();
    let mut comm: Vec<usize> = (0..n).collect();
    let mut tot: Vec<f32> = deg.to_vec();
    let order = shuffled(n, seed);

    let mut acc = vec![0.0f32; n];
    let mut tag = vec![0u32; n];
    let mut cur_tag: u32 = 1;
    let mut overall = false;

    loop {
        let mut round_moved = false;
        for &i in &order {
            let d = comm[i];
            let mut touched: Vec<usize> = Vec::new();
            for &(j, w) in &adj[i] {
                let cj = comm[j];
                if tag[cj] != cur_tag {
                    tag[cj] = cur_tag;
                    acc[cj] = 0.0;
                    touched.push(cj);
                }
                acc[cj] += w;
            }
            cur_tag = cur_tag.wrapping_add(1);
            if cur_tag == 0 {
                tag.iter_mut().for_each(|t| *t = 0);
                cur_tag = 1;
            }

            let mut best = d;
            let mut best_delta = f32::NEG_INFINITY;
            for &c in &touched {
                if c == d {
                    continue;
                }
                let delta =
                    acc[c] - acc[d] - gamma * deg[i] * (tot[c] - tot[d] + deg[i]) / (2.0 * m);
                if delta > best_delta || (delta == best_delta && c < best) {
                    best_delta = delta;
                    best = c;
                }
            }
            if best != d && best_delta > 0.0 {
                tot[d] -= deg[i];
                tot[best] += deg[i];
                comm[i] = best;
                round_moved = true;
            }
        }
        if !round_moved {
            break;
        }
        overall = true;
    }
    (comm, overall)
}

fn louvain_core(
    mut adj: Vec<Vec<(usize, f32)>>,
    mut deg: Vec<f32>,
    mut owner: Vec<usize>,
    gamma: f32,
    seed: u64,
) -> Vec<i64> {
    loop {
        let n = adj.len();
        if n <= 1 {
            break;
        }
        let m: f32 = deg.iter().sum::<f32>() * 0.5;
        if m <= 0.0 {
            break;
        }
        let (comm, changed) = local_moving(&adj, &deg, gamma, m, seed);
        if !changed {
            break;
        }

        let mut map: Vec<usize> = vec![usize::MAX; n];
        let mut new_id: Vec<usize> = vec![0; n];
        let mut k = 0usize;
        for (i, &c) in comm.iter().enumerate() {
            if map[c] == usize::MAX {
                map[c] = k;
                k += 1;
            }
            new_id[i] = map[c];
        }

        let mut new_deg = vec![0.0f32; k];
        for (i, &c) in new_id.iter().enumerate() {
            new_deg[c] += deg[i];
        }

        let mut new_owner = vec![0usize; owner.len()];
        for (orig, &cur) in owner.iter().enumerate() {
            new_owner[orig] = new_id[cur];
        }

        let mut emap: HashMap<(usize, usize), f32> = HashMap::new();
        for (u, nbrs) in adj.iter().enumerate() {
            let cu = new_id[u];
            for &(v, w) in nbrs {
                if v < u {
                    continue;
                }
                let cv = new_id[v];
                if cu != cv {
                    let key = if cu < cv { (cu, cv) } else { (cv, cu) };
                    *emap.entry(key).or_insert(0.0) += w;
                }
            }
        }
        let mut new_adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); k];
        for ((a, b), w) in emap {
            new_adj[a].push((b, w));
            new_adj[b].push((a, w));
        }

        adj = new_adj;
        deg = new_deg;
        owner = new_owner;
    }
    owner.iter().map(|&c| c as i64).collect()
}

pub fn louvain(
    connectivities: &Array2<f32>,
    resolution: f32,
    random_state: u64,
) -> Result<Vec<i64>> {
    let (n, m) = connectivities.dim();
    if n != m {
        return Err(anyhow!(
            "louvain: connectivities must be square (N×N), got {n}×{m}"
        ));
    }
    if n == 0 {
        return Ok(vec![]);
    }
    let (adj, deg) = build_adj_dense(connectivities, n);
    let owner: Vec<usize> = (0..n).collect();
    Ok(louvain_core(adj, deg, owner, resolution, random_state))
}

pub fn louvain_from_csr(
    connectivities: &CsrMatrix<f32>,
    resolution: f32,
    random_state: u64,
) -> Result<Vec<i64>> {
    let (n, m) = (connectivities.nrows(), connectivities.ncols());
    if n != m {
        return Err(anyhow!(
            "louvain: connectivities must be square (N×N), got {n}×{m}"
        ));
    }
    if n == 0 {
        return Ok(vec![]);
    }
    let (adj, deg) = build_adj_csr(connectivities, n);
    let owner: Vec<usize> = (0..n).collect();
    Ok(louvain_core(adj, deg, owner, resolution, random_state))
}

pub fn run_louvain(data: &mut SoData, resolution: f32, random_state: u64) {
    let conn = match data.result.neighbors.as_ref() {
        Some(n) => &n.connectivities,

        None => return,
    };

    let labels = match louvain_from_csr(conn, resolution, random_state) {
        Ok(l) => l,
        Err(_) => return,
    };

    if let Some(obs) = data.obs.as_mut() {
        upsert_column(obs, Column::new("louvain".into(), labels));
    }
}
