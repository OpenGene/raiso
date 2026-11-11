use crate::algorithm::rng::SplitMix64;
use ndarray::Array2;
use rayon::prelude::*;
use std::cmp::Ordering;

pub const DENSE_KNN_THRESHOLD: usize = 8192;

const MAX_ITERS: usize = 30;

const DELTA: f32 = 1e-3;

const MIN_RP_TREES: usize = 8;

fn n_rp_trees(n: usize) -> usize {
    let t = 5.0 + (n as f32).sqrt() / 20.0;
    (t.round() as usize).clamp(MIN_RP_TREES, 64)
}

const LEAF_SIZE: usize = 15;

const MAX_CANDIDATES: usize = 40;

const SEED: u64 = 0;

#[inline]
fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut z = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn sqdist_rows(xs: &[f32], a: usize, b: usize, d: usize) -> f32 {
    let xa = &xs[a..a + d];
    let xb = &xs[b..b + d];
    let (mut s0, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
    let mut c = 0usize;
    while c + 4 <= d {
        let d0 = xa[c] - xb[c];
        let d1 = xa[c + 1] - xb[c + 1];
        let d2 = xa[c + 2] - xb[c + 2];
        let d3 = xa[c + 3] - xb[c + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        c += 4;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    while c < d {
        let delta = xa[c] - xb[c];
        s += delta * delta;
        c += 1;
    }
    s
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn prefetch_row(xs: &[f32], idx: usize, d: usize) {
    let p = unsafe { xs.as_ptr().add(idx * d) } as *const i8;
    let bytes = d * std::mem::size_of::<f32>();
    let mut off = 0usize;
    while off < bytes {
        unsafe {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(p.add(off));
        }
        off += 64;
    }
}

#[inline(always)]
#[cfg(not(target_arch = "x86_64"))]
fn prefetch_row(_xs: &[f32], _idx: usize, _d: usize) {}

#[inline]
fn entry_cmp(da: f32, ia: u32, db: f32, ib: u32, self_i: u32) -> Ordering {
    da.total_cmp(&db)
        .then_with(|| (ia != self_i).cmp(&(ib != self_i)))
        .then_with(|| ia.cmp(&ib))
}

#[inline]
fn insert_sorted(
    idx: &mut [u32],
    dist: &mut [f32],
    k: usize,
    fill: &mut usize,
    j: u32,
    d2: f32,
    self_i: u32,
) {
    if *fill == k && entry_cmp(d2, j, dist[k - 1], idx[k - 1], self_i) != Ordering::Less {
        return;
    }
    let mut pos = *fill;
    while pos > 0 {
        let p = pos - 1;
        if entry_cmp(dist[p], idx[p], d2, j, self_i) == Ordering::Greater {
            pos = p;
        } else {
            break;
        }
    }
    let end = if *fill < k { *fill } else { k - 1 };
    for p in (pos..end).rev() {
        idx[p + 1] = idx[p];
        dist[p + 1] = dist[p];
    }
    idx[pos] = j;
    dist[pos] = d2;
    if *fill < k {
        *fill += 1;
    }
}

struct RpTree {
    leaves: Vec<Vec<u32>>,
}

pub fn nn_descent_knn(x: &Array2<f32>, k: usize) -> (Array2<i64>, Array2<f32>) {
    let n = x.nrows();
    debug_assert!(k <= n, "nn_descent_knn: k={k} > n={n}");
    if k >= n {
        return exact_all(x, k);
    }

    let d = x.ncols();
    let x_std = x.as_standard_layout();
    let xs = x_std
        .as_slice()
        .expect("nn_descent_knn: input must be convertible to standard layout");

    let max_candidates = MAX_CANDIDATES.max(2 * k);
    let reverse_budget = max_candidates * 3 / 4;
    let random_budget = max_candidates - reverse_budget;
    let init_random_sample = k.max(8);
    let max_iters = ((n as f32).log2().round() as usize).clamp(10, MAX_ITERS);
    let n_trees = n_rp_trees(n);

    let forest = build_forest(xs, n, d, n_trees, LEAF_SIZE);

    let (mut nn_idx, mut nn_dist) = init_graph(xs, d, n, k, &forest, init_random_sample);

    for iter in 1..=max_iters {
        let mut next_idx = vec![0u32; n * k];
        let mut next_dist = vec![f32::INFINITY; n * k];

        next_idx
            .par_chunks_mut(k)
            .zip(next_dist.par_chunks_mut(k))
            .enumerate()
            .map_init(
                || (vec![0u64; n], 0u64),
                |(seen, gen), (i, (nidx, ndist))| {
                    *gen += 1;
                    let g = *gen;
                    let mut fill = 0usize;
                    let ib = i * d;

                    seen[i] = g;
                    insert_sorted(nidx, ndist, k, &mut fill, i as u32, 0.0, i as u32);

                    let row_start = i * k;
                    for c in 0..k {
                        let j = nn_idx[row_start + c];
                        let ju = j as usize;
                        if seen[ju] == g {
                            continue;
                        }
                        seen[ju] = g;
                        let d2 = nn_dist[row_start + c];
                        insert_sorted(nidx, ndist, k, &mut fill, j, d2, i as u32);
                    }

                    let mut used = 0usize;
                    'rev: for c in 0..k {
                        let j = nn_idx[row_start + c] as usize;
                        let jrow = j * k;
                        for c2 in 0..k {
                            if used >= reverse_budget {
                                break 'rev;
                            }
                            if c2 + 2 < k {
                                prefetch_row(xs, nn_idx[jrow + c2 + 2] as usize, d);
                            }
                            let q = nn_idx[jrow + c2];
                            let qu = q as usize;
                            if seen[qu] == g {
                                continue;
                            }
                            seen[qu] = g;
                            used += 1;
                            let d2 = sqdist_rows(xs, ib, qu * d, d);
                            insert_sorted(nidx, ndist, k, &mut fill, q, d2, i as u32);
                        }
                    }

                    let mut r = SplitMix64::new(mix(SEED, i as u64, iter as u64));
                    let mut q_next = r.next_index(n) as u32;
                    prefetch_row(xs, q_next as usize, d);
                    for _ in 0..random_budget {
                        let q = q_next;
                        let qu = q as usize;
                        q_next = r.next_index(n) as u32;
                        prefetch_row(xs, q_next as usize, d);
                        if seen[qu] == g {
                            continue;
                        }
                        seen[qu] = g;
                        let d2 = sqdist_rows(xs, ib, qu * d, d);
                        insert_sorted(nidx, ndist, k, &mut fill, q, d2, i as u32);
                    }

                    let mut r2 = SplitMix64::new(mix(SEED ^ 0x5DEE_CE66D, i as u64, iter as u64));
                    let mut guard = 0usize;
                    while fill < k && guard < 2 * n {
                        let q = r2.next_index(n) as u32;
                        let qu = q as usize;
                        guard += 1;
                        if seen[qu] == g {
                            continue;
                        }
                        seen[qu] = g;
                        let d2 = sqdist_rows(xs, ib, qu * d, d);
                        insert_sorted(nidx, ndist, k, &mut fill, q, d2, i as u32);
                    }
                },
            )
            .for_each(|_| {});

        let n_changed: usize = next_idx
            .par_chunks(k)
            .zip(nn_idx.par_chunks(k))
            .map(|(b, a)| a.iter().zip(b.iter()).filter(|(x, y)| x != y).count())
            .sum();

        nn_idx = next_idx;
        nn_dist = next_dist;

        let frac = n_changed as f32 / (n * k) as f32;
        if frac <= DELTA {
            break;
        }
    }

    let mut idx = Array2::<i64>::zeros((n, k));
    let mut dists = Array2::<f32>::zeros((n, k));
    for i in 0..n {
        for c in 0..k {
            let j = nn_idx[i * k + c];
            idx[[i, c]] = j as i64;
            dists[[i, c]] = nn_dist[i * k + c].sqrt();
        }
    }
    (idx, dists)
}

fn init_graph(
    xs: &[f32],
    d: usize,
    n: usize,
    k: usize,
    forest: &[RpTree],
    init_random_sample: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut out_idx = vec![u32::MAX; n * k];
    let mut out_dist = vec![f32::INFINITY; n * k];
    let mut fill = vec![0usize; n];
    for i in 0..n {
        out_idx[i * k] = i as u32;
        out_dist[i * k] = 0.0;
        fill[i] = 1;
    }

    for tree in forest {
        let idx_p = out_idx.as_mut_ptr() as usize;
        let dist_p = out_dist.as_mut_ptr() as usize;
        let fill_p = fill.as_mut_ptr() as usize;
        tree.leaves.par_iter().for_each(|leaf| {
            let idxp = idx_p as *mut u32;
            let distp = dist_p as *mut f32;
            let fillp = fill_p as *mut usize;
            for a in 0..leaf.len() {
                let i = leaf[a] as usize;
                let ib = i * d;
                let base = i * k;
                for &j in &leaf[a + 1..] {
                    let ju = j as usize;
                    let jbase = ju * k;
                    unsafe {
                        let fi = *fillp.add(i);
                        let fj = *fillp.add(ju);
                        let mut has_ij = false;
                        for c in 0..fi {
                            if *idxp.add(base + c) == j {
                                has_ij = true;
                                break;
                            }
                        }
                        let mut has_ji = false;
                        for c in 0..fj {
                            if *idxp.add(jbase + c) == i as u32 {
                                has_ji = true;
                                break;
                            }
                        }
                        if has_ij && has_ji {
                            continue;
                        }
                        let d2 = sqdist_rows(xs, ib, ju * d, d);
                        if !has_ij {
                            let mut f = fi;
                            let ri = std::slice::from_raw_parts_mut(idxp.add(base), k);
                            let rd = std::slice::from_raw_parts_mut(distp.add(base), k);
                            insert_sorted(ri, rd, k, &mut f, j, d2, i as u32);
                            *fillp.add(i) = f;
                        }
                        if !has_ji {
                            let mut f = fj;
                            let ri = std::slice::from_raw_parts_mut(idxp.add(jbase), k);
                            let rd = std::slice::from_raw_parts_mut(distp.add(jbase), k);
                            insert_sorted(ri, rd, k, &mut f, i as u32, d2, ju as u32);
                            *fillp.add(ju) = f;
                        }
                    }
                }
            }
        });
    }

    out_idx
        .par_chunks_mut(k)
        .zip(out_dist.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, (idx_row, dist_row))| {
            let mut f = fill[i];
            let mut r = SplitMix64::new(mix(SEED, i as u64, 0));
            for _ in 0..init_random_sample {
                let j = r.next_index(n) as u32;
                if j as usize == i || idx_row[..f].contains(&j) {
                    continue;
                }
                let d2 = sqdist_rows(xs, i * d, j as usize * d, d);
                insert_sorted(idx_row, dist_row, k, &mut f, j, d2, i as u32);
            }
            if f >= k {
                return;
            }
            let mut r2 = SplitMix64::new(mix(SEED ^ 0x5DEE_CE66D, i as u64, 0));
            let mut guard = 0usize;
            while f < k && guard < 2 * n {
                let j = r2.next_index(n) as u32;
                guard += 1;
                if j as usize == i || idx_row[..f].contains(&j) {
                    continue;
                }
                let d2 = sqdist_rows(xs, i * d, j as usize * d, d);
                insert_sorted(idx_row, dist_row, k, &mut f, j, d2, i as u32);
            }
        });

    (out_idx, out_dist)
}

fn exact_all(x: &Array2<f32>, k: usize) -> (Array2<i64>, Array2<f32>) {
    let n = x.nrows();
    let d = x.ncols();
    let x_std = x.as_standard_layout();
    let xs = x_std
        .as_slice()
        .expect("exact_all: input must be convertible to standard layout");

    let mut out_idx = vec![0i64; n * k];
    let mut out_dist = vec![0.0; n * k];

    out_idx
        .par_chunks_mut(k)
        .zip(out_dist.par_chunks_mut(k))
        .enumerate()
        .for_each(|(i, (ri, rd))| {
            let ib = i * d;
            let mut entries: Vec<(f32, u32)> = Vec::with_capacity(n);
            for j in 0..n {
                let dj = if j == i {
                    0.0
                } else {
                    sqdist_rows(xs, ib, j * d, d)
                };
                entries.push((dj, j as u32));
            }
            entries.sort_by(|a, b| entry_cmp(a.0, a.1, b.0, b.1, i as u32));
            for c in 0..k {
                ri[c] = entries[c].1 as i64;
                rd[c] = entries[c].0.sqrt();
            }
        });

    let idx = Array2::from_shape_vec((n, k), out_idx).expect("exact_all: idx shape");
    let dists = Array2::from_shape_vec((n, k), out_dist).expect("exact_all: dists shape");
    (idx, dists)
}

fn build_forest(xs: &[f32], n: usize, d: usize, n_trees: usize, leaf_size: usize) -> Vec<RpTree> {
    let all: Vec<u32> = (0..n as u32).collect();
    (0..n_trees)
        .into_par_iter()
        .map(|t| {
            let mut rng = SplitMix64::new(mix(SEED, t as u64, 0xA5A5_5A5A_C3C3_3C3C));
            let mut leaves: Vec<Vec<u32>> = Vec::new();
            build_tree(xs, d, &all, leaf_size, &mut rng, &mut leaves);
            RpTree { leaves }
        })
        .collect()
}

fn build_tree(
    xs: &[f32],
    d: usize,
    node: &[u32],
    leaf_size: usize,
    rng: &mut SplitMix64,
    leaves: &mut Vec<Vec<u32>>,
) {
    if node.len() <= leaf_size {
        leaves.push(node.to_vec());
        return;
    }
    let (left, right) = hyperplane_split(xs, d, node, rng);
    build_tree(xs, d, &left, leaf_size, rng, leaves);
    build_tree(xs, d, &right, leaf_size, rng, leaves);
}

fn hyperplane_split(
    xs: &[f32],
    d: usize,
    node: &[u32],
    rng: &mut SplitMix64,
) -> (Vec<u32>, Vec<u32>) {
    let a = node[rng.next_index(node.len())] as usize;
    let mut b = node[rng.next_index(node.len())] as usize;
    let mut guard = 0;
    while b == a && guard < 8 {
        b = node[rng.next_index(node.len())] as usize;
        guard += 1;
    }
    if b == a {
        let mid = node.len() / 2;
        return (node[..mid].to_vec(), node[mid..].to_vec());
    }

    let ab = &xs[a * d..a * d + d];
    let bb = &xs[b * d..b * d + d];
    let mut proj: Vec<(f32, u32)> = Vec::with_capacity(node.len());
    for &p in node {
        let pb = &xs[p as usize * d..p as usize * d + d];
        let mut acc = [0.0; 4];
        let mut c = 0usize;
        while c + 4 <= d {
            acc[0] += (pb[c] - ab[c]) * (bb[c] - ab[c]);
            acc[1] += (pb[c + 1] - ab[c + 1]) * (bb[c + 1] - ab[c + 1]);
            acc[2] += (pb[c + 2] - ab[c + 2]) * (bb[c + 2] - ab[c + 2]);
            acc[3] += (pb[c + 3] - ab[c + 3]) * (bb[c + 3] - ab[c + 3]);
            c += 4;
        }
        while c < d {
            acc[0] += (pb[c] - ab[c]) * (bb[c] - ab[c]);
            c += 1;
        }
        proj.push((acc[0] + acc[1] + acc[2] + acc[3], p));
    }

    let mid = node.len() / 2;
    proj.select_nth_unstable_by(mid, |x, y| x.0.total_cmp(&y.0));
    let mut left = Vec::with_capacity(mid);
    let mut right = Vec::with_capacity(node.len() - mid);
    for &(_, p) in &proj[..mid] {
        left.push(p);
    }
    for &(_, p) in &proj[mid..] {
        right.push(p);
    }
    (left, right)
}
