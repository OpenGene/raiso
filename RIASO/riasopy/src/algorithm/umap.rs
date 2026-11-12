use anyhow::{anyhow, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::core::{SoData, UmapResult};

mod rng {
    use crate::algorithm::rng::SplitMix64;
    pub struct SeededRng {
        inner: SplitMix64,
    }

    impl SeededRng {
        pub fn new(seed: u64) -> Self {
            Self {
                inner: SplitMix64::new(seed),
            }
        }

        #[inline]
        fn next_f32(&mut self) -> f32 {
            (self.inner.next_u64() >> 40) as f32 / (1u32 << 24) as f32
        }

        pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
            low + (high - low) * self.next_f32()
        }

        pub fn next_index(&mut self, n: usize) -> usize {
            (self.inner.next_u64() as u32 as usize) % n
        }

        pub fn normal(&mut self, scale: f32) -> f32 {
            let u1 = self.next_f32().max(f32::MIN_POSITIVE);
            let u2 = self.next_f32();
            let z = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
            scale * z
        }
    }
}

mod graph {
    use nalgebra_sparse::csr::CsrMatrix;

    pub struct FuzzyGraph {
        pub n: usize,
        pub rows: Vec<u32>,
        pub cols: Vec<u32>,
        pub vals: Vec<f32>,
    }

    pub fn from_dense(connectivities: &ndarray::Array2<f32>) -> FuzzyGraph {
        use rayon::prelude::*;
        let n = connectivities.nrows();

        let per_row: Vec<Vec<(u32, u32, f32)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                connectivities
                    .row(i)
                    .iter()
                    .enumerate()
                    .filter(|(_, &w)| w != 0.0)
                    .map(|(j, &w)| (i as u32, j as u32, w))
                    .collect()
            })
            .collect();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for row in per_row {
            for (r, c, v) in row {
                rows.push(r);
                cols.push(c);
                vals.push(v);
            }
        }
        FuzzyGraph {
            n,
            rows,
            cols,
            vals,
        }
    }

    pub fn from_csr(connectivities: &CsrMatrix<f32>) -> FuzzyGraph {
        let n = connectivities.nrows();
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            let row = connectivities.row(i);
            for (&j, &w) in row.col_indices().iter().zip(row.values().iter()) {
                if w != 0.0 {
                    rows.push(i as u32);
                    cols.push(j as u32);
                    vals.push(w);
                }
            }
        }
        FuzzyGraph {
            n,
            rows,
            cols,
            vals,
        }
    }
}

mod laplacian {
    use super::graph::FuzzyGraph;

    pub struct Csr {
        pub n: usize,
        pub indptr: Vec<usize>,
        pub indices: Vec<u32>,
        pub vals: Vec<f32>,
    }

    impl Csr {
        pub fn from_fuzzy(g: &FuzzyGraph) -> Csr {
            let mut indptr = vec![0_usize; g.n + 1];
            for &r in &g.rows {
                indptr[r as usize + 1] += 1;
            }
            for i in 0..g.n {
                indptr[i + 1] += indptr[i];
            }
            Csr {
                n: g.n,
                indptr,
                indices: g.cols.clone(),
                vals: g.vals.to_vec(),
            }
        }

        pub fn row(&self, i: usize) -> (&[u32], &[f32]) {
            let (lo, hi) = (self.indptr[i], self.indptr[i + 1]);
            (&self.indices[lo..hi], &self.vals[lo..hi])
        }
    }

    pub struct Laplacian {
        s: Csr,

        #[allow(dead_code)]
        pub degrees: Vec<f32>,
    }

    impl Laplacian {
        pub fn new(g: &FuzzyGraph) -> Laplacian {
            let mut s = Csr::from_fuzzy(g);
            let mut degrees = vec![0.0_f32; g.n];
            for (i, deg) in degrees.iter_mut().enumerate() {
                let (lo, hi) = (s.indptr[i], s.indptr[i + 1]);
                *deg = s.vals[lo..hi].iter().sum();
            }
            let inv_sqrt: Vec<f32> = degrees.iter().map(|&d| 1.0 / d.sqrt()).collect();
            for i in 0..g.n {
                let (lo, hi) = (s.indptr[i], s.indptr[i + 1]);
                for p in lo..hi {
                    s.vals[p] *= inv_sqrt[i] * inv_sqrt[s.indices[p] as usize];
                }
            }
            Laplacian { s, degrees }
        }

        pub fn n(&self) -> usize {
            self.s.n
        }

        pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
            for i in 0..self.s.n {
                let (idx, vals) = self.s.row(i);
                let sx: f32 = idx.iter().zip(vals).map(|(&j, &v)| v * x[j as usize]).sum();
                y[i] = x[i] - sx;
            }
        }

        pub fn to_dense(&self) -> Vec<f32> {
            let n = self.s.n;
            let mut dense = vec![0.0_f32; n * n];
            for i in 0..n {
                dense[i * n + i] = 1.0;
                let (idx, vals) = self.s.row(i);
                for (&j, &v) in idx.iter().zip(vals) {
                    dense[j as usize * n + i] -= v;
                }
            }
            dense
        }
    }
}

mod components {
    use super::laplacian::Csr;

    pub struct Components {
        pub n_components: usize,
        pub labels: Vec<u32>,
    }

    pub fn connected_components(csr: &Csr) -> Components {
        const UNLABELED: u32 = u32::MAX;
        let mut labels = vec![UNLABELED; csr.n];
        let mut n_components = 0_u32;
        let mut queue = Vec::new();
        for start in 0..csr.n {
            if labels[start] != UNLABELED {
                continue;
            }
            labels[start] = n_components;
            queue.push(start);
            while let Some(v) = queue.pop() {
                let (neighbours, _) = csr.row(v);
                for &u in neighbours {
                    if labels[u as usize] == UNLABELED {
                        labels[u as usize] = n_components;
                        queue.push(u as usize);
                    }
                }
            }
            n_components += 1;
        }
        Components {
            n_components: n_components as usize,
            labels,
        }
    }
}

mod eigen {
    use super::laplacian::Laplacian;

    const DENSE_THRESHOLD: usize = 2000;

    pub struct EigenPairs {
        #[allow(dead_code)]
        pub values: Vec<f32>,
        pub vectors: Vec<f32>,
    }

    pub fn smallest_eigenpairs(lap: &Laplacian, k: usize) -> EigenPairs {
        if lap.n() <= DENSE_THRESHOLD {
            dense_eigenpairs(lap, k)
        } else {
            lanczos_eigenpairs(lap, k)
        }
    }

    fn dense_eigenpairs(lap: &Laplacian, k: usize) -> EigenPairs {
        let n = lap.n();
        let m = nalgebra::DMatrix::from_column_slice(n, n, &lap.to_dense());
        let eig = m.symmetric_eigen();

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));

        let mut values = Vec::with_capacity(k);
        let mut vectors = vec![0.0_f32; n * k];
        for (j, &col) in order.iter().take(k).enumerate() {
            values.push(eig.eigenvalues[col]);
            let v = &mut vectors[j * n..(j + 1) * n];
            for (i, x) in v.iter_mut().enumerate() {
                *x = eig.eigenvectors[(i, col)];
            }
            canonical_sign(v);
        }
        EigenPairs { values, vectors }
    }

    fn hash01(i: usize, salt: u32) -> f32 {
        let h = (i as u32).wrapping_add(salt).wrapping_mul(2654435761);
        (h >> 8) as f32 / (1_u32 << 24) as f32
    }

    fn canonical_sign(v: &mut [f32]) {
        let mut pivot = 0;
        for (i, x) in v.iter().enumerate() {
            if x.abs() > v[pivot].abs() + 1e-15 {
                pivot = i;
            }
        }
        if v[pivot] < 0.0 {
            for x in v.iter_mut() {
                *x = -*x;
            }
        }
    }

    fn lanczos_eigenpairs(lap: &Laplacian, k: usize) -> EigenPairs {
        let n = lap.n();
        let m0 = n.min((8 * k).max(30));
        const LOCK_TOL: f32 = 1e-4;
        const MAX_CYCLES: usize = 200;

        let mut locked: Vec<(f32, Vec<f32>)> = Vec::with_capacity(k);
        let mut tmp = vec![0.0_f32; n];

        let mut r: Vec<f32> = (0..n).map(|i| 1.0 + 0.01 * (hash01(i, 0) - 0.5)).collect();

        let mut cycle = 0;
        while locked.len() < k && cycle < MAX_CYCLES {
            cycle += 1;

            for _ in 0..2 {
                for (_, u) in &locked {
                    let c = dot(&r, u);
                    for (x, &ux) in r.iter_mut().zip(u) {
                        *x -= c * ux;
                    }
                }
            }
            let rnorm = dot(&r, &r).sqrt();
            if rnorm < 1e-12 {
                r = (0..n).map(|i| hash01(i, cycle as u32) - 0.5).collect();
                continue;
            }
            for x in r.iter_mut() {
                *x /= rnorm;
            }

            let m = if cycle > 15 { (m0 * 2).min(n) } else { m0 };
            let steps = m.min(n - locked.len());
            let mut basis: Vec<Vec<f32>> = vec![std::mem::take(&mut r)];
            let mut alphas: Vec<f32> = Vec::with_capacity(steps);
            let mut betas: Vec<f32> = Vec::with_capacity(steps);
            loop {
                let j = alphas.len();
                let vj = &basis[j];
                lap.matvec(vj, &mut tmp);
                let mut w: Vec<f32> = vj.iter().zip(&tmp).map(|(&x, &lx)| 2.0 * x - lx).collect();

                alphas.push(dot(&w, vj));
                for _ in 0..2 {
                    for u in basis.iter().chain(locked.iter().map(|(_, u)| u)) {
                        let c = dot(&w, u);
                        for (x, &ux) in w.iter_mut().zip(u) {
                            *x -= c * ux;
                        }
                    }
                }
                if alphas.len() == steps {
                    break;
                }
                let beta = dot(&w, &w).sqrt();
                if beta < 1e-12 {
                    break;
                }
                for x in w.iter_mut() {
                    *x /= beta;
                }
                betas.push(beta);
                basis.push(w);
            }

            let t_dim = alphas.len();
            let mut t = nalgebra::DMatrix::zeros(t_dim, t_dim);
            for (i, &a) in alphas.iter().enumerate() {
                t[(i, i)] = a;
            }
            for (i, &b) in betas.iter().enumerate() {
                t[(i, i + 1)] = b;
                t[(i + 1, i)] = b;
            }
            let eig = t.symmetric_eigen();
            let mut order: Vec<usize> = (0..t_dim).collect();
            order.sort_unstable_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));

            let mut next_r: Option<Vec<f32>> = None;
            for &col in order.iter().take(k - locked.len() + 1) {
                let theta = eig.eigenvalues[col];
                let mut x = vec![0.0_f32; n];
                for (s, u) in basis.iter().enumerate() {
                    let c = eig.eigenvectors[(s, col)];
                    for (o, &ux) in x.iter_mut().zip(u) {
                        *o += c * ux;
                    }
                }
                normalize(&mut x);
                lap.matvec(&x, &mut tmp);
                let resid: f32 = x
                    .iter()
                    .zip(&tmp)
                    .map(|(&xi, &lxi)| (2.0 * xi - lxi - theta * xi).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if resid < LOCK_TOL && locked.len() < k {
                    locked.push((2.0 - theta, x));
                } else if next_r.is_none() {
                    next_r = Some(x);
                }
            }
            r = next_r.unwrap_or_else(|| (0..n).map(|i| hash01(i, cycle as u32) - 0.5).collect());
        }

        locked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let k = k.min(locked.len());
        let mut values = Vec::with_capacity(k);
        let mut vectors = vec![0.0_f32; n * k];
        for (j, (val, vec_j)) in locked.into_iter().take(k).enumerate() {
            values.push(val);
            let out = &mut vectors[j * n..(j + 1) * n];
            out.copy_from_slice(&vec_j);
            canonical_sign(out);
        }
        EigenPairs { values, vectors }
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn normalize(v: &mut [f32]) {
        let norm = dot(v, v).sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

mod spectral {
    use super::components::connected_components;
    use super::eigen::smallest_eigenpairs;
    use super::graph::FuzzyGraph;
    use super::laplacian::{Csr, Laplacian};
    use super::rng::SeededRng;

    pub fn initial_embedding(graph: &FuzzyGraph, dim: usize, rng: &mut SeededRng) -> Vec<f32> {
        let comps = connected_components(&Csr::from_fuzzy(graph));
        let coords = if comps.n_components == 1 {
            spectral_layout_single(graph, dim)
        } else {
            multi_component_layout(graph, dim, &comps.labels, comps.n_components, rng)
        };
        noisy_scale_coords(&coords, rng)
    }

    fn spectral_layout_single(graph: &FuzzyGraph, dim: usize) -> Vec<f32> {
        let lap = Laplacian::new(graph);
        let pairs = smallest_eigenpairs(&lap, dim + 1);
        let n = graph.n;
        let mut coords = vec![0.0_f32; n * dim];
        for d in 0..dim {
            let col = &pairs.vectors[(d + 1) * n..(d + 2) * n];
            for i in 0..n {
                coords[i * dim + d] = col[i];
            }
        }
        coords
    }

    fn multi_component_layout(
        graph: &FuzzyGraph,
        dim: usize,
        labels: &[u32],
        n_components: usize,
        rng: &mut SeededRng,
    ) -> Vec<f32> {
        let n = graph.n;

        let mut meta = vec![0.0_f32; n_components * dim];
        if n_components <= 2 * dim {
            let k = n_components.div_ceil(2);
            for c in 0..n_components {
                let (row, sign) = if c < k { (c, 1.0) } else { (c - k, -1.0) };
                meta[c * dim + row] = sign;
            }
        } else {
            for c in 0..n_components {
                let mut t = c;
                for d in 0..dim {
                    meta[c * dim + d] = (t % 3) as f32;
                    t /= 3;
                }
            }
        }

        let mut result = vec![0.0_f32; n * dim];
        for label in 0..n_components {
            let mine = &meta[label * dim..(label + 1) * dim];
            let mut nearest = f32::INFINITY;
            for other in 0..n_components {
                if other == label {
                    continue;
                }
                let o = &meta[other * dim..(other + 1) * dim];
                let d: f32 = mine
                    .iter()
                    .zip(o)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                if d > 0.0 && d < nearest {
                    nearest = d;
                }
            }
            let data_range = nearest / 2.0;

            let members: Vec<usize> = (0..n).filter(|&i| labels[i] as usize == label).collect();
            if members.len() < 2 * dim || members.len() <= dim + 1 {
                for &i in &members {
                    for d in 0..dim {
                        result[i * dim + d] = rng.uniform(-data_range, data_range) + mine[d];
                    }
                }
            } else {
                let sub = extract_component(graph, labels, label as u32, &members);
                let local = spectral_layout_single(&sub, dim);
                let max_abs = local.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
                let expansion = if max_abs > 0.0 {
                    data_range / max_abs
                } else {
                    0.0
                };
                for (li, &i) in members.iter().enumerate() {
                    for d in 0..dim {
                        result[i * dim + d] = local[li * dim + d] * expansion + mine[d];
                    }
                }
            }
        }
        result
    }

    fn extract_component(
        graph: &FuzzyGraph,
        labels: &[u32],
        label: u32,
        members: &[usize],
    ) -> FuzzyGraph {
        let mut local_of = vec![u32::MAX; graph.n];
        for (li, &g) in members.iter().enumerate() {
            local_of[g] = li as u32;
        }
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for e in 0..graph.rows.len() {
            let (r, c) = (graph.rows[e] as usize, graph.cols[e] as usize);
            if labels[r] == label && labels[c] == label {
                rows.push(local_of[r]);
                cols.push(local_of[c]);
                vals.push(graph.vals[e]);
            }
        }
        FuzzyGraph {
            n: members.len(),
            rows,
            cols,
            vals,
        }
    }

    fn noisy_scale_coords(coords: &[f32], rng: &mut SeededRng) -> Vec<f32> {
        let max_abs = coords.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        let expansion = 10.0 / max_abs;
        coords
            .iter()
            .map(|&x| (x * expansion) + rng.normal(1e-4))
            .collect()
    }
}

mod sgd {
    use super::graph::FuzzyGraph;
    use super::rng::SeededRng;
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    pub fn default_n_epochs(n: usize) -> usize {
        if n <= 10_000 {
            500
        } else {
            200
        }
    }

    pub struct EdgeSchedule {
        pub head: Vec<u32>,
        pub tail: Vec<u32>,
        pub epochs_per_sample: Vec<f32>,
    }

    pub fn schedule_edges(graph: &FuzzyGraph, n_epochs: usize) -> EdgeSchedule {
        let max_w = graph.vals.iter().copied().fold(0.0_f32, f32::max);
        let prune_epochs = if n_epochs > 10 {
            n_epochs
        } else {
            default_n_epochs(graph.n)
        };
        let cutoff = max_w / prune_epochs as f32;

        let mut head = Vec::new();
        let mut tail = Vec::new();
        let mut epochs_per_sample = Vec::new();
        for e in 0..graph.vals.len() {
            let w = graph.vals[e];
            if w < cutoff {
                continue;
            }
            head.push(graph.rows[e]);
            tail.push(graph.cols[e]);
            epochs_per_sample.push(max_w / w);
        }
        EdgeSchedule {
            head,
            tail,
            epochs_per_sample,
        }
    }

    struct EdgeState {
        j: u32,
        k: u32,
        eps: f32,
        eps_neg: f32,
        next_sample: f32,
        next_negative: f32,
        rng: SeededRng,
    }

    pub fn optimize_embedding(
        embedding: &[AtomicU32],
        dim: usize,
        n_vertices: usize,
        schedule: &EdgeSchedule,
        n_epochs: usize,
        a: f32,
        b: f32,
        seed: u64,
    ) {
        const GAMMA: f32 = 1.0;
        const NEGATIVE_SAMPLE_RATE: f32 = 5.0;
        const INITIAL_ALPHA: f32 = 1.0;
        let b32 = b;

        let mut states: Vec<EdgeState> = schedule
            .head
            .iter()
            .zip(&schedule.tail)
            .zip(&schedule.epochs_per_sample)
            .enumerate()
            .map(|(i, ((&j, &k), &e))| EdgeState {
                j,
                k,
                eps: e,
                eps_neg: e / NEGATIVE_SAMPLE_RATE,
                next_sample: e,
                next_negative: e / NEGATIVE_SAMPLE_RATE,
                rng: SeededRng::new(
                    seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                ),
            })
            .collect();

        for epoch in 0..n_epochs {
            let alpha = INITIAL_ALPHA * (1.0 - epoch as f32 / n_epochs as f32);
            let n = epoch as f32;

            states.par_iter_mut().for_each(|s| {
                if s.next_sample > n {
                    return;
                }
                let j = s.j as usize;
                let k = s.k as usize;

                let d2 = rdist(embedding, dim, j, k);
                let grad_coeff = if d2 > 0.0 {
                    let d2b = pow_b(d2, b32);
                    (-2.0 * a * b * d2b / d2) / (a * d2b + 1.0)
                } else {
                    0.0
                };
                for d in 0..dim {
                    let cur = load_f32(&embedding[j * dim + d]);
                    let oth = load_f32(&embedding[k * dim + d]);
                    let grad = clip(grad_coeff * (cur - oth)) * alpha;
                    atomic_add(&embedding[j * dim + d], grad);
                    atomic_add(&embedding[k * dim + d], -grad);
                }
                s.next_sample += s.eps;

                let n_neg = ((n - s.next_negative) / s.eps_neg) as i64;
                for _ in 0..n_neg.max(0) {
                    let neg = s.rng.next_index(n_vertices);
                    let d2 = rdist(embedding, dim, j, neg);
                    let grad_coeff = if d2 > 0.0 {
                        2.0 * GAMMA * b / ((0.001 + d2) * (a * pow_b(d2, b32) + 1.0))
                    } else if j == neg {
                        continue;
                    } else {
                        0.0
                    };
                    for d in 0..dim {
                        let cur = load_f32(&embedding[j * dim + d]);
                        let grad = if grad_coeff > 0.0 {
                            let oth = load_f32(&embedding[neg * dim + d]);
                            clip(grad_coeff * (cur - oth))
                        } else {
                            0.0
                        };
                        atomic_add(&embedding[j * dim + d], grad * alpha);
                    }
                }
                s.next_negative += n_neg as f32 * s.eps_neg;
            });
        }
    }

    #[inline]
    fn atomic_add(cell: &AtomicU32, delta: f32) {
        let cur = f32::from_bits(cell.load(Ordering::Relaxed));
        cell.store((cur + delta).to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn pow_b(d2: f32, b32: f32) -> f32 {
        (d2.ln() * b32).exp()
    }

    #[inline]
    fn load_f32(cell: &AtomicU32) -> f32 {
        f32::from_bits(cell.load(Ordering::Relaxed))
    }

    #[inline]
    fn rdist(embedding: &[AtomicU32], dim: usize, j: usize, k: usize) -> f32 {
        let mut sum = 0.0_f32;
        for d in 0..dim {
            let diff = load_f32(&embedding[j * dim + d]) - load_f32(&embedding[k * dim + d]);
            sum += diff * diff;
        }
        sum
    }

    #[inline]
    fn clip(v: f32) -> f32 {
        v.clamp(-4.0, 4.0)
    }
}

mod curve {
    pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
        const N: usize = 300;
        let mut xs = [0.0_f32; N];
        let mut ys = [0.0_f32; N];
        for (i, (x, y)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
            *x = 3.0 * spread * i as f32 / (N - 1) as f32;
            *y = if *x < min_dist {
                1.0
            } else {
                (-(*x - min_dist) / spread).exp()
            };
        }

        let (mut a, mut b) = (1.0_f32, 1.0_f32);
        let mut lambda = 1e-3;
        let sse = |a: f32, b: f32| -> f32 {
            xs.iter()
                .zip(&ys)
                .map(|(&x, &y)| (model(x, a, b) - y).powi(2))
                .sum()
        };
        let mut err = sse(a, b);
        for _ in 0..200 {
            let (mut jaa, mut jab, mut jbb, mut ga, mut gb) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (&x, &y) in xs.iter().zip(&ys) {
                let xp = if x > 0.0 { x.powf(2.0 * b) } else { 0.0 };
                let denom = 1.0 + a * xp;
                let r = 1.0 / denom - y;
                let da = -xp / (denom * denom);
                let db = if x > 0.0 {
                    -2.0 * a * xp * x.ln() / (denom * denom)
                } else {
                    0.0
                };
                jaa += da * da;
                jab += da * db;
                jbb += db * db;
                ga += da * r;
                gb += db * r;
            }
            let (maa, mbb) = (jaa * (1.0 + lambda), jbb * (1.0 + lambda));
            let det = maa * mbb - jab * jab;
            if det.abs() < 1e-30 {
                break;
            }
            let (delta_a, delta_b) = ((-ga * mbb + gb * jab) / det, (-gb * maa + ga * jab) / det);
            let (na, nb) = (a + delta_a, b + delta_b);
            let new_err = if na > 0.0 && nb > 0.0 {
                sse(na, nb)
            } else {
                f32::INFINITY
            };
            if new_err < err {
                (a, b) = (na, nb);
                err = new_err;
                lambda = (lambda * 0.5).max(1e-12);
                if delta_a.abs() < 1e-12 && delta_b.abs() < 1e-12 {
                    break;
                }
            } else {
                lambda *= 2.0;
                if lambda > 1e12 {
                    break;
                }
            }
        }
        (a, b)
    }

    #[inline]
    fn model(x: f32, a: f32, b: f32) -> f32 {
        if x > 0.0 {
            1.0 / (1.0 + a * x.powf(2.0 * b))
        } else {
            1.0
        }
    }
}

pub fn umap_from_connectivities(
    connectivities: &Array2<f32>,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    random_state: u64,
) -> Result<Array2<f32>> {
    let n = connectivities.nrows();
    if connectivities.ncols() != n {
        return Err(anyhow!(
            "umap: connectivities must be square, got {}×{}",
            n,
            connectivities.ncols()
        ));
    }
    validate_umap_params(n, n_components, spread, min_dist)?;
    let graph = graph::from_dense(connectivities);
    embed_from_graph(
        &graph,
        n_components,
        min_dist,
        spread,
        n_epochs,
        random_state,
    )
}

pub fn umap_from_csr(
    connectivities: &CsrMatrix<f32>,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    random_state: u64,
) -> Result<Array2<f32>> {
    let n = connectivities.nrows();
    if connectivities.ncols() != n {
        return Err(anyhow!(
            "umap: connectivities must be square, got {}×{}",
            n,
            connectivities.ncols()
        ));
    }
    validate_umap_params(n, n_components, spread, min_dist)?;
    let graph = graph::from_csr(connectivities);
    embed_from_graph(
        &graph,
        n_components,
        min_dist,
        spread,
        n_epochs,
        random_state,
    )
}

fn validate_umap_params(n: usize, n_components: usize, spread: f32, min_dist: f32) -> Result<()> {
    if n == 0 {
        return Err(anyhow!("umap: empty connectivities (0 samples)"));
    }
    if n_components == 0 {
        return Err(anyhow!("umap: n_components must be >= 1"));
    }

    if n_components >= n {
        return Err(anyhow!(
            "umap: n_components ({}) must be < n_samples ({})",
            n_components,
            n
        ));
    }
    if spread.is_nan() || spread <= 0.0 {
        return Err(anyhow!("umap: spread must be > 0"));
    }
    if min_dist.is_nan() || min_dist < 0.0 {
        return Err(anyhow!("umap: min_dist must be >= 0"));
    }
    Ok(())
}

fn embed_from_graph(
    graph: &graph::FuzzyGraph,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    n_epochs: Option<usize>,
    random_state: u64,
) -> Result<Array2<f32>> {
    let n = graph.n;
    let n_epochs = n_epochs.unwrap_or_else(|| sgd::default_n_epochs(n));

    let mut rng = rng::SeededRng::new(random_state);
    let embedding = spectral::initial_embedding(graph, n_components, &mut rng);
    let embedding: Vec<AtomicU32> = embedding
        .iter()
        .map(|&x| AtomicU32::new(x.to_bits()))
        .collect();

    let (a, b) = curve::find_ab_params(spread, min_dist);
    let schedule = sgd::schedule_edges(graph, n_epochs);
    sgd::optimize_embedding(
        &embedding,
        n_components,
        n,
        &schedule,
        n_epochs,
        a,
        b,
        random_state,
    );

    let embedding: Vec<f32> = embedding
        .iter()
        .map(|c| f32::from_bits(c.load(Ordering::Relaxed)))
        .collect();
    Array2::from_shape_vec((n, n_components), embedding)
        .map_err(|e| anyhow!("umap: reshape failed: {e}"))
        .map(|a| a.mapv(|v| v))
}

pub fn run_umap(
    data: &mut SoData,
    n_components: usize,
    min_dist: f32,
    spread: f32,
    random_state: u64,
) {
    data.result.umap = None;

    let conn = match &data.result.neighbors {
        Some(n) => &n.connectivities,

        None => return,
    };

    let emb = match umap_from_csr(conn, n_components, min_dist, spread, None, random_state) {
        Ok(e) => e,
        Err(_) => return,
    };

    data.result.umap = Some(UmapResult { embedding: emb });
}
