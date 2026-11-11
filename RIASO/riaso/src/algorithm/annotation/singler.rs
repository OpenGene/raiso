use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::core::{AnnotationResult, ExpMatrix, SingleRRef, SoData};
use crate::parallel::{csr_to_dense, SendPtr};

#[derive(Debug, Clone)]
pub struct SingleRParams {
    pub quantile: f32,

    pub fine_tune_threshold: f32,

    pub fine_tune_times: usize,
}

impl Default for SingleRParams {
    fn default() -> Self {
        Self {
            quantile: 80.0,
            fine_tune_threshold: 0.05,
            fine_tune_times: 0,
        }
    }
}

pub fn singler(
    reference: &SingleRRef,
    test_x: &Array2<f32>,
    test_genes: &[String],
    params: &SingleRParams,
) -> Result<AnnotationResult> {
    let ref_x = reference.ref_x();
    let ref_genes = reference.ref_genes();
    let ref_labels = reference.ref_labels();

    if test_x.ncols() != test_genes.len() {
        bail!(
            "singler: test_x cols ({}) != test_genes len ({})",
            test_x.ncols(),
            test_genes.len()
        );
    }

    let ref_gene_idx: HashMap<&str, usize> = ref_genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    let mut interact_test_cols: Vec<usize> = Vec::new();
    let mut interact_ref_cols: Vec<usize> = Vec::new();
    let mut interact_names: Vec<String> = Vec::new();
    for (ti, g) in test_genes.iter().enumerate() {
        if let Some(&ri) = ref_gene_idx.get(g.as_str()) {
            interact_test_cols.push(ti);
            interact_ref_cols.push(ri);
            interact_names.push(g.clone());
        }
    }
    if interact_names.is_empty() {
        bail!("singler: no gene intersection between ref_genes and test_genes");
    }
    let ref_interact = select_cols(ref_x, &interact_ref_cols);
    let test_interact = select_cols(test_x, &interact_test_cols);

    let mut labels: Vec<String> = ref_labels.to_vec();
    labels.sort();
    labels.dedup();
    let n_labels = labels.len();
    if n_labels < 2 {
        bail!(
            "singler: need at least 2 reference labels, got {}",
            n_labels
        );
    }
    let label_idx: HashMap<&str, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i))
        .collect();
    let mut group_cells: Vec<Vec<usize>> = vec![Vec::new(); n_labels];
    for (r, l) in ref_labels.iter().enumerate() {
        group_cells[label_idx[l.as_str()]].push(r);
    }

    let de_n = round_de_n(n_labels);
    let (trained, common) = train_ref(&ref_interact, &group_cells, &interact_names, n_labels, de_n);

    let (scores, first) = score_test(
        &test_interact,
        &ref_interact,
        &group_cells,
        &common,
        params.quantile,
    );

    let final_idx = fine_tune(
        &test_interact,
        &ref_interact,
        &group_cells,
        &interact_names,
        &trained,
        &scores,
        params,
    );

    let first_labels = first.iter().map(|&i| labels[i].clone()).collect();
    let final_labels = final_idx.iter().map(|&i| labels[i].clone()).collect();

    Ok(AnnotationResult {
        first_labels,
        final_labels,
        scores,
        labels,
        clusters: Vec::new(),
    })
}

fn aggregate_by_cluster(
    test_x: &Array2<f32>,
    cluster_labels: &[i64],
) -> Result<(Array2<f32>, Vec<i64>, Vec<usize>)> {
    let n_test = test_x.nrows();
    if cluster_labels.len() != n_test {
        bail!(
            "singler: cluster_labels len ({}) != test_x rows ({})",
            cluster_labels.len(),
            n_test
        );
    }
    let mut cluster_ids: Vec<i64> = cluster_labels.to_vec();
    cluster_ids.sort_unstable();
    cluster_ids.dedup();
    let id_to_row: HashMap<i64, usize> = cluster_ids
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let g = test_x.ncols();
    let n_clusters = cluster_ids.len();

    let mut cell_to_row = vec![0usize; n_test];
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (t, &c) in cluster_labels.iter().enumerate() {
        let r = id_to_row[&c];
        cell_to_row[t] = r;
        members[r].push(t);
    }

    let mut buf = vec![0.0f32; n_clusters * g];
    let ptr = SendPtr(buf.as_mut_ptr());
    (0..n_clusters).into_par_iter().for_each(|r| {
        let row = unsafe { std::slice::from_raw_parts_mut(ptr.add(r * g), g) };
        for &t in &members[r] {
            let src = test_x.row(t);
            for gg in 0..g {
                row[gg] += src[gg];
            }
        }
    });
    let agg = Array2::from_shape_vec((n_clusters, g), buf).expect("aggregate_by_cluster: shape");
    Ok((agg, cluster_ids, cell_to_row))
}

pub fn singler_cluster(
    reference: &SingleRRef,
    test_x: &Array2<f32>,
    test_genes: &[String],
    cluster_labels: &[i64],
    params: &SingleRParams,
) -> Result<AnnotationResult> {
    let (agg, cluster_ids, cell_to_row) = aggregate_by_cluster(test_x, cluster_labels)?;
    let res = singler(reference, &agg, test_genes, params)?;

    let n_test = cluster_labels.len();
    let mut first_labels = vec![String::new(); n_test];
    let mut final_labels = vec![String::new(); n_test];
    for (t, &r) in cell_to_row.iter().enumerate() {
        first_labels[t] = res.first_labels[r].clone();
        final_labels[t] = res.final_labels[r].clone();
    }

    Ok(AnnotationResult {
        first_labels,
        final_labels,
        scores: res.scores,
        labels: res.labels,
        clusters: cluster_ids,
    })
}

fn densify(exp: &ExpMatrix) -> Array2<f32> {
    match exp {
        ExpMatrix::Dense(m) => m.mapv(|v| v),
        ExpMatrix::Sparse(csr) => csr_to_dense(&crate::parallel::csr_f32_to_f32(csr)),
    }
}

pub fn run_singler(
    data: &mut SoData,
    reference: &SingleRRef,
    cluster_labels: Option<&[i64]>,
    params: &SingleRParams,
) {
    data.result.annotation = None;

    let exp = match data.exp_matrix.as_ref() {
        Some(exp) => exp,
        None => return,
    };
    let test_x = densify(exp);
    let test_genes = data.symbol_names();

    let result = match cluster_labels {
        Some(cl) => singler_cluster(reference, &test_x, &test_genes, cl, params),
        None => singler(reference, &test_x, &test_genes, params),
    };

    if let Ok(res) = result {
        data.result.annotation = Some(res);
    }
}

fn rankdata1d(a: &[f32]) -> Vec<f32> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    if a.iter().any(|x| x.is_nan()) {
        return vec![f32::NAN; n];
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_unstable_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap().then(i.cmp(&j)));
    let mut ranks = vec![0.0f32; n];
    let mut s = 0usize;
    while s < n {
        let mut e = s;
        while e + 1 < n && a[idx[e + 1]] == a[idx[s]] {
            e += 1;
        }
        let rank = (s + e + 2) as f32 * 0.5_f32;
        for &k in &idx[s..=e] {
            ranks[k] = rank;
        }
        s = e + 1;
    }
    ranks
}

fn rank_rows(mat: &Array2<f32>) -> Array2<f32> {
    let (nr, nc) = mat.dim();
    let mut out = Array2::<f32>::zeros((nr, nc));
    let ptr = SendPtr(out.as_mut_ptr());
    (0..nr).into_par_iter().for_each(|r| {
        let mut row = vec![0.0f32; nc];
        for c in 0..nc {
            row[c] = mat[[r, c]];
        }
        let ranks = rankdata1d(&row);

        #[allow(clippy::needless_range_loop)]
        for c in 0..nc {
            unsafe { *ptr.add(r * nc + c) = ranks[c] };
        }
    });
    out
}

fn normalize_rows_inplace(ranked: &mut Array2<f32>) {
    let (n, g) = ranked.dim();
    if n == 0 || g == 0 {
        return;
    }
    debug_assert!(
        ranked.is_standard_layout(),
        "normalize_rows_inplace requires a row-major contiguous layout"
    );
    let mean = (g as f32 + 1.0) / 2.0;
    let ptr = SendPtr(ranked.as_mut_ptr());
    (0..n).into_par_iter().for_each(|i| {
        let row = unsafe { std::slice::from_raw_parts_mut(ptr.add(i * g), g) };

        let mut ss = 0.0f32;
        for v in row.iter_mut() {
            let d = *v - mean;
            *v = d;
            ss += d * d;
        }
        if ss > 0.0 {
            let inv = 1.0 / ss.sqrt();
            for v in row.iter_mut() {
                *v *= inv;
            }
        } else {
            for v in row.iter_mut() {
                *v = f32::NAN;
            }
        }
    });
}

fn corr_from_norm(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (n1, g) = a.dim();
    let n2 = b.nrows();
    debug_assert_eq!(g, b.ncols());
    if n1 == 0 || n2 == 0 {
        return Array2::<f32>::zeros((n1, n2));
    }
    let bt = b.t().to_owned();
    let mut out = Array2::<f32>::zeros((n1, n2));

    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n1.div_ceil(nthreads).max(1);
    let starts: Vec<usize> = (0..n1).step_by(chunk).collect();
    let ptr = SendPtr(out.as_mut_ptr());
    starts.par_iter().for_each(|&r0| {
        let r1 = (r0 + chunk).min(n1);
        let blk = a.slice(ndarray::s![r0..r1, ..]).dot(&bt);
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr.add(r0 * n2), (r1 - r0) * n2) };
        for (i, row) in blk.rows().into_iter().enumerate() {
            dst[i * n2..(i + 1) * n2]
                .copy_from_slice(row.to_slice().expect("GEMM output rows are contiguous"));
        }
    });
    out
}

fn corr_spearman(ranked_a: &Array2<f32>, ranked_b: &Array2<f32>) -> Array2<f32> {
    let mut a = ranked_a.to_owned();
    let mut b = ranked_b.to_owned();
    normalize_rows_inplace(&mut a);
    normalize_rows_inplace(&mut b);
    corr_from_norm(&a, &b)
}

fn percentile_linear(a: &[f32], q: f32) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NAN;
    }
    let mut s = a.to_vec();
    s.sort_by(|x, y| match (x.is_nan(), y.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => x.partial_cmp(y).unwrap(),
    });
    let idx = (n as f32 - 1.0) * q / 100.0;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        s[lower]
    } else {
        let w_upper = idx - lower as f32;
        let w_lower = upper as f32 - idx;
        let vl = s[lower];
        let vu = s[upper];
        if vl.is_nan() || vu.is_nan() {
            f32::NAN
        } else {
            vl * w_lower + vu * w_upper
        }
    }
}

fn median_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return f32::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn median_unsorted(vals: &[f32]) -> f32 {
    let mut v = vals.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    median_sorted(&v)
}

fn population_std(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NAN;
    }
    let mean = a.iter().sum::<f32>() / n as f32;
    let var = a.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    var.sqrt()
}

fn round_de_n(n: usize) -> usize {
    (500.0f32 * (2.0f32 / 3.0f32).powf((n as f32).log2())).round() as usize
}

const TRAINED_KEEP: usize = 500;

fn keep_top_trained(pairs: &mut Vec<(usize, f32)>) {
    let cmp =
        |a: &(usize, f32), b: &(usize, f32)| b.1.partial_cmp(&a.1).unwrap().then(b.0.cmp(&a.0));
    if pairs.len() > TRAINED_KEEP {
        pairs.select_nth_unstable_by(TRAINED_KEEP - 1, cmp);
        pairs.truncate(TRAINED_KEEP);

        pairs.shrink_to_fit();
    }
    pairs.sort_by(cmp);
}

#[allow(clippy::if_same_then_else)]
fn round6(x: f32) -> f32 {
    let scale = 1e6;
    let y = x * scale;
    let floor = y.floor();
    let frac = y - floor;
    if frac > 0.5 {
        (floor + 1.0) / scale
    } else if frac < 0.5 {
        floor / scale
    } else if (floor as i64).rem_euclid(2) == 0 {
        floor / scale
    } else {
        (floor + 1.0) / scale
    }
}

fn select_cols(x: &Array2<f32>, cols: &[usize]) -> Array2<f32> {
    x.select(Axis(1), cols).to_owned()
}

fn select_rows_cols(x: &Array2<f32>, rows: &[usize], cols: &[usize]) -> Array2<f32> {
    x.select(Axis(0), rows).select(Axis(1), cols).to_owned()
}

fn train_ref(
    ref_interact: &Array2<f32>,
    group_cells: &[Vec<usize>],
    interact_names: &[String],
    n_labels: usize,
    de_n: usize,
) -> (Vec<Vec<Vec<(usize, f32)>>>, Vec<usize>) {
    let g = ref_interact.ncols();

    let max_group = group_cells.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut median_exp = vec![0.0f32; g * n_labels];
    {
        let ptr = SendPtr(median_exp.as_mut_ptr());
        (0..g)
            .into_par_iter()
            .map_init(
                || vec![0.0f32; max_group],
                |buf, gg| {
                    let dst =
                        unsafe { std::slice::from_raw_parts_mut(ptr.add(gg * n_labels), n_labels) };
                    for l in 0..n_labels {
                        let rows = &group_cells[l];
                        for (k, &r) in rows.iter().enumerate() {
                            buf[k] = ref_interact[[r, gg]];
                        }
                        let s = &mut buf[..rows.len()];
                        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        dst[l] = median_sorted(s);
                    }
                },
            )
            .for_each(|_| {});
    }
    let med_at = |gg: usize, l: usize| median_exp[gg * n_labels + l];

    let pair_keys: Vec<(usize, usize)> = (0..n_labels)
        .flat_map(|i| (0..n_labels).map(move |j| (i, j)))
        .filter(|&(i, j)| i != j)
        .collect();
    let computed: Vec<(usize, usize, Vec<(usize, f32)>)> = pair_keys
        .par_iter()
        .map(|&(i, j)| {
            let mut pairs: Vec<(usize, f32)> = (0..g)
                .map(|gg| (gg, round6(med_at(gg, i) - med_at(gg, j))))
                .collect();
            keep_top_trained(&mut pairs);
            (i, j, pairs)
        })
        .collect();

    let mut trained = vec![vec![Vec::new(); n_labels]; n_labels];
    let mut common: Vec<usize> = Vec::new();
    let mut common_seen = HashSet::new();
    for (i, j, pairs) in computed {
        for &(gg, _) in pairs.iter().take(de_n) {
            if common_seen.insert(gg) {
                common.push(gg);
            }
        }
        trained[i][j] = pairs;
    }
    common.sort_by(|&a, &b| interact_names[a].cmp(&interact_names[b]));
    (trained, common)
}

fn score_test(
    test_interact: &Array2<f32>,
    ref_interact: &Array2<f32>,
    group_cells: &[Vec<usize>],
    common: &[usize],
    quantile: f32,
) -> (Array2<f32>, Vec<usize>) {
    let n_test = test_interact.nrows();
    let n_labels = group_cells.len();
    let test_mat = select_cols(test_interact, common);
    let mut ranked_ref = rank_rows(&test_mat);

    normalize_rows_inplace(&mut ranked_ref);

    let score_cols: Vec<Vec<f32>> = (0..n_labels)
        .into_par_iter()
        .map_init(Vec::<f32>::new, |buf, l| {
            let ref_sub = select_rows_cols(ref_interact, &group_cells[l], common);
            let mut ranked_qry = rank_rows(&ref_sub);
            normalize_rows_inplace(&mut ranked_qry);
            let sim = corr_from_norm(&ranked_ref, &ranked_qry);
            (0..n_test)
                .map(|t| {
                    buf.clear();
                    buf.extend_from_slice(
                        sim.row(t)
                            .to_slice()
                            .expect("corr output rows are contiguous"),
                    );
                    percentile_linear(buf, quantile)
                })
                .collect()
        })
        .collect();

    let mut scores = Array2::<f32>::zeros((n_test, n_labels));
    for (l, col) in score_cols.into_iter().enumerate() {
        for t in 0..n_test {
            scores[[t, l]] = col[t];
        }
    }

    let mut first = vec![0usize; n_test];
    for t in 0..n_test {
        let mut best = 0usize;
        let mut bestv = f32::NEG_INFINITY;
        for l in 0..n_labels {
            let v = scores[[t, l]];
            if v > bestv {
                bestv = v;
                best = l;
            }
        }
        first[t] = best;
    }
    (scores, first)
}

fn fine_tune(
    test_interact: &Array2<f32>,
    ref_interact: &Array2<f32>,
    group_cells: &[Vec<usize>],
    interact_names: &[String],
    trained: &[Vec<Vec<(usize, f32)>>],
    scores: &Array2<f32>,
    params: &SingleRParams,
) -> Vec<usize> {
    let n_test = test_interact.nrows();
    let n_labels = group_cells.len();
    (0..n_test)
        .into_par_iter()
        .map(|t| {
            let test_cell: Vec<f32> = test_interact.row(t).to_vec();
            let rowmax = (0..n_labels)
                .map(|l| scores[[t, l]])
                .fold(f32::NEG_INFINITY, f32::max);
            let thr = rowmax - params.fine_tune_threshold;
            let mut candidates: Vec<usize> = (0..n_labels)
                .filter(|&l| scores[[t, l]] >= thr && scores[[t, l]] != 0.0)
                .collect();
            if candidates.is_empty() {
                let mut best = 0usize;
                let mut bestv = f32::NEG_INFINITY;
                for l in 0..n_labels {
                    let v = scores[[t, l]];
                    if v > bestv {
                        bestv = v;
                        best = l;
                    }
                }
                candidates.push(best);
            }
            fine_tune_parallel(
                &test_cell,
                &candidates,
                ref_interact,
                group_cells,
                interact_names,
                trained,
                params,
            )
        })
        .collect()
}

fn fine_tune_parallel(
    test_cell: &[f32],
    candidates: &[usize],
    ref_interact: &Array2<f32>,
    group_cells: &[Vec<usize>],
    interact_names: &[String],
    trained: &[Vec<Vec<(usize, f32)>>],
    params: &SingleRParams,
) -> usize {
    let mut labels = candidates.to_vec();
    if params.fine_tune_times > 0 {
        let mut try_num = 0usize;
        while try_num < params.fine_tune_times && labels.len() > 1 {
            labels = fine_tune_one_time(
                &labels,
                test_cell,
                ref_interact,
                group_cells,
                interact_names,
                trained,
                params,
                Some(try_num),
            );
            try_num += 1;
        }
    } else {
        while labels.len() > 1 {
            labels = fine_tune_one_time(
                &labels,
                test_cell,
                ref_interact,
                group_cells,
                interact_names,
                trained,
                params,
                None,
            );
        }
    }
    labels[0]
}

fn fine_tune_one_time(
    top_labels: &[usize],
    test_cell: &[f32],
    ref_interact: &Array2<f32>,
    group_cells: &[Vec<usize>],
    interact_names: &[String],
    trained: &[Vec<Vec<(usize, f32)>>],
    params: &SingleRParams,
    try_num: Option<usize>,
) -> Vec<usize> {
    let de_n = round_de_n(top_labels.len());

    let mut gene_set: Vec<usize> = Vec::new();
    {
        let mut seen = HashSet::new();
        for &i in top_labels {
            for &j in top_labels {
                if i == j {
                    continue;
                }
                let mut taken = 0usize;
                for &(g, v) in &trained[i][j] {
                    if taken >= de_n {
                        break;
                    }
                    if v > 0.0 {
                        if seen.insert(g) {
                            gene_set.push(g);
                        }
                        taken += 1;
                    }
                }
            }
        }
    }
    gene_set.sort_by(|&a, &b| interact_names[a].cmp(&interact_names[b]));
    if gene_set.len() < 20 {
        return vec![top_labels[0]];
    }

    let test_filtered: Vec<f32> = gene_set.iter().map(|&g| test_cell[g]).collect();
    if population_std(&test_filtered) <= 0.0 {
        return vec![top_labels[0]];
    }
    let ranked_test = rank_rows(
        &Array2::from_shape_vec((1, gene_set.len()), test_filtered.clone())
            .expect("ranked_test shape"),
    );

    let mut res: Vec<(usize, f32)> = Vec::new();
    for &p in top_labels {
        let cells = &group_cells[p];
        if cells.is_empty() {
            continue;
        }
        let ref_sub = select_rows_cols(ref_interact, cells, &gene_set);
        let ranked_qry = rank_rows(&ref_sub);
        let sim = corr_spearman(&ranked_test, &ranked_qry);
        let row: Vec<f32> = (0..sim.ncols()).map(|c| sim[[0, c]]).collect();
        res.push((p, percentile_linear(&row, params.quantile)));
    }
    if res.is_empty() {
        return vec![top_labels[0]];
    }

    let min_v = res.iter().map(|&(_, v)| v).fold(f32::INFINITY, f32::min);
    let median_v = median_unsorted(&res.iter().map(|&(_, v)| v).collect::<Vec<f32>>());
    let mid = (min_v + median_v) / 2.0;
    let mut kept: Vec<(usize, f32)> = res.into_iter().filter(|&(_, v)| v > mid).collect();
    if kept.is_empty() {
        return vec![top_labels[0]];
    }

    let max_v = kept
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let thr = max_v - params.fine_tune_threshold;
    kept.retain(|&(_, v)| v >= thr);
    if kept.is_empty() {
        return vec![top_labels[0]];
    }

    if let Some(t) = try_num {
        if t > 0 && t == params.fine_tune_times - 1 {
            let mut best = kept[0].0;
            let mut bestv = kept[0].1;
            for &(p, v) in kept.iter().skip(1) {
                if v >= bestv {
                    bestv = v;
                    best = p;
                }
            }
            return vec![best];
        }
    }

    kept.into_iter().map(|(p, _)| p).collect()
}
