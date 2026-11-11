use std::collections::HashMap;

use rayon::prelude::*;

use crate::algorithm::rng::Mt19937;
use crate::core::gene_set::{GeneSet, GeneSetDb};
use crate::core::sodata::SoData;
use crate::core::{ClusterMarkerResult, MarkerResult};
use crate::core::{EnrichmentResult, GseaGroupResult, GseaResult, GseaTerm};

use super::bh_adjust;

const TRACE_MAX_PTS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankBy {
    Log2fc,

    TStat,

    Score,
}

impl Default for RankBy {
    fn default() -> Self {
        RankBy::Log2fc
    }
}

#[derive(Debug, Clone)]
pub struct GseaParams {
    pub category: Option<String>,

    pub rank_by: RankBy,

    pub gsea_param: f32,

    pub min_gene_count: usize,

    pub max_gene_count: usize,

    pub n_perm: usize,

    pub seed: u64,

    pub pvalue_cutoff: f32,

    pub qvalue_cutoff: f32,

    pub top_n_terms: usize,
}

impl Default for GseaParams {
    fn default() -> Self {
        Self {
            category: None,
            rank_by: RankBy::Log2fc,
            gsea_param: 1.0,
            min_gene_count: 10,
            max_gene_count: 500,
            n_perm: 1000,
            seed: 0,
            pvalue_cutoff: 0.05,
            qvalue_cutoff: 0.25,
            top_n_terms: 20,
        }
    }
}

pub fn run_gsea(data: &mut SoData, db: &GeneSetDb, params: &GseaParams) {
    let marker = match data.result.marker.as_ref() {
        Some(m) => m,
        None => {
            data.result.enrichment = None;
            return;
        }
    };
    let syms = data.symbol_names();
    let groups = gsea_from_marker(marker, &syms, db, params);

    let mut res = data.result.enrichment.take().unwrap_or(EnrichmentResult {
        groups: HashMap::new(),
        gsea: HashMap::new(),
    });
    res.gsea.insert(db.name.clone(), GseaResult { groups });
    data.result.enrichment = Some(res);
}

pub(super) fn gsea_from_marker(
    marker: &MarkerResult,
    syms: &[String],
    db: &GeneSetDb,
    params: &GseaParams,
) -> Vec<GseaGroupResult> {
    let cands = candidate_sets(db, params.category.as_deref());

    marker
        .clusters
        .par_iter()
        .map(|cl| gsea_one_group(cl, syms, db, &cands, params))
        .collect()
}

fn candidate_sets<'a>(
    db: &'a GeneSetDb,
    category: Option<&str>,
) -> Vec<(&'a GeneSet, Vec<&'a str>)> {
    let mut cands: Vec<(&GeneSet, Vec<&str>)> = db
        .sets
        .values()
        .filter(|s| category.is_none_or(|c| s.category.as_deref() == Some(c)))
        .map(|s| (s, s.genes.iter().map(|g| g.as_str()).collect::<Vec<&str>>()))
        .collect();
    cands.sort_unstable_by(|a, b| a.0.id.cmp(&b.0.id));
    cands
}

struct Ranking {
    names: Vec<String>,

    weights: Vec<f32>,

    pos: HashMap<String, usize>,
}

fn rank_values(cl: &ClusterMarkerResult, by: RankBy) -> Vec<f32> {
    match by {
        RankBy::Log2fc => cl.log2fc.clone(),
        RankBy::TStat => cl.t_stats.clone(),
        RankBy::Score => cl
            .scores
            .iter()
            .zip(cl.log2fc.iter())
            .map(|(&s, &fc)| if fc < 0.0 { -s } else { s })
            .collect(),
    }
}

fn build_ranking(
    cl: &ClusterMarkerResult,
    syms: &[String],
    db: &GeneSetDb,
    by: RankBy,
    p: f32,
) -> Ranking {
    let stats = rank_values(cl, by);
    let m = stats.len().min(cl.gene_indices.len());
    let mut rows: Vec<(f32, String)> = Vec::with_capacity(m);
    for (&v, &gi) in stats.iter().zip(cl.gene_indices.iter()) {
        if !v.is_finite() {
            continue;
        }
        let Some(raw) = syms.get(gi) else { continue };
        rows.push((v, db.resolve(raw).unwrap_or(raw).to_string()));
    }
    ranking_from_rows(cl.cluster_id, rows, p)
}

fn ranking_from_rows(cluster_id: i64, mut rows: Vec<(f32, String)>, p: f32) -> Ranking {
    rows.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    rows.dedup_by(|a, b| a.1 == b.1);

    let worst = max_tie_run(&rows);
    if !rows.is_empty() && worst * 10 > rows.len() {
        eprintln!(
            "[run_gsea] warning: cluster {} ranked list has {}/{} ({:.0}%) tied values; ties are broken by gene name, switch rank_by if needed",
            cluster_id,
            worst,
            rows.len(),
            100.0 * worst as f32 / rows.len() as f32
        );
    }

    let n = rows.len();
    let weights: Vec<f32> = rows.iter().map(|(s, _)| s.abs().powf(p)).collect();
    let mut pos = HashMap::with_capacity(n);
    let mut names = Vec::with_capacity(n);
    for (i, (_, g)) in rows.into_iter().enumerate() {
        pos.insert(g.clone(), i);
        names.push(g);
    }
    Ranking {
        names,
        weights,
        pos,
    }
}

fn max_tie_run(rows: &[(f32, String)]) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let mut best = 1usize;
    let mut cur = 1usize;
    for w in rows.windows(2) {
        if w[0].0 == w[1].0 {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 1;
        }
    }
    best
}

fn gsea_one_group(
    cl: &ClusterMarkerResult,
    syms: &[String],
    db: &GeneSetDb,
    cands: &[(&GeneSet, Vec<&str>)],
    params: &GseaParams,
) -> GseaGroupResult {
    let empty = GseaGroupResult {
        group_id: cl.cluster_id,
        terms: Vec::new(),
    };

    let ranking = build_ranking(cl, syms, db, params.rank_by, params.gsea_param);
    let n = ranking.names.len();
    if n < 2 {
        return empty;
    }
    if cl.gene_indices.len() < syms.len() {
        eprintln!(
            "[run_gsea] warning: cluster {} ranked list has only {} genes ({} in the dataset); truncating via the marker n_genes weakens GSEA, use n_genes=None",
            cl.cluster_id,
            cl.gene_indices.len(),
            syms.len()
        );
    }

    let mut hits: Vec<Vec<usize>> = Vec::with_capacity(cands.len());
    let mut kept: Vec<(&GeneSet, usize)> = Vec::with_capacity(cands.len());
    for (set, genes) in cands.iter() {
        let mut h: Vec<usize> = genes
            .iter()
            .filter_map(|g| ranking.pos.get(*g).copied())
            .collect();
        if h.len() < params.min_gene_count || h.len() > params.max_gene_count || h.len() >= n {
            continue;
        }
        h.sort_unstable();
        kept.push((set, h.len()));
        hits.push(h);
    }
    if kept.is_empty() {
        return empty;
    }

    let mut rng = Mt19937::new(params.seed);
    let mut obs: Vec<(f32, usize)> = Vec::with_capacity(kept.len());
    let mut nulls: Vec<Vec<f32>> = Vec::with_capacity(kept.len());
    let mut seen = SeenStamp::new(n);
    let mut buf: Vec<usize> = Vec::with_capacity(params.max_gene_count);
    for h in hits.iter() {
        obs.push(es_from_sorted(h, &ranking.weights, n));
        let k = h.len();
        let mut null = Vec::with_capacity(params.n_perm);
        if k == 0 {
            null.resize(params.n_perm, 0.0);
        } else {
            for _ in 0..params.n_perm {
                sample_sorted(&mut rng, n, k, &mut seen, &mut buf);
                null.push(es_from_sorted(&buf, &ranking.weights, n).0);
            }
        }
        nulls.push(null);
    }

    let ps: Vec<f32> = obs
        .iter()
        .zip(nulls.iter())
        .map(|(&(es, _), null)| pvalue_from_null(es, null).1)
        .collect();
    let adj = bh_adjust(&ps);

    let mut rows: Vec<GseaTerm> = Vec::new();
    for (i, &(set, k)) in kept.iter().enumerate() {
        let (es, rank_at_es) = obs[i];
        let (nes, p) = pvalue_from_null(es, &nulls[i]);
        let padj = adj[i];
        if p > params.pvalue_cutoff || padj > params.qvalue_cutoff {
            continue;
        }

        let (es2, rank_at_es2, running_sum) =
            es_trace(&hits[i], &ranking.weights, n, TRACE_MAX_PTS);
        debug_assert_eq!(
            (es, rank_at_es),
            (es2, rank_at_es2),
            "the trajectory and the observed ES must come from the same source (single es_core implementation)"
        );
        rows.push(GseaTerm {
            id: set.id.clone(),
            name: set.name.clone(),
            category: set.category.clone(),
            es,
            nes,
            pvalue: p,
            pvalue_adj: padj,
            size: k,
            rank_at_es,
            leading_edge: leading_edge(es, rank_at_es, &hits[i], &ranking.names),
            running_sum,
            hits: hits[i].clone(),
        });
    }
    rows.sort_by(|a, b| {
        a.pvalue_adj
            .partial_cmp(&b.pvalue_adj)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    rows.truncate(params.top_n_terms);

    GseaGroupResult {
        group_id: cl.cluster_id,
        terms: rows,
    }
}

fn es_core(
    hits: &[usize],
    w: &[f32],
    n_total: usize,
    mut trace: Option<&mut Vec<(usize, f32)>>,
) -> (f32, usize) {
    let k = hits.len();
    if k == 0 || k >= n_total {
        return (0.0, 0);
    }
    let n_r: f32 = hits.iter().map(|&q| w[q]).sum();

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(n_r > 0.0) {
        return (0.0, 0);
    }
    let inv_r = 1.0 / n_r;
    let inv_miss = 1.0 / (n_total - k) as f32;

    if let Some(t) = trace.as_deref_mut() {
        t.push((0, 0.0));
    }

    let mut cum = 0.0f32;
    let mut best_max = f32::NEG_INFINITY;
    let mut arg_max = 0usize;
    let mut best_min = f32::INFINITY;
    let mut arg_min = 0usize;
    for (i, &q) in hits.iter().enumerate() {
        if q > 0 {
            let before = cum - (q - i) as f32 * inv_miss;
            if let Some(t) = trace.as_deref_mut() {
                t.push((q - 1, before));
            }
            if before < best_min {
                best_min = before;
                arg_min = q - 1;
            }
        }
        cum += w[q] * inv_r;
        let after = cum - (q - i) as f32 * inv_miss;
        if let Some(t) = trace.as_deref_mut() {
            t.push((q, after));
        }
        if after > best_max {
            best_max = after;
            arg_max = q;
        }
    }

    let tail = cum - 1.0;
    if let Some(t) = trace {
        t.push((n_total - 1, tail));
    }
    if tail < best_min {
        best_min = tail;
        arg_min = n_total - 1;
    }

    if best_max.abs() > best_min.abs() {
        (best_max, arg_max)
    } else {
        (best_min, arg_min)
    }
}

fn es_from_sorted(hits: &[usize], w: &[f32], n_total: usize) -> (f32, usize) {
    es_core(hits, w, n_total, None)
}

fn es_trace(
    hits: &[usize],
    w: &[f32],
    n_total: usize,
    max_pts: usize,
) -> (f32, usize, Vec<(usize, f32)>) {
    let mut trace: Vec<(usize, f32)> = Vec::with_capacity(2 * hits.len() + 2);
    let (es, rank_at) = es_core(hits, w, n_total, Some(&mut trace));
    if trace.is_empty() {
        trace.push((0, 0.0));
        return (es, rank_at, trace);
    }
    decimate_trace(&mut trace, max_pts, rank_at);
    (es, rank_at, trace)
}

fn decimate_trace(trace: &mut Vec<(usize, f32)>, max_pts: usize, rank_at_es: usize) {
    if max_pts == 0 || trace.len() <= max_pts {
        return;
    }

    let peak = trace
        .iter()
        .rposition(|&(p, _)| p == rank_at_es)
        .unwrap_or(trace.len() - 1);
    let stride = trace.len().div_ceil(max_pts);
    let mut keep: Vec<usize> = (0..trace.len()).step_by(stride).collect();
    keep.push(trace.len() - 1);
    keep.push(peak);
    keep.sort_unstable();
    keep.dedup();
    let old = std::mem::take(trace);
    *trace = keep.into_iter().map(|i| old[i]).collect();
}

struct SeenStamp {
    stamp: Vec<u32>,
    gen: u32,
}

impl SeenStamp {
    fn new(n: usize) -> Self {
        Self {
            stamp: vec![0; n],
            gen: 0,
        }
    }

    fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.stamp.fill(0);
            self.gen = 1;
        }
    }

    #[inline]
    fn insert(&mut self, v: usize) -> bool {
        if self.stamp[v] == self.gen {
            false
        } else {
            self.stamp[v] = self.gen;
            true
        }
    }
}

fn sample_sorted(
    rng: &mut Mt19937,
    n: usize,
    k: usize,
    seen: &mut SeenStamp,
    buf: &mut Vec<usize>,
) {
    seen.reset();
    buf.clear();
    while buf.len() < k {
        let v = rng.bounded_below(0, n as u64) as usize;
        if seen.insert(v) {
            buf.push(v);
        }
    }
    buf.sort_unstable();
}

fn pvalue_from_null(es: f32, null: &[f32]) -> (f32, f32) {
    if null.is_empty() {
        return (0.0, 1.0);
    }
    let mut ge0 = 0usize;
    let mut lt0 = 0usize;
    let mut sum_pos = 0.0f32;
    let mut sum_neg = 0.0f32;
    for &e in null {
        if e >= 0.0 {
            ge0 += 1;
            sum_pos += e;
        } else {
            lt0 += 1;
            sum_neg += e;
        }
    }
    if es >= 0.0 {
        let mean = if ge0 > 0 { sum_pos / ge0 as f32 } else { 0.0 };
        let nes = if mean > 0.0 { es / mean } else { 0.0 };
        let n_ge = null.iter().filter(|&&x| x >= es).count();
        (nes, (1.0 + n_ge as f32) / (1.0 + ge0 as f32))
    } else {
        let mean = if lt0 > 0 { sum_neg / lt0 as f32 } else { 0.0 };
        let nes = if mean < 0.0 { es / mean.abs() } else { 0.0 };
        let n_le = null.iter().filter(|&&x| x <= es).count();
        (nes, (1.0 + n_le as f32) / (1.0 + lt0 as f32))
    }
}

fn leading_edge(es: f32, rank_at_es: usize, hits: &[usize], names: &[String]) -> Vec<String> {
    hits.iter()
        .filter(|&&q| {
            if es >= 0.0 {
                q <= rank_at_es
            } else {
                q > rank_at_es
            }
        })
        .filter_map(|&q| names.get(q).cloned())
        .collect()
}
