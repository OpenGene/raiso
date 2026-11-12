use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::core::gene_set::{GeneSet, GeneSetDb};
use crate::core::sodata::SoData;
use crate::core::{ClusterMarkerResult, EnrichmentResult, MarkerResult, OraGroupResult, OraTerm};

use super::bh_adjust;

#[derive(Debug, Clone)]
pub struct OraParams {
    pub category: Option<String>,

    pub gene_pvalue_cutoff: f32,

    pub gene_log2fc_cutoff: f32,

    pub pvalue_cutoff: f32,

    pub qvalue_cutoff: f32,

    pub min_gene_count: usize,

    pub max_gene_count: usize,

    pub top_n_terms: usize,
}

impl Default for OraParams {
    fn default() -> Self {
        Self {
            category: None,
            gene_pvalue_cutoff: 0.05,
            gene_log2fc_cutoff: 0.5,
            pvalue_cutoff: 0.05,
            qvalue_cutoff: 0.05,
            min_gene_count: 5,
            max_gene_count: 500,
            top_n_terms: 20,
        }
    }
}

pub fn run_ora(data: &mut SoData, db: &GeneSetDb, params: &OraParams) {
    let marker = match data.result.marker.as_ref() {
        Some(m) => m,
        None => {
            data.result.enrichment = None;
            return;
        }
    };
    let syms = data.symbol_names();
    let groups = match ora_from_marker(marker, &syms, db, params) {
        Ok(g) => g,
        Err(_) => {
            data.result.enrichment = None;
            return;
        }
    };

    let mut res = data.result.enrichment.take().unwrap_or(EnrichmentResult {
        groups: HashMap::new(),
        gsea: HashMap::new(),
    });
    res.groups.insert(db.name.clone(), groups);
    data.result.enrichment = Some(res);
}

fn ora_from_marker(
    marker: &MarkerResult,
    syms: &[String],
    db: &GeneSetDb,
    params: &OraParams,
) -> anyhow::Result<Vec<OraGroupResult>> {
    let mut background: HashSet<String> = HashSet::new();
    let mut unresolved = 0usize;
    for s in syms {
        match db.resolve(s) {
            Some(c) => {
                background.insert(c.to_string());
            }
            None => unresolved += 1,
        }
    }
    let n_bg = background.len();
    if n_bg == 0 {
        return Ok(Vec::new());
    }
    if unresolved > syms.len() / 2 {
        eprintln!(
            "[run_ora] warning: {unresolved}/{} genes could not be mapped to the annotation database; check that the database and species match",
            syms.len()
        );
    }

    let lf = ln_factorials(n_bg);

    let category = params.category.as_deref();
    let candidates: Vec<(&GeneSet, usize)> = db
        .sets
        .values()
        .filter(|s| category.is_none_or(|c| s.category.as_deref() == Some(c)))
        .map(|s| (s, intersect_size(&s.genes, &background)))
        .filter(|(_, k)| *k >= params.min_gene_count && *k <= params.max_gene_count)
        .collect();

    let group_results: Vec<OraGroupResult> = marker
        .clusters
        .par_iter()
        .map(|cl| ora_one_group(cl, syms, db, &background, &candidates, &lf, params))
        .collect();

    Ok(group_results)
}

#[allow(clippy::too_many_arguments)]
fn ora_one_group(
    cl: &ClusterMarkerResult,
    syms: &[String],
    db: &GeneSetDb,
    background: &HashSet<String>,
    candidates: &[(&GeneSet, usize)],
    lf: &[f32],
    params: &OraParams,
) -> OraGroupResult {
    let empty = OraGroupResult {
        group_id: cl.cluster_id,
        terms: Vec::new(),
    };

    let mut input: Vec<String> = cl
        .gene_indices
        .iter()
        .zip(cl.log2fc.iter())
        .zip(cl.pvalues.iter())
        .filter(|((_, fc), p)| **fc > params.gene_log2fc_cutoff && **p < params.gene_pvalue_cutoff)
        .filter_map(|((&gi, _), _)| {
            let s = syms.get(gi)?;
            db.resolve(s).map(|c| c.to_string())
        })
        .collect();
    input.sort();
    input.dedup();
    let n = input.len();
    if n == 0 {
        return empty;
    }
    let n_bg = background.len();

    let input_set: HashSet<&str> = input.iter().map(|s| s.as_str()).collect();

    let (mut ps, mut ks): (Vec<f32>, Vec<usize>) = (
        Vec::with_capacity(candidates.len()),
        Vec::with_capacity(candidates.len()),
    );
    for &(set, k_bg) in candidates.iter() {
        let k = if set.genes.len() <= input_set.len() {
            set.genes
                .iter()
                .filter(|g| input_set.contains(g.as_str()))
                .count()
        } else {
            input_set.iter().filter(|g| set.genes.contains(**g)).count()
        };
        let p = if k == 0 {
            1.0
        } else {
            hypergeom_tail(n_bg, k_bg, n, k, lf)
        };
        ps.push(p);
        ks.push(k);
    }

    let adj = bh_adjust(&ps);

    let mut rows: Vec<OraTerm> = Vec::new();
    for (i, &(set, k_bg)) in candidates.iter().enumerate() {
        let p = ps[i];
        let padj = adj[i];
        if p > params.pvalue_cutoff || padj > params.qvalue_cutoff {
            continue;
        }
        let overlap: Vec<String> = input
            .iter()
            .filter(|g| set.genes.contains(*g))
            .cloned()
            .collect();
        rows.push(OraTerm {
            id: set.id.clone(),
            name: set.name.clone(),
            category: set.category.clone(),
            pvalue: p,
            pvalue_adj: padj,
            count: ks[i],
            overlap,
            gene_ratio: ks[i] as f32 / n as f32,
            bg_ratio: k_bg as f32 / n_bg as f32,
        });
    }

    rows.sort_by(|a, b| {
        a.pvalue_adj
            .partial_cmp(&b.pvalue_adj)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    rows.truncate(params.top_n_terms);

    OraGroupResult {
        group_id: cl.cluster_id,
        terms: rows,
    }
}

fn intersect_size(a: &HashSet<String>, b: &HashSet<String>) -> usize {
    let (small, big) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small.iter().filter(|x| big.contains(*x)).count()
}

fn ln_factorials(max: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; max + 1];
    for i in 1..=max {
        v[i] = v[i - 1] + (i as f32).ln();
    }
    v
}

fn hypergeom_tail(n_total: usize, k_total: usize, n_draw: usize, k_obs: usize, lf: &[f32]) -> f32 {
    if k_obs == 0 {
        return 1.0;
    }
    let kmax = k_total.min(n_draw);
    if k_obs > kmax {
        return 0.0;
    }

    let kmin = k_total.saturating_add(n_draw).saturating_sub(n_total);
    let ln_c = |a: usize, b: usize| -> f32 {
        if b == 0 || b == a {
            0.0
        } else {
            lf[a] - lf[b] - lf[a - b]
        }
    };
    let mut p = 0.0f32;
    for x in kmin..=kmax {
        let lnpmf = ln_c(k_total, x) + ln_c(n_total - k_total, n_draw - x) - ln_c(n_total, n_draw);
        let mass = lnpmf.exp();
        if x >= k_obs {
            p += mass;
        }
    }
    p.min(1.0)
}
