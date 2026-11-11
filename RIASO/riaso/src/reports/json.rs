use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{json, Map, Value};

use super::{file_name_of, label_counts, stats, Report};
use crate::pipeline::{display_numbers, obs_f32, QC_METRICS};

pub fn write(path: &Path, r: &Report<'_>) -> Result<()> {
    let mut root = Map::new();
    root.insert("meta".to_string(), meta(r));
    root.insert("steps".to_string(), steps(r));
    root.insert("params".to_string(), params(r));
    root.insert("qc".to_string(), qc(r));
    if let Some(hvg) = r.sd.result.hvg.as_ref() {
        let syms = r.sd.symbol_names();
        let hv: Vec<&str> = hvg
            .highly_variable
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| syms.get(i).map(String::as_str).unwrap_or(""))
            .collect();
        root.insert(
            "hvg".to_string(),
            json!({
                "method": r.config.hvg_method,
                "n_bins": r.config.n_bins,
                "n_top_genes": r.config.n_top_genes,
                "n_highly_variable": hv.len(),
                "genes": hv,
            }),
        );
    }
    if let Some(pca) = r.sd.result.pca.as_ref() {
        let ratio: Vec<f32> = pca.explained_variance_ratio.iter().copied().collect();
        root.insert(
            "pca".to_string(),
            json!({
                "n_components": r.config.pca.n_components,
                "explained_variance_ratio": ratio,
            }),
        );
    }
    if let Some(batch) = r.sd.result.batch.as_ref() {
        root.insert(
            "harmony".to_string(),
            json!({ "ran": true, "n_iter": batch.n_iter, "converged": batch.converged }),
        );
    }
    if r.sd.result.neighbors.is_some() {
        root.insert(
            "neighbors".to_string(),
            json!({ "n_neighbors": r.config.n_neighbors }),
        );
    }
    if r.sd.result.umap.is_some() {
        root.insert(
            "umap".to_string(),
            json!({ "n_components": r.config.umap_components }),
        );
    }
    if let Some(labels) = r.labels {
        let display = display_numbers(labels);
        let sizes: Vec<(i64, i64)> = super::cluster_sizes(labels, &display);
        let mut size_map = Map::new();
        for (d, n) in &sizes {
            size_map.insert(d.to_string(), json!(n));
        }
        root.insert(
            "clustering".to_string(),
            json!({
                "algorithm": r.alg.key(),
                "resolution": r.config.resolution,
                "n_clusters": sizes.len(),
                "cluster_sizes": Value::Object(size_map),
            }),
        );
    }
    if !r.markers.is_empty() {
        let rows: Vec<Value> = r
            .markers
            .iter()
            .map(|m| {
                json!({
                    "cluster": m.cluster,
                    "gene": m.gene,
                    "scores": m.scores,
                    "log2fc": m.log2fc,
                    "pvalues": m.pvalues,
                    "pvalues_adj": m.pvalues_adj,
                    "pct": m.pct,
                    "pct_rest": m.pct_rest,
                })
            })
            .collect();
        root.insert(
            "marker".to_string(),
            json!({ "top_n": r.config.top_n, "rows": rows }),
        );
    }
    if !r.ann_by_ref.is_empty() {
        let ann: Vec<Value> = r
            .ann_by_ref
            .iter()
            .map(|(ref_name, labels)| {
                let counts = label_counts(labels);
                let mut m = Map::new();
                for (k, v) in counts {
                    m.insert(k, json!(v));
                }
                json!({ "reference": ref_name, "cell_type_counts": Value::Object(m) })
            })
            .collect();
        root.insert("annotation".to_string(), Value::Array(ann));
    }
    if let Some(enr) = r.sd.result.enrichment.as_ref() {
        root.insert("enrichment".to_string(), enrichment(r, enr));
    }

    let text = serde_json::to_string_pretty(&Value::Object(root))?;
    fs::write(path, text.as_bytes())
        .with_context(|| format!("failed to write report.json: {}", path.display()))?;
    Ok(())
}

fn meta(r: &Report<'_>) -> Value {
    let generated_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    json!({
        "tool": "RIASO",
        "version": env!("CARGO_PKG_VERSION"),
        "generated_at": generated_at,
        "sample": r.sample,
        "multi_sample": r.multi,
        "inputs": r.config.io.inputs.iter().map(|s| file_name_of(s)).collect::<Vec<_>>(),
        "out_root": file_name_of(&r.config.io.out_root),
    })
}

fn steps(r: &Report<'_>) -> Value {
    let s = &r.config.steps;
    json!({
        "qc": s.qc,
        "filter_cells": s.filter_cells,
        "filter_genes": s.filter_genes,
        "filter_coordinates": s.filter_coordinates,
        "normalize_total": s.normalize_total,
        "log1p": s.log1p,
        "scale": s.scale,
        "pca": s.pca,
        "harmony": s.harmony,
        "neighbors": s.neighbors,
        "umap": s.umap,
        "clustering": s.clustering,
        "marker": s.marker,
        "singler": s.singler,
        "ora_go": s.ora_go,
        "ora_kegg": s.ora_kegg,
        "ora_reactome": s.ora_reactome,
        "gsea": s.gsea,
    })
}

fn params(r: &Report<'_>) -> Value {
    let c = r.config;
    json!({
        "bin_size": c.io.bin_size,
        "bin_type": c.io.bin_type,
        "dedup_symbols": c.io.dedup_symbols,
        "min_counts": c.min_counts,
        "max_counts": c.max_counts,
        "min_genes": c.min_genes,
        "max_genes": c.max_genes,
        "pct_counts_mt": c.pct_counts_mt,
        "target_sum": c.target_sum,
        "zero_center": c.zero_center,
        "max_value": c.max_value,
        "n_top_genes": c.n_top_genes,
        "n_bins": c.n_bins,
        "hvg_method": c.hvg_method,
        "use_hvg": c.use_hvg,
        "n_components": c.pca.n_components,
        "svd_solver": format!("{:?}", c.pca.svd_solver),
        "n_neighbors": c.n_neighbors,
        "n_pcs_neighbors": c.n_pcs_neighbors,
        "umap_components": c.umap_components,
        "min_dist": c.min_dist,
        "spread": c.spread,
        "cluster_alg": c.cluster_alg.key(),
        "resolution": c.resolution,
        "top_n": c.top_n,
        "random_state": c.random_state,
        "label_key": c.label_key,
        "references": c
            .references
            .iter()
            .map(|p| p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned()))
            .collect::<Vec<_>>(),
        "ora_pvalue_cutoff": c.ora_pvalue_cutoff,
        "ora_qvalue_cutoff": c.ora_qvalue_cutoff,
        "kegg_org": c.kegg_org,
        "reactome_species": c.reactome_species,
    })
}

fn qc(r: &Report<'_>) -> Value {
    let mut m = Map::new();
    m.insert("n_cells".to_string(), json!(r.sd.n_obs()));
    m.insert("n_genes".to_string(), json!(r.sd.n_vars()));
    for metric in QC_METRICS {
        if let Ok(v) = obs_f32(r.sd, metric) {
            let (med, avg) = stats(&v);
            m.insert(
                metric.to_string(),
                json!({ "median": med, "mean": avg, "min": min_of(&v), "max": max_of(&v) }),
            );
        }
    }
    Value::Object(m)
}

fn min_of(v: &[f32]) -> f32 {
    v.iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NAN, f32::min)
}

fn max_of(v: &[f32]) -> f32 {
    v.iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NAN, f32::max)
}

fn enrichment(r: &Report<'_>, enr: &crate::core::EnrichmentResult) -> Value {
    let display: HashMap<i64, i64> = match r.labels {
        Some(l) => display_numbers(l),
        None => HashMap::new(),
    };
    let disp = |raw: i64| display.get(&raw).copied().unwrap_or(raw);

    let mut ora_map = Map::new();
    for (db, groups) in &enr.groups {
        let mut gmap = Map::new();
        for g in groups {
            let terms: Vec<Value> = g
                .terms
                .iter()
                .map(|t| {
                    json!({
                        "id": t.id,
                        "name": t.name,
                        "category": t.category,
                        "pvalue": t.pvalue,
                        "pvalue_adj": t.pvalue_adj,
                        "count": t.count,
                        "gene_ratio": t.gene_ratio,
                        "bg_ratio": t.bg_ratio,
                    })
                })
                .collect();
            gmap.insert(disp(g.group_id).to_string(), Value::Array(terms));
        }
        ora_map.insert(db.clone(), Value::Object(gmap));
    }

    let mut gsea_map = Map::new();
    for (db, res) in &enr.gsea {
        let mut gmap = Map::new();
        for g in &res.groups {
            let terms: Vec<Value> = g
                .terms
                .iter()
                .map(|t| {
                    json!({
                        "id": t.id,
                        "name": t.name,
                        "category": t.category,
                        "es": t.es,
                        "nes": t.nes,
                        "pvalue": t.pvalue,
                        "pvalue_adj": t.pvalue_adj,
                        "size": t.size,
                        "rank_at_es": t.rank_at_es,
                        "leading_edge": t.leading_edge,
                    })
                })
                .collect();
            gmap.insert(disp(g.group_id).to_string(), Value::Array(terms));
        }
        gsea_map.insert(db.clone(), Value::Object(gmap));
    }

    json!({ "ora": Value::Object(ora_map), "gsea": Value::Object(gsea_map) })
}
