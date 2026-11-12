pub mod html;
pub mod json;

use std::collections::HashMap;
use std::path::Path;

use crate::config::{ClusterAlg, Config};
use crate::core::SoData;
use crate::pipeline::ClusterRows;

pub struct Report<'a> {
    pub sd: &'a SoData,

    pub config: &'a Config,

    pub name: &'a str,

    pub sample: &'a str,

    pub multi: bool,

    pub alg: ClusterAlg,

    pub labels: Option<&'a [i64]>,

    pub markers: &'a [MarkerRow],

    pub ann_by_ref: &'a [(String, Vec<String>)],
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarkerRow {
    pub cluster: i64,
    pub gene: String,
    pub scores: f32,
    pub log2fc: f32,
    pub pvalues: f32,
    pub pvalues_adj: f32,
    pub pct: f32,
    pub pct_rest: f32,
}

pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

pub fn median(v: &[f32]) -> f32 {
    let mut xs: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.is_empty() {
        return f32::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

pub fn mean(v: &[f32]) -> f32 {
    let xs: Vec<f32> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if xs.is_empty() {
        return f32::NAN;
    }
    xs.iter().sum::<f32>() / xs.len() as f32
}

pub fn stats(v: &[f32]) -> (f32, f32) {
    (median(v), mean(v))
}

pub fn file_name_of(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
}

pub fn cluster_sizes(labels: &[i64], display: &HashMap<i64, i64>) -> Vec<(i64, i64)> {
    let mut counts: HashMap<i64, i64> = HashMap::new();
    for &l in labels {
        *counts.entry(l).or_insert(0) += 1;
    }
    let mut out: Vec<(i64, i64)> = counts
        .into_iter()
        .map(|(raw, n)| (display.get(&raw).copied().unwrap_or(raw), n))
        .collect();
    out.sort_by_key(|(d, _)| *d);
    out
}

pub fn label_counts(labels: &[String]) -> Vec<(String, i64)> {
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for l in labels {
        *counts.entry(l.as_str()).or_insert(0) += 1;
    }
    let mut out: Vec<(String, i64)> = counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

pub(crate) fn marker_rows(
    per: &HashMap<i64, ClusterRows>,
    display: &HashMap<i64, i64>,
    top_n: usize,
) -> Vec<MarkerRow> {
    let mut order: Vec<i64> = per.keys().copied().collect();
    order.sort_by_key(|c| display.get(c).copied().unwrap_or(*c));
    let mut out = Vec::new();
    for cid in order {
        let rows = &per[&cid];
        let d = display.get(&cid).copied().unwrap_or(cid);
        for i in 0..rows.len().min(top_n) {
            out.push(MarkerRow {
                cluster: d,
                gene: rows.genes[i].clone(),
                scores: rows.scores[i],
                log2fc: rows.log2fc[i],
                pvalues: rows.pvalues[i],
                pvalues_adj: rows.pvalues_adj[i],
                pct: rows.pct[i],
                pct_rest: rows.pct_rest[i],
            });
        }
    }
    out
}
