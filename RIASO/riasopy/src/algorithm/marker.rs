use std::borrow::Cow;
use std::cmp::Ordering;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use polars::prelude::{Column, CsvWriter, DataFrame, SerWriter};
use rayon::prelude::*;

use crate::core::{ClusterMarkerResult, ExpMatrix, MarkerResult, SoData};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrMethod {
    BH,

    Bonferroni,
}

#[derive(Debug, Clone)]
pub struct MarkerParams {
    pub corr_method: CorrMethod,

    pub n_genes: Option<usize>,
}

impl Default for MarkerParams {
    fn default() -> Self {
        Self {
            corr_method: CorrMethod::BH,
            n_genes: None,
        }
    }
}

fn log_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let x0 = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    let mut y = x0;
    for &c in &COF {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x0).ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: i32 = 200;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;

        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = log_gamma(a) + log_gamma(b) - log_gamma(a + b);
    let bt = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn t_two_sided_p(df: f64, t: f64) -> f64 {
    if df.is_nan() || t.is_nan() {
        return f64::NAN;
    }
    let x = df / (df + t * t);
    let p = regularized_beta(df * 0.5, 0.5, x);

    if p == 0.0 {
        f64::MIN_POSITIVE
    } else {
        p
    }
}

fn welch_t(ma: f64, va: f64, na: f64, mb: f64, vb: f64, nb: f64) -> (f64, f64) {
    if na < 2.0 || nb < 2.0 {
        return (f64::NAN, f64::NAN);
    }
    let se_sq = va / na + vb / nb;
    if se_sq <= 0.0 || se_sq.is_nan() {
        return (f64::NAN, f64::NAN);
    }
    let t = (ma - mb) / se_sq.sqrt();
    let num = se_sq * se_sq;
    let den = (va / na).powi(2) / (na - 1.0) + (vb / nb).powi(2) / (nb - 1.0);
    let df = if den == 0.0 { f64::NAN } else { num / den };
    (t, df)
}

struct ClusterAcc {
    n_g: usize,
    n_r: usize,
    sum_g: Vec<f64>,
    sum_r: Vec<f64>,
    sq_g: Vec<f64>,
    sq_r: Vec<f64>,
    cnt_pos_g: Vec<usize>,
    cnt_pos_r: Vec<usize>,
}

fn accumulate_cluster_dense(x: &Array2<f32>, labels: &[i64], cid: i64) -> ClusterAcc {
    let n_genes = x.ncols();
    let n_cells = x.nrows();
    let mut acc = ClusterAcc {
        n_g: 0,
        n_r: 0,
        sum_g: vec![0.0; n_genes],
        sum_r: vec![0.0; n_genes],
        sq_g: vec![0.0; n_genes],
        sq_r: vec![0.0; n_genes],
        cnt_pos_g: vec![0usize; n_genes],
        cnt_pos_r: vec![0usize; n_genes],
    };

    #[allow(clippy::needless_range_loop)]
    for c in 0..n_cells {
        if labels[c] == cid {
            acc.n_g += 1;
            let row = &x.row(c);
            for (g, &v) in row.iter().enumerate() {
                acc.sum_g[g] += v as f64;
                if v > 0.0 {
                    acc.cnt_pos_g[g] += 1;
                }
            }
        } else {
            acc.n_r += 1;
            let row = &x.row(c);
            for (g, &v) in row.iter().enumerate() {
                acc.sum_r[g] += v as f64;
                if v > 0.0 {
                    acc.cnt_pos_r[g] += 1;
                }
            }
        }
    }

    let mean_g: Vec<f64> = acc.sum_g.iter().map(|&s| s / acc.n_g as f64).collect();
    let mean_r: Vec<f64> = acc.sum_r.iter().map(|&s| s / acc.n_r as f64).collect();

    #[allow(clippy::needless_range_loop)]
    for c in 0..n_cells {
        if labels[c] == cid {
            let row = &x.row(c);
            for (g, &v) in row.iter().enumerate() {
                let d = v as f64 - mean_g[g];
                acc.sq_g[g] += d * d;
            }
        } else {
            let row = &x.row(c);
            for (g, &v) in row.iter().enumerate() {
                let d = v as f64 - mean_r[g];
                acc.sq_r[g] += d * d;
            }
        }
    }

    acc
}

fn accumulate_cluster_sparse(csr: &CsrMatrix<f32>, labels: &[i64], cid: i64) -> ClusterAcc {
    let n_genes = csr.ncols();
    let n_cells = csr.nrows();
    let mut acc = ClusterAcc {
        n_g: 0,
        n_r: 0,
        sum_g: vec![0.0; n_genes],
        sum_r: vec![0.0; n_genes],
        sq_g: vec![0.0; n_genes],
        sq_r: vec![0.0; n_genes],
        cnt_pos_g: vec![0usize; n_genes],
        cnt_pos_r: vec![0usize; n_genes],
    };

    for &l in labels {
        if l == cid {
            acc.n_g += 1;
        } else {
            acc.n_r += 1;
        }
    }

    let offsets = csr.row_offsets();
    let cols = csr.col_indices();
    let vals = csr.values();

    for c in 0..n_cells {
        for idx in offsets[c]..offsets[c + 1] {
            let g = cols[idx];
            let v = vals[idx];
            if labels[c] == cid {
                acc.sum_g[g] += v as f64;
                acc.cnt_pos_g[g] += 1;
            } else {
                acc.sum_r[g] += v as f64;
                acc.cnt_pos_r[g] += 1;
            }
        }
    }

    let mean_g: Vec<f64> = acc.sum_g.iter().map(|&s| s / acc.n_g as f64).collect();
    let mean_r: Vec<f64> = acc.sum_r.iter().map(|&s| s / acc.n_r as f64).collect();

    for c in 0..n_cells {
        for idx in offsets[c]..offsets[c + 1] {
            let g = cols[idx];
            let v = vals[idx];
            if labels[c] == cid {
                let d = v as f64 - mean_g[g];
                acc.sq_g[g] += d * d;
            } else {
                let d = v as f64 - mean_r[g];
                acc.sq_r[g] += d * d;
            }
        }
    }

    for g in 0..n_genes {
        acc.sq_g[g] += (acc.n_g - acc.cnt_pos_g[g]) as f64 * mean_g[g] * mean_g[g];
        acc.sq_r[g] += (acc.n_r - acc.cnt_pos_r[g]) as f64 * mean_r[g] * mean_r[g];
    }

    acc
}

fn cal_log2fc(mg: f64, mr: f64) -> f64 {
    let g = mg.exp_m1() + 1e-9;
    let r = mr.exp_m1() + 1e-9;
    (g / r).log2()
}

fn bh_correct(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    let mut out = vec![f64::NAN; n];
    let finite: Vec<usize> = p
        .iter()
        .enumerate()
        .filter(|(_, &v)| !v.is_nan())
        .map(|(i, _)| i)
        .collect();
    if finite.is_empty() {
        return out;
    }
    let m = finite.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| p[finite[i]].partial_cmp(&p[finite[j]]).unwrap());
    let mut adj = vec![0.0f64; m];
    for (rank, &oi) in order.iter().enumerate() {
        let rank = rank + 1;
        adj[oi] = (p[finite[oi]] * m as f64 / rank as f64).min(1.0);
    }

    let mut run = f64::INFINITY;
    for i in (0..m).rev() {
        run = run.min(adj[order[i]]);
        adj[order[i]] = run;
    }
    for (k, &fi) in finite.iter().enumerate() {
        out[fi] = adj[k];
    }
    out
}

fn bonferroni_correct(p: &[f64]) -> Vec<f64> {
    let n = p.len() as f64;
    p.iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else {
                (v * n).min(1.0)
            }
        })
        .collect()
}

pub fn marker_genes(x: &ExpMatrix, labels: &[i64], params: &MarkerParams) -> Result<MarkerResult> {
    let n_cells = x.n_rows();
    if labels.len() != n_cells {
        bail!(
            "marker_genes: labels len ({}) != rows ({})",
            labels.len(),
            n_cells
        );
    }
    let n_genes = x.n_cols();
    if n_genes == 0 {
        return Ok(MarkerResult {
            clusters: Vec::new(),
        });
    }

    let mut cluster_ids: Vec<i64> = labels.to_vec();
    cluster_ids.sort_unstable();
    cluster_ids.dedup();

    let clusters: Vec<ClusterMarkerResult> = cluster_ids
        .par_iter()
        .map(|&cid| {
            let acc = match x {
                ExpMatrix::Dense(arr) => accumulate_cluster_dense(arr, labels, cid),
                ExpMatrix::Sparse(csr) => accumulate_cluster_sparse(csr, labels, cid),
            };
            let ng = acc.n_g as f64;
            let nr = acc.n_r as f64;

            let mut t_stats = vec![0.0f64; n_genes];
            let mut pvalues = vec![0.0f64; n_genes];
            let mut log2fcs = vec![0.0f64; n_genes];
            let mut pctg = vec![0.0f64; n_genes];
            let mut pctr = vec![0.0f64; n_genes];
            let mut mean_counts = vec![0.0f64; n_genes];

            for g in 0..n_genes {
                let mg = acc.sum_g[g] / ng;
                let mr = acc.sum_r[g] / nr;
                let vg = if acc.n_g >= 2 {
                    acc.sq_g[g] / (ng - 1.0)
                } else {
                    f64::NAN
                };
                let vr = if acc.n_r >= 2 {
                    acc.sq_r[g] / (nr - 1.0)
                } else {
                    f64::NAN
                };
                let (t, df) = welch_t(mg, vg, ng, mr, vr, nr);
                t_stats[g] = t;
                pvalues[g] = t_two_sided_p(df, t);
                log2fcs[g] = cal_log2fc(mg, mr);
                pctg[g] = acc.cnt_pos_g[g] as f64 / ng;
                pctr[g] = acc.cnt_pos_r[g] as f64 / nr;
                mean_counts[g] = mg;
            }

            let scores: Vec<f64> = pvalues
                .iter()
                .map(|&p| if p.is_nan() { f64::NAN } else { -p.log10() })
                .collect();
            let pvalues_adj = match params.corr_method {
                CorrMethod::BH => bh_correct(&pvalues),
                CorrMethod::Bonferroni => bonferroni_correct(&pvalues),
            };

            let mut order: Vec<usize> = (0..n_genes).collect();
            order.sort_by(|&i, &j| match (scores[i].is_nan(), scores[j].is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => scores[j].partial_cmp(&scores[i]).unwrap(),
            });
            let n_keep = params.n_genes.unwrap_or(n_genes).min(n_genes);
            let keep: Vec<usize> = order[..n_keep].to_vec();
            let pick = |v: &Vec<f64>| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();

            ClusterMarkerResult {
                cluster_id: cid,
                gene_indices: keep.clone(),
                t_stats: pick(&t_stats),
                scores: pick(&scores),
                pvalues: pick(&pvalues),
                pvalues_adj: pick(&pvalues_adj),
                log2fc: pick(&log2fcs),
                pct: pick(&pctg),
                pct_rest: pick(&pctr),
                mean_count: pick(&mean_counts),
            }
        })
        .collect();

    Ok(MarkerResult { clusters })
}

pub(crate) fn marker_input(data: &SoData) -> Option<Cow<'_, ExpMatrix>> {
    if let Some(layer) = data.get_layer(crate::algorithm::normalize::LOG1P_LAYER) {
        return Some(Cow::Borrowed(layer));
    }
    match &data.exp_matrix {
        Some(e) => Some(Cow::Borrowed(e)),
        None => None,
    }
}

pub fn run_marker(data: &mut SoData, labels: &[i64], params: &MarkerParams) {
    data.result.marker = None;

    let x = match marker_input(data) {
        Some(x) => x,
        None => return,
    };

    if let Ok(res) = marker_genes(x.as_ref(), labels, params) {
        data.result.marker = Some(res);
    }
}

pub fn write_marker_csv(
    result: &MarkerResult,
    gene_names: &[String],
    path: &Path,
    top: Option<usize>,
) -> Result<()> {
    let mut col_cluster: Vec<i64> = Vec::new();
    let mut col_gene: Vec<String> = Vec::new();
    let mut col_scores: Vec<f64> = Vec::new();
    let mut col_p: Vec<f64> = Vec::new();
    let mut col_padj: Vec<f64> = Vec::new();
    let mut col_log2fc: Vec<f64> = Vec::new();
    let mut col_pct: Vec<f64> = Vec::new();
    let mut col_pct_rest: Vec<f64> = Vec::new();
    let mut col_mean: Vec<f64> = Vec::new();

    for c in &result.clusters {
        let n = top
            .unwrap_or(c.gene_indices.len())
            .min(c.gene_indices.len());
        for (k, &gi) in c.gene_indices.iter().take(n).enumerate() {
            let name = gene_names
                .get(gi)
                .cloned()
                .unwrap_or_else(|| format!("g{}", gi));
            col_cluster.push(c.cluster_id);
            col_gene.push(name);
            col_scores.push(c.scores[k]);
            col_p.push(c.pvalues[k]);
            col_padj.push(c.pvalues_adj[k]);
            col_log2fc.push(c.log2fc[k]);
            col_pct.push(c.pct[k]);
            col_pct_rest.push(c.pct_rest[k]);
            col_mean.push(c.mean_count[k]);
        }
    }

    let mut df = DataFrame::new(vec![
        Column::new("cluster".into(), col_cluster),
        Column::new("gene".into(), col_gene),
        Column::new("scores".into(), col_scores),
        Column::new("pvalues".into(), col_p),
        Column::new("pvalues_adj".into(), col_padj),
        Column::new("log2fc".into(), col_log2fc),
        Column::new("pct".into(), col_pct),
        Column::new("pct_rest".into(), col_pct_rest),
        Column::new("mean_count".into(), col_mean),
    ])
    .map_err(|e| anyhow!("write_marker_csv: DataFrame::new: {e}"))?;

    let f = File::create(path)
        .map_err(|e| anyhow!("write_marker_csv: create {}: {e}", path.display()))?;
    let mut writer = CsvWriter::new(f);
    writer
        .finish(&mut df)
        .map_err(|e| anyhow!("write_marker_csv: finish: {e}"))?;
    Ok(())
}
