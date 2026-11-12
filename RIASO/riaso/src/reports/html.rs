use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use super::{file_name_of, html_escape, label_counts, stats, Report};
use crate::pipeline::{display_numbers, obs_f32, ORA_DBS, QC_METRICS};

const REP_MARKER_GENES: usize = 2;

fn representative_cluster(r: &Report<'_>) -> Option<(i64, i64)> {
    let labels = r.labels?;
    let sizes = super::cluster_sizes(labels, &display_numbers(labels));
    sizes
        .into_iter()
        .max_by_key(|&(d, n)| (n, std::cmp::Reverse(d)))
}

pub fn write(path: &Path, r: &Report<'_>) -> Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let exists = |f: &str| dir.join(f).is_file();
    let html = render(r, &exists);
    fs::write(path, html.as_bytes())
        .with_context(|| format!("failed to write report.html: {}", path.display()))?;
    Ok(())
}

pub fn render(r: &Report<'_>, exists: &dyn Fn(&str) -> bool) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = r.name;
    let multi = r.multi;

    let mut body = String::new();
    body.push_str(&qc_section(r, name, exists));
    body.push_str(&hvg_section(r, name, exists));
    body.push_str(&pca_section(r, name, multi, exists));
    body.push_str(&cluster_section(r, name, multi, exists));
    body.push_str(&marker_section(r, name, multi, exists));
    body.push_str(&annotation_section(r, name, multi, exists));
    body.push_str(&enrichment_section(r, exists));

    let toc = [
        ("sec1", "1. Quality Control"),
        ("sec2", "2. Highly Variable Genes"),
        ("sec3", "3. Dimensionality Reduction"),
        ("sec4", "4. Clustering"),
        ("sec5", "5. Marker Genes"),
        ("sec6", "6. Cell Type Annotation"),
        ("sec7", "7. Functional Enrichment"),
    ]
    .iter()
    .map(|(id, t)| format!("<li><a href=\"#{id}\">{}</a></li>", html_escape(t)))
    .collect::<Vec<_>>()
    .join("\n");

    let mut out = String::new();
    out.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    out.push_str(&format!(
        "<title>{} Analysis Report</title>\n",
        html_escape(name)
    ));
    out.push_str(STYLE);
    out.push_str("</head>\n<body>\n<div class=\"layout\">\n");
    out.push_str(&format!(
        "<nav class=\"sidebar\">\n<h1>{} Single-Cell Spatial Transcriptomics Analysis Report</h1>\n",
        html_escape(name)
    ));
    out.push_str(&format!(
        "<p class=\"subtitle\">RIASO v{}</p>\n",
        env!("CARGO_PKG_VERSION")
    ));
    out.push_str(&format!(
        "<h2 class=\"plain\">Contents</h2>\n<ul class=\"toc\">\n{toc}\n</ul>\n</nav>\n"
    ));
    out.push_str("<main>\n");
    out.push_str(&table(
        &["Item", "Value"],
        &[
            vec!["Sample".into(), esc(r.sample)],
            vec![
                "Input files".into(),
                r.config
                    .io
                    .inputs
                    .iter()
                    .map(|s| esc(file_name_of(s)))
                    .collect::<Vec<_>>()
                    .join("<br>"),
            ],
            vec!["Report time".into(), fmt_utc(now)],
        ],
    ));
    out.push_str(&body);
    out.push_str("</main>\n</div>\n");

    out.push_str(SCRIPT);
    out.push_str("</html>\n");
    out
}

const STYLE: &str = r#"<style>
html { scroll-behavior: smooth; }
body { font-family: "Helvetica Neue", Arial, sans-serif;
       margin: 2em auto; max-width: 1280px; line-height: 1.6; color: #222; padding: 0 1em; }
div.layout { display: flex; gap: 2em; align-items: flex-start; }
nav.sidebar { position: sticky; top: 1em; flex: 0 0 230px; max-height: calc(100vh - 2em);
       overflow-y: auto; background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 6px;
       padding: .6em 1em; }
nav.sidebar h1 { font-size: 1.1em; line-height: 1.35; border-bottom: 3px solid #2b6cb0;
       padding-bottom: .3em; margin: 0 0 .2em; }
nav.sidebar h2.plain { font-size: .95em; margin: .8em 0 .1em; border: none; color: #4a5568; }
nav.sidebar p.subtitle { font-size: .82em; margin: 0 0 .4em; }
ul.toc { column-count: 1; margin: .2em 0; padding-left: 1.2em; font-size: .92em; }
main { flex: 1 1 auto; min-width: 0; }
@media (max-width: 800px) {
  div.layout { display: block; }
  nav.sidebar { position: static; max-height: none; margin-bottom: 1.5em; }
}
h2 { border-bottom: 1px solid #cbd5e0; padding-bottom: .2em; margin-top: 2.2em; color: #2b6cb0;
     scroll-margin-top: 1em; }
h2.plain { border: none; color: #222; }
h3 { margin-top: 1.6em; color: #2d3748; }
p.subtitle { color: #718096; margin-top: -.6em; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; font-size: .92em; }
th, td { border: 1px solid #dde3ea; padding: .35em .6em; text-align: left; vertical-align: top; }
th { background: #edf2f7; }
tbody tr:nth-child(even) { background: #fafcfe; }
div.filterbar { margin: .8em 0 .2em; }
div.filterbar label { color: #4a5568; margin-right: .4em; font-size: .95em; }
div.filterbar select { padding: .15em .4em; font-size: .95em; max-width: 12em; }
table.filterable caption.rowcnt { caption-side: top; text-align: left; color: #718096;
       font-size: .88em; padding-bottom: .3em; }
table.filterable tbody tr:nth-child(even) { background: transparent; }
table.filterable tbody tr.alt { background: #fafcfe; }
figure { margin: 1.2em 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 4px; }
div.figrow { display: flex; gap: 1em; align-items: flex-start; }
div.figrow figure { flex: 1 1 0; min-width: 0; margin: .6em 0; }
div.figrow figure img { width: 100%; }
figcaption { color: #718096; font-size: .9em; margin-top: .4em; }
.note { color: #718096; font-size: .9em; }
</style>
"#;

fn esc(s: &str) -> String {
    html_escape(s)
}

fn num(x: f32) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "NA".to_string()
    }
}

fn num4(x: f32) -> String {
    if x.is_finite() {
        format!("{x:.4}")
    } else {
        "NA".to_string()
    }
}

fn nume(x: f32) -> String {
    if x.is_finite() {
        format!("{x:.3e}")
    } else {
        "NA".to_string()
    }
}

fn section(id: &str, title: &str, body: &str) -> String {
    format!("<h2 id=\"{id}\">{}</h2>\n{body}\n", html_escape(title))
}

fn table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut s = String::from("<table>\n<thead><tr>");
    for h in headers {
        s.push_str(&format!("<th>{}</th>", html_escape(h)));
    }
    s.push_str("</tr></thead>\n<tbody>\n");
    for row in rows {
        s.push_str("<tr>");
        for c in row {
            s.push_str(&format!("<td>{c}</td>"));
        }
        s.push_str("</tr>\n");
    }
    s.push_str("</tbody>\n</table>\n");
    s
}

fn figure(file: &str, caption: &str, exists: &dyn Fn(&str) -> bool) -> String {
    if !exists(file) {
        return String::new();
    }
    format!(
        "<figure><img src=\"{}\" alt=\"{}\"><figcaption>{}</figcaption></figure>\n",
        html_escape(file),
        html_escape(caption),
        html_escape(caption)
    )
}

fn figure_row(figs: &[(&str, &str)], exists: &dyn Fn(&str) -> bool) -> String {
    let mut inner = String::new();
    for (file, caption) in figs {
        inner.push_str(&figure(file, caption, exists));
    }
    if inner.is_empty() {
        return String::new();
    }
    format!("<div class=\"figrow\">{inner}</div>\n")
}

fn kv_table(rows: &[(&str, String)]) -> String {
    let body: Vec<Vec<String>> = rows.iter().map(|(k, v)| vec![esc(k), v.clone()]).collect();
    table(&["Item", "Value"], &body)
}

fn list_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    table(headers, rows)
}

fn filterable_table(
    id: &str,
    label: &str,
    mode: &str,
    options: &[(String, String)],
    headers: &[&str],
    rows: &[(String, Vec<String>)],
) -> String {
    let opts: String = options
        .iter()
        .map(|(v, t)| format!("<option value=\"{}\">{}</option>", esc(v), esc(t)))
        .collect();
    let mut s = format!(
        "<div class=\"filterbar\"><label for=\"{id}-sel\">{}</label>\
         <select id=\"{id}-sel\" data-filter=\"{mode}\" data-table=\"{id}\">{opts}</select></div>\n",
        esc(label)
    );
    s.push_str(&format!(
        "<table id=\"{id}\" class=\"filterable\">\n<caption class=\"rowcnt\">{} rows total, showing <span class=\"cnt\">{}</span></caption>\n",
        rows.len(),
        rows.len()
    ));
    s.push_str("<thead><tr>");
    for h in headers {
        s.push_str(&format!("<th>{}</th>", html_escape(h)));
    }
    s.push_str("</tr></thead>\n<tbody>\n");
    let attr = if mode == "cluster" { "cluster" } else { "idx" };
    for (i, (key, cells)) in rows.iter().enumerate() {
        let data = if mode == "cluster" {
            key.clone()
        } else {
            i.to_string()
        };
        s.push_str(&format!("<tr data-{attr}=\"{}\">", esc(&data)));
        for c in cells {
            s.push_str(&format!("<td>{c}</td>"));
        }
        s.push_str("</tr>\n");
    }
    s.push_str("</tbody>\n</table>\n");
    s
}

fn cluster_options(keys: impl IntoIterator<Item = i64>) -> Vec<(String, String)> {
    let mut ks: Vec<i64> = keys.into_iter().collect();
    ks.sort_unstable();
    ks.dedup();
    let mut opts: Vec<(String, String)> = vec![("*".into(), "All".into())];
    opts.extend(
        ks.into_iter()
            .map(|k| (k.to_string(), format!("cluster{k}"))),
    );
    opts
}

const SCRIPT: &str = r#"<script>
(function () {
  function apply(sel) {
    var tbl = document.getElementById(sel.dataset.table);
    if (!tbl) return;
    var mode = sel.dataset.filter, v = sel.value;
    var rows = tbl.tBodies[0].rows, shown = 0;
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      var ok = (v === '*');
      if (!ok) ok = (mode === 'cluster') ? (r.dataset.cluster === v) : (shown < parseInt(v, 10));
      r.style.display = ok ? '' : 'none';
      if (ok) { r.className = (shown % 2) ? 'alt' : ''; shown++; }
    }
    var n = tbl.querySelector('caption .cnt');
    if (n) n.textContent = String(shown);
  }
  var sels = document.querySelectorAll('select[data-filter]');
  for (var i = 0; i < sels.length; i++) {
    (function (s) {
      s.addEventListener('change', function () { apply(s); });
      apply(s);
    })(sels[i]);
  }
})();
</script>
</body>
"#;

fn qc_section(r: &Report<'_>, name: &str, exists: &dyn Fn(&str) -> bool) -> String {
    let mut rows = Vec::new();
    for metric in QC_METRICS {
        if let Ok(v) = obs_f32(r.sd, metric) {
            let (med, avg) = stats(&v);
            rows.push(vec![
                format!("<code>{metric}</code>"),
                num(med),
                num(avg),
                num(v
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(f32::NAN, f32::min)),
                num(v
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(f32::NAN, f32::max)),
            ]);
        }
    }
    let mut body = kv_table(&[
        ("Cells after filtering", format!("{}", r.sd.n_obs())),
        ("Genes after filtering", format!("{}", r.sd.n_vars())),
    ]);
    if !rows.is_empty() {
        body.push_str(&list_table(
            &["Metric", "Median", "Mean", "Min", "Max"],
            &rows,
        ));
    } else {
        body.push_str(
            "<p class=\"note\">(no QC metric column in obs; cal_qc/qc may not have run)</p>\n",
        );
    }
    body.push_str(&figure(
        &format!("{name}_Violin.png"),
        "QC Metric Violin Plots",
        exists,
    ));

    let spatial: Vec<(String, String)> = QC_METRICS
        .iter()
        .map(|m| {
            (
                format!("{name}_Spatial_Scatter_{m}.png"),
                format!("{m} spatial distribution"),
            )
        })
        .collect();
    let spatial_refs: Vec<(&str, &str)> = spatial
        .iter()
        .map(|(f, c)| (f.as_str(), c.as_str()))
        .collect();
    body.push_str(&figure_row(&spatial_refs, exists));
    body.push_str(&figure(
        &format!("{name}_Genes_Count.png"),
        "Gene Count Distribution Histogram (single sample)",
        exists,
    ));
    section("sec1", "1. Quality Control", &body)
}

fn hvg_section(r: &Report<'_>, name: &str, exists: &dyn Fn(&str) -> bool) -> String {
    let body = kv_table(&[
        ("Method", esc(&r.config.hvg_method)),
        ("Number of bins n_bins", format!("{}", r.config.n_bins)),
        (
            "Target count n_top_genes",
            format!("{}", r.config.n_top_genes),
        ),
    ]) + &figure(
        &format!("{name}_Highly_Variable_Genes.png"),
        "Highly Variable Genes Scatter",
        exists,
    );
    section("sec2", "2. Highly Variable Genes", &body)
}

fn pca_section(r: &Report<'_>, name: &str, multi: bool, exists: &dyn Fn(&str) -> bool) -> String {
    let mut body = kv_table(&[
        (
            "Number of components",
            format!("{}", r.config.pca.n_components),
        ),
        (
            "PCA uses HVG",
            if r.config.use_hvg { "Yes" } else { "No" }.into(),
        ),
    ]);
    if let Some(pca) = r.sd.result.pca.as_ref() {
        let rows: Vec<(String, Vec<String>)> = pca
            .explained_variance_ratio
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), vec![format!("PC{}", i + 1), num4(*v)]))
            .collect();
        let mut opts: Vec<(String, String)> = Vec::new();
        for n in [10usize, 20, 30] {
            if n < rows.len() {
                opts.push((n.to_string(), format!("Top {n}")));
            }
        }
        opts.push(("*".into(), "All".into()));
        body.push_str(&filterable_table(
            "pca-tbl",
            "Showing",
            "rows",
            &opts,
            &["Component", "Explained variance ratio"],
            &rows,
        ));
    }

    let elbow_cap = if multi {
        "PCA Scree Plot (multi-sample integration)"
    } else {
        "PCA Scree Plot (single sample)"
    };
    body.push_str(&figure(&format!("{name}_Elbow.png"), elbow_cap, exists));
    if multi {
        body.push_str(&figure(
            &format!("{name}_PCA.png"),
            "PCA Scatter (multi-sample integration)",
            exists,
        ));
    }
    section("sec3", "3. Dimensionality Reduction", &body)
}

fn cluster_section(
    r: &Report<'_>,
    name: &str,
    multi: bool,
    exists: &dyn Fn(&str) -> bool,
) -> String {
    let sizes = r.labels.map(|labels| {
        let display = display_numbers(labels);
        super::cluster_sizes(labels, &display)
    });

    let mut kv: Vec<(&str, String)> = vec![
        ("Algorithm", esc(r.config.cluster_alg.display())),
        ("resolution", num(r.config.resolution)),
    ];
    if let Some(sizes) = sizes.as_ref() {
        kv.push(("Number of clusters", format!("{}", sizes.len())));
    }
    let mut body = kv_table(&kv);
    if let Some(sizes) = sizes.as_ref() {
        let rows: Vec<(String, Vec<String>)> = sizes
            .iter()
            .map(|(d, n)| (d.to_string(), vec![format!("cluster{d}"), format!("{n}")]))
            .collect();
        body.push_str(&filterable_table(
            "cl-tbl",
            "Cluster",
            "cluster",
            &cluster_options(sizes.iter().map(|(d, _)| *d)),
            &["Cluster", "Number of cells"],
            &rows,
        ));
    } else {
        body.push_str("<p class=\"note\">(clustering did not run)</p>\n");
    }
    if multi {
        body.push_str(&figure(
            &format!("{name}_Cluster_Scatter_batch.png"),
            "Spatial distribution of cells by sample batch",
            exists,
        ));
        body.push_str(&figure(
            &format!("{name}_Cluster_Scatter_{}.png", r.alg.key()),
            "Spatial distribution of the clustering result",
            exists,
        ));
        body.push_str(&figure(
            &format!("{name}_Cluster_Umap.png"),
            "UMAP of the clustering result",
            exists,
        ));
    } else {
        body.push_str(&figure(
            &format!("{name}_Cluster_{}.png", r.alg.display()),
            "Spatial distribution of the clustering result",
            exists,
        ));
        body.push_str(&figure(
            &format!("{name}_Cluster_{}_Umap.png", r.alg.display()),
            "UMAP of the clustering result",
            exists,
        ));
    }
    section("sec4", "4. Clustering", &body)
}

fn marker_section(
    r: &Report<'_>,
    name: &str,
    multi: bool,
    exists: &dyn Fn(&str) -> bool,
) -> String {
    let mut body = String::new();
    if r.markers.is_empty() {
        body.push_str("<p class=\"note\">(no marker gene passed the filters)</p>\n");
    } else {
        let rows: Vec<(String, Vec<String>)> = r
            .markers
            .iter()
            .map(|m| {
                (
                    m.cluster.to_string(),
                    vec![
                        format!("cluster{}", m.cluster),
                        esc(&m.gene),
                        nume(m.scores),
                        num4(m.log2fc),
                        nume(m.pvalues),
                        nume(m.pvalues_adj),
                        num4(m.pct),
                        num4(m.pct_rest),
                    ],
                )
            })
            .collect();
        body.push_str(&format!(
            "<p class=\"note\">Top {} genes per cluster by descending log2fc (filters: log2fc >= 1, pct >= 0.25, pct_rest <= 0.5).</p>\n",
            r.config.top_n
        ));
        body.push_str(&filterable_table(
            "marker-tbl",
            "Cluster",
            "cluster",
            &cluster_options(r.markers.iter().map(|m| m.cluster)),
            &[
                "cluster",
                "genes",
                "scores",
                "log2fc",
                "pvalues",
                "pvalues_adj",
                "pct",
                "pct_rest",
            ],
            &rows,
        ));
    }
    let top_n = r.config.top_n;
    body.push_str(&figure(
        &format!("{name}_Marker_Gene_Filter_Top{top_n}_heatmap.png"),
        "Marker Gene Heatmap",
        exists,
    ));
    let suffix = if multi { "dotplot" } else { "scatter" };
    body.push_str(&figure(
        &format!("{name}_Marker_Gene_Filter_Top{top_n}_{suffix}.png"),
        "Marker Gene Expression (per cluster)",
        exists,
    ));

    if let Some((rep, n_cells)) = representative_cluster(r) {
        let figs: Vec<(String, String)> = r
            .markers
            .iter()
            .filter(|m| m.cluster == rep)
            .take(REP_MARKER_GENES)
            .map(|m| {
                (
                    format!("{name}_MarkerTop{top_n}_cluster{rep}_{}.png", m.gene),
                    format!("cluster{rep} · {} spatial distribution", m.gene),
                )
            })
            .collect();
        if !figs.is_empty() {
            body.push_str(&format!(
                "<h3>Spatial distribution of marker genes for the representative cluster{rep}</h3>\n<p class=\"note\">Representative cluster = the cluster with the most cells (cluster{rep}, {n_cells} cells); the top {} genes by log2fc in that cluster.</p>\n",
                figs.len()
            ));
            let refs: Vec<(&str, &str)> =
                figs.iter().map(|(f, c)| (f.as_str(), c.as_str())).collect();
            body.push_str(&figure_row(&refs, exists));
        }
    }
    section("sec5", "5. Marker Genes", &body)
}

fn annotation_section(
    r: &Report<'_>,
    name: &str,
    multi: bool,
    exists: &dyn Fn(&str) -> bool,
) -> String {
    let mut body = String::new();
    if r.ann_by_ref.is_empty() {
        body.push_str("<p class=\"note\">(SingleR annotation did not run)</p>\n");
    }
    for (ref_name, labels) in r.ann_by_ref {
        body.push_str(&format!("<h3>Reference: {}</h3>\n", esc(ref_name)));
        let rows: Vec<Vec<String>> = label_counts(labels)
            .into_iter()
            .map(|(k, v)| vec![esc(&k), format!("{v}")])
            .collect();
        body.push_str(&list_table(&["Cell type", "Number of cells"], &rows));
        body.push_str(&figure(
            &format!("{name}_SingleR_Heatmap_{ref_name}.png"),
            &format!("SingleR Score Heatmap ({ref_name})"),
            exists,
        ));
        let scatter = if multi {
            format!("{name}_Cluster_Scatter_{ref_name}.png")
        } else {
            format!("{name}_scatter_{ref_name}.png")
        };
        body.push_str(&figure(
            &scatter,
            &format!("Spatial distribution of the cell type annotation ({ref_name})"),
            exists,
        ));
    }
    section("sec6", "6. Cell Type Annotation", &body)
}

const ORA_ROWS_PER_GROUP: usize = 20;

fn enrichment_section(r: &Report<'_>, exists: &dyn Fn(&str) -> bool) -> String {
    let mut body = String::new();
    let Some(enr) = r.sd.result.enrichment.as_ref() else {
        return section(
            "sec7",
            "7. Functional Enrichment",
            "<p class=\"note\">(enrichment analysis did not run)</p>\n",
        );
    };
    let display = r.labels.map(display_numbers).unwrap_or_default();
    let disp = |raw: i64| display.get(&raw).copied().unwrap_or(raw);

    let mut any = false;
    let mut dbs: Vec<&String> = enr.groups.keys().chain(enr.gsea.keys()).collect();
    dbs.sort();
    dbs.dedup();
    for db in dbs {
        if let Some(groups) = enr.groups.get(db) {
            body.push_str(&format!("<h3>ORA -- {}</h3>\n", esc(db)));
            let mut rows = Vec::new();
            let mut keys = Vec::new();
            for g in groups {
                let d = disp(g.group_id);
                for t in g.terms.iter().take(ORA_ROWS_PER_GROUP) {
                    rows.push((
                        d.to_string(),
                        vec![
                            format!("cluster{d}"),
                            esc(&t.id),
                            esc(&t.name),
                            t.category.clone().map(|c| esc(&c)).unwrap_or_default(),
                            nume(t.pvalue),
                            nume(t.pvalue_adj),
                            format!("{}", t.count),
                        ],
                    ));
                    keys.push(d);
                    any = true;
                }
            }
            if rows.is_empty() {
                body.push_str("<p class=\"note\">(no significant terms)</p>\n");
            } else {
                body.push_str(&filterable_table(
                    &format!("ora-{db}-tbl"),
                    "Cluster",
                    "cluster",
                    &cluster_options(keys),
                    &[
                        "cluster",
                        "id",
                        "name",
                        "category",
                        "pvalue",
                        "pvalue_adj",
                        "count",
                    ],
                    &rows,
                ));
            }
        }
        if let Some(res) = enr.gsea.get(db) {
            body.push_str(&format!("<h3>GSEA -- {}</h3>\n", esc(db)));
            let mut rows = Vec::new();
            let mut keys = Vec::new();
            for g in &res.groups {
                let d = disp(g.group_id);
                for t in g.terms.iter().take(ORA_ROWS_PER_GROUP) {
                    rows.push((
                        d.to_string(),
                        vec![
                            format!("cluster{d}"),
                            esc(&t.id),
                            esc(&t.name),
                            num4(t.es),
                            num4(t.nes),
                            nume(t.pvalue),
                            nume(t.pvalue_adj),
                            format!("{}", t.size),
                        ],
                    ));
                    keys.push(d);
                    any = true;
                }
            }
            if rows.is_empty() {
                body.push_str("<p class=\"note\">(no significant terms)</p>\n");
            } else {
                body.push_str(&filterable_table(
                    &format!("gsea-{db}-tbl"),
                    "Cluster",
                    "cluster",
                    &cluster_options(keys),
                    &[
                        "cluster",
                        "id",
                        "name",
                        "es",
                        "nes",
                        "pvalue",
                        "pvalue_adj",
                        "size",
                    ],
                    &rows,
                ));
            }
        }
    }

    if let Some((rep, n_cells)) = representative_cluster(r) {
        let figs: Vec<(String, String)> = ORA_DBS
            .iter()
            .filter(|(_, key, _, _, _)| enr.groups.contains_key(*key))
            .map(|(display, _, png, _, _)| {
                (
                    format!("../{}_Pathways/cluster{rep}/{png}", r.sample),
                    format!("cluster{rep} · {display} enrichment"),
                )
            })
            .collect();
        if !figs.is_empty() {
            body.push_str(&format!(
                "<h3>Enrichment plots for the representative cluster{rep}</h3>\n<p class=\"note\">Representative cluster = the cluster with the most cells (cluster{rep}, {n_cells} cells).</p>\n"
            ));

            for (file, caption) in &figs {
                body.push_str(&figure(file, caption, exists));
            }
        }
    }
    if !any {
        body.push_str("<p class=\"note\">(enrichment results are empty)</p>\n");
    }
    body.push_str(&format!(
        "<p class=\"note\">Use the drop-down above each table to filter by cluster; only the top {ORA_ROWS_PER_GROUP} rows per cluster per database are listed. Full results are in <code>*_Pathways/cluster{{N}}/Pathway_*.xls</code> and <code>report.json</code> under the output directory.</p>\n"
    ));
    section("sec7", "7. Functional Enrichment", &body)
}

pub fn fmt_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}:{s:02} UTC")
}
