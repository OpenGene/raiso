use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use polars::prelude::{Column, CsvWriter, DataFrame, QuoteStyle, SerWriter};

use crate::algorithm::annotation::{run_singler, SingleRParams};
use crate::algorithm::batch::{run_harmony, HarmonyParams};
use crate::algorithm::cluster::{run_leiden, run_louvain};
use crate::algorithm::enrichment::{run_gsea, run_ora, OraParams};
use crate::algorithm::filter::{filter_cells, filter_coordinates, filter_genes};
use crate::algorithm::hvg::select_top_variable_genes;
use crate::algorithm::marker::{run_marker, write_marker_csv, MarkerParams};
use crate::algorithm::neighbors::run_neighbors;
use crate::algorithm::normalize::{log1p, normalize_total, scale};
use crate::algorithm::pca::{run_pca, PcaParams};
use crate::algorithm::qc::cal_qc;
use crate::algorithm::umap::run_umap;
use crate::config::{ClusterAlg, Config};
use crate::core::{GeneSetDb, MarkerResult, OraTerm, SoData};
use crate::io::{
    read_gef, read_gem, read_go_db, read_h5ad, read_kegg_db, read_reactome_db, read_ref,
};
use crate::plot::cluster_scatter::{plot_annotation_scatter, plot_cluster_scatter};
use crate::plot::gsea::{plot_gsea_barplot, plot_gsea_dotplot};
use crate::plot::heatmap::{plot_marker_dotplot, plot_marker_heatmap};
use crate::plot::hvg::plot_hvg;
use crate::plot::marker_spatial::plot_marker_spatial;
use crate::plot::ora::plot_ora_dotplot;
use crate::plot::pca::{plot_pca, plot_pca_variance_ratio};
use crate::plot::singler_heatmap::plot_singler_score_heatmap;
use crate::plot::umap::plot_umap;
use crate::plot::{plot_batch_spatial_scatter, plot_qc, plot_spatial_scatter, PlotKind};
use crate::tools::integrate;

pub const SAMPLE: &str = "Combined";

const MIN_FOLD_CHANGE: f64 = 1.0;
const MIN_IN_GROUP_FRACTION: f64 = 0.25;
const MAX_OUT_GROUP_FRACTION: f64 = 0.5;

pub(crate) const QC_METRICS: [&str; 3] = ["total_counts", "n_genes_by_counts", "pct_counts_mt"];

const MARKER_FIELDS: [&str; 7] = [
    "genes",
    "scores",
    "pvalues",
    "pvalues_adj",
    "log2fc",
    "pct",
    "pct_rest",
];
const MARKER_KEYS: [&str; 7] = [
    "gene",
    "scores",
    "pvalues",
    "pvalues_adj",
    "log2fc",
    "pct",
    "pct_rest",
];

pub(crate) const ORA_DBS: [(&str, &str, &str, f64, f64); 3] = [
    ("GO", "go", "Pathways_GO_Top10.png", 0.05, 0.2),
    ("KEGG", "kegg", "Pathways_KEGG.png", 0.05, 0.2),
    ("Reactome", "reactome", "Pathways_Reactome.png", 1.0, 1.0),
];

pub fn ref_name_of(ref_path: &str) -> Result<String> {
    Path::new(ref_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("cannot derive a file stem from the reference path: {ref_path}"))
}

fn done(name: &str, t: Instant) {
    eprintln!(
        "[riaso] ✓ {name} completed ({:.1}s)",
        t.elapsed().as_secs_f64()
    );
}

pub fn run(config: &Config) -> Result<()> {
    if config.io.inputs.is_empty() {
        bail!("at least one input file or directory must be given (-I/--input)");
    }
    // Directory inputs are expanded into file lists first; everything downstream works on files.
    let expanded = expand_inputs(&config.io.inputs)?;
    let mut cfg = config.clone();
    cfg.io.inputs = expanded;
    let config = &cfg;
    crate::validate::validate_static(config)?;
    let io = &config.io;
    if io.integrate {
        if io.inputs.len() < 2 {
            bail!("-i/--integrate needs at least 2 input files (drop -i to analyze each sample separately)");
        }
        ensure_same_input_type(&io.inputs)?;
        return run_one(config);
    }
    let mut seen = std::collections::HashSet::new();
    for path in &io.inputs {
        let stem = sample_stem(path)?;
        if !seen.insert(stem.clone()) {
            bail!("duplicate input file stem ({stem}); per-sample output directories would overwrite each other, rename the files or use -i to integrate");
        }
    }
    for path in &io.inputs {
        let mut one = config.clone();
        one.io.inputs = vec![path.clone()];
        run_one(&one).with_context(|| format!("sample failed: {path}"))?;
    }
    Ok(())
}

fn run_one(config: &Config) -> Result<()> {
    let io = &config.io;

    crate::parallel::init_rayon(config.threads);
    let multi = io.integrate;
    let sample = if multi {
        SAMPLE.to_string()
    } else {
        sample_stem(&io.inputs[0])?
    };
    let name = if matches!(input_ext(&io.inputs[0]).as_str(), "gef" | "gem") {
        format!("{sample}_{}", io.bin_size)
    } else {
        sample.clone()
    };
    let run_dir = PathBuf::from(&io.out_root).join(format!("run_{sample}"));
    let odir = run_dir.join(format!("{sample}_Riaso"));
    let pathways_dir = run_dir.join(format!("{sample}_Pathways"));

    let mut sd = if multi {
        let t = Instant::now();
        let mut datas = Vec::with_capacity(io.inputs.len());
        for path in &io.inputs {
            datas.push(read_input(path, config).with_context(|| format!("read failed: {path}"))?);
        }
        done("Reading input data", t);
        let t = Instant::now();
        let sd = integrate(&datas, false).context("integrate failed")?;
        done("Multi-sample integration", t);
        sd
    } else {
        let t = Instant::now();
        let sd = read_input(&io.inputs[0], config)
            .with_context(|| format!("read failed: {}", io.inputs[0]))?;
        done("Reading input data", t);
        sd
    };

    cal_qc(&mut sd);

    crate::validate::validate_against_data(config, &sd)?;

    reset_dirs(&[&odir, &pathways_dir])?;

    let (ncols, batch_size) = if multi {
        let n_batches = crate::plot::batch_count(&sd).unwrap_or(1);
        let ncols = 3usize.min(n_batches).max(1);
        let nrows = n_batches.div_ceil(ncols);
        (ncols, (400 * ncols as u32, 400 * nrows as u32))
    } else {
        (1usize, (800u32, 800u32))
    };
    let canvas = if multi { batch_size } else { (800, 800) };

    if config.steps.qc {
        let t = Instant::now();
        let qc_metrics: Vec<String> = QC_METRICS.iter().map(|s| s.to_string()).collect();
        let qc_size = (420 * qc_metrics.len() as u32, 360);
        plot_qc(
            &sd,
            p(&odir, &format!("{name}_Violin.png")).to_str().unwrap(),
            PlotKind::Violin,
            &qc_metrics,
            None,
            qc_size,
        )?;

        if !multi {
            plot_qc(
                &sd,
                p(&odir, &format!("{name}_Genes_Count.png"))
                    .to_str()
                    .unwrap(),
                PlotKind::Histogram,
                &qc_metrics,
                None,
                qc_size,
            )?;
        }

        for metric in &qc_metrics {
            let path = p(&odir, &format!("{name}_Spatial_Scatter_{metric}.png"));
            if multi {
                plot_batch_spatial_scatter(
                    &sd,
                    path.to_str().unwrap(),
                    metric.as_str(),
                    ncols,
                    None,
                    batch_size,
                    None,
                    None,
                    None,
                )?;
            } else {
                plot_spatial_scatter(
                    &sd,
                    path.to_str().unwrap(),
                    metric.as_str(),
                    None,
                    (800, 800),
                    None,
                    None,
                    None,
                )?;
            }
        }
        done("Quality control (QC)", t);
    }

    let cell_list = read_list_file(&config.cell_list)?;
    let gene_list = read_list_file(&config.gene_list)?;
    let any_filter =
        config.steps.filter_cells || config.steps.filter_genes || config.steps.filter_coordinates;
    let t = Instant::now();
    if config.steps.filter_cells {
        filter_cells(
            &mut sd,
            config.min_counts,
            config.max_counts,
            config.min_genes,
            config.max_genes,
            config.pct_counts_mt,
            cell_list.as_deref(),
            config.exclude_cells,
            false,
        );
    }
    if config.steps.filter_genes {
        filter_genes(
            &mut sd,
            config.min_cells,
            config.max_cells,
            config.gene_min_counts,
            config.gene_max_counts,
            config.gene_mean_umi,
            gene_list.as_deref(),
            config.exclude_genes,
            config.filter_mt_genes,
            false,
        );
    }
    if config.steps.filter_coordinates {
        filter_coordinates(
            &mut sd,
            config.min_x,
            config.max_x,
            config.min_y,
            config.max_y,
            false,
        );
    }
    if any_filter {
        done("Filtering cells/genes/coordinates", t);
    }

    if config.steps.normalize_total {
        let t = Instant::now();
        normalize_total(&mut sd, config.target_sum, false);
        done("Normalization", t);
    }
    if config.steps.log1p {
        let t = Instant::now();
        log1p(&mut sd, false);
        done("log1p", t);
    }

    let t = Instant::now();
    select_top_variable_genes(
        &mut sd,
        config.n_top_genes,
        config.n_bins,
        &config.hvg_method,
    );
    plot_hvg(
        &sd,
        p(&odir, &format!("{name}_Highly_Variable_Genes.png"))
            .to_str()
            .unwrap(),
        (1200, 600),
    )?;
    done("Highly variable genes", t);

    if config.steps.scale {
        let t = Instant::now();
        scale(&mut sd, config.zero_center, config.max_value, false);
        done("Scaling", t);
    }
    if config.steps.pca {
        let t = Instant::now();
        run_pca(
            &mut sd,
            &PcaParams {
                n_components: config.pca.n_components,
                random_state: config.random_state,
                svd_solver: config.pca.svd_solver,
            },
            config.use_hvg,
        );

        plot_pca_variance_ratio(
            &sd,
            p(&odir, &format!("{name}_Elbow.png")).to_str().unwrap(),
            config.pca.n_components,
            false,
            false,
            (900, 600),
        )?;
        done("PCA", t);
    }

    if config.steps.pca && (multi || config.steps.harmony) {
        let t = Instant::now();
        let mut hp: HarmonyParams = config.harmony.clone();
        hp.random_state = config.random_state;
        run_harmony(&mut sd, &hp);
        done("Harmony", t);
    }

    if multi && config.steps.pca {
        plot_pca(
            &sd,
            p(&odir, &format!("{name}_PCA.png")).to_str().unwrap(),
            None,
            (0, 1),
            None,
            (800, 720),
        )?;
    }

    if config.steps.neighbors {
        let t = Instant::now();
        run_neighbors(&mut sd, config.n_neighbors, config.n_pcs_neighbors);
        done("Neighbors", t);
    }
    if config.steps.umap {
        let t = Instant::now();
        run_umap(
            &mut sd,
            config.umap_components,
            config.min_dist,
            config.spread,
            config.random_state,
        );
        done("UMAP", t);
    }
    let alg = config.cluster_alg;
    let mut labels_opt: Option<Vec<i64>> = None;
    if config.steps.clustering {
        let t = Instant::now();
        match alg {
            ClusterAlg::Leiden => run_leiden(&mut sd, config.resolution, config.random_state),
            ClusterAlg::Louvain => run_louvain(&mut sd, config.resolution, config.random_state),
        }
        labels_opt = Some(obs_labels(&sd, alg.key())?);
        done("Clustering", t);
    }
    let display_of = labels_opt.as_ref().map(|l| display_numbers(l));

    if config.steps.clustering {
        if multi {
            plot_cluster_scatter(
                &sd,
                p(&odir, &format!("{name}_Cluster_Scatter_batch.png"))
                    .to_str()
                    .unwrap(),
                "batch",
                None,
                None,
                None,
                None,
                batch_size,
                sd.uns.contains_key("resolution"),
                None,
                None,
                0.0,
                0.0,
                true,
            )?;
            plot_cluster_scatter(
                &sd,
                p(&odir, &format!("{name}_Cluster_Scatter_{}.png", alg.key()))
                    .to_str()
                    .unwrap(),
                alg.key(),
                None,
                None,
                None,
                None,
                batch_size,
                sd.uns.contains_key("resolution"),
                None,
                None,
                0.0,
                0.0,
                true,
            )?;
            if config.steps.umap {
                plot_umap(
                    &sd,
                    p(&odir, &format!("{name}_Cluster_Umap.png"))
                        .to_str()
                        .unwrap(),
                    alg.key(),
                    None,
                    "umap1",
                    "umap2",
                    None,
                    (800, 800),
                )?;
            }
        } else {
            plot_cluster_scatter(
                &sd,
                p(&odir, &format!("{name}_Cluster_{}.png", alg.display()))
                    .to_str()
                    .unwrap(),
                alg.key(),
                None,
                None,
                None,
                None,
                canvas,
                sd.uns.contains_key("resolution"),
                None,
                None,
                0.0,
                0.0,
                false,
            )?;
            if config.steps.umap {
                plot_umap(
                    &sd,
                    p(&odir, &format!("{name}_Cluster_{}_Umap.png", alg.display()))
                        .to_str()
                        .unwrap(),
                    alg.key(),
                    None,
                    "umap1",
                    "umap2",
                    None,
                    (800, 800),
                )?;
            }
        }
    }

    let top_n = config.top_n;
    let mut per_cluster_opt: Option<HashMap<i64, ClusterRows>> = None;
    if config.steps.marker {
        let t = Instant::now();
        let labels = labels_opt
            .as_ref()
            .ok_or_else(|| anyhow!("marker requires clustering first"))?;
        let disp = display_of.as_ref().unwrap();
        run_marker(
            &mut sd,
            labels,
            &MarkerParams {
                n_genes: None,
                ..Default::default()
            },
        );
        let marker = sd
            .result
            .marker
            .as_ref()
            .ok_or_else(|| anyhow!("marker computation failed"))?;
        let syms = sd.symbol_names();

        write_marker_csv(
            marker,
            &syms,
            &p(&odir, &format!("{name}_Marker_Gene.csv")),
            Some(top_n),
        )?;
        let per = write_stereo_marker_wide(
            marker,
            &syms,
            &p(&odir, &format!("{name}_Marker_Gene_Filter.csv")),
            disp,
        )?;

        plot_marker_heatmap(
            &sd,
            p(
                &odir,
                &format!("{name}_Marker_Gene_Filter_Top{top_n}_heatmap.png"),
            )
            .to_str()
            .unwrap(),
            alg.key(),
            top_n,
            None,
            None,
            (1200, 1200),
        )?;

        let dot_suffix = if multi { "dotplot" } else { "scatter" };
        plot_marker_dotplot(
            &sd,
            p(
                &odir,
                &format!("{name}_Marker_Gene_Filter_Top{top_n}_{dot_suffix}.png"),
            )
            .to_str()
            .unwrap(),
            alg.key(),
            top_n,
            None,
            None,
            (1200, 1200),
        )?;

        let mut ordered: Vec<(i64, &ClusterRows)> = per.iter().map(|(k, v)| (*k, v)).collect();
        ordered.sort_by_key(|(cid, _)| disp[cid]);
        for (cid, rows) in &ordered {
            let num = disp[cid];
            for gene in rows.top_genes(top_n) {
                plot_marker_spatial(
                    &sd,
                    std::slice::from_ref(&gene),
                    p(
                        &odir,
                        &format!("{name}_MarkerTop{top_n}_cluster{num}_{gene}.png"),
                    )
                    .to_str()
                    .unwrap(),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )?;
            }
        }
        per_cluster_opt = Some(per);
        done("Marker genes", t);
    }

    if let (Some(per), Some(disp)) = (per_cluster_opt.as_ref(), display_of.as_ref()) {
        let t = Instant::now();
        let mut ordered: Vec<(i64, &ClusterRows)> = per.iter().map(|(k, v)| (*k, v)).collect();
        ordered.sort_by_key(|(cid, _)| disp[cid]);
        for (cid, rows) in ordered {
            let num = disp[&cid];
            let cdir = pathways_dir.join(format!("cluster{num}"));
            fs::create_dir_all(&cdir)?;
            write_cluster_marker_tsv(&cdir.join("MarkerGeneFilter.xls"), rows, num)?;
        }
        done("Per-cluster marker files", t);
    }

    let mut ann_by_ref: Vec<(String, Vec<String>)> = Vec::new();
    if config.steps.singler {
        let t = Instant::now();
        let labels = labels_opt
            .as_ref()
            .ok_or_else(|| anyhow!("SingleR requires clustering first"))?;
        let disp = display_of.as_ref().unwrap();
        ann_by_ref = run_refs(
            &mut sd,
            labels,
            &config.references,
            &odir,
            &name,
            &config.label_key,
            &config.singler,
        )?;
        write_cell_metadata(
            &p(&odir, &format!("{name}_SingleR_Metadata.csv")),
            &sd,
            &ann_by_ref,
            disp,
            alg.key(),
        )?;
        for (ref_name, ref_labels) in &ann_by_ref {
            let fname = if multi {
                format!("{name}_Cluster_Scatter_{ref_name}.png")
            } else {
                format!("{name}_scatter_{ref_name}.png")
            };
            plot_annotation_scatter(
                &sd,
                p(&odir, &fname).to_str().unwrap(),
                ref_labels,
                None,
                None,
                None,
                None,
                canvas,
                sd.uns.contains_key("resolution"),
                None,
                None,
                0.0,
                0.0,
                multi,
            )?;
        }
        done("Cell type annotation (SingleR)", t);
    }

    let enabled = [
        config.steps.ora_go,
        config.steps.ora_kegg,
        config.steps.ora_reactome,
    ];
    let any_ora = enabled.iter().any(|&b| b);
    if any_ora {
        let t = Instant::now();
        let disp = display_of
            .as_ref()
            .ok_or_else(|| anyhow!("enrichment requires clustering first"))?;
        let dbs = load_enabled_dbs(config, &enabled)?;
        enrich_ora(&mut sd, config, &dbs, &pathways_dir, disp)?;
        if config.steps.gsea {
            enrich_gsea(&mut sd, config, &dbs, &pathways_dir, disp)?;
        }
        done("Enrichment (ORA/GSEA)", t);
    }

    let t = Instant::now();
    let report_markers = match (per_cluster_opt.as_ref(), display_of.as_ref()) {
        (Some(per), Some(disp)) => crate::reports::marker_rows(per, disp, top_n),
        _ => Vec::new(),
    };
    let report = crate::reports::Report {
        sd: &sd,
        config,
        name: name.as_str(),
        sample: sample.as_str(),
        multi,
        alg,
        labels: labels_opt.as_deref(),
        markers: report_markers.as_slice(),
        ann_by_ref: ann_by_ref.as_slice(),
    };
    crate::reports::json::write(&p(&odir, "report.json"), &report)?;
    crate::reports::html::write(&p(&odir, "report.html"), &report)?;
    done("Generating report", t);

    Ok(())
}

fn sample_stem(path: &str) -> Result<String> {
    Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("cannot derive a file stem from the input path: {path}"))
}

fn input_ext(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default()
}

/// Expand **directory** entries in `-I` into the files they contain (**top level only**).
///
/// - Only files with a `.gef` / `.gem` / `.h5ad` extension are kept; anything else
///   (readme files, hidden files, sub-directories) is ignored;
/// - files inside one directory are sorted by **file name** so batch order is stable
///   and reproducible; multiple directories are expanded in command-line order;
/// - a directory with no supported file is an error instead of being skipped silently;
/// - non-directory entries are passed through unchanged (extension checks happen later).
fn expand_inputs(inputs: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(inputs.len());
    for raw in inputs {
        if !Path::new(raw).is_dir() {
            out.push(raw.clone());
            continue;
        }
        let mut files = Vec::new();
        let entries = std::fs::read_dir(raw)
            .with_context(|| format!("cannot open input directory: {raw}"))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("cannot read input directory: {raw}"))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let shown = path.to_string_lossy().into_owned();
            if matches!(input_ext(&shown).as_str(), "gef" | "gem" | "h5ad") {
                files.push(shown);
            }
        }
        if files.is_empty() {
            bail!("the input directory contains no supported file (.gef/.gem/.h5ad): {raw}");
        }
        files.sort();
        out.extend(files);
    }
    Ok(out)
}

fn ensure_same_input_type(inputs: &[String]) -> Result<()> {
    let mut first: Option<(String, &str)> = None;
    for path in inputs {
        let ext = input_ext(path);
        if !matches!(ext.as_str(), "gef" | "gem" | "h5ad") {
            bail!("unsupported input extension {ext:?} (supported: .gef/.gem/.h5ad): {path}");
        }
        match &first {
            None => first = Some((ext, path.as_str())),
            Some((first_ext, first_path)) if first_ext != &ext => bail!(
                "integration needs all inputs of the same type: {first_path} is .{first_ext}, {path} is .{ext}; unify the file types or run them separately without -i"
            ),
            _ => {}
        }
    }
    Ok(())
}

fn read_input(path: &str, config: &Config) -> Result<SoData> {
    let io = &config.io;
    let ext = input_ext(path);
    match ext.as_str() {
        "gef" => read_gef(
            path,
            &io.bin_type,
            io.bin_size,
            !io.dense,
            io.dedup_symbols,
            false,
            false,
        ),
        "gem" => read_gem(path, io.bin_size, !io.dense, io.dedup_symbols, false, false),
        "h5ad" => read_h5ad(path, &io.spatial_key, false),
        other => {
            bail!("unsupported input extension {other:?} (supported: .gef/.gem/.h5ad): {path}")
        }
    }
}

fn read_list_file(path: &Option<PathBuf>) -> Result<Option<Vec<String>>> {
    match path {
        None => Ok(None),
        Some(p) => {
            let text = fs::read_to_string(p)
                .with_context(|| format!("failed to read the list file: {}", p.display()))?;
            Ok(Some(
                text.lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .collect(),
            ))
        }
    }
}

fn run_refs(
    sd: &mut SoData,
    labels: &[i64],
    ref_paths: &[PathBuf],
    odir: &Path,
    name: &str,
    label_key: &str,
    params: &SingleRParams,
) -> Result<Vec<(String, Vec<String>)>> {
    let paths: Vec<String> = ref_paths
        .iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    let refstr: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    let names = ref_names(&refstr)?;
    let mut out = Vec::with_capacity(refstr.len());
    for (ref_path, ref_name) in refstr.iter().zip(&names) {
        let rf = read_ref(ref_path, label_key, None)
            .with_context(|| format!("read_ref failed: {ref_path}"))?;
        run_singler(sd, &rf, Some(labels), params);
        let ann = sd
            .result
            .annotation
            .as_ref()
            .ok_or_else(|| anyhow!("singler failed ({ref_name})"))?;
        let final_labels = ann.final_labels.clone();
        plot_singler_score_heatmap(
            sd,
            p(odir, &format!("{name}_SingleR_Heatmap_{ref_name}.png"))
                .to_str()
                .unwrap(),
            40,
            true,
            None,
            (1400, 700),
        )?;
        out.push((ref_name.clone(), final_labels));
    }
    Ok(out)
}

fn ref_names(ref_paths: &[&str]) -> Result<Vec<String>> {
    if ref_paths.is_empty() {
        bail!("at least one SingleR reference path must be given");
    }
    let mut names: Vec<String> = Vec::with_capacity(ref_paths.len());
    for path in ref_paths {
        let n = ref_name_of(path)?;
        if names.contains(&n) {
            bail!("duplicated reference file stem: {n:?} (artifact names only carry the stem, so they would overwrite each other)");
        }
        names.push(n);
    }
    Ok(names)
}

struct LoadedDb<'a> {
    display: &'a str,
    key: &'a str,
    png: &'a str,
    def_p: f64,
    def_q: f64,
    db: GeneSetDb,
}

fn load_enabled_dbs(config: &Config, enabled: &[bool; 3]) -> Result<Vec<LoadedDb<'static>>> {
    let mut out = Vec::new();
    for (i, (display, key, png, def_p, def_q)) in ORA_DBS.iter().enumerate() {
        if !enabled[i] {
            continue;
        }
        out.push(LoadedDb {
            display,
            key,
            png,
            def_p: *def_p,
            def_q: *def_q,
            db: load_db(display, config)?,
        });
    }
    Ok(out)
}

fn load_db(display: &str, config: &Config) -> Result<GeneSetDb> {
    let org = config.org_db.to_str().ok_or_else(|| {
        anyhow!(
            "--org-db path is not valid UTF-8: {}",
            config.org_db.display()
        )
    })?;
    match display {
        "GO" => read_go_db(
            org,
            config
                .go_db
                .to_str()
                .ok_or_else(|| anyhow!("--go-db path is not valid UTF-8"))?,
        )
        .context("read_go_db failed"),
        "KEGG" => read_kegg_db(
            config
                .kegg_dir
                .to_str()
                .ok_or_else(|| anyhow!("--kegg-dir path is not valid UTF-8"))?,
            org,
            &config.kegg_org,
        )
        .context("read_kegg_db failed"),
        "Reactome" => read_reactome_db(
            config
                .reactome_dir
                .to_str()
                .ok_or_else(|| anyhow!("--reactome-dir path is not valid UTF-8"))?,
            org,
            &config.reactome_species,
        )
        .context("read_reactome_db failed"),
        other => bail!("unknown enrichment database: {other}"),
    }
}

fn enrich_ora(
    sd: &mut SoData,
    config: &Config,
    dbs: &[LoadedDb<'_>],
    pathways_dir: &Path,
    display_of: &HashMap<i64, i64>,
) -> Result<()> {
    for ld in dbs {
        let LoadedDb {
            display,
            key: db_key,
            def_p,
            def_q,
            db,
            ..
        } = ld;
        let params = OraParams {
            category: if *display == "GO" {
                config.ora_category.clone()
            } else {
                None
            },
            gene_pvalue_cutoff: config.ora_gene_pvalue_cutoff,
            gene_log2fc_cutoff: config.ora_gene_log2fc_cutoff,
            pvalue_cutoff: config.ora_pvalue_cutoff.unwrap_or(*def_p),
            qvalue_cutoff: config.ora_qvalue_cutoff.unwrap_or(*def_q),
            min_gene_count: config.ora_min_gene_count,
            max_gene_count: config.ora_max_gene_count,
            top_n_terms: config.ora_top_n_terms,
        };
        run_ora(sd, db, &params);
        let groups = sd
            .result
            .enrichment
            .as_ref()
            .ok_or_else(|| anyhow!("run_ora failed ({display})"))?
            .groups
            .get(*db_key)
            .ok_or_else(|| anyhow!("enrichment results are missing the database slot {db_key}"))?;

        for g in groups {
            let num = display_of.get(&g.group_id).copied().unwrap_or(g.group_id);
            let cdir = pathways_dir.join(format!("cluster{num}"));
            fs::create_dir_all(&cdir)?;
            if g.terms.is_empty() {
                continue;
            }
            write_terms_tsv(&cdir.join(format!("Pathway_{display}.xls")), &g.terms)?;
            plot_ora_dotplot(
                sd,
                db_key,
                cdir.join(ld.png).to_str().unwrap(),
                Some(g.group_id),
                15,
                None,
                (1200, 900),
            )?;
        }
    }
    Ok(())
}

fn enrich_gsea(
    sd: &mut SoData,
    config: &Config,
    dbs: &[LoadedDb<'_>],
    pathways_dir: &Path,
    display_of: &HashMap<i64, i64>,
) -> Result<()> {
    for ld in dbs {
        let LoadedDb {
            display,
            key: db_key,
            db,
            ..
        } = ld;
        run_gsea(sd, db, &config.gsea);
        let groups = &sd
            .result
            .enrichment
            .as_ref()
            .ok_or_else(|| anyhow!("run_gsea failed ({display})"))?
            .gsea
            .get(*db_key)
            .ok_or_else(|| anyhow!("GSEA results are missing the database slot {db_key}"))?
            .groups;

        for g in groups {
            let num = display_of.get(&g.group_id).copied().unwrap_or(g.group_id);
            let cdir = pathways_dir.join(format!("cluster{num}"));
            fs::create_dir_all(&cdir)?;
            if g.terms.is_empty() {
                continue;
            }
            let top = config.gsea.top_n_terms;
            plot_gsea_dotplot(
                sd,
                db_key,
                cdir.join(format!("GSEA_{display}_dotplot.png"))
                    .to_str()
                    .unwrap(),
                Some(g.group_id),
                top,
                None,
                (1200, 900),
            )?;
            plot_gsea_barplot(
                sd,
                db_key,
                cdir.join(format!("GSEA_{display}_barplot.png"))
                    .to_str()
                    .unwrap(),
                Some(g.group_id),
                top,
                None,
                (1200, 900),
            )?;
        }
    }
    Ok(())
}

fn obs_labels(data: &SoData, key: &str) -> Result<Vec<i64>> {
    let obs = data.obs.as_ref().ok_or_else(|| anyhow!("obs is empty"))?;
    let col = obs
        .column(key)
        .map_err(|_| anyhow!("obs is missing column {key}: run clustering first"))?;
    let labels: Vec<i64> = col
        .i64()
        .map_err(|_| anyhow!("obs[{key}] is not Int64"))?
        .into_no_null_iter()
        .collect();
    Ok(labels)
}

pub(crate) fn display_numbers(labels: &[i64]) -> HashMap<i64, i64> {
    labels.iter().map(|&l| (l, l)).collect()
}

fn reset_dirs(paths: &[&Path]) -> Result<()> {
    for path in paths {
        if path.is_dir() {
            for _ in 0..10 {
                let _ = fs::remove_dir_all(path);
                if !path.exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(300));
            }
            if path.exists() {
                let stale = path.with_extension(format!("stale.{}", std::process::id()));
                let _ = fs::rename(path, &stale);
            }
        }
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn p(dir: &Path, name: &str) -> PathBuf {
    dir.join(name)
}

pub(crate) fn obs_f64(sd: &SoData, col: &str) -> Result<Vec<f64>> {
    let obs = sd.obs.as_ref().ok_or_else(|| anyhow!("obs is empty"))?;
    let s = obs
        .column(col)
        .map_err(|e| anyhow!("obs is missing column {col}: {e}"))?;
    if let Ok(ca) = s.f64() {
        return Ok((0..ca.len())
            .map(|i| ca.get(i).unwrap_or(f64::NAN))
            .collect());
    }
    if let Ok(ca) = s.i64() {
        return Ok((0..ca.len())
            .map(|i| ca.get(i).map(|v| v as f64).unwrap_or(f64::NAN))
            .collect());
    }
    Err(anyhow!("obs[{col}] is neither f64 nor Int64"))
}

fn obs_i64(sd: &SoData, col: &str) -> Result<Vec<i64>> {
    let obs = sd.obs.as_ref().ok_or_else(|| anyhow!("obs is empty"))?;
    let ca = obs
        .column(col)
        .map_err(|e| anyhow!("obs is missing column {col}: {e}"))?
        .i64()
        .map_err(|e| anyhow!("obs[{col}] is not i64: {e}"))?;
    Ok((0..ca.len()).map(|i| ca.get(i).unwrap_or(0)).collect())
}

fn obs_str(sd: &SoData, col: &str) -> Result<Vec<String>> {
    let obs = sd.obs.as_ref().ok_or_else(|| anyhow!("obs is empty"))?;
    let ca = obs
        .column(col)
        .map_err(|e| anyhow!("obs is missing column {col}: {e}"))?
        .str()
        .map_err(|e| anyhow!("obs[{col}] is not a string: {e}"))?;
    Ok((0..ca.len())
        .map(|i| ca.get(i).unwrap_or("").to_string())
        .collect())
}

pub(crate) struct ClusterRows {
    pub(crate) genes: Vec<String>,
    pub(crate) scores: Vec<f64>,
    pub(crate) pvalues: Vec<f64>,
    pub(crate) pvalues_adj: Vec<f64>,
    pub(crate) log2fc: Vec<f64>,
    pub(crate) pct: Vec<f64>,
    pub(crate) pct_rest: Vec<f64>,
}

impl ClusterRows {
    fn top_genes(&self, n: usize) -> Vec<String> {
        self.genes.iter().take(n).cloned().collect()
    }

    fn field(&self, key: &str) -> ClusterField<'_> {
        match key {
            "gene" => ClusterField::Str(&self.genes),
            "scores" => ClusterField::F64(&self.scores),
            "pvalues" => ClusterField::F64(&self.pvalues),
            "pvalues_adj" => ClusterField::F64(&self.pvalues_adj),
            "log2fc" => ClusterField::F64(&self.log2fc),
            "pct" => ClusterField::F64(&self.pct),
            "pct_rest" => ClusterField::F64(&self.pct_rest),
            _ => ClusterField::Str(&self.genes),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.genes.len()
    }
}

enum ClusterField<'a> {
    Str(&'a [String]),
    F64(&'a [f64]),
}

fn write_stereo_marker_wide(
    marker: &MarkerResult,
    syms: &[String],
    out_csv: &Path,
    display_of: &HashMap<i64, i64>,
) -> Result<HashMap<i64, ClusterRows>> {
    let mut per: HashMap<i64, ClusterRows> = HashMap::new();
    for c in &marker.clusters {
        let mut rows = ClusterRows {
            genes: Vec::new(),
            scores: Vec::new(),
            pvalues: Vec::new(),
            pvalues_adj: Vec::new(),
            log2fc: Vec::new(),
            pct: Vec::new(),
            pct_rest: Vec::new(),
        };
        for (k, &gi) in c.gene_indices.iter().enumerate() {
            if c.log2fc[k] < MIN_FOLD_CHANGE
                || c.pct[k] < MIN_IN_GROUP_FRACTION
                || c.pct_rest[k] > MAX_OUT_GROUP_FRACTION
            {
                continue;
            }
            let name = syms.get(gi).cloned().unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            rows.genes.push(name);
            rows.scores.push(c.scores[k]);
            rows.pvalues.push(c.pvalues[k]);
            rows.pvalues_adj.push(c.pvalues_adj[k]);
            rows.log2fc.push(c.log2fc[k]);
            rows.pct.push(c.pct[k]);
            rows.pct_rest.push(c.pct_rest[k]);
        }

        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| {
            rows.log2fc[b]
                .partial_cmp(&rows.log2fc[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let reorder = |v: &mut Vec<f64>| {
            let old = std::mem::take(v);
            v.extend(idx.iter().map(|&i| old[i]));
        };
        let gene_old = std::mem::take(&mut rows.genes);
        rows.genes = idx.iter().map(|&i| gene_old[i].clone()).collect();
        reorder(&mut rows.scores);
        reorder(&mut rows.pvalues);
        reorder(&mut rows.pvalues_adj);
        reorder(&mut rows.log2fc);
        reorder(&mut rows.pct);
        reorder(&mut rows.pct_rest);
        per.insert(c.cluster_id, rows);
    }

    let mut order: Vec<i64> = per.keys().copied().collect();
    order.sort_by_key(|c| display_of.get(c).copied().unwrap_or(*c));

    let mut columns: Vec<Column> = Vec::new();
    let n_rows = order.iter().map(|c| per[c].len()).max().unwrap_or(0);
    columns.push(Column::new(
        "".into(),
        (0..n_rows as i64).collect::<Vec<i64>>(),
    ));

    for &cid in &order {
        let rows = &per[&cid];
        let disp = display_of.get(&cid).copied().unwrap_or(cid);
        for (fi, field) in MARKER_FIELDS.iter().enumerate() {
            let col_name = format!("{disp}_{field}");
            let key = MARKER_KEYS[fi];
            let col = match rows.field(key) {
                ClusterField::Str(v) => {
                    let padded: Vec<Option<String>> =
                        (0..n_rows).map(|i| v.get(i).cloned()).collect();
                    Column::new(col_name.into(), padded)
                }
                ClusterField::F64(v) => {
                    let padded: Vec<Option<f64>> = (0..n_rows).map(|i| v.get(i).copied()).collect();
                    Column::new(col_name.into(), padded)
                }
            };
            columns.push(col);
        }
    }

    write_csv(out_csv, columns, b',')?;
    Ok(per)
}

fn write_terms_tsv(path: &Path, terms: &[OraTerm]) -> Result<()> {
    let ids: Vec<String> = terms.iter().map(|t| t.id.clone()).collect();
    let names: Vec<String> = terms.iter().map(|t| t.name.clone()).collect();
    let cats: Vec<Option<String>> = terms.iter().map(|t| t.category.clone()).collect();
    let pv: Vec<f64> = terms.iter().map(|t| t.pvalue).collect();
    let padj: Vec<f64> = terms.iter().map(|t| t.pvalue_adj).collect();
    let counts: Vec<u64> = terms.iter().map(|t| t.count as u64).collect();
    let gratio: Vec<f64> = terms.iter().map(|t| t.gene_ratio).collect();
    let bratio: Vec<f64> = terms.iter().map(|t| t.bg_ratio).collect();
    let overlap: Vec<String> = terms.iter().map(|t| t.overlap.join("/")).collect();

    let columns = vec![
        Column::new("ID".into(), ids),
        Column::new("Description".into(), names),
        Column::new("category".into(), cats),
        Column::new("pvalue".into(), pv),
        Column::new("pvalue_adj".into(), padj),
        Column::new("count".into(), counts),
        Column::new("gene_ratio".into(), gratio),
        Column::new("bg_ratio".into(), bratio),
        Column::new("overlap".into(), overlap),
    ];
    write_csv(path, columns, b'\t')
}

fn write_cluster_marker_tsv(path: &Path, rows: &ClusterRows, display: i64) -> Result<()> {
    let mut columns: Vec<Column> = Vec::new();
    for (fi, field) in MARKER_FIELDS.iter().enumerate() {
        let col_name = format!("X{display}_{field}");
        let key = MARKER_KEYS[fi];
        let col = match rows.field(key) {
            ClusterField::Str(v) => Column::new(col_name.into(), v.to_vec()),
            ClusterField::F64(v) => Column::new(col_name.into(), v.to_vec()),
        };
        columns.push(col);
    }
    write_csv(path, columns, b'\t')
}

fn write_cell_metadata(
    path: &Path,
    sd: &SoData,
    ann_by_ref: &[(String, Vec<String>)],
    display_of: &HashMap<i64, i64>,
    cluster_key: &str,
) -> Result<()> {
    let cell = obs_str(sd, "cell")?;
    let total_counts = obs_f64(sd, "total_counts")?;
    let n_genes = obs_i64(sd, "n_genes_by_counts")?;
    let pct_mt = obs_f64(sd, "pct_counts_mt")?;
    let clusters: Vec<i64> = obs_labels(sd, cluster_key)?
        .iter()
        .map(|x| display_of.get(x).copied().unwrap_or(*x))
        .collect();

    let mut columns: Vec<Column> = vec![
        Column::new("".into(), cell),
        Column::new("total_counts".into(), total_counts),
        Column::new("n_genes_by_counts".into(), n_genes),
        Column::new("pct_counts_mt".into(), pct_mt),
        Column::new("clustering".into(), clusters),
    ];
    for (ref_name, ref_labels) in ann_by_ref {
        columns.push(Column::new(ref_name.as_str().into(), ref_labels.clone()));
    }
    write_csv(path, columns, b',')
}

fn write_csv(path: &Path, columns: Vec<Column>, separator: u8) -> Result<()> {
    const SENTINEL: u8 = 0x1f;
    let n_cols = columns.len();
    let mut df = DataFrame::new(columns).map_err(|e| anyhow!("DataFrame::new: {e}"))?;
    let mut buf: Vec<u8> = Vec::new();
    CsvWriter::new(&mut buf)
        .with_separator(SENTINEL)
        .with_quote_style(QuoteStyle::Never)
        .finish(&mut df)
        .map_err(|e| anyhow!("csv finish: {e}"))?;
    let text = String::from_utf8(buf).map_err(|e| anyhow!("csv utf8: {e}"))?;

    let mut out = String::with_capacity(text.len() + 64);
    for raw in text.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        let fields: Vec<&str> = line.split(char::from(SENTINEL)).collect();
        if fields.len() != n_cols {
            return Err(anyhow!(
                "csv row has {} fields but {n_cols} columns (a cell contains U+001F or a newline)",
                fields.len()
            ));
        }
        for (i, f) in fields.iter().enumerate() {
            if i > 0 {
                out.push(separator as char);
            }
            push_python_field(&mut out, f, separator);
        }
        out.push('\n');
    }
    fs::write(path, out).map_err(|e| anyhow!("write {}: {e}", path.display()))?;
    Ok(())
}

fn push_python_field(out: &mut String, field: &str, separator: u8) {
    let sep = separator as char;
    if field.contains(sep) || field.contains('"') || field.contains('\n') || field.contains('\r') {
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
    } else {
        out.push_str(field);
    }
}
