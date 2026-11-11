use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use polars::prelude::{AnyValue, DataFrame};

use crate::algorithm::pca::SvdSolver;
use crate::config::Config;
use crate::core::SoData;

const COMMON_BIN_SIZES: [u32; 10] = [1, 10, 20, 50, 100, 150, 200, 250, 500, 1000];

struct Checker {
    errors: Vec<String>,
    warnings: Vec<String>,
}

impl Checker {
    fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn err(
        &mut self,
        title: impl Into<String>,
        reason: impl Into<String>,
        legal: impl Into<String>,
    ) {
        self.errors.push(format!(
            "parameter error: {}\n  reason: {}\n  valid range: {}",
            title.into(),
            reason.into(),
            legal.into()
        ));
    }

    fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    fn finish(self) -> Result<()> {
        for w in &self.warnings {
            eprintln!("riaso: warning: {w}");
        }
        if self.errors.is_empty() {
            return Ok(());
        }
        let n = self.errors.len();
        let mut msg = self.errors.join("\n\n");
        msg.push_str(&format!(
            "\n\nriaso: {n} parameter error(s); aborted. No computation was run and output directories were left untouched."
        ));
        Err(anyhow!(msg))
    }
}

fn input_ext(path: &str) -> String {
    std::path::Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default()
}

pub fn validate_static(c: &Config) -> Result<()> {
    let mut ck = Checker::new();

    check_io(c, &mut ck);
    check_filter_cells(c, &mut ck);
    check_filter_genes(c, &mut ck);
    check_coordinates(c, &mut ck);
    check_normalize(c, &mut ck);
    check_hvg(c, &mut ck);
    check_pca_cluster(c, &mut ck);
    check_harmony(c, &mut ck);
    check_singler(c, &mut ck);
    check_ora(c, &mut ck);
    check_gsea(c, &mut ck);
    check_steps(c, &mut ck);

    ck.finish()
}

fn check_io(c: &Config, ck: &mut Checker) {
    for path in &c.io.inputs {
        if !std::path::Path::new(path).is_file() {
            ck.err(
                format!("--input = {path}"),
                "the input path does not exist, or is not a regular file.",
                "--input must point to an existing .gef / .gem / .h5ad file",
            );
            continue;
        }
        let ext = input_ext(path);
        if !matches!(ext.as_str(), "gef" | "gem" | "h5ad") {
            ck.err(
                format!("--input = {path}"),
                format!("unsupported input extension {ext:?}."),
                "only .gef / .gem / .h5ad are supported",
            );
        }
    }

    if c.io.out_root.trim().is_empty() {
        ck.err(
            "--output = ''",
            "the output root directory cannot be empty.",
            "--output must be a non-empty path",
        );
    }

    let all_h5ad = !c.io.inputs.is_empty() && c.io.inputs.iter().all(|p| input_ext(p) == "h5ad");
    if all_h5ad && c.io.spatial_key.trim().is_empty() {
        ck.err(
            "--spatial-key = ''",
            "the obsm key holding the spatial coordinates in the h5ad cannot be empty.",
            "--spatial-key must be a non-empty string (default: spatial)",
        );
    }
    if all_h5ad && c.io.dense {
        ck.warn(
            "--dense only applies to GEF/GEM input; it is ignored for h5ad (the matrix is read with the storage layout stored in the h5ad).",
        );
    }
    if all_h5ad && !c.io.dedup_symbols {
        ck.warn("--no-dedup-symbols only applies to GEF/GEM input; it is ignored for h5ad.");
    }

    if c.io.bin_size == 0 {
        ck.err(
            "--bin-size = 0",
            "the bin edge length must be a positive integer; 0 would divide by zero when merging coordinates.",
            "1 ≤ --bin-size ≤ 4294967295",
        );
    }

    if c.threads > 32 {
        ck.err(
            format!("--threads = {}", c.threads),
            "the thread count is capped at 32.",
            "0 ≤ --threads ≤ 32 (0 = do not touch)",
        );
    } else if c.threads > 0 {
        if let Ok(cores) = std::thread::available_parallelism() {
            if c.threads > cores.get() {
                ck.warn(format!(
                    "--threads={} exceeds the parallelism available on this machine; the process may be throttled by the scheduler.",
                    c.threads
                ));
            }
        }
    }

    let mut gem_warned = false;
    let mut h5ad_warned = false;
    for path in &c.io.inputs {
        match input_ext(path).as_str() {
            "gem" if !gem_warned && !COMMON_BIN_SIZES.contains(&c.io.bin_size) => {
                gem_warned = true;
                ck.warn(format!(
                    "--bin-size={} is not a common preset; GEM will merge coordinates directly with this value.",
                    c.io.bin_size
                ));
            }
            "h5ad" if !h5ad_warned && (c.io.bin_size != 100 || c.io.bin_type != "bins") => {
                h5ad_warned = true;
                ck.warn("--bin-size / --bin-type only apply to GEF/GEM input; they are ignored for h5ad (only the output file names are affected).");
            }
            _ => {}
        }
    }
}

fn check_filter_cells(c: &Config, ck: &mut Checker) {
    if !c.steps.filter_cells {
        check_list_file(ck, "--cell-list", &c.cell_list);
        return;
    }
    if let Some(v) = c.min_counts {
        check_nonneg_f32(ck, "--min-counts", v, "the minimum total counts per cell");
    }
    if let Some(v) = c.max_counts {
        check_nonneg_f32(ck, "--max-counts", v, "the maximum total counts per cell");
    }
    if let (Some(lo), Some(hi)) = (c.min_counts, c.max_counts) {
        if lo > hi {
            ck.err(
                format!("--min-counts = {lo} is greater than --max-counts = {hi}"),
                "the lower bound on cell counts is greater than the upper bound.",
                "--min-counts ≤ --max-counts",
            );
        }
    }
    if let Some(v) = c.min_genes {
        if v < 0 {
            ck.err(
                format!("--min-genes = {v}"),
                "the minimum number of detected genes per cell cannot be negative.",
                "0 ≤ --min-genes ≤ 9223372036854775807",
            );
        }
    }
    if let Some(v) = c.max_genes {
        if v < 0 {
            ck.err(
                format!("--max-genes = {v}"),
                "the maximum number of detected genes per cell cannot be negative.",
                "0 ≤ --max-genes ≤ 9223372036854775807",
            );
        }
    }
    if let (Some(lo), Some(hi)) = (c.min_genes, c.max_genes) {
        if lo > hi {
            ck.err(
                format!("--min-genes = {lo} is greater than --max-genes = {hi}"),
                "the lower bound on the per-cell gene count is greater than the upper bound.",
                "--min-genes ≤ --max-genes",
            );
        }
    }
    if let Some(v) = c.pct_counts_mt {
        if !v.is_finite() || !(0.0..=100.0).contains(&v) {
            ck.err(
                format!("--pct-mt = {v}"),
                "the mitochondrial count percentage cap is expressed in percent; it cannot be negative and cannot exceed 100.",
                "0 ≤ --pct-mt ≤ 100",
            );
        }
    }
    check_list_file(ck, "--cell-list", &c.cell_list);
}

fn check_filter_genes(c: &Config, ck: &mut Checker) {
    if !c.steps.filter_genes {
        check_list_file(ck, "--gene-list", &c.gene_list);
        return;
    }
    if let Some(v) = c.min_cells {
        if v < 0 {
            ck.err(
                format!("--min-cells = {v}"),
                "the minimum number of cells expressing a gene cannot be negative.",
                "0 ≤ --min-cells ≤ 9223372036854775807",
            );
        }
    }
    if let Some(v) = c.max_cells {
        if v < 0 {
            ck.err(
                format!("--max-cells = {v}"),
                "the maximum number of cells expressing a gene cannot be negative.",
                "0 ≤ --max-cells ≤ 9223372036854775807",
            );
        }
    }
    if let (Some(lo), Some(hi)) = (c.min_cells, c.max_cells) {
        if lo > hi {
            ck.err(
                format!("--min-cells = {lo} is greater than --max-cells = {hi}"),
                "the lower bound on the number of expressing cells is greater than the upper bound.",
                "--min-cells ≤ --max-cells",
            );
        }
    }
    if let Some(v) = c.gene_min_counts {
        check_nonneg_f32(ck, "--gene-min-counts", v, "the minimum total UMI per gene");
    }
    if let Some(v) = c.gene_max_counts {
        check_nonneg_f32(ck, "--gene-max-counts", v, "the maximum total UMI per gene");
    }
    if let (Some(lo), Some(hi)) = (c.gene_min_counts, c.gene_max_counts) {
        if lo > hi {
            ck.err(
                format!("--gene-min-counts = {lo} is greater than --gene-max-counts = {hi}"),
                "the lower bound on per-gene UMI is greater than the upper bound.",
                "--gene-min-counts ≤ --gene-max-counts",
            );
        }
    }
    if let Some(v) = c.gene_mean_umi {
        check_nonneg_f32(ck, "--gene-mean-umi", v, "the minimum mean UMI per gene");
    }
    check_list_file(ck, "--gene-list", &c.gene_list);
}

fn check_coordinates(c: &Config, ck: &mut Checker) {
    if !c.steps.filter_coordinates {
        return;
    }
    for (flag, v) in [
        ("--min-x", c.min_x),
        ("--max-x", c.max_x),
        ("--min-y", c.min_y),
        ("--max-y", c.max_y),
    ] {
        if let Some(v) = v {
            if !v.is_finite() {
                ck.err(
                    format!("{flag} = {v}"),
                    "coordinate bounds must be finite numbers.",
                    "-inf < {flag} < +inf (negative values are allowed)",
                );
            }
        }
    }
    if let (Some(lo), Some(hi)) = (c.min_x, c.max_x) {
        if lo > hi {
            ck.err(
                format!("--min-x = {lo} is greater than --max-x = {hi}"),
                "the X lower bound is greater than the X upper bound.",
                "--min-x ≤ --max-x",
            );
        }
    }
    if let (Some(lo), Some(hi)) = (c.min_y, c.max_y) {
        if lo > hi {
            ck.err(
                format!("--min-y = {lo} is greater than --max-y = {hi}"),
                "the Y lower bound is greater than the Y upper bound.",
                "--min-y ≤ --max-y",
            );
        }
    }
}

fn check_normalize(c: &Config, ck: &mut Checker) {
    if c.steps.normalize_total {
        if let Some(v) = c.target_sum {
            if !v.is_finite() || v <= 0.0 {
                ck.err(
                    format!("--target-sum = {v}"),
                    "the normalization target sum must be greater than 0 and must be finite.",
                    "0 < --target-sum < +inf",
                );
            } else if !(1.0..=10_000_000.0).contains(&v) {
                ck.warn(format!(
                    "--target-sum={v} is outside the usual range; the scale after normalization may not suit the downstream log transform."
                ));
            }
        }
    }
    if c.steps.scale {
        if let Some(v) = c.max_value {
            if !v.is_finite() || v <= 0.0 {
                ck.err(
                    format!("--max-value = {v}"),
                    "the clipping upper bound must be greater than 0 and must be finite (-inf would turn the whole matrix into -inf).",
                    "0 < --max-value < +inf",
                );
            } else if !(1.0..=1000.0).contains(&v) {
                ck.warn(format!(
                    "--max-value={v} is outside the usual range; it may squash most expression values into the same number."
                ));
            }
        }
    }
}

fn check_hvg(c: &Config, ck: &mut Checker) {
    if c.n_top_genes == 0 {
        ck.err(
            "--n-top-genes = 0",
            "the number of highly variable genes must be at least 1; 0 triggers an index underflow inside HVG.",
            "1 ≤ --n-top-genes < number of genes after filtering",
        );
    } else if c.n_top_genes > 100_000_000 {
        ck.err(
            format!("--n-top-genes = {}", c.n_top_genes),
            "the number of highly variable genes is out of the supported range.",
            "1 ≤ --n-top-genes ≤ 100000000",
        );
    }
    if c.n_bins == 0 {
        ck.err(
            "--n-bins = 0",
            "the number of dispersion bins must be at least 1.",
            "1 ≤ --n-bins ≤ 1000000",
        );
    } else if c.n_bins > 1_000_000 {
        ck.err(
            format!("--n-bins = {}", c.n_bins),
            "the number of dispersion bins is out of the supported range.",
            "1 ≤ --n-bins ≤ 1000000",
        );
    }
}

fn check_pca_cluster(c: &Config, ck: &mut Checker) {
    if c.steps.pca {
        if c.pca.n_components == 0 {
            ck.err(
                "--n-pcs = 0",
                "the number of principal components must be at least 1.",
                "1 ≤ --n-pcs ≤ 1000000",
            );
        } else if c.pca.n_components > 1_000_000 {
            ck.err(
                format!("--n-pcs = {}", c.pca.n_components),
                "the number of principal components is out of the supported range.",
                "1 ≤ --n-pcs ≤ 1000000",
            );
        }
        if let Some(v) = c.n_pcs_neighbors {
            if v > c.pca.n_components {
                ck.err(
                    format!(
                        "--n-pcs-neighbors = {v} is greater than --n-pcs = {}",
                        c.pca.n_components
                    ),
                    "the number of PCs used by the neighbor graph cannot exceed the number of principal components computed by PCA.",
                    "--n-pcs-neighbors ≤ --n-pcs",
                );
            }
        }
    }
    if c.steps.neighbors {
        if c.n_neighbors == 0 {
            ck.err(
                "--n-neighbors = 0",
                "the number of neighbors must be at least 1 and must be less than the number of cells.",
                "1 ≤ --n-neighbors < number of cells",
            );
        }
        if let Some(v) = c.n_pcs_neighbors {
            if v == 0 {
                ck.err(
                    "--n-pcs-neighbors = 0",
                    "the number of PCs used by the neighbor graph must be at least 1.",
                    "1 ≤ --n-pcs-neighbors ≤ --n-pcs",
                );
            }
        }
    }
    if c.steps.umap {
        if c.umap_components == 0 {
            ck.err(
                "--umap-components = 0",
                "the UMAP output dimension must be at least 1 and must be less than the number of cells.",
                "1 ≤ --umap-components < number of cells",
            );
        }
        if !c.min_dist.is_finite() || !(0.0..1.0).contains(&c.min_dist) {
            ck.err(
                format!("--min-dist = {}", c.min_dist),
                "the UMAP minimum distance must lie in [0, 1).",
                "0 ≤ --min-dist < 1",
            );
        }
        if !c.spread.is_finite() || c.spread <= 0.0 {
            ck.err(
                format!("--spread = {}", c.spread),
                "the UMAP embedding scale must be greater than 0.",
                "0 < --spread < +inf",
            );
        }
    }
    if !c.resolution.is_finite() || c.resolution <= 0.0 {
        ck.err(
            format!("--resolution = {}", c.resolution),
            "the clustering resolution must be greater than 0.",
            "0 < --resolution < +inf",
        );
    } else if !(0.01..=20.0).contains(&c.resolution) {
        ck.warn(format!(
            "--resolution={} is outside the usual range; it may produce too many or too few clusters.",
            c.resolution
        ));
    }
    if c.steps.marker {
        if c.top_n == 0 {
            ck.err(
                "--top-n = 0",
                "the number of markers per cluster must be at least 1.",
                "1 ≤ --top-n ≤ 1000000",
            );
        } else if c.top_n > 1_000_000 {
            ck.err(
                format!("--top-n = {}", c.top_n),
                "the number of markers per cluster is out of the supported range.",
                "1 ≤ --top-n ≤ 1000000",
            );
        }
    }
}

fn check_harmony(c: &Config, ck: &mut Checker) {
    if !(c.steps.pca && (c.io.integrate || c.steps.harmony)) {
        return;
    }
    let h = &c.harmony;
    if !h.sigma.is_finite() || h.sigma <= 0.0 {
        ck.err(
            format!("--sigma = {}", h.sigma),
            "the soft-assignment width must be greater than 0.",
            "0 < --sigma < +inf",
        );
    }
    if let Some(v) = h.theta {
        if !v.is_finite() || v < 0.0 {
            ck.err(
                format!("--theta = {v}"),
                "the batch diversity penalty cannot be negative.",
                "0 ≤ --theta < +inf",
            );
        }
    }
    if let Some(v) = h.lamb {
        if !v.is_finite() || v < 0.0 {
            ck.err(
                format!("--lamb = {v}"),
                "the cluster diversity penalty cannot be negative.",
                "0 ≤ --lamb < +inf",
            );
        }
    }
    if !h.tau.is_finite() || h.tau < 0.0 {
        ck.err(
            format!("--tau = {}", h.tau),
            "the assignment matrix power cannot be negative.",
            "0 ≤ --tau < +inf",
        );
    }
    if !h.block_size.is_finite() || h.block_size <= 0.0 || h.block_size > 1.0 {
        ck.err(
            format!("--block-size = {}", h.block_size),
            "the fraction of cells processed per iteration must lie in (0, 1].",
            "0 < --block-size ≤ 1",
        );
    }
    if h.max_iter_harmony == 0 {
        ck.err(
            "--max-iter-harmony = 0",
            "the number of outer iterations must be at least 1.",
            "1 ≤ --max-iter-harmony ≤ 1000000",
        );
    } else if h.max_iter_harmony > 1_000_000 {
        ck.err(
            format!("--max-iter-harmony = {}", h.max_iter_harmony),
            "the number of outer iterations is out of the supported range.",
            "1 ≤ --max-iter-harmony ≤ 1000000",
        );
    } else if h.max_iter_harmony > 10_000 {
        ck.warn(format!(
            "--max-iter-harmony={} is too large; the runtime of a single step will increase noticeably.",
            h.max_iter_harmony
        ));
    }
    if h.max_iter_kmeans == 0 {
        ck.err(
            "--max-iter-kmeans = 0",
            "the number of inner kmeans iterations must be at least 1.",
            "1 ≤ --max-iter-kmeans ≤ 1000000",
        );
    } else if h.max_iter_kmeans > 1_000_000 {
        ck.err(
            format!("--max-iter-kmeans = {}", h.max_iter_kmeans),
            "the number of inner kmeans iterations is out of the supported range.",
            "1 ≤ --max-iter-kmeans ≤ 1000000",
        );
    } else if h.max_iter_kmeans > 10_000 {
        ck.warn(format!(
            "--max-iter-kmeans={} is too large; the runtime of a single step will increase noticeably.",
            h.max_iter_kmeans
        ));
    }
    if !h.epsilon_cluster.is_finite() || h.epsilon_cluster <= 0.0 {
        ck.err(
            format!("--epsilon-cluster = {}", h.epsilon_cluster),
            "the kmeans convergence threshold must be greater than 0.",
            "0 < --epsilon-cluster < +inf",
        );
    }
    if !h.epsilon_harmony.is_finite() || h.epsilon_harmony <= 0.0 {
        ck.err(
            format!("--epsilon-harmony = {}", h.epsilon_harmony),
            "the outer convergence threshold must be greater than 0.",
            "0 < --epsilon-harmony < +inf",
        );
    }
    if let Some(v) = h.nclust {
        if v == 0 {
            ck.err(
                "--nclust = 0",
                "the harmony cluster count must be at least 1.",
                "1 ≤ --nclust ≤ number of cells",
            );
        }
    }
}

fn check_singler(c: &Config, ck: &mut Checker) {
    if !c.steps.singler {
        return;
    }
    let s = &c.singler;
    if c.label_key.trim().is_empty() {
        ck.err(
            "--label-key = ''",
            "the label column name in the reference obs cannot be empty.",
            "--label-key must be a non-empty column name (default: label.fine)",
        );
    }
    if !s.quantile.is_finite() || !(0.0..=100.0).contains(&s.quantile) {
        ck.err(
            format!("--quantile = {}", s.quantile),
            "the coarse scoring quantile must lie in [0, 100]; values above 100 trigger an out-of-range quantile index.",
            "0 ≤ --quantile ≤ 100",
        );
    }
    if !s.fine_tune_threshold.is_finite() || s.fine_tune_threshold < 0.0 {
        ck.err(
            format!("--fine-tune-threshold = {}", s.fine_tune_threshold),
            "the fine-tuning threshold cannot be negative.",
            "0 ≤ --fine-tune-threshold < +inf",
        );
    }
    if s.fine_tune_times > 100_000_000 {
        ck.err(
            format!("--fine-tune-times = {}", s.fine_tune_times),
            "the number of fine-tuning rounds is out of the supported range; 0 means unlimited.",
            "0 ≤ --fine-tune-times ≤ 100000000",
        );
    }
}

fn check_ora(c: &Config, ck: &mut Checker) {
    let any_ora = c.steps.ora_go || c.steps.ora_kegg || c.steps.ora_reactome;
    if !any_ora {
        return;
    }
    check_unit_f32(
        ck,
        "--gene-pvalue-cutoff",
        c.ora_gene_pvalue_cutoff,
        "the input gene p-value cutoff",
    );
    check_nonneg_f32(
        ck,
        "--gene-log2fc-cutoff",
        c.ora_gene_log2fc_cutoff,
        "the input gene log2fc cutoff",
    );
    if let Some(v) = c.ora_pvalue_cutoff {
        check_unit_f32(ck, "--ora-pvalue-cutoff", v, "the term p-value cutoff");
    }
    if let Some(v) = c.ora_qvalue_cutoff {
        check_unit_f32(ck, "--ora-qvalue-cutoff", v, "the term q-value cutoff");
    }
    if c.ora_min_gene_count == 0 {
        ck.err(
            "--min-gene-count = 0",
            "the gene set size lower bound must be at least 1 and cannot exceed the upper bound.",
            "1 ≤ --min-gene-count ≤ --max-gene-count",
        );
    } else if c.ora_min_gene_count > c.ora_max_gene_count {
        ck.err(
            format!(
                "--min-gene-count = {} is greater than --max-gene-count = {}",
                c.ora_min_gene_count, c.ora_max_gene_count
            ),
            "the gene set size lower bound is greater than the upper bound.",
            "--min-gene-count ≤ --max-gene-count",
        );
    }
    if c.ora_max_gene_count == 0 || c.ora_max_gene_count > 10_000_000 {
        ck.err(
            format!("--max-gene-count = {}", c.ora_max_gene_count),
            "the gene set size upper bound is out of the supported range.",
            "1 ≤ --max-gene-count ≤ 10000000",
        );
    }
    if c.ora_top_n_terms == 0 {
        ck.err(
            "--top-n-terms = 0",
            "the number of returned terms per cluster must be at least 1.",
            "1 ≤ --top-n-terms ≤ 10000000",
        );
    } else if c.ora_top_n_terms > 10_000_000 {
        ck.err(
            format!("--top-n-terms = {}", c.ora_top_n_terms),
            "the number of returned terms per cluster is out of the supported range.",
            "1 ≤ --top-n-terms ≤ 10000000",
        );
    }
    if c.steps.ora_kegg && c.kegg_org.trim().is_empty() {
        ck.err(
            "--kegg-org = ''",
            "the KEGG organism code cannot be empty.",
            "--kegg-org must be a non-empty three-letter organism code (default: mmu)",
        );
    }
    if c.steps.ora_reactome && c.reactome_species.trim().is_empty() {
        ck.err(
            "--reactome-species = ''",
            "the Reactome species name cannot be empty.",
            "--reactome-species must be a non-empty species name (default: Mus musculus)",
        );
    }
}

fn check_gsea(c: &Config, ck: &mut Checker) {
    if !c.steps.gsea {
        return;
    }
    let g = &c.gsea;
    if !g.gsea_param.is_finite() || g.gsea_param < 0.0 {
        ck.err(
            format!("--gsea-param = {}", g.gsea_param),
            "the weighting exponent cannot be negative and cannot be nan / inf.",
            "0 ≤ --gsea-param < +inf",
        );
    }
    if g.min_gene_count == 0 {
        ck.err(
            "--gsea-min-gene-count = 0",
            "the gene set size lower bound must be at least 1 and cannot exceed the upper bound.",
            "1 ≤ --gsea-min-gene-count ≤ --gsea-max-gene-count",
        );
    } else if g.min_gene_count > g.max_gene_count {
        ck.err(
            format!(
                "--gsea-min-gene-count = {} is greater than --gsea-max-gene-count = {}",
                g.min_gene_count, g.max_gene_count
            ),
            "the gene set size lower bound is greater than the upper bound.",
            "--gsea-min-gene-count ≤ --gsea-max-gene-count",
        );
    }
    if g.max_gene_count == 0 || g.max_gene_count > 10_000_000 {
        ck.err(
            format!("--gsea-max-gene-count = {}", g.max_gene_count),
            "the gene set size upper bound is out of the supported range.",
            "1 ≤ --gsea-max-gene-count ≤ 10000000",
        );
    }
    if g.n_perm == 0 {
        ck.err(
            "--gsea-n-perm = 0",
            "the number of permutations must be at least 1.",
            "1 ≤ --gsea-n-perm ≤ 10000000",
        );
    } else if g.n_perm > 10_000_000 {
        ck.err(
            format!("--gsea-n-perm = {}", g.n_perm),
            "the number of permutations is out of the supported range.",
            "1 ≤ --gsea-n-perm ≤ 10000000",
        );
    } else {
        if g.n_perm < 100 {
            ck.warn(format!(
                "--gsea-n-perm={} is too small; the p-value resolution is insufficient.",
                g.n_perm
            ));
        }
        if g.n_perm > 1_000_000 {
            ck.warn(format!(
                "--gsea-n-perm={} is too large; the null distribution matrix may use a lot of memory.",
                g.n_perm
            ));
        }
    }
    check_unit_f32(
        ck,
        "--gsea-pvalue-cutoff",
        g.pvalue_cutoff,
        "the term p-value cutoff",
    );
    check_unit_f32(
        ck,
        "--gsea-qvalue-cutoff",
        g.qvalue_cutoff,
        "the term q-value cutoff",
    );
    if g.top_n_terms == 0 {
        ck.err(
            "--gsea-top-n-terms = 0",
            "the number of returned terms per cluster must be at least 1.",
            "1 ≤ --gsea-top-n-terms ≤ 10000000",
        );
    } else if g.top_n_terms > 10_000_000 {
        ck.err(
            format!("--gsea-top-n-terms = {}", g.top_n_terms),
            "the number of returned terms per cluster is out of the supported range.",
            "1 ≤ --gsea-top-n-terms ≤ 10000000",
        );
    }
}

fn check_steps(c: &Config, ck: &mut Checker) {
    if c.steps.singler {
        if c.references.is_empty() {
            ck.err(
                "--singler",
                "singler is enabled but no reference set was given (--reference).",
                "at least 1 --reference is required when --singler is enabled",
            );
        } else {
            for r in &c.references {
                if !r.is_file() {
                    ck.err(
                        format!("--reference = {}", r.display()),
                        "the reference file does not exist or is not a regular file.",
                        "every --reference must point to an existing .h5ad file",
                    );
                }
            }
        }
    } else if !c.references.is_empty() {
        ck.err(
            "--reference",
            "reference sets were given but --singler is not enabled; they would never be read and cell type annotation would be skipped entirely.",
            "add -a / --singler to run cell type annotation, or drop --reference",
        );
    }
    let any_ora = c.steps.ora_go || c.steps.ora_kegg || c.steps.ora_reactome;
    if c.steps.gsea && !any_ora {
        ck.err(
            "--gsea",
            "GSEA needs at least one reference set: -G / -K / -R (all three databases are currently disabled).",
            "when --gsea is enabled, at least one of -G / -K / -R must also be enabled",
        );
    }
    if any_ora {
        if c.org_db.as_os_str().is_empty() {
            ck.err(
                "--org-db",
                "enrichment needs the species annotation database --org-db (shared by GO/KEGG/Reactome).",
                "when any enrichment database is enabled, --org-db must be given",
            );
        } else if !c.org_db.is_file() {
            ck.err(
                format!("--org-db = {}", c.org_db.display()),
                "the species annotation database file does not exist or is not a regular file.",
                "--org-db must point to an existing .sqlite file",
            );
        }
    } else if !c.org_db.as_os_str().is_empty() {
        ck.err(
            "--org-db",
            "--org-db was given but no enrichment database is enabled (-G / -K / -R); it would never be read and enrichment would be skipped entirely.",
            "enable at least one of -G / -K / -R to run enrichment, or drop --org-db",
        );
    }
    if c.steps.ora_go {
        if c.go_db.as_os_str().is_empty() {
            ck.err(
                "--go-db",
                "--go is enabled and requires --go-db.",
                "when -G is enabled, --go-db must be given",
            );
        } else if !c.go_db.is_file() {
            ck.err(
                format!("--go-db = {}", c.go_db.display()),
                "the GO annotation database file does not exist or is not a regular file.",
                "--go-db must point to an existing .sqlite file",
            );
        }
    } else if !c.go_db.as_os_str().is_empty() {
        ck.err(
            "--go-db",
            "--go-db was given but -G / --go is not enabled; it would never be read and GO enrichment would be skipped.",
            "add -G / --go to run GO enrichment, or drop --go-db",
        );
    }
    if c.steps.ora_kegg {
        if c.kegg_dir.as_os_str().is_empty() {
            ck.err(
                "--kegg-dir",
                "--kegg is enabled and requires --kegg-dir.",
                "when -K is enabled, --kegg-dir must be given",
            );
        } else if !c.kegg_dir.is_dir() {
            ck.err(
                format!("--kegg-dir = {}", c.kegg_dir.display()),
                "the KEGG raw data directory does not exist or is not a directory.",
                "--kegg-dir must point to an existing directory",
            );
        }
    } else if !c.kegg_dir.as_os_str().is_empty() {
        ck.err(
            "--kegg-dir",
            "--kegg-dir was given but -K / --kegg is not enabled; it would never be read and KEGG enrichment would be skipped.",
            "add -K / --kegg to run KEGG enrichment, or drop --kegg-dir",
        );
    }
    if c.steps.ora_reactome {
        if c.reactome_dir.as_os_str().is_empty() {
            ck.err(
                "--reactome-dir",
                "--reactome is enabled and requires --reactome-dir.",
                "when -R is enabled, --reactome-dir must be given",
            );
        } else if !c.reactome_dir.is_dir() {
            ck.err(
                format!("--reactome-dir = {}", c.reactome_dir.display()),
                "the Reactome raw data directory does not exist or is not a directory.",
                "--reactome-dir must point to an existing directory",
            );
        }
    } else if !c.reactome_dir.as_os_str().is_empty() {
        ck.err(
            "--reactome-dir",
            "--reactome-dir was given but -R / --reactome is not enabled; it would never be read and Reactome enrichment would be skipped.",
            "add -R / --reactome to run Reactome enrichment, or drop --reactome-dir",
        );
    }

    let mut seen = HashSet::new();
    for r in &c.references {
        let stem = r
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        if !seen.insert(stem) {
            ck.err(
                "--reference",
                "two reference sets share the same file stem; their outputs would overwrite each other.",
                "every --reference must have a distinct file stem",
            );
            break;
        }
    }
}

fn check_nonneg_f32(ck: &mut Checker, flag: &str, v: f32, what: &str) {
    if !v.is_finite() {
        ck.err(
            format!("{flag} = {v}"),
            format!("{what} must be a finite number (nan / inf / -inf are invalid)."),
            format!("0 ≤ {flag} < +inf"),
        );
    } else if v < 0.0 {
        ck.err(
            format!("{flag} = {v}"),
            format!("{what} cannot be negative."),
            format!("0 ≤ {flag} < +inf"),
        );
    }
}

fn check_unit_f32(ck: &mut Checker, flag: &str, v: f32, what: &str) {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        ck.err(
            format!("{flag} = {v}"),
            format!("{what} must lie in [0, 1]."),
            format!("0 ≤ {flag} ≤ 1"),
        );
    }
}

fn check_list_file(ck: &mut Checker, flag: &str, path: &Option<PathBuf>) {
    if let Some(p) = path {
        if !p.is_file() {
            ck.err(
                format!("{flag} = {}", p.display()),
                "cannot read the list file.",
                format!("{flag} must point to an existing, readable file"),
            );
        }
    }
}

pub fn validate_against_data(c: &Config, sd: &SoData) -> Result<()> {
    let mut ck = Checker::new();
    let n_obs = sd.n_obs();
    let n_vars = sd.n_vars();

    let cell_list = read_list(&c.cell_list);
    let gene_list = read_list(&c.gene_list);

    let errs_before_cells = ck.errors.len();
    if let Some(obs) = &sd.obs {
        if c.steps.filter_cells {
            check_cell_thresholds(c, obs, n_obs, cell_list.as_deref(), &mut ck);
        }
    }

    let coord_mask = if c.steps.filter_coordinates {
        match &sd.position {
            Some(pos) => {
                check_coordinate_thresholds(c, pos, n_obs, &mut ck);
                Some(coordinate_keep(pos, n_obs, c))
            }
            None => {
                ck.warn("the input has no spatial coordinates; coordinate filtering was skipped.");
                None
            }
        }
    } else {
        None
    };
    let cell_group_errors = ck.errors.len() - errs_before_cells;

    let errs_before_genes = ck.errors.len();
    if let Some(var) = &sd.var {
        if c.steps.filter_genes {
            check_gene_thresholds(c, var, n_vars, gene_list.as_deref(), &mut ck);
        }
    }
    let gene_group_errors = ck.errors.len() - errs_before_genes;

    let cell_mask = if c.steps.filter_cells {
        sd.obs
            .as_ref()
            .map(|obs| cell_keep(obs, n_obs, c, cell_list.as_deref()))
    } else {
        None
    };
    let gene_mask = if c.steps.filter_genes {
        sd.var
            .as_ref()
            .map(|var| gene_keep(var, n_vars, c, gene_list.as_deref(), sd))
    } else {
        None
    };

    let cells_eff = (0..n_obs)
        .filter(|&i| {
            cell_mask.as_ref().is_none_or(|m| m[i]) && coord_mask.as_ref().is_none_or(|m| m[i])
        })
        .count();
    let genes_eff = gene_mask
        .as_ref()
        .map_or(n_vars, |m| m.iter().filter(|&&k| k).count());

    if c.steps.filter_cells || c.steps.filter_coordinates {
        let any_cell_filter = c.steps.filter_cells || c.steps.filter_coordinates;
        if cells_eff == 0 && any_cell_filter {
            if cell_group_errors == 0 {
                ck.err(
                    "no cells would remain under the current cell filtering parameters",
                    "the combination of conditions covers every cell.",
                    "retained cells ≥ 1; check --min-counts / --max-counts / --min-genes / --max-genes / --pct-mt / the coordinate window against the actual data range",
                );
            }
        } else if cells_eff < 3 {
            ck.err(
                "too few cells would remain under the current filtering parameters",
                "the retained cell count is not enough for downstream dimensionality reduction and clustering.",
                "retained cells ≥ 3",
            );
        } else if (cells_eff as f32) / (n_obs as f32) < 0.01 {
            ck.warn("the current cell filtering parameters retain a very low fraction of cells (below 1%); downstream results may be unstable.");
        }
    }

    if c.steps.filter_genes {
        if genes_eff == 0 {
            if gene_group_errors == 0 {
                ck.err(
                    "no genes would remain under the current gene filtering parameters",
                    "the combination of conditions removes every gene.",
                    "retained genes ≥ 1; check --min-cells / --max-cells / --gene-min-counts / --gene-max-counts / --gene-mean-umi against the actual data range",
                );
            }
        } else if (genes_eff as f32) / (n_vars as f32) < 0.01 {
            ck.warn("the current gene filtering parameters retain a very low fraction of genes (below 1%); downstream HVG/PCA/marker may be unstable.");
        }
    }

    check_downstream(c, cells_eff, genes_eff, &mut ck);

    ck.finish()
}

fn check_cell_thresholds(
    c: &Config,
    obs: &DataFrame,
    n_obs: usize,
    list: Option<&[String]>,
    ck: &mut Checker,
) {
    if let Some(vals) = col_f32(obs, "total_counts") {
        let r = range_f32(&vals);
        if let (Some(v), Some((_, max))) = (c.min_counts, r) {
            if v > max {
                ck.err(
                    format!("--min-counts = {v}"),
                    "this value is greater than the largest total_counts in the data; no cell would remain after filtering.",
                    "0 ≤ --min-counts ≤ the largest total_counts in the data",
                );
            }
        }
        if let (Some(v), Some((min, max))) = (c.max_counts, r) {
            if v < min {
                ck.err(
                    format!("--max-counts = {v}"),
                    "this value is smaller than the smallest total_counts in the data; no cell would remain after filtering.",
                    "the smallest total_counts in the data ≤ --max-counts < +inf",
                );
            } else if v >= max {
                ck.warn(format!(
                    "--max-counts={v} is not smaller than the largest total_counts in the data; no cell will be filtered out."
                ));
            }
        }
    }

    if let Some(vals) = col_i64(obs, "n_genes_by_counts") {
        let r = range_i64(&vals);
        if let (Some(v), Some((_, max))) = (c.min_genes, r) {
            if v > max {
                ck.err(
                    format!("--min-genes = {v}"),
                    "this value is greater than the largest detected gene count in the data; no cell would remain after filtering.",
                    "0 ≤ --min-genes ≤ the largest n_genes_by_counts in the data",
                );
            }
        }
        if let (Some(v), Some((min, max))) = (c.max_genes, r) {
            if v < min {
                ck.err(
                    format!("--max-genes = {v}"),
                    "this value is smaller than the smallest detected gene count in the data; no cell would remain after filtering.",
                    "the smallest n_genes_by_counts in the data ≤ --max-genes < +inf",
                );
            } else if v >= max {
                ck.warn(format!(
                    "--max-genes={v} is not smaller than the largest detected gene count in the data; no cell will be filtered out."
                ));
            }
        }
    }

    if let Some(vals) = col_f32(obs, "pct_counts_mt") {
        let r = range_f32(&vals);
        if let (Some(v), Some((min, max))) = (c.pct_counts_mt, r) {
            if v < min {
                ck.err(
                    format!("--pct-mt = {v}"),
                    "this value is smaller than the smallest mitochondrial percentage in the data; no cell would remain after filtering.",
                    "the smallest pct_counts_mt in the data ≤ --pct-mt ≤ 100",
                );
            } else if v >= max {
                ck.warn(format!(
                    "--pct-mt={v} is not smaller than the largest mitochondrial percentage in the data; no cell will be filtered out."
                ));
            }
        }
    }

    if let Some(list) = list {
        check_cell_list(obs, n_obs, c, list, ck);
    }
}

fn check_cell_list(obs: &DataFrame, n_obs: usize, c: &Config, list: &[String], ck: &mut Checker) {
    let Some(names) = names_of(obs, "cell") else {
        if c.exclude_cells {
            ck.warn("the cell blacklist cannot be applied: the data has no cell column, so no cell will be filtered out.");
        } else {
            ck.err(
                "--cell-list",
                "the cell list cannot be applied: obs of this data has no cell column.",
                "--cell-list requires input whose obs contains a cell name column (cell)",
            );
        }
        return;
    };
    let set: HashSet<&str> = list.iter().map(String::as_str).collect();
    let matched = names
        .iter()
        .flatten()
        .filter(|n| set.contains(n.as_str()))
        .count();
    if c.exclude_cells {
        if matched >= n_obs {
            ck.err(
                "--cell-list (blacklist)",
                "the blacklist covers every cell; no cell would remain after filtering.",
                "the blacklist match count must be smaller than the total number of cells",
            );
        }
    } else if matched == 0 {
        ck.err(
            "--cell-list (whitelist)",
            "no name in the cell list matches any cell name in the data.",
            "at least 1 whitelist name must match the cell column of obs",
        );
    } else if matched >= n_obs {
        ck.warn("the cell list covers every cell, which is the same as not filtering at all.");
    }
}

fn check_gene_thresholds(
    c: &Config,
    var: &DataFrame,
    n_vars: usize,
    list: Option<&[String]>,
    ck: &mut Checker,
) {
    if let Some(vals) = col_i64(var, "n_cells") {
        let r = range_i64(&vals);
        if let (Some(v), Some((_, max))) = (c.min_cells, r) {
            if v > max {
                ck.err(
                    format!("--min-cells = {v}"),
                    "this value is greater than the largest number of expressing cells in the data; no gene would remain after filtering.",
                    "0 ≤ --min-cells ≤ the largest n_cells in the data",
                );
            }
        }
        if let (Some(v), Some((min, max))) = (c.max_cells, r) {
            if v < min {
                ck.err(
                    format!("--max-cells = {v}"),
                    "this value is smaller than the smallest number of expressing cells in the data; no gene would remain after filtering.",
                    "the smallest n_cells in the data ≤ --max-cells < +inf",
                );
            } else if v >= max {
                ck.warn(format!(
                    "--max-cells={v} is not smaller than the largest number of expressing cells in the data; no gene will be filtered out."
                ));
            }
        }
    }

    if let Some(vals) = col_f32(var, "n_counts") {
        let r = range_f32(&vals);
        if let (Some(v), Some((_, max))) = (c.gene_min_counts, r) {
            if v > max {
                ck.err(
                    format!("--gene-min-counts = {v}"),
                    "this value is greater than the largest total UMI per gene in the data; no gene would remain after filtering.",
                    "0 ≤ --gene-min-counts ≤ the largest n_counts in the data",
                );
            }
        }
        if let (Some(v), Some((min, max))) = (c.gene_max_counts, r) {
            if v < min {
                ck.err(
                    format!("--gene-max-counts = {v}"),
                    "this value is smaller than the smallest total UMI per gene in the data; no gene would remain after filtering.",
                    "the smallest n_counts in the data ≤ --gene-max-counts < +inf",
                );
            } else if v >= max {
                ck.warn(format!(
                    "--gene-max-counts={v} is not smaller than the largest total UMI per gene in the data; no gene will be filtered out."
                ));
            }
        }
    }

    if let Some(vals) = col_f32(var, "mean_umi") {
        let r = range_f32(&vals);
        if let (Some(v), Some((min, max))) = (c.gene_mean_umi, r) {
            if v >= max {
                ck.err(
                    format!("--gene-mean-umi = {v}"),
                    "this test is a strict greater-than: when the threshold is not smaller than the largest mean UMI in the data, no gene would remain after filtering.",
                    "--gene-mean-umi < the largest mean_umi in the data",
                );
            } else if v < min {
                ck.warn(format!(
                    "--gene-mean-umi={v} is smaller than the smallest mean UMI in the data; no gene will be filtered out."
                ));
            }
        }
    }

    if let Some(list) = list {
        check_gene_list(var, n_vars, c, list, ck);
    }

    if c.filter_mt_genes {
        let mt = count_mt_genes(var);
        if mt == 0 {
            ck.warn(
                "no mitochondrial gene was detected; --filter-mt-genes will not remove any gene.",
            );
        }
    }
}

fn check_gene_list(var: &DataFrame, n_vars: usize, c: &Config, list: &[String], ck: &mut Checker) {
    let Some(names) = names_of(var, "gene") else {
        if c.exclude_genes {
            ck.warn("the gene blacklist cannot be applied: the data has no gene column, so no gene will be filtered out.");
        } else {
            ck.err(
                "--gene-list",
                "the gene list cannot be applied: var of this data has no gene column.",
                "--gene-list requires input whose var contains a gene name column (gene)",
            );
        }
        return;
    };
    let set: HashSet<&str> = list.iter().map(String::as_str).collect();
    let matched = names
        .iter()
        .flatten()
        .filter(|n| set.contains(n.as_str()))
        .count();
    if c.exclude_genes {
        if matched >= n_vars {
            ck.err(
                "--gene-list (blacklist)",
                "the blacklist covers every gene; no gene would remain after filtering.",
                "the blacklist match count must be smaller than the total number of genes",
            );
        }
    } else if matched == 0 {
        ck.err(
            "--gene-list (whitelist)",
            "no name in the gene list matches any gene name in the data (note the difference between symbols and Ensembl IDs).",
            "at least 1 whitelist name must match the gene column of var",
        );
    } else if matched >= n_vars {
        ck.warn("the gene list covers every gene, which is the same as not filtering at all.");
    }
}

fn check_coordinate_thresholds(
    c: &Config,
    pos: &ndarray::Array2<f32>,
    n_obs: usize,
    ck: &mut Checker,
) {
    let (mut x_min, mut x_max, mut y_min, mut y_max) = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    let mut any = false;
    for i in 0..n_obs {
        let (x, y) = (pos[[i, 0]], pos[[i, 1]]);
        if x.is_finite() {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            any = true;
        }
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            any = true;
        }
    }
    if !any {
        ck.err(
            "--filter-coordinates",
            "the data has no usable finite coordinate values; coordinate filtering cannot be performed.",
            "the input must contain finite spatial coordinates",
        );
        return;
    }

    if let Some(v) = c.min_x {
        if v > x_max {
            ck.err(
                format!("--min-x = {v}"),
                "the X lower bound is greater than the largest X in the data; no cell would remain after filtering.",
                "--min-x ≤ the largest X in the data",
            );
        }
    }
    if let Some(v) = c.max_x {
        if v < x_min {
            ck.err(
                format!("--max-x = {v}"),
                "the X upper bound is smaller than the smallest X in the data; no cell would remain after filtering.",
                "--max-x ≥ the smallest X in the data",
            );
        }
    }
    if let Some(v) = c.min_y {
        if v > y_max {
            ck.err(
                format!("--min-y = {v}"),
                "the Y lower bound is greater than the largest Y in the data; no cell would remain after filtering.",
                "--min-y ≤ the largest Y in the data",
            );
        }
    }
    if let Some(v) = c.max_y {
        if v < y_min {
            ck.err(
                format!("--max-y = {v}"),
                "the Y upper bound is smaller than the smallest Y in the data; no cell would remain after filtering.",
                "--max-y ≥ the smallest Y in the data",
            );
        }
    }

    let covers = c.min_x.is_none_or(|v| v <= x_min)
        && c.max_x.is_none_or(|v| v >= x_max)
        && c.min_y.is_none_or(|v| v <= y_min)
        && c.max_y.is_none_or(|v| v >= y_max)
        && (c.min_x.is_some() || c.max_x.is_some() || c.min_y.is_some() || c.max_y.is_some());
    if covers {
        ck.warn(
            "the coordinate window covers every cell in the data; no cell will be filtered out.",
        );
    }
}

fn check_downstream(c: &Config, cells_eff: usize, genes_eff: usize, ck: &mut Checker) {
    if genes_eff == 0 {
        return;
    }

    if c.n_top_genes >= genes_eff {
        ck.err(
            format!("--n-top-genes = {}", c.n_top_genes),
            "the number of highly variable genes is not smaller than the number of genes left after filtering, so HVG cannot select any and the subsequent PCA will certainly fail.",
            "1 ≤ --n-top-genes < number of genes after filtering",
        );
    }
    if c.n_bins > genes_eff {
        ck.warn(format!(
            "--n-bins={} exceeds the number of genes left after filtering; most bins will be empty.",
            c.n_bins
        ));
    }

    if cells_eff < 3 {
        return;
    }

    let eff_pcs = c.pca.n_components.min(cells_eff).min(genes_eff);
    if c.steps.pca {
        if eff_pcs == 0 {
            ck.err(
                "--n-pcs",
                "the effective dimensionality is 0, so PCA cannot run.",
                "min(--n-pcs, number of cells, number of genes) ≥ 1",
            );
        } else {
            if c.pca.n_components > eff_pcs {
                ck.warn(format!(
                    "--n-pcs={} exceeds the effective dimensionality; it will be truncated to min(number of cells, number of genes).",
                    c.pca.n_components
                ));
            }
            if matches!(c.pca.svd_solver, SvdSolver::Arpack) && c.pca.n_components + 2 > eff_pcs {
                ck.warn(format!(
                    "--n-pcs={} is close to the effective dimensionality; the ARPACK path will fall back to full SVD.",
                    c.pca.n_components
                ));
            }
        }
    }
    if c.steps.neighbors {
        if c.n_neighbors >= cells_eff {
            ck.err(
                format!("--n-neighbors = {}", c.n_neighbors),
                "the number of neighbors is not smaller than the number of cells; KNN degrades to all-pairs and produces zero-filled columns.",
                "1 ≤ --n-neighbors < number of cells",
            );
        }
        if let Some(v) = c.n_pcs_neighbors {
            if v > eff_pcs {
                ck.err(
                    format!("--n-pcs-neighbors = {v}"),
                    "the number of PCs used by the neighbor graph exceeds the available dimensionality, so the neighbor computation fails silently.",
                    "1 ≤ --n-pcs-neighbors ≤ min(--n-pcs, number of cells, number of genes)",
                );
            }
        }
    }
    if c.steps.umap {
        if c.umap_components >= cells_eff {
            ck.err(
                format!("--umap-components = {}", c.umap_components),
                "the UMAP output dimension must be smaller than the number of cells.",
                "1 ≤ --umap-components < number of cells",
            );
        } else if c.steps.pca && c.umap_components > eff_pcs {
            ck.warn(format!(
                "--umap-components={} exceeds the PCA dimensionality; the embedding will degrade.",
                c.umap_components
            ));
        }
    }
    if c.steps.pca && (c.io.integrate || c.steps.harmony) {
        if let Some(v) = c.harmony.nclust {
            if v > cells_eff {
                ck.err(
                    format!("--nclust = {v}"),
                    "the harmony cluster count exceeds the number of cells.",
                    "1 ≤ --nclust ≤ number of cells",
                );
            }
        }
    }
    if c.steps.marker && c.top_n > genes_eff {
        ck.warn(format!(
            "--top-n={} exceeds the number of genes left after filtering; each cluster will output at most as many markers as there are genes.",
            c.top_n
        ));
    }
}

fn col_f32(df: &DataFrame, name: &str) -> Option<Vec<Option<f32>>> {
    let ca = df.column(name).ok()?.f32().ok()?;
    Some(ca.into_iter().collect())
}

fn col_i64(df: &DataFrame, name: &str) -> Option<Vec<Option<i64>>> {
    let ca = df.column(name).ok()?.i64().ok()?;
    Some(ca.into_iter().collect())
}

fn range_f32(vals: &[Option<f32>]) -> Option<(f32, f32)> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut any = false;
    for v in vals.iter().flatten() {
        if v.is_finite() {
            any = true;
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    any.then_some((min, max))
}

fn range_i64(vals: &[Option<i64>]) -> Option<(i64, i64)> {
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut any = false;
    for v in vals.iter().flatten() {
        any = true;
        min = min.min(*v);
        max = max.max(*v);
    }
    any.then_some((min, max))
}

fn names_of(df: &DataFrame, name: &str) -> Option<Vec<Option<String>>> {
    let col = df.column(name).ok()?;
    let s = col.as_materialized_series();
    match s.dtype() {
        polars::datatypes::DataType::String => {
            let ca = s.str().ok()?;
            Some(ca.into_iter().map(|v| v.map(|s| s.to_string())).collect())
        }
        _ => Some(
            (0..s.len())
                .map(|i| {
                    let v = s.get(i).ok()?;
                    if matches!(v, AnyValue::Null) {
                        None
                    } else {
                        Some(format!("{v}"))
                    }
                })
                .collect(),
        ),
    }
}

fn read_list(path: &Option<PathBuf>) -> Option<Vec<String>> {
    let p = path.as_ref()?;
    let text = fs::read_to_string(p).ok()?;
    Some(
        text.lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
    )
}

fn apply_f32(keep: &mut [bool], vals: &[Option<f32>], pred: impl Fn(f32) -> bool) {
    for (k, v) in keep.iter_mut().zip(vals) {
        if let Some(v) = v {
            *k &= pred(*v);
        }
    }
}

fn apply_i64(keep: &mut [bool], vals: &[Option<i64>], pred: impl Fn(i64) -> bool) {
    for (k, v) in keep.iter_mut().zip(vals) {
        if let Some(v) = v {
            *k &= pred(*v);
        }
    }
}

fn cell_keep(obs: &DataFrame, n_obs: usize, c: &Config, list: Option<&[String]>) -> Vec<bool> {
    let mut keep = vec![true; n_obs];
    if let Some(vals) = col_f32(obs, "total_counts") {
        if let Some(v) = c.min_counts {
            apply_f32(&mut keep, &vals, |x| x >= v);
        }
        if let Some(v) = c.max_counts {
            apply_f32(&mut keep, &vals, |x| x <= v);
        }
    }
    if let Some(vals) = col_i64(obs, "n_genes_by_counts") {
        if let Some(v) = c.min_genes {
            apply_i64(&mut keep, &vals, |x| x >= v);
        }
        if let Some(v) = c.max_genes {
            apply_i64(&mut keep, &vals, |x| x <= v);
        }
    }
    if let Some(vals) = col_f32(obs, "pct_counts_mt") {
        if let Some(v) = c.pct_counts_mt {
            apply_f32(&mut keep, &vals, |x| x <= v);
        }
    }
    if let (Some(list), Some(names)) = (list, names_of(obs, "cell")) {
        let set: HashSet<&str> = list.iter().map(String::as_str).collect();
        for (k, name) in keep.iter_mut().zip(names.iter()) {
            let in_list = name.as_ref().is_some_and(|n| set.contains(n.as_str()));
            *k &= if c.exclude_cells { !in_list } else { in_list };
        }
    }
    keep
}

fn gene_keep(
    var: &DataFrame,
    n_vars: usize,
    c: &Config,
    list: Option<&[String]>,
    sd: &SoData,
) -> Vec<bool> {
    let mut keep = vec![true; n_vars];
    if let Some(vals) = col_i64(var, "n_cells") {
        if let Some(v) = c.min_cells {
            apply_i64(&mut keep, &vals, |x| x >= v);
        }
        if let Some(v) = c.max_cells {
            apply_i64(&mut keep, &vals, |x| x <= v);
        }
    }
    if let Some(vals) = col_f32(var, "n_counts") {
        if let Some(v) = c.gene_min_counts {
            apply_f32(&mut keep, &vals, |x| x >= v);
        }
        if let Some(v) = c.gene_max_counts {
            apply_f32(&mut keep, &vals, |x| x <= v);
        }
    }
    if let Some(vals) = col_f32(var, "mean_umi") {
        if let Some(v) = c.gene_mean_umi {
            apply_f32(&mut keep, &vals, |x| x > v);
        }
    }
    if let (Some(list), Some(names)) = (list, names_of(var, "gene")) {
        let set: HashSet<&str> = list.iter().map(String::as_str).collect();
        for (k, name) in keep.iter_mut().zip(names.iter()) {
            let in_list = name.as_ref().is_some_and(|n| set.contains(n.as_str()));
            *k &= if c.exclude_genes { !in_list } else { in_list };
        }
    }
    if c.filter_mt_genes {
        let names = sd.mt_gene_names();
        if names.len() == n_vars {
            for (k, name) in keep.iter_mut().zip(names.iter()) {
                let lower = name.to_lowercase();
                if lower.starts_with("mt-") || lower.starts_with("mt.") {
                    *k = false;
                }
            }
        }
    }
    keep
}

fn coordinate_keep(pos: &ndarray::Array2<f32>, n_obs: usize, c: &Config) -> Vec<bool> {
    let mut keep = vec![true; n_obs];
    for (i, k) in keep.iter_mut().enumerate() {
        let x = pos[[i, 0]];
        let y = pos[[i, 1]];
        if let Some(v) = c.min_x {
            *k &= x >= v;
        }
        if let Some(v) = c.max_x {
            *k &= x <= v;
        }
        if let Some(v) = c.min_y {
            *k &= y >= v;
        }
        if let Some(v) = c.max_y {
            *k &= y <= v;
        }
    }
    keep
}

fn count_mt_genes(var: &DataFrame) -> usize {
    let Some(names) = names_of(var, "gene") else {
        return 0;
    };
    names
        .iter()
        .flatten()
        .filter(|n| {
            let lower = n.to_lowercase();
            lower.starts_with("mt-") || lower.starts_with("mt.")
        })
        .count()
}
