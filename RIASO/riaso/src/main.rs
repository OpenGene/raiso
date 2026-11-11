use std::path::PathBuf;

use clap::{ArgAction, Parser};

use riaso::algorithm::annotation::SingleRParams;
use riaso::algorithm::batch::HarmonyParams;
use riaso::algorithm::enrichment::{GseaParams, RankBy};
use riaso::algorithm::pca::{PcaParams, SvdSolver};
use riaso::config::{ClusterAlg, Config, Io, Steps};

#[derive(Parser, Debug)]
#[command(
    name = "riaso",
    version,
    about = "RIASO -- spatial omics analysis pipeline (Rust core)",
    disable_help_subcommand = true,
    disable_help_flag = true,
    allow_negative_numbers = true,
    term_width = 150
)]
struct Cli {
    #[arg(short = 'H', long = "HELP", action = ArgAction::Help, help = "print this message (-h is taken by --hvg)", help_heading = "Global options")]
    help: Option<bool>,

    #[arg(
        short = 'I',
        long = "input",
        value_name = "path",
        required = true,
        help = "input file or directory, repeatable; file: type auto-detected by extension (.gef/.gem -> GEF, .h5ad -> H5ad); directory: top-level *.gef/*.gem/*.h5ad, each file = one sample, sorted by name; without -i every input is analyzed independently as its own single sample",
        help_heading = "I/O options"
    )]
    inputs: Vec<String>,

    #[arg(long = "bin-type", default_value = "bins", value_parser = ["bins", "cell_bins"], help = "GEF bin type: bins|cell_bins", help_heading = "I/O options")]
    bin_type: String,

    #[arg(
        long = "bin-size",
        default_value_t = 100,
        help = "GEF bin edge length, unit DNB / 500nm",
        help_heading = "I/O options"
    )]
    bin_size: u32,

    #[arg(
        long = "spatial-key",
        default_value = "spatial",
        help = "obsm key holding spatial coords in h5ad",
        help_heading = "I/O options"
    )]
    spatial_key: String,

    #[arg(long = "dense", action = ArgAction::SetTrue, help = "read dense matrix instead of sparse (default sparse)", help_heading = "I/O options")]
    dense: bool,

    #[arg(long = "no-dedup-symbols", action = ArgAction::SetTrue, help = "keep duplicated gene symbols (default dedup, aligned with stereo)", help_heading = "I/O options")]
    no_dedup_symbols: bool,

    #[arg(short = 'i', long = "integrate", action = ArgAction::SetTrue, help = "integrate all -I inputs into one multi-sample object (off by default; without it every input is analyzed as an independent single sample); needs at least 2 inputs of the same file type (.gef/.gem/.h5ad, no mixing)", help_heading = "I/O options")]
    integrate: bool,

    #[arg(
        short = 'O',
        long = "output",
        default_value = "./result",
        value_name = "dir",
        help = "output root directory",
        help_heading = "I/O options"
    )]
    out_root: String,

    #[arg(short = 'q', long = "qc", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "run quality control (total_counts / n_genes_by_counts / pct_counts_mt)", help_heading = "Data preprocessing options")]
    qc: bool,

    #[arg(short = 'f', long = "filter-cells", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "filter low-quality cells", help_heading = "Data preprocessing options")]
    filter_cells: bool,

    #[arg(
        long = "min-counts",
        default_value_t = 200.0,
        help = "min total counts per cell",
        help_heading = "Data preprocessing options"
    )]
    min_counts: f32,

    #[arg(
        long = "max-counts",
        help = "max total counts per cell",
        help_heading = "Data preprocessing options"
    )]
    max_counts: Option<f32>,

    #[arg(
        long = "min-genes",
        default_value_t = 3,
        help = "min detected genes per cell",
        help_heading = "Data preprocessing options"
    )]
    min_genes: i64,

    #[arg(
        long = "max-genes",
        help = "max detected genes per cell",
        help_heading = "Data preprocessing options"
    )]
    max_genes: Option<i64>,

    #[arg(
        long = "pct-mt",
        default_value_t = 10.0,
        help = "max mitochondrial percentage",
        help_heading = "Data preprocessing options"
    )]
    pct_counts_mt: f32,

    #[arg(
        long = "cell-list",
        value_name = "file",
        help = "cell whitelist, one name per line",
        help_heading = "Data preprocessing options"
    )]
    cell_list: Option<PathBuf>,

    #[arg(long = "exclude-cells", action = ArgAction::SetTrue, help = "treat --cell-list as blacklist instead of whitelist", help_heading = "Data preprocessing options")]
    exclude_cells: bool,

    #[arg(short = 'g', long = "filter-genes", action = ArgAction::SetTrue, help = "filter lowly-expressed genes", help_heading = "Data preprocessing options")]
    filter_genes: bool,

    #[arg(
        long = "min-cells",
        help = "min cells expressing a gene",
        help_heading = "Data preprocessing options"
    )]
    min_cells: Option<i64>,

    #[arg(
        long = "max-cells",
        help = "max cells expressing a gene",
        help_heading = "Data preprocessing options"
    )]
    max_cells: Option<i64>,

    #[arg(
        long = "gene-min-counts",
        help = "min total UMI of a gene",
        help_heading = "Data preprocessing options"
    )]
    gene_min_counts: Option<f32>,

    #[arg(
        long = "gene-max-counts",
        help = "max total UMI of a gene",
        help_heading = "Data preprocessing options"
    )]
    gene_max_counts: Option<f32>,

    #[arg(
        long = "gene-mean-umi",
        help = "min mean UMI of a gene",
        help_heading = "Data preprocessing options"
    )]
    gene_mean_umi: Option<f32>,

    #[arg(
        long = "gene-list",
        value_name = "file",
        help = "gene whitelist, one name per line",
        help_heading = "Data preprocessing options"
    )]
    gene_list: Option<PathBuf>,

    #[arg(long = "exclude-genes", action = ArgAction::SetTrue, help = "treat --gene-list as blacklist instead of whitelist", help_heading = "Data preprocessing options")]
    exclude_genes: bool,

    #[arg(long = "filter-mt-genes", action = ArgAction::SetTrue, help = "drop mt- / mt. genes unconditionally", help_heading = "Data preprocessing options")]
    filter_mt_genes: bool,

    #[arg(short = 'x', long = "filter-coordinates", action = ArgAction::SetTrue, help = "crop cells by a spatial bounding box", help_heading = "Data preprocessing options")]
    filter_coordinates: bool,

    #[arg(
        long = "min-x",
        help = "X lower bound (closed interval, float)",
        help_heading = "Data preprocessing options"
    )]
    min_x: Option<f32>,
    #[arg(
        long = "max-x",
        help = "X upper bound (closed interval, float)",
        help_heading = "Data preprocessing options"
    )]
    max_x: Option<f32>,
    #[arg(
        long = "min-y",
        help = "Y lower bound (closed interval, float)",
        help_heading = "Data preprocessing options"
    )]
    min_y: Option<f32>,
    #[arg(
        long = "max-y",
        help = "Y upper bound (closed interval, float)",
        help_heading = "Data preprocessing options"
    )]
    max_y: Option<f32>,

    #[arg(short = 'n', long = "normalize-total", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "normalize total counts (CP10K by default)", help_heading = "Data preprocessing options")]
    normalize_total: bool,

    #[arg(
        long = "target-sum",
        default_value_t = 10000.0,
        help = "target total count after normalization",
        help_heading = "Data preprocessing options"
    )]
    target_sum: f32,

    #[arg(short = 'l', long = "log1p", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "apply ln(1+x)", help_heading = "Data preprocessing options")]
    log1p: bool,

    #[arg(short = 's', long = "scale", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "z-score scale; subtracts the mean before scaling, aligned with stereo", help_heading = "Data preprocessing options")]
    scale: bool,

    #[arg(long = "no-zero-center", action = ArgAction::SetTrue, help = "keep the mean: skip mean subtraction before scaling", help_heading = "Data preprocessing options")]
    no_zero_center: bool,

    #[arg(
        long = "max-value",
        help = "clip scaled values to [-max-value, max-value]",
        help_heading = "Data preprocessing options"
    )]
    max_value: Option<f32>,

    #[arg(
        long = "n-top-genes",
        default_value_t = 2000,
        help = "number of top variable genes",
        help_heading = "Data preprocessing options"
    )]
    n_top_genes: usize,

    #[arg(
        long = "n-bins",
        default_value_t = 20,
        help = "bins for dispersion normalization",
        help_heading = "Data preprocessing options"
    )]
    n_bins: usize,

    #[arg(short = 'h', long = "hvg", default_value = "seurat", value_parser = ["seurat", "cell_ranger"], help = "HVG selection method: seurat | cell_ranger", help_heading = "Data preprocessing options")]
    hvg_method: String,

    #[arg(long = "no-hvg", action = ArgAction::SetTrue, help = "PCA ignores the HVG subset and runs on the full gene matrix (HVG is still computed and plotted)", help_heading = "Data preprocessing options")]
    no_hvg: bool,

    #[arg(short = 'p', long = "pca", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "run PCA", help_heading = "Clustering options")]
    pca: bool,

    #[arg(
        long = "n-pcs",
        default_value_t = 30,
        help = "number of principal components",
        help_heading = "Clustering options"
    )]
    n_pcs: usize,

    #[arg(long = "svd-solver", default_value = "auto", value_parser = ["auto", "full", "arpack", "randomized"], help = "SVD solver: auto|full|arpack|randomized", help_heading = "Clustering options")]
    svd_solver: String,

    #[arg(short = 'b', long = "harmony", action = ArgAction::SetTrue, help = "batch effect correction (multi-sample only)", help_heading = "Clustering options")]
    harmony: bool,

    #[arg(
        long = "sigma",
        default_value_t = 0.1,
        help = "cluster soft-assignment width",
        help_heading = "Clustering options"
    )]
    sigma: f32,

    #[arg(
        long = "theta",
        help = "batch diversity penalty",
        help_heading = "Clustering options"
    )]
    theta: Option<f32>,

    #[arg(
        long = "lamb",
        help = "cluster diversity penalty",
        help_heading = "Clustering options"
    )]
    lamb: Option<f32>,

    #[arg(
        long = "tau",
        default_value_t = 0.0,
        help = "assignment matrix power",
        help_heading = "Clustering options"
    )]
    tau: f32,

    #[arg(
        long = "block-size",
        default_value_t = 0.05,
        help = "fraction of cells per iteration",
        help_heading = "Clustering options"
    )]
    block_size: f32,

    #[arg(
        long = "max-iter-harmony",
        default_value_t = 10,
        help = "outer iterations",
        help_heading = "Clustering options"
    )]
    max_iter_harmony: usize,

    #[arg(
        long = "max-iter-kmeans",
        default_value_t = 20,
        help = "inner kmeans iterations",
        help_heading = "Clustering options"
    )]
    max_iter_kmeans: usize,

    #[arg(
        long = "epsilon-cluster",
        default_value_t = 1e-5,
        help = "kmeans convergence",
        help_heading = "Clustering options"
    )]
    epsilon_cluster: f32,

    #[arg(
        long = "epsilon-harmony",
        default_value_t = 1e-4,
        help = "outer convergence",
        help_heading = "Clustering options"
    )]
    epsilon_harmony: f32,

    #[arg(
        long = "nclust",
        help = "number of clusters",
        help_heading = "Clustering options"
    )]
    nclust: Option<usize>,

    #[arg(short = 'k', long = "neighbors", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "build neighbors graph", help_heading = "Clustering options")]
    neighbors: bool,

    #[arg(
        long = "n-neighbors",
        default_value_t = 10,
        help = "number of neighbors",
        help_heading = "Clustering options"
    )]
    n_neighbors: usize,

    #[arg(
        long = "n-pcs-neighbors",
        help = "use first N PCs only",
        help_heading = "Clustering options"
    )]
    n_pcs_neighbors: Option<usize>,

    #[arg(short = 'u', long = "umap", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "UMAP embedding", help_heading = "Clustering options")]
    umap: bool,

    #[arg(
        long = "umap-components",
        default_value_t = 2,
        help = "output dimensions",
        help_heading = "Clustering options"
    )]
    umap_components: usize,

    #[arg(
        long = "min-dist",
        default_value_t = 0.5,
        help = "min distance in low-dim space",
        help_heading = "Clustering options"
    )]
    min_dist: f32,

    #[arg(
        long = "spread",
        default_value_t = 1.0,
        help = "embedding scale",
        help_heading = "Clustering options"
    )]
    spread: f32,

    #[arg(short = 'c', long = "clustering", default_value = "leiden", value_parser = ["leiden", "louvain"], help = "clustering algorithm: leiden|louvain", help_heading = "Clustering options")]
    clustering: String,

    #[arg(
        long = "resolution",
        default_value_t = 1.0,
        help = "clustering resolution",
        help_heading = "Clustering options"
    )]
    resolution: f32,

    #[arg(short = 'm', long = "marker-genes", action = ArgAction::Set, num_args = 0..=1, default_missing_value = "true", default_value_t = true, help = "find marker genes", help_heading = "Clustering options")]
    marker_genes: bool,

    #[arg(
        long = "top-n",
        default_value_t = 5,
        help = "top N marker genes per cluster, shared by the long table, heatmap, dotplot and per-cluster spatial plots",
        help_heading = "Clustering options"
    )]
    top_n: usize,

    #[arg(short = 'a', long = "singler", action = ArgAction::SetTrue, help = "SingleR cell annotation (off by default; pass this flag to enable, needs --reference)", help_heading = "Cell annotation options")]
    singler: bool,

    #[arg(
        long = "reference",
        value_name = "file",
        help = "SingleR reference h5ad, repeatable (needed while annotation is on)",
        help_heading = "Cell annotation options"
    )]
    references: Vec<PathBuf>,

    #[arg(
        long = "label-key",
        default_value = "label.fine",
        help = "SingleR label column in reference obs",
        help_heading = "Cell annotation options"
    )]
    label_key: String,

    #[arg(
        long = "quantile",
        default_value_t = 80.0,
        help = "score quantile for coarse scoring",
        help_heading = "Cell annotation options"
    )]
    quantile: f32,

    #[arg(
        long = "fine-tune-threshold",
        default_value_t = 0.05,
        help = "absolute score-gap threshold for fine tuning",
        help_heading = "Cell annotation options"
    )]
    fine_tune_threshold: f32,

    #[arg(
        long = "fine-tune-times",
        default_value_t = 0,
        help = "max fine-tuning rounds, 0 = unlimited",
        help_heading = "Cell annotation options"
    )]
    fine_tune_times: usize,

    #[arg(short = 'G', long = "go", action = ArgAction::SetTrue, help = "run ORA against GO (off by default; pass this flag to enable, needs --go-db/--org-db)", help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)")]
    go: bool,

    #[arg(
        long = "go-db",
        value_name = "file",
        help = "GO.sqlite (needed while GO enrichment is on)",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    go_db: Option<PathBuf>,

    #[arg(long = "ora-category", default_value = "all", value_parser = ["all", "BP", "CC", "MF"], help = "GO ontology only: BP|CC|MF", help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)")]
    ora_category: String,

    #[arg(short = 'K', long = "kegg", action = ArgAction::SetTrue, help = "run ORA against KEGG (off by default; pass this flag to enable, needs --kegg-dir)", help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)")]
    kegg: bool,

    #[arg(
        long = "kegg-dir",
        value_name = "dir",
        help = "KEGG raw directory (needed while KEGG enrichment is on)",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    kegg_dir: Option<PathBuf>,

    #[arg(
        long = "kegg-org",
        default_value = "mmu",
        help = "KEGG three-letter organism code, `mmu` = mouse; mouse by default, for any other species pass this flag (e.g. `hsa` = human, `rno` = rat), or the lookup fails with \"failed to open the KEGG gene-to-pathway file\"",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    kegg_org: String,

    #[arg(short = 'R', long = "reactome", action = ArgAction::SetTrue, help = "run ORA against Reactome (off by default; pass this flag to enable, needs --reactome-dir)", help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)")]
    reactome: bool,

    #[arg(
        long = "reactome-dir",
        value_name = "dir",
        help = "Reactome raw directory (needed while Reactome enrichment is on)",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    reactome_dir: Option<PathBuf>,

    #[arg(
        long = "reactome-species",
        default_value = "Mus musculus",
        help = "species name as spelled in NCBI2Reactome, default \"Mus musculus\" (mouse); the name is matched literally, so a mismatch silently yields an empty database (no error), e.g. use \"Homo sapiens\" for human",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    reactome_species: String,

    #[arg(
        long = "org-db",
        value_name = "file",
        help = "org.*.eg.sqlite for symbol/Entrez mapping (needed while any of GO/KEGG/Reactome is on); must be the org database of the same species",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    org_db: Option<PathBuf>,

    #[arg(
        long = "gene-pvalue-cutoff",
        default_value_t = 0.05,
        help = "input-gene pvalue cutoff",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    gene_pvalue_cutoff: f32,

    #[arg(
        long = "gene-log2fc-cutoff",
        default_value_t = 1.0,
        help = "input-gene log2fc cutoff",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    gene_log2fc_cutoff: f32,

    #[arg(
        long = "ora-pvalue-cutoff",
        help = "term pvalue cutoff; when given, overrides all three DBs, otherwise the default is per-DB (GO/KEGG 0.05, Reactome 1.0)",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    ora_pvalue_cutoff: Option<f32>,

    #[arg(
        long = "ora-qvalue-cutoff",
        help = "term qvalue cutoff; when given, overrides all three DBs, otherwise the default is per-DB (GO/KEGG 0.2, Reactome 1.0)",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    ora_qvalue_cutoff: Option<f32>,

    #[arg(
        long = "min-gene-count",
        default_value_t = 10,
        help = "min geneset size",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    min_gene_count: usize,

    #[arg(
        long = "max-gene-count",
        default_value_t = 500,
        help = "max geneset size",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    max_gene_count: usize,

    #[arg(
        long = "top-n-terms",
        default_value_t = 500,
        help = "max terms per cluster",
        help_heading = "Gene enrichment options (ORA: GO/KEGG/Reactome)"
    )]
    top_n_terms: usize,

    #[arg(short = 'E', long = "gsea", action = ArgAction::SetTrue, help = "run pre-ranked GSEA (off by default); reuses the ORA libraries, so at least one of GO/KEGG/Reactome must stay on", help_heading = "Gene enrichment options (GSEA)")]
    gsea: bool,

    #[arg(long = "gsea-rank-by", default_value = "log2fc", value_parser = ["log2fc", "tstat", "score"], help = "gene ranking metric: log2fc|tstat|score", help_heading = "Gene enrichment options (GSEA)")]
    gsea_rank_by: String,

    #[arg(
        long = "gsea-param",
        default_value_t = 1.0,
        help = "weight exponent p",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_param: f32,

    #[arg(
        long = "gsea-min-gene-count",
        default_value_t = 10,
        help = "min geneset size",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_min_gene_count: usize,

    #[arg(
        long = "gsea-max-gene-count",
        default_value_t = 500,
        help = "max geneset size",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_max_gene_count: usize,

    #[arg(
        long = "gsea-n-perm",
        default_value_t = 1000,
        help = "permutation count",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_n_perm: usize,

    #[arg(
        long = "gsea-seed",
        default_value_t = 0,
        help = "permutation seed",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_seed: u64,

    #[arg(
        long = "gsea-pvalue-cutoff",
        default_value_t = 0.05,
        help = "term pvalue cutoff",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_pvalue_cutoff: f32,

    #[arg(
        long = "gsea-qvalue-cutoff",
        default_value_t = 0.25,
        help = "term qvalue cutoff",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_qvalue_cutoff: f32,

    #[arg(
        long = "gsea-top-n-terms",
        default_value_t = 20,
        help = "max terms per cluster",
        help_heading = "Gene enrichment options (GSEA)"
    )]
    gsea_top_n_terms: usize,

    #[arg(
        long = "random-seed",
        default_value_t = 0,
        help = "random seed, shared by PCA / harmony / UMAP / leiden / louvain",
        help_heading = "Global options"
    )]
    random_state: u64,

    #[arg(
        long = "threads",
        default_value_t = 0,
        help = "rayon global thread pool size; 0 = leave it alone",
        help_heading = "Global options"
    )]
    threads: usize,
}

impl From<Cli> for Config {
    fn from(c: Cli) -> Self {
        let svd_solver = match c.svd_solver.as_str() {
            "auto" => SvdSolver::Auto,
            "full" => SvdSolver::Full,
            "arpack" => SvdSolver::Arpack,
            "randomized" => SvdSolver::Randomized,
            other => unreachable!("clap already restricts the value of svd-solver: {other}"),
        };
        let rank_by = match c.gsea_rank_by.as_str() {
            "log2fc" => RankBy::Log2fc,
            "tstat" => RankBy::TStat,
            "score" => RankBy::Score,
            other => unreachable!("clap already restricts the value of gsea-rank-by: {other}"),
        };
        let cluster_alg = ClusterAlg::parse(&c.clustering)
            .expect("clap already restricts the value of clustering");

        Config {
            io: Io {
                inputs: c.inputs,
                out_root: c.out_root,
                bin_type: c.bin_type,
                bin_size: c.bin_size,
                spatial_key: c.spatial_key,
                dense: c.dense,
                dedup_symbols: !c.no_dedup_symbols,
                integrate: c.integrate,
            },
            steps: Steps {
                qc: c.qc,
                filter_cells: c.filter_cells,
                filter_genes: c.filter_genes,
                filter_coordinates: c.filter_coordinates,
                normalize_total: c.normalize_total,
                log1p: c.log1p,
                scale: c.scale,
                pca: c.pca,
                harmony: c.harmony,
                neighbors: c.neighbors,
                umap: c.umap,
                clustering: true,
                marker: c.marker_genes,
                singler: c.singler,
                ora_go: c.go,
                ora_kegg: c.kegg,
                ora_reactome: c.reactome,
                gsea: c.gsea,
            },

            min_counts: Some(c.min_counts),
            max_counts: c.max_counts,
            min_genes: Some(c.min_genes),
            max_genes: c.max_genes,
            pct_counts_mt: Some(c.pct_counts_mt),
            cell_list: c.cell_list,
            exclude_cells: c.exclude_cells,

            min_cells: c.min_cells,
            max_cells: c.max_cells,
            gene_min_counts: c.gene_min_counts,
            gene_max_counts: c.gene_max_counts,
            gene_mean_umi: c.gene_mean_umi,
            gene_list: c.gene_list,
            exclude_genes: c.exclude_genes,
            filter_mt_genes: c.filter_mt_genes,

            min_x: c.min_x,
            max_x: c.max_x,
            min_y: c.min_y,
            max_y: c.max_y,

            target_sum: Some(c.target_sum),
            zero_center: !c.no_zero_center,
            max_value: c.max_value,

            n_top_genes: c.n_top_genes,
            n_bins: c.n_bins,
            hvg_method: c.hvg_method,

            use_hvg: !c.no_hvg,

            pca: PcaParams {
                n_components: c.n_pcs,
                random_state: c.random_state,
                svd_solver,
            },
            n_neighbors: c.n_neighbors,
            n_pcs_neighbors: c.n_pcs_neighbors,
            umap_components: c.umap_components,
            min_dist: c.min_dist,
            spread: c.spread,

            cluster_alg,
            resolution: c.resolution,

            top_n: c.top_n,

            random_state: c.random_state,

            threads: c.threads,

            harmony: HarmonyParams {
                nclust: c.nclust,
                sigma: c.sigma,
                theta: c.theta,
                lamb: c.lamb,
                tau: c.tau,
                block_size: c.block_size,
                max_iter_harmony: c.max_iter_harmony,
                max_iter_kmeans: c.max_iter_kmeans,
                epsilon_cluster: c.epsilon_cluster,
                epsilon_harmony: c.epsilon_harmony,
                random_state: c.random_state,
            },

            references: c.references,
            label_key: c.label_key,
            singler: SingleRParams {
                quantile: c.quantile,
                fine_tune_threshold: c.fine_tune_threshold,
                fine_tune_times: c.fine_tune_times,
            },

            ora_gene_pvalue_cutoff: c.gene_pvalue_cutoff,
            ora_gene_log2fc_cutoff: c.gene_log2fc_cutoff,
            ora_min_gene_count: c.min_gene_count,
            ora_max_gene_count: c.max_gene_count,
            ora_top_n_terms: c.top_n_terms,
            ora_category: if c.ora_category == "all" {
                None
            } else {
                Some(c.ora_category)
            },
            ora_pvalue_cutoff: c.ora_pvalue_cutoff,
            ora_qvalue_cutoff: c.ora_qvalue_cutoff,

            gsea: GseaParams {
                category: None,
                rank_by,
                gsea_param: c.gsea_param,
                min_gene_count: c.gsea_min_gene_count,
                max_gene_count: c.gsea_max_gene_count,
                n_perm: c.gsea_n_perm,
                seed: c.gsea_seed,
                pvalue_cutoff: c.gsea_pvalue_cutoff,
                qvalue_cutoff: c.gsea_qvalue_cutoff,
                top_n_terms: c.gsea_top_n_terms,
            },

            go_db: c.go_db.unwrap_or_default(),
            org_db: c.org_db.unwrap_or_default(),
            kegg_dir: c.kegg_dir.unwrap_or_default(),
            reactome_dir: c.reactome_dir.unwrap_or_default(),
            kegg_org: c.kegg_org,
            reactome_species: c.reactome_species,
        }
    }
}

const LOGO: &str = r" ____    ___      _      ____     ___
|  _ \  |_ _|    / \    / ___|   / _ \
| |_) |  | |    / _ \   \___ \  | | | |
|  _ <   | |   / ___ \   ___) | | |_| |
|_| \_\ |___| /_/   \_\ |____/   \___/";

const BANNER_WIDTH: usize = 62;

fn banner_line(s: &str) -> String {
    let pad = BANNER_WIDTH.saturating_sub(s.chars().count());
    let left = pad / 2;
    format!("#{}{}{}#", " ".repeat(left), s, " ".repeat(pad - left))
}

fn banner_indented(s: &str, left: usize) -> String {
    let pad = BANNER_WIDTH.saturating_sub(left + s.chars().count());
    format!("#{}{}{}#", " ".repeat(left), s, " ".repeat(pad))
}

fn print_banner() {
    let border = "#".repeat(BANNER_WIDTH + 2);
    let logo_w = LOGO.lines().map(|l| l.chars().count()).max().unwrap_or(0);
    let left = BANNER_WIDTH.saturating_sub(logo_w) / 2;
    println!("{border}");
    println!("{}", banner_line(""));
    for line in LOGO.lines() {
        println!("{}", banner_indented(line, left));
    }
    println!("{}", banner_line(""));
    println!(
        "{}",
        banner_line("Rapid And Intelligent Analyzer For Spatial Omics")
    );
    println!(
        "{}",
        banner_line(&format!("version {}", env!("CARGO_PKG_VERSION")))
    );
    println!("{}", banner_line(""));
    println!("{border}");
    println!();
}

fn main() {
    let cli = Cli::parse();
    let config = Config::from(cli);
    print_banner();
    if let Err(e) = riaso::pipeline::run(&config) {
        eprintln!("riaso: {e:#}");
        std::process::exit(1);
    }
}
