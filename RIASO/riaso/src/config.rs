use std::path::PathBuf;

use crate::algorithm::annotation::SingleRParams;
use crate::algorithm::batch::HarmonyParams;
use crate::algorithm::enrichment::{GseaParams, RankBy};
use crate::algorithm::pca::{PcaParams, SvdSolver};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterAlg {
    Leiden,
    Louvain,
}

impl ClusterAlg {
    pub fn key(self) -> &'static str {
        match self {
            ClusterAlg::Leiden => "leiden",
            ClusterAlg::Louvain => "louvain",
        }
    }

    pub fn display(self) -> &'static str {
        match self {
            ClusterAlg::Leiden => "Leiden",
            ClusterAlg::Louvain => "Louvain",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "leiden" => Some(ClusterAlg::Leiden),
            "louvain" => Some(ClusterAlg::Louvain),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Io {
    pub inputs: Vec<String>,

    pub out_root: String,

    pub bin_type: String,

    pub bin_size: u32,

    pub spatial_key: String,

    pub dense: bool,

    pub dedup_symbols: bool,

    pub integrate: bool,
}

impl Default for Io {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            out_root: "./result".to_string(),
            bin_type: "bins".to_string(),
            bin_size: 100,
            spatial_key: "spatial".to_string(),
            dense: false,
            dedup_symbols: true,
            integrate: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Steps {
    pub qc: bool,

    pub filter_cells: bool,
    pub filter_genes: bool,
    pub filter_coordinates: bool,
    pub normalize_total: bool,
    pub log1p: bool,
    pub scale: bool,
    pub pca: bool,

    pub harmony: bool,
    pub neighbors: bool,
    pub umap: bool,
    pub clustering: bool,
    pub marker: bool,
    pub singler: bool,
    pub ora_go: bool,
    pub ora_kegg: bool,
    pub ora_reactome: bool,
    pub gsea: bool,
}

impl Default for Steps {
    fn default() -> Self {
        Self {
            qc: true,
            filter_cells: true,
            filter_genes: false,
            filter_coordinates: false,
            normalize_total: true,
            log1p: true,
            scale: true,
            pca: true,
            harmony: false,
            neighbors: true,
            umap: true,
            clustering: true,
            marker: true,
            singler: false,
            ora_go: false,
            ora_kegg: false,
            ora_reactome: false,
            gsea: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub io: Io,
    pub steps: Steps,

    pub min_counts: Option<f32>,
    pub max_counts: Option<f32>,
    pub min_genes: Option<i64>,
    pub max_genes: Option<i64>,
    pub pct_counts_mt: Option<f32>,

    pub cell_list: Option<PathBuf>,

    pub exclude_cells: bool,

    pub min_cells: Option<i64>,
    pub max_cells: Option<i64>,
    pub gene_min_counts: Option<f32>,
    pub gene_max_counts: Option<f32>,
    pub gene_mean_umi: Option<f32>,

    pub gene_list: Option<PathBuf>,
    pub exclude_genes: bool,
    pub filter_mt_genes: bool,

    pub min_x: Option<f32>,
    pub max_x: Option<f32>,
    pub min_y: Option<f32>,
    pub max_y: Option<f32>,

    pub target_sum: Option<f32>,
    pub zero_center: bool,
    pub max_value: Option<f32>,

    pub n_top_genes: usize,

    pub n_bins: usize,

    pub hvg_method: String,

    pub use_hvg: bool,

    pub pca: PcaParams,
    pub n_neighbors: usize,
    pub n_pcs_neighbors: Option<usize>,
    pub umap_components: usize,
    pub min_dist: f32,
    pub spread: f32,

    pub cluster_alg: ClusterAlg,
    pub resolution: f32,

    pub top_n: usize,

    pub random_state: u64,

    pub threads: usize,

    pub harmony: HarmonyParams,

    pub references: Vec<PathBuf>,
    pub label_key: String,

    pub singler: SingleRParams,

    pub ora_gene_pvalue_cutoff: f32,
    pub ora_gene_log2fc_cutoff: f32,
    pub ora_min_gene_count: usize,
    pub ora_max_gene_count: usize,
    pub ora_top_n_terms: usize,

    pub ora_category: Option<String>,

    pub ora_pvalue_cutoff: Option<f32>,

    pub ora_qvalue_cutoff: Option<f32>,

    pub gsea: GseaParams,

    pub go_db: PathBuf,

    pub org_db: PathBuf,

    pub kegg_dir: PathBuf,

    pub reactome_dir: PathBuf,

    pub kegg_org: String,

    pub reactome_species: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            io: Io::default(),
            steps: Steps::default(),

            min_counts: Some(200.0),
            max_counts: None,
            min_genes: Some(3),
            max_genes: None,
            pct_counts_mt: Some(10.0),
            cell_list: None,
            exclude_cells: false,

            min_cells: None,
            max_cells: None,
            gene_min_counts: None,
            gene_max_counts: None,
            gene_mean_umi: None,
            gene_list: None,
            exclude_genes: false,
            filter_mt_genes: false,

            min_x: None,
            max_x: None,
            min_y: None,
            max_y: None,

            target_sum: Some(10000.0),
            zero_center: true,
            max_value: None,

            n_top_genes: 2000,
            n_bins: 20,
            hvg_method: "seurat".to_string(),
            use_hvg: true,

            pca: PcaParams {
                n_components: 30,
                random_state: 0,
                svd_solver: SvdSolver::Auto,
            },
            n_neighbors: 10,
            n_pcs_neighbors: None,
            umap_components: 2,
            min_dist: 0.5,
            spread: 1.0,

            cluster_alg: ClusterAlg::Leiden,
            resolution: 1.0,

            top_n: 5,

            random_state: 0,

            threads: 0,

            harmony: HarmonyParams::default(),

            references: Vec::new(),
            label_key: "label.fine".to_string(),
            singler: SingleRParams::default(),

            ora_gene_pvalue_cutoff: 0.05,
            ora_gene_log2fc_cutoff: 1.0,
            ora_min_gene_count: 10,
            ora_max_gene_count: 500,
            ora_top_n_terms: 500,
            ora_category: None,
            ora_pvalue_cutoff: None,
            ora_qvalue_cutoff: None,

            gsea: GseaParams {
                rank_by: RankBy::Log2fc,
                ..Default::default()
            },

            go_db: PathBuf::new(),
            org_db: PathBuf::new(),
            kegg_dir: PathBuf::new(),
            reactome_dir: PathBuf::new(),
            kegg_org: "mmu".to_string(),
            reactome_species: "Mus musculus".to_string(),
        }
    }
}
