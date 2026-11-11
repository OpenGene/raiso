use anyhow::anyhow;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use polars::datatypes::DataType;
use polars::frame::DataFrame;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::parallel::csr_to_dense;

use crate::algorithm::annotation::run_singler as core_run_singler;
use crate::algorithm::annotation::SingleRParams;
use crate::algorithm::batch::harmony::run_harmony as core_run_harmony;
use crate::algorithm::batch::harmony::HarmonyParams;
use crate::algorithm::cluster::{run_leiden as core_run_leiden, run_louvain as core_run_louvain};
use crate::algorithm::enrichment::{run_gsea as core_run_gsea, run_ora as core_run_ora};
use crate::algorithm::enrichment::{GseaParams, OraParams, RankBy};
use crate::algorithm::filter::{
    filter_cells as core_filter_cells, filter_coordinates as core_filter_coordinates,
    filter_genes as core_filter_genes,
};
use crate::algorithm::hvg::select_top_variable_genes as core_hvg;
use crate::algorithm::knn::DENSE_KNN_THRESHOLD;
use crate::algorithm::marker::run_marker as core_run_marker;
use crate::algorithm::marker::write_marker_csv as core_write_marker_csv;
use crate::algorithm::marker::MarkerParams;
use crate::algorithm::neighbors::run_neighbors as core_run_neighbors;
use crate::algorithm::normalize::{
    log1p as core_log1p, normalize_total as core_normalize_total, scale as core_scale,
};
use crate::algorithm::pca::run_pca as core_run_pca;
use crate::algorithm::pca::{PcaParams, SvdSolver};
use crate::algorithm::qc::cal_qc as core_cal_qc;
use crate::algorithm::umap::run_umap as core_run_umap;
use crate::core::EnrichmentResult;
use crate::core::GeneSetDb;
use crate::core::{ExpMatrix, PcaResult, SingleRRef, SoData};
use crate::io::read_gef as io_read_gef;
use crate::io::read_gem as io_read_gem;
use crate::io::read_h5ad as io_read_h5ad;
use crate::io::singler_ref::read_ref as io_read_singler_ref;
use crate::io::{
    read_go_db as io_read_go_db, read_kegg_db as io_read_kegg_db,
    read_reactome_db as io_read_reactome_db,
};
use crate::plot::cluster_scatter::{
    plot_annotation_scatter as plt_annotation_scatter,
    plot_annotation_scatter_png as plt_annotation_scatter_png,
    plot_cluster_scatter as plt_cluster_scatter,
    plot_cluster_scatter_png as plt_cluster_scatter_png,
};
use crate::plot::gsea::{
    plot_gsea_barplot as plt_gsea_barplot, plot_gsea_barplot_png as plt_gsea_barplot_png,
    plot_gsea_dotplot as plt_gsea_dotplot, plot_gsea_dotplot_png as plt_gsea_dotplot_png,
    plot_gsea_enrichment as plt_gsea_enrichment,
    plot_gsea_enrichment_png as plt_gsea_enrichment_png,
};
use crate::plot::heatmap::{
    plot_marker_dotplot as plt_marker_dotplot, plot_marker_dotplot_png as plt_marker_dotplot_png,
    plot_marker_heatmap as plt_marker_heatmap, plot_marker_heatmap_png as plt_marker_heatmap_png,
};
use crate::plot::hvg::{plot_hvg as plt_hvg, plot_hvg_png as plt_hvg_png};
use crate::plot::marker_spatial::{
    plot_marker_spatial as plt_marker_spatial, plot_marker_spatial_png as plt_marker_spatial_png,
};
use crate::plot::ora::{
    plot_ora_barplot as plt_ora_barplot, plot_ora_barplot_png as plt_ora_barplot_png,
    plot_ora_dotplot as plt_ora_dotplot, plot_ora_dotplot_png as plt_ora_dotplot_png,
};
use crate::plot::pca::{
    plot_pca as plt_pca, plot_pca_loadings as plt_pca_loadings,
    plot_pca_loadings_png as plt_pca_loadings_png, plot_pca_png as plt_pca_png,
    plot_pca_variance_ratio as plt_pca_variance_ratio,
    plot_pca_variance_ratio_png as plt_pca_variance_ratio_png,
};
use crate::plot::singler_heatmap::{
    plot_singler_score_heatmap as plt_singler_heatmap,
    plot_singler_score_heatmap_png as plt_singler_heatmap_png,
};
use crate::plot::umap::{plot_umap as plt_umap, plot_umap_png as plt_umap_png};
use crate::plot::{
    plot_batch_spatial_scatter as plt_batch_spatial_scatter,
    plot_batch_spatial_scatter_png as plt_batch_spatial_scatter_png, plot_qc as plt_qc,
    plot_qc_png as plt_qc_png, plot_spatial_scatter as plt_spatial_scatter,
    plot_spatial_scatter_png as plt_spatial_scatter_png, PlotKind,
};
use crate::tools::{integrate as tools_integrate, integrate_union as tools_integrate_union};

fn to_pyerr(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

fn neighbors_matrix(py: Python<'_>, csr: &CsrMatrix<f32>) -> PyResult<Py<PyAny>> {
    let n = csr.nrows();
    if n <= DENSE_KNN_THRESHOLD {
        Ok(csr_to_dense(csr).into_pyarray(py).into_any().unbind())
    } else {
        let dict = PyDict::new(py);
        let data: Vec<f32> = csr.values().iter().copied().collect();
        let indices: Vec<i64> = csr.col_indices().iter().map(|&x| x as i64).collect();
        let indptr: Vec<i64> = csr.row_offsets().iter().map(|&x| x as i64).collect();
        dict.set_item("data", data.into_pyarray(py))?;
        dict.set_item("indices", indices.into_pyarray(py))?;
        dict.set_item("indptr", indptr.into_pyarray(py))?;
        Ok(dict.into_any().unbind())
    }
}

fn dataframe_to_dict(py: Python<'_>, df: &DataFrame) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for col in df.get_columns() {
        let name = col.name().to_string();
        match col.dtype() {
            DataType::Float64 => {
                let ca = col
                    .f32()
                    .map_err(|e| PyRuntimeError::new_err(format!("column '{name}': {e}")))?;
                let v: Vec<f32> = (0..ca.len())
                    .map(|i| ca.get(i).unwrap_or(f32::NAN))
                    .collect();
                dict.set_item(name, Array1::from_vec(v).into_pyarray(py))?;
            }
            DataType::Int64 => {
                let ca = col
                    .i64()
                    .map_err(|e| PyRuntimeError::new_err(format!("column '{name}': {e}")))?;
                let v: Vec<i64> = (0..ca.len()).map(|i| ca.get(i).unwrap_or(0)).collect();
                dict.set_item(name, Array1::from_vec(v).into_pyarray(py))?;
            }
            DataType::String => {
                let ca = col
                    .str()
                    .map_err(|e| PyRuntimeError::new_err(format!("column '{name}': {e}")))?;
                let v: Vec<String> = (0..ca.len())
                    .map(|i| ca.get(i).unwrap_or("").to_string())
                    .collect();
                dict.set_item(name, PyList::new(py, v)?)?;
            }

            DataType::Boolean => {
                let ca = col
                    .bool()
                    .map_err(|e| PyRuntimeError::new_err(format!("column '{name}': {e}")))?;
                let v: Vec<bool> = (0..ca.len()).map(|i| ca.get(i).unwrap_or(false)).collect();
                dict.set_item(name, v)?;
            }
            _ => {}
        }
    }
    Ok(dict.unbind())
}

fn parse_svd_solver(s: &str) -> PyResult<SvdSolver> {
    match s {
        "auto" => Ok(SvdSolver::Auto),
        "full" => Ok(SvdSolver::Full),
        "arpack" => Ok(SvdSolver::Arpack),
        "randomized" => Ok(SvdSolver::Randomized),
        other => Err(PyValueError::new_err(format!(
            "svd_solver must be \"auto\" / \"full\" / \"arpack\" / \"randomized\", got {other:?}"
        ))),
    }
}

fn parse_components(s: &str, name: &str) -> PyResult<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let t = part.trim();
        if t.is_empty() {
            continue;
        }
        let n: usize = t.parse().map_err(|_| {
            PyValueError::new_err(format!(
                "{name} must be comma-separated positive integers (1-based), e.g. \"1,2\", got {s:?}"
            ))
        })?;
        if n == 0 {
            return Err(PyValueError::new_err(format!(
                "{name} is 1-based (PC1 means 1); 0 is not allowed, got {s:?}"
            )));
        }
        out.push(n - 1);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err(format!("{name} cannot be empty")));
    }
    Ok(out)
}

fn obs_labels(data: &SoData, key: &str) -> anyhow::Result<Vec<i64>> {
    let obs = data.obs.as_ref().ok_or_else(|| anyhow!("obs is empty"))?;
    let col = obs
        .column(key)
        .map_err(|_| anyhow!("obs is missing column {key}: run leiden() first (or pass cluster_key=<another cluster label column>)"))?;
    let labels: Vec<i64> = col
        .i64()
        .map_err(|_| anyhow!("obs[{key}] is not Int64"))?
        .into_no_null_iter()
        .collect();
    Ok(labels)
}

fn pca_result_to_dict(py: Python<'_>, result: &PcaResult, dtype: &str) -> PyResult<Py<PyDict>> {
    let to_f32 = match dtype {
        "float32" => true,
        "float64" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "dtype must be \"float32\" or \"float64\", got {other:?}"
            )))
        }
    };
    let dict = PyDict::new(py);
    if to_f32 {
        dict.set_item(
            "projected",
            result.projected.mapv(|v| v as f32).into_pyarray(py),
        )?;
        dict.set_item(
            "components",
            result.components.mapv(|v| v as f32).into_pyarray(py),
        )?;
        dict.set_item(
            "explained_variance",
            result
                .explained_variance
                .mapv(|v| v as f32)
                .into_pyarray(py),
        )?;
        dict.set_item(
            "explained_variance_ratio",
            result
                .explained_variance_ratio
                .mapv(|v| v as f32)
                .into_pyarray(py),
        )?;
        dict.set_item("mean", result.mean.mapv(|v| v as f32).into_pyarray(py))?;
    } else {
        dict.set_item("projected", result.projected.clone().into_pyarray(py))?;
        dict.set_item("components", result.components.clone().into_pyarray(py))?;
        dict.set_item(
            "explained_variance",
            result.explained_variance.clone().into_pyarray(py),
        )?;
        dict.set_item(
            "explained_variance_ratio",
            result.explained_variance_ratio.clone().into_pyarray(py),
        )?;
        dict.set_item("mean", result.mean.clone().into_pyarray(py))?;
    }
    dict.set_item("total_variance", result.total_variance)?;
    Ok(dict.unbind())
}

fn ora_result_to_dict(py: Python<'_>, res: &EnrichmentResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (db_name, groups) in &res.groups {
        let clusters = PyList::empty(py);
        for g in groups {
            let terms = PyList::empty(py);
            for t in &g.terms {
                let td = PyDict::new(py);
                td.set_item("id", t.id.as_str())?;
                td.set_item("name", t.name.as_str())?;
                td.set_item("category", t.category.clone())?;
                td.set_item("pvalue", t.pvalue)?;
                td.set_item("pvalue_adj", t.pvalue_adj)?;
                td.set_item("count", t.count)?;
                td.set_item("overlap", t.overlap.clone())?;
                td.set_item("gene_ratio", t.gene_ratio)?;
                td.set_item("bg_ratio", t.bg_ratio)?;
                terms.append(td)?;
            }
            let cd = PyDict::new(py);
            cd.set_item("cluster_id", g.group_id)?;
            cd.set_item("terms", terms)?;
            clusters.append(cd)?;
        }
        let d = PyDict::new(py);
        d.set_item("clusters", clusters)?;
        d.set_item("n_groups", groups.len())?;
        dict.set_item(db_name.as_str(), d)?;
    }
    Ok(dict.unbind())
}

fn gsea_result_to_dict(py: Python<'_>, res: &EnrichmentResult) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    for (db_name, gsea) in &res.gsea {
        let clusters = PyList::empty(py);
        for g in &gsea.groups {
            let terms = PyList::empty(py);
            for t in &g.terms {
                let td = PyDict::new(py);
                td.set_item("id", t.id.as_str())?;
                td.set_item("name", t.name.as_str())?;
                td.set_item("category", t.category.clone())?;
                td.set_item("es", t.es)?;
                td.set_item("nes", t.nes)?;
                td.set_item("pvalue", t.pvalue)?;
                td.set_item("pvalue_adj", t.pvalue_adj)?;
                td.set_item("size", t.size)?;
                td.set_item("rank_at_es", t.rank_at_es)?;
                td.set_item("leading_edge", t.leading_edge.clone())?;
                terms.append(td)?;
            }
            let cd = PyDict::new(py);
            cd.set_item("cluster_id", g.group_id)?;
            cd.set_item("terms", terms)?;
            clusters.append(cd)?;
        }
        let d = PyDict::new(py);
        d.set_item("clusters", clusters)?;
        d.set_item("n_groups", gsea.groups.len())?;
        dict.set_item(db_name.as_str(), d)?;
    }
    Ok(dict.unbind())
}

fn parse_rank_by(rank_by: &str) -> PyResult<RankBy> {
    match rank_by.to_ascii_lowercase().as_str() {
        "log2fc" | "log2_fc" => Ok(RankBy::Log2fc),
        "tstat" | "t_stat" | "t" => Ok(RankBy::TStat),
        "score" => Ok(RankBy::Score),
        other => Err(PyValueError::new_err(format!(
            "rank_by must be \"log2fc\" / \"tstat\" / \"score\", got {other:?}"
        ))),
    }
}

#[pyclass(name = "SoData")]
pub struct PySoData {
    inner: SoData,

    var_type: String,
}

#[pymethods]
impl PySoData {
    #[new]
    fn new() -> Self {
        Self {
            inner: SoData::new(),
            var_type: "intersect".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SoData(n_cells={}, n_genes={}, n_samples={})",
            self.inner.n_obs(),
            self.inner.n_vars(),
            self.inner.samples_names.len()
        )
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn n_cells(&self) -> usize {
        self.inner.n_obs()
    }

    #[getter]
    fn n_genes(&self) -> usize {
        self.inner.n_vars()
    }

    #[getter]
    fn is_sparse(&self) -> bool {
        matches!(self.inner.exp_matrix, Some(ExpMatrix::Sparse(_)))
    }

    #[allow(non_snake_case)]
    #[getter]
    fn X(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f32>>> {
        let exp =
            self.inner.exp_matrix.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyAttributeError::new_err("exp_matrix is empty")
            })?;
        Ok(exp.to_dense().into_pyarray(py).unbind())
    }

    #[getter]
    fn obs(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        match &self.inner.obs {
            Some(df) => dataframe_to_dict(py, df),
            None => Ok(PyDict::new(py).unbind()),
        }
    }

    #[getter]
    fn var(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        match &self.inner.var {
            Some(df) => dataframe_to_dict(py, df),
            None => Ok(PyDict::new(py).unbind()),
        }
    }

    #[getter]
    fn position(&self, py: Python<'_>) -> Option<Py<PyArray2<f32>>> {
        self.inner
            .position
            .as_ref()
            .map(|p| p.clone().into_pyarray(py).unbind())
    }

    #[getter]
    fn raw(&self) -> Option<PySoData> {
        self.inner.raw.as_ref().map(|r| PySoData {
            inner: (**r).clone(),
            var_type: self.var_type.clone(),
        })
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.samples_names.clone()
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.samples_names.len()
    }

    #[getter]
    fn var_type(&self) -> String {
        self.var_type.clone()
    }

    fn cal_qc(&mut self, py: Python<'_>) {
        py.detach(|| core_cal_qc(&mut self.inner));
    }

    #[pyo3(signature = (min_counts=None, max_counts=None, min_genes=None, max_genes=None,
                        pct_counts_mt=None, cell_list=None, excluded=false, *, write_layer=false))]
    fn filter_cells(
        &mut self,
        py: Python<'_>,
        min_counts: Option<f32>,
        max_counts: Option<f32>,
        min_genes: Option<i64>,
        max_genes: Option<i64>,
        pct_counts_mt: Option<f32>,
        cell_list: Option<Vec<String>>,
        excluded: bool,
        write_layer: bool,
    ) {
        py.detach(|| {
            core_filter_cells(
                &mut self.inner,
                min_counts,
                max_counts,
                min_genes,
                max_genes,
                pct_counts_mt,
                cell_list.as_deref(),
                excluded,
                write_layer,
            )
        });
    }

    #[pyo3(signature = (min_cells=None, max_cells=None, min_counts=None, max_counts=None,
                        mean_umi_gt=None, gene_list=None, excluded=false, filter_mt_genes=false,
                        *, write_layer=false))]
    fn filter_genes(
        &mut self,
        py: Python<'_>,
        min_cells: Option<i64>,
        max_cells: Option<i64>,
        min_counts: Option<f32>,
        max_counts: Option<f32>,
        mean_umi_gt: Option<f32>,
        gene_list: Option<Vec<String>>,
        excluded: bool,
        filter_mt_genes: bool,
        write_layer: bool,
    ) {
        py.detach(|| {
            core_filter_genes(
                &mut self.inner,
                min_cells,
                max_cells,
                min_counts,
                max_counts,
                mean_umi_gt,
                gene_list.as_deref(),
                excluded,
                filter_mt_genes,
                write_layer,
            )
        });
    }

    #[pyo3(signature = (min_x=None, max_x=None, min_y=None, max_y=None, *, write_layer=false))]
    fn filter_coordinates(
        &mut self,
        py: Python<'_>,
        min_x: Option<f32>,
        max_x: Option<f32>,
        min_y: Option<f32>,
        max_y: Option<f32>,
        write_layer: bool,
    ) {
        py.detach(|| {
            core_filter_coordinates(&mut self.inner, min_x, max_x, min_y, max_y, write_layer)
        });
    }

    #[pyo3(signature = (target_sum, *, write_layer=false))]
    fn normalize_total(&mut self, py: Python<'_>, target_sum: f32, write_layer: bool) {
        py.detach(|| core_normalize_total(&mut self.inner, Some(target_sum), write_layer));
    }

    #[pyo3(signature = (*, write_layer=false))]
    fn log1p(&mut self, py: Python<'_>, write_layer: bool) {
        py.detach(|| core_log1p(&mut self.inner, write_layer));
    }

    #[pyo3(signature = (*, zero_center=false, max_value=None, write_layer=false))]
    fn scale(
        &mut self,
        py: Python<'_>,
        zero_center: bool,
        max_value: Option<f32>,
        write_layer: bool,
    ) {
        py.detach(|| core_scale(&mut self.inner, zero_center, max_value, write_layer));
    }

    #[pyo3(signature = (method="seurat", n_top_genes=2000, n_bins=20))]
    fn highly_variable_genes(
        &mut self,
        py: Python<'_>,
        method: &str,
        n_top_genes: usize,
        n_bins: usize,
    ) {
        py.detach(|| core_hvg(&mut self.inner, n_top_genes, n_bins, method));
    }

    #[pyo3(signature = (n_components=30, svd_solver="auto", dtype="float32", random_state=0))]
    fn pca(
        &mut self,
        py: Python<'_>,
        n_components: usize,
        svd_solver: &str,
        dtype: &str,
        random_state: u64,
    ) -> PyResult<Py<PyDict>> {
        let params = PcaParams {
            n_components,
            random_state,
            svd_solver: parse_svd_solver(svd_solver)?,
        };
        py.detach(|| core_run_pca(&mut self.inner, &params, true));
        let result = self.inner.result.pca.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "pca failed: the expression matrix is empty or has too few genes",
            )
        })?;
        pca_result_to_dict(py, result, dtype)
    }

    #[pyo3(signature = (sigma=0.1, theta=None, lamb=None, tau=0.0, block_size=0.05,
                        max_iter_harmony=10, max_iter_kmeans=20,
                        epsilon_cluster=1e-5, epsilon_harmony=1e-4, random_state=0, nclust=None))]
    fn harmony(
        &mut self,
        py: Python<'_>,
        sigma: f32,
        theta: Option<f32>,
        lamb: Option<f32>,
        tau: f32,
        block_size: f32,
        max_iter_harmony: usize,
        max_iter_kmeans: usize,
        epsilon_cluster: f32,
        epsilon_harmony: f32,
        random_state: u64,
        nclust: Option<usize>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let params = HarmonyParams {
            nclust,
            sigma,
            theta,
            lamb,
            tau,
            block_size,
            max_iter_harmony,
            max_iter_kmeans,
            epsilon_cluster,
            epsilon_harmony,
            random_state,
        };
        py.detach(|| core_run_harmony(&mut self.inner, &params));
        let result = self
            .inner
            .result
            .batch
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("harmony failed"))?;
        Ok(result.corrected.clone().into_pyarray(py).unbind())
    }

    #[pyo3(signature = (pca_res_key=None, n_neighbors=10, n_pcs=None))]
    fn neighbors(
        &mut self,
        py: Python<'_>,
        pca_res_key: Option<String>,
        n_neighbors: usize,
        n_pcs: Option<usize>,
    ) -> PyResult<Py<PyDict>> {
        let _ = pca_res_key;
        py.detach(|| core_run_neighbors(&mut self.inner, n_neighbors, n_pcs));
        let result = self
            .inner
            .result
            .neighbors
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("neighbors failed"))?;
        let dict = PyDict::new(py);
        dict.set_item("neighbor", result.knn_indices.clone().into_pyarray(py))?;
        dict.set_item(
            "connectivities",
            neighbors_matrix(py, &result.connectivities)?,
        )?;
        dict.set_item("nn_dist", neighbors_matrix(py, &result.distances)?)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (n_components=2, min_dist=0.5, spread=1.0, random_state=0))]
    fn umap(
        &mut self,
        py: Python<'_>,
        n_components: usize,
        min_dist: f32,
        spread: f32,
        random_state: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            core_run_umap(
                &mut self.inner,
                n_components,
                min_dist,
                spread,
                random_state,
            )
        });
        Ok(())
    }

    #[pyo3(signature = (resolution=1.0, random_state=0))]
    fn leiden(&mut self, py: Python<'_>, resolution: f32, random_state: u64) -> PyResult<()> {
        py.detach(|| core_run_leiden(&mut self.inner, resolution, random_state));
        Ok(())
    }

    #[pyo3(signature = (resolution=1.0, random_state=0))]
    fn louvain(&mut self, py: Python<'_>, resolution: f32, random_state: u64) -> PyResult<()> {
        py.detach(|| core_run_louvain(&mut self.inner, resolution, random_state));
        Ok(())
    }

    #[pyo3(signature = (cluster_key="leiden", *, top_n=10))]
    fn marker_genes(
        &mut self,
        py: Python<'_>,
        cluster_key: &str,
        top_n: usize,
    ) -> PyResult<Py<PyDict>> {
        let cluster_key = cluster_key.to_string();
        let syms = py
            .detach(|| {
                let labels = obs_labels(&self.inner, &cluster_key)?;
                let params = MarkerParams {
                    n_genes: None,
                    ..Default::default()
                };
                core_run_marker(&mut self.inner, &labels, &params);
                Ok::<_, anyhow::Error>(self.inner.symbol_names())
            })
            .map_err(to_pyerr)?;
        let marker = self.inner.result.marker.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("marker_genes failed: obs has no cluster label column, or the expression matrix is empty")
        })?;

        let dict = PyDict::new(py);
        let clusters = PyList::empty(py);
        for cl in &marker.clusters {
            let mut genes = Vec::new();
            let mut log2fc = Vec::new();
            let mut p_adj = Vec::new();
            let mut mean_count = Vec::new();
            let mut pct = Vec::new();
            let mut pct_rest = Vec::new();
            for (i, &gi) in cl.gene_indices.iter().enumerate() {
                let name = syms.get(gi).cloned().unwrap_or_default();
                if name.is_empty() {
                    continue;
                }
                genes.push(name);
                log2fc.push(cl.log2fc[i]);
                p_adj.push(cl.pvalues_adj[i]);
                mean_count.push(cl.mean_count[i]);
                pct.push(cl.pct[i]);
                pct_rest.push(cl.pct_rest[i]);
                if genes.len() >= top_n {
                    break;
                }
            }
            let cdict = PyDict::new(py);
            cdict.set_item("cluster_id", cl.cluster_id)?;
            cdict.set_item("genes", genes)?;
            cdict.set_item("log2fc", Array1::from_vec(log2fc).into_pyarray(py))?;
            cdict.set_item("p_adj", Array1::from_vec(p_adj).into_pyarray(py))?;
            cdict.set_item("mean_count", Array1::from_vec(mean_count).into_pyarray(py))?;
            cdict.set_item("pct", Array1::from_vec(pct).into_pyarray(py))?;
            cdict.set_item("pct_rest", Array1::from_vec(pct_rest).into_pyarray(py))?;
            clusters.append(cdict)?;
        }
        dict.set_item("clusters", clusters)?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (path, *, top=None))]
    fn write_marker_csv(&self, py: Python<'_>, path: &str, top: Option<usize>) -> PyResult<()> {
        let path = path.to_string();
        py.detach(|| {
            let marker = self.inner.result.marker.as_ref().ok_or_else(|| {
                anyhow!("write_marker_csv: result.marker is empty; call marker_genes() first")
            })?;
            let syms = self.inner.symbol_names();
            core_write_marker_csv(marker, &syms, std::path::Path::new(&path), top)
        })
        .map_err(to_pyerr)
    }

    #[pyo3(signature = (reference, cluster_labels=None, quantile=80.0,
                        fine_tune_threshold=0.05, fine_tune_times=0))]
    fn singler(
        &mut self,
        py: Python<'_>,
        reference: &PySingleRRef,
        cluster_labels: Option<Vec<i64>>,
        quantile: f32,
        fine_tune_threshold: f32,
        fine_tune_times: usize,
    ) -> PyResult<Py<PyDict>> {
        let reference = reference.inner.clone();
        let params = SingleRParams {
            quantile,
            fine_tune_threshold,
            fine_tune_times,
        };
        py.detach(|| {
            core_run_singler(
                &mut self.inner,
                &reference,
                cluster_labels.as_deref(),
                &params,
            )
        });
        let res = self
            .inner
            .result
            .annotation
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("singler failed"))?;
        let dict = PyDict::new(py);
        dict.set_item("first_labels", res.first_labels.clone())?;
        dict.set_item("final_labels", res.final_labels.clone())?;
        dict.set_item("labels", res.labels.clone())?;
        if cluster_labels.is_some() {
            dict.set_item("clusters", res.clusters.clone())?;
        }
        dict.set_item("scores", res.scores.clone().into_pyarray(py))?;
        Ok(dict.unbind())
    }

    #[pyo3(signature = (path=None, *, kind="violin", metrics=None, title=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_qc(
        &self,
        py: Python<'_>,
        path: Option<String>,
        kind: &str,
        metrics: Option<Vec<String>>,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let plot_kind = match kind {
            "violin" => PlotKind::Violin,
            "hist" => PlotKind::Histogram,
            other => {
                return Err(PyValueError::new_err(format!(
                    "kind must be \"violin\" or \"hist\", got {other:?}"
                )))
            }
        };
        let metrics = metrics.unwrap_or_else(|| {
            ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        });
        let n = metrics.len().max(1) as u32;
        let size = (width.unwrap_or(420 * n), height.unwrap_or(360));
        let title = title.as_deref();

        if return_bytes {
            let bytes = py
                .detach(|| plt_qc_png(&self.inner, plot_kind, &metrics, title, size))
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| plt_qc(&self.inner, &path, plot_kind, &metrics, title, size))
                .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, color_by="total_counts", ncols=3, dot_size=None,
                        title=None, width=None, height=None, vmin=None, vmax=None,
                        return_bytes=false))]
    fn plot_spatial_scatter(
        &self,
        py: Python<'_>,
        path: Option<String>,
        color_by: &str,
        ncols: usize,
        dot_size: Option<f32>,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        vmin: Option<f32>,
        vmax: Option<f32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let color_by = color_by.to_string();
        let title = title.as_deref();
        let n_batches = crate::plot::batch_count(&self.inner).unwrap_or(1);
        let multi = n_batches > 1;
        let ncols = if multi {
            ncols.min(n_batches).max(1)
        } else {
            1
        };
        let nrows = (n_batches + ncols - 1) / ncols;
        let size = if multi {
            (
                width.unwrap_or(400 * ncols as u32),
                height.unwrap_or(400 * nrows as u32),
            )
        } else {
            (width.unwrap_or(800), height.unwrap_or(800))
        };

        if return_bytes {
            let bytes = if multi {
                py.detach(|| {
                    plt_batch_spatial_scatter_png(
                        &self.inner,
                        &color_by,
                        ncols,
                        title,
                        size,
                        vmin,
                        vmax,
                        dot_size,
                    )
                })
            } else {
                py.detach(|| {
                    plt_spatial_scatter_png(
                        &self.inner,
                        &color_by,
                        title,
                        size,
                        vmin,
                        vmax,
                        dot_size,
                    )
                })
            }
            .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            if multi {
                py.detach(|| {
                    plt_batch_spatial_scatter(
                        &self.inner,
                        &path,
                        &color_by,
                        ncols,
                        title,
                        size,
                        vmin,
                        vmax,
                        dot_size,
                    )
                })
            } else {
                py.detach(|| {
                    plt_spatial_scatter(
                        &self.inner,
                        &path,
                        &color_by,
                        title,
                        size,
                        vmin,
                        vmax,
                        dot_size,
                    )
                })
            }
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, width=None, height=None, return_bytes=false))]
    fn plot_highly_variable_genes(
        &self,
        py: Python<'_>,
        path: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(600));
        if return_bytes {
            let bytes = py
                .detach(|| plt_hvg_png(&self.inner, size))
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| plt_hvg(&self.inner, &path, size))
                .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, cluster_key="leiden", top_n=5, groups=None,
                        title=None, width=None, height=None, return_bytes=false))]
    fn plot_marker_heatmap(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        cluster_key: &str,
        top_n: usize,
        groups: Option<Vec<i64>>,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(1200));
        let cluster_key = cluster_key.to_string();
        let title = title.as_deref();
        let groups = groups.as_deref();

        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_marker_heatmap_png(&self.inner, &cluster_key, top_n, groups, title, size)
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_marker_heatmap(&self.inner, &path, &cluster_key, top_n, groups, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, max_labels=40, normalize=true,
                        title=None, width=None, height=None, return_bytes=false))]
    fn plot_singler_score_heatmap(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        max_labels: usize,
        normalize: bool,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1400), height.unwrap_or(700));
        let title = title.as_deref();

        if return_bytes {
            let bytes = py
                .detach(|| plt_singler_heatmap_png(&self.inner, max_labels, normalize, title, size))
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_singler_heatmap(&self.inner, &path, max_labels, normalize, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, cluster_key="leiden", top_n=5, groups=None,
                        title=None, width=None, height=None, return_bytes=false))]
    fn plot_marker_dotplot(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        cluster_key: &str,
        top_n: usize,
        groups: Option<Vec<i64>>,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(1200));
        let cluster_key = cluster_key.to_string();
        let title = title.as_deref();
        let groups = groups.as_deref();

        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_marker_dotplot_png(&self.inner, &cluster_key, top_n, groups, title, size)
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_marker_dotplot(&self.inner, &path, &cluster_key, top_n, groups, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (genes, path=None, *, title=None, width=None, height=None,
                        dot_size=None, vmin=None, vmax=None, return_bytes=false))]
    fn plot_marker_spatial(
        &self,
        py: Python<'_>,
        genes: Vec<String>,
        path: Option<String>,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        dot_size: Option<f32>,
        vmin: Option<f32>,
        vmax: Option<f32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let title = title.as_deref();

        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_marker_spatial_png(
                        &self.inner,
                        &genes,
                        title,
                        width,
                        height,
                        dot_size,
                        vmin,
                        vmax,
                    )
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_marker_spatial(
                    &self.inner,
                    &genes,
                    &path,
                    title,
                    width,
                    height,
                    dot_size,
                    vmin,
                    vmax,
                )
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, cluster_key="leiden", title=None, x_label="umap1",
                        y_label="umap2", dot_size=None, width=None, height=None,
                        return_bytes=false))]
    fn plot_umap(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        cluster_key: &str,
        title: Option<String>,
        x_label: &str,
        y_label: &str,
        dot_size: Option<f32>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(800), height.unwrap_or(800));
        let cluster_key = cluster_key.to_string();
        let x_label = x_label.to_string();
        let y_label = y_label.to_string();
        let title = title.as_deref();

        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_umap_png(
                        &self.inner,
                        &cluster_key,
                        title,
                        &x_label,
                        &y_label,
                        dot_size,
                        size,
                    )
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_umap(
                    &self.inner,
                    &path,
                    &cluster_key,
                    title,
                    &x_label,
                    &y_label,
                    dot_size,
                    size,
                )
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, n_pcs=30, cum=false, log=false,
                        width=None, height=None, return_bytes=false))]
    fn plot_pca_variance_ratio(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        n_pcs: usize,
        cum: bool,
        log: bool,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(900), height.unwrap_or(600));
        if return_bytes {
            let bytes = py
                .detach(|| plt_pca_variance_ratio_png(&self.inner, n_pcs, cum, log, size))
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| plt_pca_variance_ratio(&self.inner, &path, n_pcs, cum, log, size))
                .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, color=None, components="1,2", dot_size=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_pca(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        color: Option<String>,
        components: &str,
        dot_size: Option<f32>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let comps = parse_components(components, "components")?;
        if comps.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "plot_pca needs exactly 2 PCs (e.g. components=\"1,2\"), got {components:?}"
            )));
        }
        let pair = (comps[0], comps[1]);
        let size = (width.unwrap_or(800), height.unwrap_or(720));
        if return_bytes {
            let bytes = py
                .detach(|| plt_pca_png(&self.inner, color.as_deref(), pair, dot_size, size))
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| plt_pca(&self.inner, &path, color.as_deref(), pair, dot_size, size))
                .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, components="1", n_top=30, include_lowest=false,
                        width=None, height=None, return_bytes=false))]
    fn plot_pca_loadings(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        components: &str,
        n_top: usize,
        include_lowest: bool,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let comps = parse_components(components, "components")?;
        let size = if comps.len() > 1 {
            (width.unwrap_or(1400), height.unwrap_or(800))
        } else {
            (width.unwrap_or(1000), height.unwrap_or(800))
        };
        if return_bytes {
            let bytes = py
                .detach(|| plt_pca_loadings_png(&self.inner, &comps, n_top, include_lowest, size))
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| plt_pca_loadings(&self.inner, &path, &comps, n_top, include_lowest, size))
                .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (path=None, *, res_key="leiden", title=None, x_label=None, y_label=None,
                        dot_size=None, width=None, height=None, show_plotting_scale=None,
                        plotting_scale_width=None, reorganize_coordinate=None,
                        horizontal_offset_additional=0.0, vertical_offset_additional=0.0,
                        split_by_batch=true, return_bytes=false))]
    fn plot_cluster_scatter(
        &self,
        py: Python<'_>,
        path: Option<String>,
        res_key: &str,
        title: Option<String>,
        x_label: Option<String>,
        y_label: Option<String>,
        dot_size: Option<f32>,
        width: Option<u32>,
        height: Option<u32>,
        show_plotting_scale: Option<bool>,
        plotting_scale_width: Option<f32>,
        reorganize_coordinate: Option<usize>,
        horizontal_offset_additional: f32,
        vertical_offset_additional: f32,
        split_by_batch: bool,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let res_key = res_key.to_string();
        let title = title.as_deref();
        let x_label = x_label.as_deref();
        let y_label = y_label.as_deref();

        let show_plotting_scale =
            show_plotting_scale.unwrap_or_else(|| self.inner.uns.contains_key("resolution"));

        let reorganize_coordinate = reorganize_coordinate.filter(|&n| n > 0);
        let n = crate::plot::batch_count(&self.inner).unwrap_or(1);
        let size = if split_by_batch {
            let ncols = reorganize_coordinate.unwrap_or(3).max(1).min(n);
            let nrows = (n + ncols - 1) / ncols;
            (
                width.unwrap_or(400 * ncols as u32),
                height.unwrap_or(400 * nrows as u32),
            )
        } else {
            (width.unwrap_or(800), height.unwrap_or(800))
        };

        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_cluster_scatter_png(
                        &self.inner,
                        &res_key,
                        title,
                        x_label,
                        y_label,
                        dot_size,
                        size,
                        show_plotting_scale,
                        plotting_scale_width,
                        reorganize_coordinate,
                        horizontal_offset_additional,
                        vertical_offset_additional,
                        split_by_batch,
                    )
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_cluster_scatter(
                    &self.inner,
                    &path,
                    &res_key,
                    title,
                    x_label,
                    y_label,
                    dot_size,
                    size,
                    show_plotting_scale,
                    plotting_scale_width,
                    reorganize_coordinate,
                    horizontal_offset_additional,
                    vertical_offset_additional,
                    split_by_batch,
                )
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (labels=None, path=None, *, title=None, x_label=None, y_label=None,
                        dot_size=None, width=None, height=None, show_plotting_scale=None,
                        plotting_scale_width=None, reorganize_coordinate=None,
                        horizontal_offset_additional=0.0, vertical_offset_additional=0.0,
                        split_by_batch=true, return_bytes=false))]
    fn plot_annotation_scatter(
        &self,
        py: Python<'_>,
        labels: Option<Vec<String>>,
        path: Option<String>,
        title: Option<String>,
        x_label: Option<String>,
        y_label: Option<String>,
        dot_size: Option<f32>,
        width: Option<u32>,
        height: Option<u32>,
        show_plotting_scale: Option<bool>,
        plotting_scale_width: Option<f32>,
        reorganize_coordinate: Option<usize>,
        horizontal_offset_additional: f32,
        vertical_offset_additional: f32,
        split_by_batch: bool,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let labels: Vec<String> = match labels {
            Some(v) => v,
            None => {
                let ann = self.inner.result.annotation.as_ref().ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "plot_annotation_scatter: no annotation result yet; run singler() first, or pass labels= explicitly",
                    )
                })?;
                if ann.final_labels.is_empty() {
                    return Err(PyRuntimeError::new_err(
                        "plot_annotation_scatter: result.annotation.final_labels is empty",
                    ));
                }
                let n_cells = self
                    .inner
                    .obs
                    .as_ref()
                    .map(|o| o.height())
                    .unwrap_or(ann.final_labels.len());
                if ann.final_labels.len() != n_cells {
                    return Err(PyRuntimeError::new_err(format!(
                        "plot_annotation_scatter: final_labels length ({}) does not match the cell count ({})",
                        ann.final_labels.len(),
                        n_cells
                    )));
                }
                ann.final_labels.clone()
            }
        };
        let title = title.as_deref();
        let x_label = x_label.as_deref();
        let y_label = y_label.as_deref();

        let show_plotting_scale =
            show_plotting_scale.unwrap_or_else(|| self.inner.uns.contains_key("resolution"));

        let reorganize_coordinate = reorganize_coordinate.filter(|&n| n > 0);
        let n = crate::plot::batch_count(&self.inner).unwrap_or(1);
        let size = if split_by_batch {
            let ncols = reorganize_coordinate.unwrap_or(3).max(1).min(n);
            let nrows = (n + ncols - 1) / ncols;
            (
                width.unwrap_or(400 * ncols as u32),
                height.unwrap_or(400 * nrows as u32),
            )
        } else {
            (width.unwrap_or(800), height.unwrap_or(800))
        };

        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_annotation_scatter_png(
                        &self.inner,
                        &labels,
                        title,
                        x_label,
                        y_label,
                        dot_size,
                        size,
                        show_plotting_scale,
                        plotting_scale_width,
                        reorganize_coordinate,
                        horizontal_offset_additional,
                        vertical_offset_additional,
                        split_by_batch,
                    )
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_annotation_scatter(
                    &self.inner,
                    &path,
                    &labels,
                    title,
                    x_label,
                    y_label,
                    dot_size,
                    size,
                    show_plotting_scale,
                    plotting_scale_width,
                    reorganize_coordinate,
                    horizontal_offset_additional,
                    vertical_offset_additional,
                    split_by_batch,
                )
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (db, *, category=None, gene_pvalue_cutoff=0.05,
                        gene_log2fc_cutoff=0.5, pvalue_cutoff=0.05,
                        qvalue_cutoff=0.05, min_gene_count=5, max_gene_count=500,
                        top_n_terms=20))]
    #[allow(clippy::too_many_arguments)]
    fn run_ora(
        &mut self,
        py: Python<'_>,
        db: Py<PyGeneSetDb>,
        category: Option<String>,
        gene_pvalue_cutoff: f32,
        gene_log2fc_cutoff: f32,
        pvalue_cutoff: f32,
        qvalue_cutoff: f32,
        min_gene_count: usize,
        max_gene_count: usize,
        top_n_terms: usize,
    ) -> PyResult<Py<PyDict>> {
        let params = OraParams {
            category,
            gene_pvalue_cutoff,
            gene_log2fc_cutoff,
            pvalue_cutoff,
            qvalue_cutoff,
            min_gene_count,
            max_gene_count,
            top_n_terms,
        };
        let db_inner = db.borrow(py).inner.clone();
        py.detach(|| core_run_ora(&mut self.inner, &db_inner, &params));
        let res = self.inner.result.enrichment.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "run_ora failed: marker_genes() must run first, or the gene set database is empty",
            )
        })?;
        ora_result_to_dict(py, res)
    }

    #[pyo3(signature = (db, *, category=None, rank_by="log2fc", gsea_param=1.0,
                        min_gene_count=10, max_gene_count=500, n_perm=1000, seed=0,
                        pvalue_cutoff=0.05, qvalue_cutoff=0.25, top_n_terms=20))]
    #[allow(clippy::too_many_arguments)]
    fn run_gsea(
        &mut self,
        py: Python<'_>,
        db: Py<PyGeneSetDb>,
        category: Option<String>,
        rank_by: &str,
        gsea_param: f32,
        min_gene_count: usize,
        max_gene_count: usize,
        n_perm: usize,
        seed: u64,
        pvalue_cutoff: f32,
        qvalue_cutoff: f32,
        top_n_terms: usize,
    ) -> PyResult<Py<PyDict>> {
        let params = GseaParams {
            category,
            rank_by: parse_rank_by(rank_by)?,
            gsea_param,
            min_gene_count,
            max_gene_count,
            n_perm,
            seed,
            pvalue_cutoff,
            qvalue_cutoff,
            top_n_terms,
        };
        let db_inner = db.borrow(py).inner.clone();
        py.detach(|| core_run_gsea(&mut self.inner, &db_inner, &params));
        let res = self.inner.result.enrichment.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "run_gsea failed: marker_genes() must run first, or the gene set database is empty",
            )
        })?;
        gsea_result_to_dict(py, res)
    }

    #[pyo3(signature = (db_name, path=None, *, cluster_id=None, top_n=15, title=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_ora_dotplot(
        &self,
        py: Python<'_>,
        db_name: &str,
        path: Option<String>,
        cluster_id: Option<i64>,
        top_n: usize,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(900));
        let title = title.as_deref();
        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_ora_dotplot_png(&self.inner, db_name, cluster_id, top_n, title, size)
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_ora_dotplot(&self.inner, db_name, &path, cluster_id, top_n, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (db_name, path=None, *, cluster_id=None, top_n=15, title=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_ora_barplot(
        &self,
        py: Python<'_>,
        db_name: &str,
        path: Option<String>,
        cluster_id: Option<i64>,
        top_n: usize,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(900));
        let title = title.as_deref();
        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_ora_barplot_png(&self.inner, db_name, cluster_id, top_n, title, size)
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_ora_barplot(&self.inner, db_name, &path, cluster_id, top_n, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (db_name, path=None, *, cluster_id=None, term_index=0, title=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_gsea_enrichment(
        &self,
        py: Python<'_>,
        db_name: &str,
        path: Option<String>,
        cluster_id: Option<i64>,
        term_index: usize,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1100), height.unwrap_or(680));
        let title = title.as_deref();
        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_gsea_enrichment_png(
                        &self.inner,
                        db_name,
                        cluster_id,
                        term_index,
                        title,
                        size,
                    )
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_gsea_enrichment(
                    &self.inner,
                    db_name,
                    &path,
                    cluster_id,
                    term_index,
                    title,
                    size,
                )
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (db_name, path=None, *, cluster_id=None, top_n=15, title=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_gsea_dotplot(
        &self,
        py: Python<'_>,
        db_name: &str,
        path: Option<String>,
        cluster_id: Option<i64>,
        top_n: usize,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(900));
        let title = title.as_deref();
        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_gsea_dotplot_png(&self.inner, db_name, cluster_id, top_n, title, size)
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_gsea_dotplot(&self.inner, db_name, &path, cluster_id, top_n, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }

    #[pyo3(signature = (db_name, path=None, *, cluster_id=None, top_n=15, title=None,
                        width=None, height=None, return_bytes=false))]
    fn plot_gsea_barplot(
        &self,
        py: Python<'_>,
        db_name: &str,
        path: Option<String>,
        cluster_id: Option<i64>,
        top_n: usize,
        title: Option<String>,
        width: Option<u32>,
        height: Option<u32>,
        return_bytes: bool,
    ) -> PyResult<Py<PyAny>> {
        let size = (width.unwrap_or(1200), height.unwrap_or(900));
        let title = title.as_deref();
        if return_bytes {
            let bytes = py
                .detach(|| {
                    plt_gsea_barplot_png(&self.inner, db_name, cluster_id, top_n, title, size)
                })
                .map_err(to_pyerr)?;
            Ok(PyBytes::new(py, &bytes).into_any().unbind())
        } else {
            let path = path.ok_or_else(|| {
                PyValueError::new_err("path must be provided when return_bytes=False")
            })?;
            py.detach(|| {
                plt_gsea_barplot(&self.inner, db_name, &path, cluster_id, top_n, title, size)
            })
            .map_err(to_pyerr)?;
            Ok(py.None())
        }
    }
}

#[pyclass(name = "GeneSetDb")]
pub struct PyGeneSetDb {
    inner: GeneSetDb,
}

#[pymethods]
impl PyGeneSetDb {
    fn __repr__(&self) -> String {
        format!("GeneSetDb(n_sets={})", self.inner.n_sets())
    }

    #[getter]
    fn n_sets(&self) -> usize {
        self.inner.n_sets()
    }
}

#[pyclass(name = "SingleRRef")]
pub struct PySingleRRef {
    inner: SingleRRef,
}

#[pymethods]
impl PySingleRRef {
    #[new]
    #[pyo3(signature = (ref_x, ref_genes, ref_labels))]
    fn new(
        _py: Python<'_>,
        ref_x: PyReadonlyArray2<'_, f32>,
        ref_genes: Vec<String>,
        ref_labels: Vec<String>,
    ) -> PyResult<Self> {
        let ref_x = ref_x.as_array().to_owned();
        let inner = SingleRRef::new(ref_x, ref_genes, ref_labels).map_err(to_pyerr)?;
        Ok(PySingleRRef { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "SingleRRef(n_cells={}, n_genes={})",
            self.inner.n_cells(),
            self.inner.n_genes()
        )
    }

    #[getter]
    fn ref_x(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.ref_x().clone().into_pyarray(py).unbind()
    }

    #[getter]
    fn ref_genes(&self) -> Vec<String> {
        self.inner.ref_genes().to_vec()
    }

    #[getter]
    fn ref_labels(&self) -> Vec<String> {
        self.inner.ref_labels().to_vec()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.n_cells(), self.inner.n_genes())
    }

    #[getter]
    fn n_cells(&self) -> usize {
        self.inner.n_cells()
    }

    #[getter]
    fn n_genes(&self) -> usize {
        self.inner.n_genes()
    }
}

#[pyfunction]
#[pyo3(signature = (path, *, bin_type="bins", bin_size=100, is_sparse=true, dedup_symbols=false,
                    save_counts_layer=false, save_exon_layer=false))]
fn read_gef(
    py: Python<'_>,
    path: &str,
    bin_type: &str,
    bin_size: u32,
    is_sparse: bool,
    dedup_symbols: bool,
    save_counts_layer: bool,
    save_exon_layer: bool,
) -> PyResult<PySoData> {
    let path = path.to_string();
    let bin_type = bin_type.to_string();
    let sd = py
        .detach(|| {
            io_read_gef(
                &path,
                &bin_type,
                bin_size,
                is_sparse,
                dedup_symbols,
                save_counts_layer,
                save_exon_layer,
            )
        })
        .map_err(to_pyerr)?;
    Ok(PySoData {
        inner: sd,
        var_type: "intersect".to_string(),
    })
}

#[pyfunction]
#[pyo3(signature = (path, *, bin_size=100, is_sparse=true, dedup_symbols=false,
                    save_counts_layer=false, save_exon_layer=false))]
fn read_gem(
    py: Python<'_>,
    path: &str,
    bin_size: u32,
    is_sparse: bool,
    dedup_symbols: bool,
    save_counts_layer: bool,
    save_exon_layer: bool,
) -> PyResult<PySoData> {
    let path = path.to_string();
    let sd = py
        .detach(|| {
            io_read_gem(
                &path,
                bin_size,
                is_sparse,
                dedup_symbols,
                save_counts_layer,
                save_exon_layer,
            )
        })
        .map_err(to_pyerr)?;
    Ok(PySoData {
        inner: sd,
        var_type: "intersect".to_string(),
    })
}

#[pyfunction]
#[pyo3(signature = (path, *, spatial_key="spatial", save_counts_layer=false))]
fn read_h5ad(
    py: Python<'_>,
    path: &str,
    spatial_key: &str,
    save_counts_layer: bool,
) -> PyResult<PySoData> {
    let path = path.to_string();
    let spatial_key = spatial_key.to_string();
    let sd = py
        .detach(|| io_read_h5ad(&path, &spatial_key, save_counts_layer))
        .map_err(to_pyerr)?;
    Ok(PySoData {
        inner: sd,
        var_type: "intersect".to_string(),
    })
}

#[pyfunction]
#[pyo3(signature = (path, *, label_key="label.main", gene_key=None))]
fn read_singler_ref(
    py: Python<'_>,
    path: &str,
    label_key: &str,
    gene_key: Option<&str>,
) -> PyResult<PySingleRRef> {
    let path = path.to_string();
    let label_key = label_key.to_string();
    let gene_key = gene_key.map(|s| s.to_string());
    let rf = py
        .detach(|| io_read_singler_ref(&path, &label_key, gene_key.as_deref()))
        .map_err(to_pyerr)?;
    Ok(PySingleRRef { inner: rf })
}

#[pyfunction]
#[pyo3(signature = (org_path, go_path))]
fn read_go_db(py: Python<'_>, org_path: &str, go_path: &str) -> PyResult<PyGeneSetDb> {
    let org_path = org_path.to_string();
    let go_path = go_path.to_string();
    let inner = py
        .detach(|| io_read_go_db(&org_path, &go_path))
        .map_err(to_pyerr)?;
    Ok(PyGeneSetDb { inner })
}

#[pyfunction]
#[pyo3(signature = (kegg_dir, org_path, org))]
fn read_kegg_db(
    py: Python<'_>,
    kegg_dir: &str,
    org_path: &str,
    org: &str,
) -> PyResult<PyGeneSetDb> {
    let kegg_dir = kegg_dir.to_string();
    let org_path = org_path.to_string();
    let org = org.to_string();
    let inner = py
        .detach(|| io_read_kegg_db(&kegg_dir, &org_path, &org))
        .map_err(to_pyerr)?;
    Ok(PyGeneSetDb { inner })
}

#[pyfunction]
#[pyo3(signature = (reactome_dir, org_path, species))]
fn read_reactome_db(
    py: Python<'_>,
    reactome_dir: &str,
    org_path: &str,
    species: &str,
) -> PyResult<PyGeneSetDb> {
    let reactome_dir = reactome_dir.to_string();
    let org_path = org_path.to_string();
    let species = species.to_string();
    let inner = py
        .detach(|| io_read_reactome_db(&reactome_dir, &org_path, &species))
        .map_err(to_pyerr)?;
    Ok(PyGeneSetDb { inner })
}

#[pyfunction]
#[pyo3(signature = (samples, *, var_type="intersect", save_counts_layer=false))]
fn integrate(
    py: Python<'_>,
    samples: Vec<Py<PySoData>>,
    var_type: &str,
    save_counts_layer: bool,
) -> PyResult<PySoData> {
    if var_type != "intersect" && var_type != "union" {
        return Err(PyValueError::new_err(
            "var_type must be \"intersect\" or \"union\"",
        ));
    }
    let vec: Vec<SoData> = samples.iter().map(|s| s.borrow(py).inner.clone()).collect();
    let var_type_owned = var_type.to_string();
    let merged = py
        .detach(move || {
            if var_type_owned == "intersect" {
                tools_integrate(&vec, save_counts_layer)
            } else {
                tools_integrate_union(&vec, save_counts_layer)
            }
        })
        .map_err(to_pyerr)?;
    Ok(PySoData {
        inner: merged,
        var_type: var_type.to_string(),
    })
}

#[pyfunction]
fn set_threads(n: usize) {
    crate::parallel::set_rayon_threads(n);
}

#[pymodule]
fn riasopy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySoData>()?;
    m.add_class::<PySingleRRef>()?;
    m.add_class::<PyGeneSetDb>()?;
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    m.add_function(wrap_pyfunction!(read_gef, m)?)?;
    m.add_function(wrap_pyfunction!(read_gem, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5ad, m)?)?;
    m.add_function(wrap_pyfunction!(read_singler_ref, m)?)?;
    m.add_function(wrap_pyfunction!(read_go_db, m)?)?;
    m.add_function(wrap_pyfunction!(read_kegg_db, m)?)?;
    m.add_function(wrap_pyfunction!(read_reactome_db, m)?)?;
    m.add_function(wrap_pyfunction!(integrate, m)?)?;
    Ok(())
}
