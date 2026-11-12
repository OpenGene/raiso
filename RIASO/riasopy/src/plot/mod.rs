pub mod cluster_scatter;
pub mod gsea;
pub mod heatmap;
pub mod hvg;
pub mod marker_spatial;
pub mod ora;
pub mod pca;
pub mod singler_heatmap;
pub mod umap;

mod qc;
mod spatial;

use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::SoData;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlotKind {
    Violin,

    Histogram,
}

pub fn plot_qc(
    data: &SoData,
    path: &str,
    kind: PlotKind,
    metrics: &[String],
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let series = extract_series(data, metrics)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_qc(root, &series, kind, title).map_err(|e| anyhow!("failed to draw the QC plot: {e}"))
}

pub fn plot_qc_png(
    data: &SoData,
    kind: PlotKind,
    metrics: &[String],
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let series = extract_series(data, metrics)?;

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_qc(root, &series, kind, title)
            .map_err(|e| anyhow!("failed to draw the QC plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn extract_series(data: &SoData, metrics: &[String]) -> anyhow::Result<Vec<(String, Vec<f32>)>> {
    let obs = data
        .obs
        .as_ref()
        .ok_or_else(|| anyhow!("obs is empty: call cal_qc() first"))?;
    metrics
        .iter()
        .map(|m| qc::metric_vec(obs, m).map(|v| (m.clone(), v)))
        .collect()
}

fn render_qc<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    series: &[(String, Vec<f32>)],
    kind: PlotKind,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    let n = series.len();
    if n == 0 {
        root.present()?;
        return Ok(());
    }
    let areas = root.split_evenly((1, n));
    for (i, (name, vals)) in series.iter().enumerate() {
        match kind {
            PlotKind::Violin => qc::plot_violin_panel(&areas[i], name, vals)?,
            PlotKind::Histogram => qc::plot_hist_panel(&areas[i], name, vals)?,
        }
    }

    areas[0].present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_spatial_scatter(
    data: &SoData,
    path: &str,
    color_by: &str,
    title: Option<&str>,
    size: (u32, u32),
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> anyhow::Result<()> {
    let vals = extract_single(data, color_by)?;
    let (xs, ys) = spatial::extract_position(data)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_spatial(root, color_by, &xs, &ys, &vals, title, vmin, vmax, dot_size)
        .map_err(|e| anyhow!("failed to draw the spatial scatter plot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_spatial_scatter_png(
    data: &SoData,
    color_by: &str,
    title: Option<&str>,
    size: (u32, u32),
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> anyhow::Result<Vec<u8>> {
    let vals = extract_single(data, color_by)?;
    let (xs, ys) = spatial::extract_position(data)?;

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_spatial(root, color_by, &xs, &ys, &vals, title, vmin, vmax, dot_size)
            .map_err(|e| anyhow!("failed to draw the spatial scatter plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn extract_single(data: &SoData, color_by: &str) -> anyhow::Result<Vec<f32>> {
    let obs = data
        .obs
        .as_ref()
        .ok_or_else(|| anyhow!("obs is empty: call cal_qc() first"))?;
    qc::metric_vec(obs, color_by)
}

#[allow(clippy::too_many_arguments)]
fn render_spatial<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    name: &str,
    xs: &[f32],
    ys: &[f32],
    vals: &[f32],
    title: Option<&str>,
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    spatial::spatial_scatter_panel(&root, name, xs, ys, vals, vmin, vmax, dot_size)?;
    root.present()?;
    Ok(())
}

pub fn batch_count(data: &SoData) -> anyhow::Result<usize> {
    let obs = data
        .obs
        .as_ref()
        .ok_or_else(|| anyhow!("obs is empty: call cal_qc() first"))?;
    let batch = obs
        .column("batch")
        .map_err(|_| anyhow!("obs is missing the batch column: the color_by mode needs data after a multi-sample integrate()"))?
        .i64()
        .map_err(|_| anyhow!("obs[\"batch\"] is not Int64"))?;
    let mut uniq: Vec<i64> = batch.into_no_null_iter().collect();
    uniq.sort_unstable();
    uniq.dedup();
    Ok(uniq.len())
}

fn extract_batch_series(
    data: &SoData,
    color_by: &str,
) -> anyhow::Result<Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>)>> {
    let obs = data
        .obs
        .as_ref()
        .ok_or_else(|| anyhow!("obs is empty: call cal_qc() first"))?;
    let batch: Vec<i64> = obs
        .column("batch")
        .map_err(|_| anyhow!("obs is missing the batch column: the color_by mode needs data after a multi-sample integrate()"))?
        .i64()
        .map_err(|_| anyhow!("obs[\"batch\"] is not Int64"))?
        .into_no_null_iter()
        .collect();
    let vals_all = qc::metric_vec(obs, color_by)?;
    let (xs, ys) = spatial::extract_position(data)?;
    if batch.len() != xs.len() || batch.len() != vals_all.len() {
        return Err(anyhow!(
            "batch count ({}) / spatial point count ({}) / metric value count ({}) are inconsistent",
            batch.len(),
            xs.len(),
            vals_all.len()
        ));
    }

    let mut uniq = batch.clone();
    uniq.sort_unstable();
    uniq.dedup();

    let out: Vec<(String, Vec<f32>, Vec<f32>, Vec<f32>)> = uniq
        .par_iter()
        .map(|&b| {
            let mut bxs = Vec::new();
            let mut bys = Vec::new();
            let mut bvals = Vec::new();
            for (((&bb, &x), &y), &v) in batch
                .iter()
                .zip(xs.iter())
                .zip(ys.iter())
                .zip(vals_all.iter())
            {
                if bb == b {
                    bxs.push(x);
                    bys.push(y);
                    bvals.push(v);
                }
            }
            (format!("{b}"), bxs, bys, bvals)
        })
        .collect();
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn plot_batch_spatial_scatter(
    data: &SoData,
    path: &str,
    color_by: &str,
    ncols: usize,
    title: Option<&str>,
    size: (u32, u32),
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> anyhow::Result<()> {
    let series = extract_batch_series(data, color_by)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_ms_spatial(root, &series, ncols, title, vmin, vmax, dot_size)
        .map_err(|e| anyhow!("failed to draw the spatial scatter plot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_batch_spatial_scatter_png(
    data: &SoData,
    color_by: &str,
    ncols: usize,
    title: Option<&str>,
    size: (u32, u32),
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> anyhow::Result<Vec<u8>> {
    let series = extract_batch_series(data, color_by)?;

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_ms_spatial(root, &series, ncols, title, vmin, vmax, dot_size)
            .map_err(|e| anyhow!("failed to draw the spatial scatter plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

#[allow(clippy::too_many_arguments)]
fn render_ms_spatial<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    series: &[(String, Vec<f32>, Vec<f32>, Vec<f32>)],
    ncols: usize,
    title: Option<&str>,
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    let n = series.len();
    if n == 0 {
        root.present()?;
        return Ok(());
    }
    let ncols = ncols.min(n).max(1);
    let nrows = n.div_ceil(ncols);

    let (mut glo, mut ghi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (_, _, _, vals) in series {
        for &v in vals {
            glo = glo.min(v);
            ghi = ghi.max(v);
        }
    }
    let (vmin, vmax) = if glo <= ghi {
        (vmin.unwrap_or(glo), vmax.unwrap_or(ghi))
    } else {
        (vmin.unwrap_or(0.0), vmax.unwrap_or(1.0))
    };

    let areas = root.split_evenly((nrows, ncols));
    for (i, (name, xs, ys, vals)) in series.iter().enumerate() {
        spatial::spatial_scatter_panel(
            &areas[i],
            &format!("sample {name}"),
            xs,
            ys,
            vals,
            Some(vmin),
            Some(vmax),
            dot_size,
        )?;
    }
    areas[0].present()?;
    Ok(())
}
