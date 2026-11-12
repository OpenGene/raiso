use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::{PcaResult, SoData};
use crate::parallel::par_bbox_opt;

use super::spatial::default_dot_size_px;
use super::umap::{cluster_colors, draw_legend, extract_clusters, legend_width, unique_sorted};

const GRAY_DOT: RGBColor = RGBColor(0xcc, 0xcc, 0xcc);
const BAR_BLUE: RGBColor = RGBColor(0x2f, 0x8f, 0xcf);
const CUM_RED: RGBColor = RGBColor(0xd6, 0x27, 0x28);
const POS_BLUE: RGBColor = RGBColor(0x2f, 0x8f, 0xcf);
const NEG_RED: RGBColor = RGBColor(0xd6, 0x27, 0x28);

fn extract_pca(data: &SoData) -> anyhow::Result<&PcaResult> {
    data.result
        .pca
        .as_ref()
        .ok_or_else(|| anyhow!("SoData.result.pca has no PcaResult; call run_pca first"))
}

fn trans_y(v: f32, log: bool) -> f32 {
    if log {
        v.max(1e-300).log10()
    } else {
        v
    }
}

fn fmt_y(v: f32, log: bool) -> String {
    if log {
        let raw = 10f32.powf(v);
        if raw >= 0.01 {
            format!("{raw:.3}")
        } else {
            format!("{raw:.1e}")
        }
    } else {
        format!("{v:.4}")
    }
}

pub fn plot_pca_variance_ratio(
    data: &SoData,
    path: &str,
    n_pcs: usize,
    cum: bool,
    log: bool,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let ratio = extract_pca(data)?
        .explained_variance_ratio
        .as_slice()
        .unwrap_or_default();
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_pca_variance_ratio(root, ratio, n_pcs, cum, log)
        .map_err(|e| anyhow!("failed to draw the PCA scree plot: {e}"))
}

pub fn plot_pca_variance_ratio_png(
    data: &SoData,
    n_pcs: usize,
    cum: bool,
    log: bool,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let ratio = extract_pca(data)?
        .explained_variance_ratio
        .as_slice()
        .unwrap_or_default();
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_pca_variance_ratio(root, ratio, n_pcs, cum, log)
            .map_err(|e| anyhow!("failed to draw the PCA scree plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn render_pca_variance_ratio<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    ratio: &[f32],
    n_pcs: usize,
    cum: bool,
    log: bool,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let k = ratio.len();
    if k == 0 {
        root.present()?;
        return Ok(());
    }
    let n = n_pcs.min(k);

    let xs: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let vals: Vec<f32> = ratio[..n].to_vec();
    let mut cum_vals: Vec<f32> = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &v in &vals {
        acc += v;
        cum_vals.push(acc);
    }

    let mut ylo = f32::INFINITY;
    let mut yhi = f32::NEG_INFINITY;
    for &v in vals.iter().chain(cum_vals.iter()) {
        let t = trans_y(v, log);
        ylo = ylo.min(t);
        yhi = yhi.max(t);
    }
    if !yhi.is_finite() {
        root.present()?;
        return Ok(());
    }
    if log {
        ylo = (ylo - 0.3).min(-0.3);
        let pad = ((yhi - ylo) * 0.18).max(0.1);
        yhi += pad;
    } else {
        ylo = 0.0;
        if yhi <= 0.0 {
            yhi = 1.0;
        } else {
            yhi *= 1.18;
        }
    }
    if ylo >= yhi {
        ylo = 0.0;
        yhi = 1.0;
    }

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.5..(n as f32 + 0.5), ylo..yhi)?;

    chart
        .configure_mesh()
        .light_line_style(RGBColor(0xdd, 0xdd, 0xdd).stroke_width(1))
        .x_label_style(("sans-serif", 13.0).into_font())
        .y_label_style(("sans-serif", 13.0).into_font())
        .x_label_formatter(&|&x: &f32| {
            let r = x.round();
            if (x - r).abs() < 1e-9 && r >= 1.0 && r <= n as f32 {
                format!("{r:.0}")
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|&v: &f32| fmt_y(v, log))
        .x_desc("Principal Component")
        .y_desc(if log {
            "log10 Variance ratio"
        } else {
            "Variance ratio"
        })
        .axis_desc_style(("sans-serif", 15.0).into_font())
        .draw()?;

    let bar_w = 0.42;
    chart.draw_series(xs.iter().zip(vals.iter()).map(|(&xi, &v)| {
        let top = trans_y(v, log);
        Rectangle::new([(xi - bar_w, ylo), (xi + bar_w, top)], BAR_BLUE.filled())
    }))?;

    if cum {
        chart.draw_series(LineSeries::new(
            xs.iter()
                .zip(cum_vals.iter())
                .map(|(&xi, &v)| (xi, trans_y(v, log))),
            CUM_RED.stroke_width(2),
        ))?;
        chart.draw_series(
            xs.iter()
                .zip(cum_vals.iter())
                .map(|(&xi, &v)| Circle::new((xi, trans_y(v, log)), 3, CUM_RED.filled())),
        )?;
    }

    let (ppw_px, pph_px) = chart.plotting_area().dim_in_pixel();
    if pph_px > 0 && ppw_px > 0 {
        let axis_span = yhi - ylo;
        let px2data = axis_span / pph_px as f32;
        let x2data = n as f32 / ppw_px as f32;

        let clear_px = if cum { 7.0 } else { 2.0 };
        let anchor_y0 = (10.0 + clear_px) * px2data;

        let style = ("sans-serif", 10.0).into_font().color(&BLACK);
        for (&xi, &v) in xs.iter().zip(vals.iter()) {
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            let pct = v * 100.0;
            let lab = format!("{pct:.1}%");
            let top = trans_y(v, log);

            let half_w_data = (lab.len() as f32 * 5.6 * 0.5) * x2data;
            let x_lab = xi - half_w_data;
            let y_lab = top + anchor_y0;
            chart.draw_series(std::iter::once(Text::new(lab, (x_lab, y_lab), &style)))?;
        }
    }

    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_pca(
    data: &SoData,
    path: &str,
    color: Option<&str>,
    components: (usize, usize),
    dot_size: Option<f32>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let pca = extract_pca(data)?;
    let (x, y) = proj_columns(pca, components)?;
    let labels = resolve_color_labels(data, color)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_pca_scatter(root, &x, &y, labels.as_deref(), components, dot_size)
        .map_err(|e| anyhow!("failed to draw the PCA scatter plot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_pca_png(
    data: &SoData,
    color: Option<&str>,
    components: (usize, usize),
    dot_size: Option<f32>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let pca = extract_pca(data)?;
    let (x, y) = proj_columns(pca, components)?;
    let labels = resolve_color_labels(data, color)?;
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_pca_scatter(root, &x, &y, labels.as_deref(), components, dot_size)
            .map_err(|e| anyhow!("failed to draw the PCA scatter plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn proj_columns(
    pca: &PcaResult,
    components: (usize, usize),
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let k = pca.projected.ncols();
    let (cx, cy) = components;
    if cx >= k || cy >= k {
        return Err(anyhow!(
            "components index out of range: PCA has only {k} components but PC{} / PC{} were requested (0-based {cx}/{cy})",
            cx + 1,
            cy + 1
        ));
    }
    Ok((
        pca.projected.column(cx).to_vec(),
        pca.projected.column(cy).to_vec(),
    ))
}

fn resolve_color_labels(data: &SoData, color: Option<&str>) -> anyhow::Result<Option<Vec<i64>>> {
    let key = color.unwrap_or("batch");
    match extract_clusters(data, key) {
        Ok(labels) => {
            if let Some(x) = data.exp_matrix.as_ref() {
                if labels.len() != x.n_rows() {
                    return Err(anyhow!(
                        "column {key} has {} rows but there are {} cells",
                        labels.len(),
                        x.n_rows()
                    ));
                }
            }
            Ok(Some(labels))
        }
        Err(_) if color.is_none() => Ok(None),
        Err(e) => Err(e),
    }
}

fn data_extent(x: &[f32], y: &[f32]) -> (f32, f32, f32, f32) {
    let (mut x0, mut x1, mut y0, mut y1) = par_bbox_opt(x, y, true);
    if !x0.is_finite() {
        return (0.0, 1.0, 0.0, 1.0);
    }
    if x0 == x1 {
        x0 -= 0.5;
        x1 += 0.5;
    }
    if y0 == y1 {
        y0 -= 0.5;
        y1 += 0.5;
    }
    (x0, x1, y0, y1)
}

#[allow(clippy::too_many_arguments)]
fn render_pca_scatter<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    x: &[f32],
    y: &[f32],
    labels: Option<&[i64]>,
    components: (usize, usize),
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    if x.is_empty() {
        root.present()?;
        return Ok(());
    }
    let (x0, x1, y0, y1) = data_extent(x, y);

    let (w, _h) = root.dim_in_pixel();
    let w = w as i32;

    let (main_area, legend_area) = match labels {
        Some(labels) if !labels.is_empty() => {
            let sorted = unique_sorted(labels);
            let sorted_str: Vec<String> = sorted.iter().map(|v| v.to_string()).collect();
            let legend_w = legend_width(w, &sorted_str);
            let (main_area, legend_area) = root.split_horizontally((w - legend_w).max(200));
            (main_area, Some((sorted, legend_area)))
        }
        _ => (root, None),
    };

    const MARGIN: i32 = 12;
    const X_LABEL: i32 = 40;
    const Y_LABEL: i32 = 50;

    let mut chart = ChartBuilder::on(&main_area)
        .margin(MARGIN)
        .x_label_area_size(X_LABEL)
        .y_label_area_size(Y_LABEL)
        .build_cartesian_2d(x0..x1, y1..y0)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_label_style(("sans-serif", 15.0).into_font().color(&BLACK))
        .y_label_style(("sans-serif", 15.0).into_font().color(&BLACK))
        .x_desc(format!("PC{}", components.0 + 1))
        .y_desc(format!("PC{}", components.1 + 1))
        .axis_desc_style(("sans-serif", 15.0).into_font())
        .draw()?;

    let (pw_px, ph_px) = chart.plotting_area().dim_in_pixel();
    let dot_size =
        dot_size.unwrap_or_else(|| default_dot_size_px(x.len(), (pw_px as f32) * (ph_px as f32)));
    let hx = dot_size * 0.5 * (x1 - x0) / (pw_px as f32).max(1.0);
    let hy = dot_size * 0.5 * (y1 - y0) / (ph_px as f32).max(1.0);

    match labels {
        Some(labels) => {
            if labels.len() != x.len() {
                return Ok(());
            }
            let sorted = unique_sorted(labels);
            let colors = cluster_colors(sorted.len());
            let color_of: std::collections::HashMap<i64, RGBColor> =
                sorted.iter().copied().zip(colors.iter().copied()).collect();

            let pts: Vec<(f32, f32, RGBColor)> = x
                .par_iter()
                .zip(y.par_iter())
                .zip(labels.par_iter())
                .map(|((&px, &py), lab)| {
                    let c = *color_of.get(lab).unwrap_or(&RGBColor(0x88, 0x88, 0x88));
                    (px, py, c)
                })
                .collect();
            chart.draw_series(pts.iter().map(|&(px, py, c)| {
                Rectangle::new([(px - hx, py - hy), (px + hx, py + hy)], c.filled())
            }))?;
            if let Some((sorted, legend_area)) = legend_area {
                let sorted_str: Vec<String> = sorted.iter().map(|v| v.to_string()).collect();
                draw_legend(&legend_area, &sorted_str, &colors)?;
            }
        }
        None => {
            chart.draw_series(x.iter().zip(y.iter()).map(|(&px, &py)| {
                Rectangle::new([(px - hx, py - hy), (px + hx, py + hy)], GRAY_DOT.filled())
            }))?;
        }
    }

    main_area.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_pca_loadings(
    data: &SoData,
    path: &str,
    components: &[usize],
    n_top: usize,
    include_lowest: bool,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let pca = extract_pca(data)?;
    let syms = loading_gene_names(data, pca);
    let comps = loadings_series(pca, components, n_top, include_lowest, &syms)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_pca_loadings(root, &comps)
        .map_err(|e| anyhow!("failed to draw the PCA loading plot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_pca_loadings_png(
    data: &SoData,
    components: &[usize],
    n_top: usize,
    include_lowest: bool,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let pca = extract_pca(data)?;
    let syms = loading_gene_names(data, pca);
    let comps = loadings_series(pca, components, n_top, include_lowest, &syms)?;
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_pca_loadings(root, &comps)
            .map_err(|e| anyhow!("failed to draw the PCA loading plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn loading_gene_names(data: &SoData, pca: &PcaResult) -> Vec<String> {
    let ncols = pca.components.ncols();
    let syms = data.symbol_names();
    if syms.len() == ncols {
        return syms;
    }
    if let Some(hv) = data.result.hvg.as_ref() {
        if hv.highly_variable.len() == syms.len() {
            let sel: Vec<usize> = (0..hv.highly_variable.len())
                .filter(|&j| hv.highly_variable[j])
                .collect();
            if sel.len() == ncols {
                return sel.into_iter().map(|j| syms[j].clone()).collect();
            }
        }
    }
    (0..ncols).map(|j| format!("gene{j}")).collect()
}

struct LoadingSeries {
    pc_no: usize,

    bars: Vec<(String, f32)>,
}

fn loadings_series(
    pca: &PcaResult,
    components: &[usize],
    n_top: usize,
    include_lowest: bool,
    syms: &[String],
) -> anyhow::Result<Vec<LoadingSeries>> {
    let k = pca.components.nrows();
    let n_genes = pca.components.ncols();
    let comps: Vec<usize> = if components.is_empty() {
        vec![0]
    } else {
        components.to_vec()
    };
    if let Some(&bad) = comps.iter().find(|&&ci| ci >= k) {
        return Err(anyhow!(
            "components index out of range: PCA has only {k} components but PC{} was requested (0-based {bad})",
            bad + 1
        ));
    }

    let out: Vec<LoadingSeries> = comps
        .par_iter()
        .map(|&ci| {
            let row = pca.components.row(ci);
            let mut idx: Vec<usize> = (0..n_genes).collect();
            idx.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let n_top = n_top.min(n_genes);
            let chosen: Vec<usize> = if include_lowest && n_top < n_genes {
                let high = n_top / 2;
                let low = n_top - high;
                idx[..high]
                    .iter()
                    .chain(idx[n_genes - low..].iter())
                    .copied()
                    .collect()
            } else {
                idx[..n_top].to_vec()
            };
            let bars = chosen
                .into_iter()
                .filter_map(|gi| {
                    let name = syms.get(gi).cloned().unwrap_or_default();
                    if name.is_empty() {
                        None
                    } else {
                        Some((name, row[gi]))
                    }
                })
                .collect();
            LoadingSeries {
                pc_no: ci + 1,
                bars,
            }
        })
        .collect();
    Ok(out)
}

fn render_pca_loadings<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    comps: &[LoadingSeries],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    if comps.is_empty() {
        root.present()?;
        return Ok(());
    }

    let label_w = comps
        .iter()
        .flat_map(|c| c.bars.iter().map(|(name, _)| name.len()))
        .max()
        .unwrap_or(1)
        .max(1) as i32;
    let label_w = (label_w * 7 + 10).clamp(50, 260) as u32;

    let areas = root.split_evenly((1, comps.len()));
    for (i, comp) in comps.iter().enumerate() {
        loading_panel(&areas[i], comp, label_w)?;
    }
    root.present()?;
    Ok(())
}

fn loading_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    comp: &LoadingSeries,
    label_w: u32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let n = comp.bars.len();
    if n == 0 {
        return Ok(());
    }

    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    if label_w as i32 >= w - 120 {
        return Ok(());
    }
    let title_h = 30i32;
    let (title_area, rest) = area.split_vertically(title_h);
    let title_style = ("sans-serif", 16.0).into_font().color(&BLACK);
    title_area.draw_text(&format!("PC{}", comp.pc_no), &title_style, (10, 8))?;
    title_area.draw(&PathElement::new(
        vec![(0, title_h - 1), (w - 1, title_h - 1)],
        RGBColor(0x99, 0x99, 0x99).stroke_width(1),
    ))?;

    let (label_area, plot_area) = rest.split_horizontally(label_w);

    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (_, v) in &comp.bars {
        let v = *v;
        if !v.is_finite() {
            continue;
        }
        lo = lo.min(v.min(0.0));
        hi = hi.max(v.max(0.0));
    }
    if !lo.is_finite() {
        return Ok(());
    }
    if lo == hi {
        lo -= 0.5;
        hi += 0.5;
    }
    let pad = (hi - lo) * 0.05;
    let x_lo = lo - pad;
    let x_hi = hi + pad;
    if x_lo >= x_hi {
        return Ok(());
    }
    let _ = h;

    let mut chart = ChartBuilder::on(&plot_area)
        .margin(0)
        .x_label_area_size(30)
        .y_label_area_size(0)
        .build_cartesian_2d(x_lo..x_hi, 0.0..n as f32)?;

    chart
        .configure_mesh()
        .disable_y_mesh()
        .disable_y_axis()
        .light_line_style(RGBColor(0xe8, 0xe8, 0xe8).stroke_width(1))
        .x_label_style(("sans-serif", 12.0).into_font())
        .draw()?;

    let (_ppw, pph) = chart.plotting_area().dim_in_pixel();
    if pph == 0 {
        return Ok(());
    }

    for (i, (_, v)) in comp.bars.iter().enumerate() {
        let v = *v;
        let y_top = n as f32 - i as f32;
        let y_bot = y_top - 1.0;
        let color = if v >= 0.0 { POS_BLUE } else { NEG_RED };
        if v != 0.0 {
            chart.draw_series(std::iter::once(Rectangle::new(
                [(0.0_f32, y_top), (v, y_bot)],
                color.filled(),
            )))?;
        }

        let cy = (((i as f32 + 0.5) * pph as f32) / n as f32) as i32;
        let style = ("sans-serif", 11.0).into_font().color(&BLACK);
        let name = &comp.bars[i].0;
        let tw = name.len() as i32 * 7;
        label_area.draw_text(name, &style, (label_w as i32 - 4 - tw, cy - 7))?;
    }

    if lo <= 0.0 && 0.0 <= hi {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.0_f32, n as f32), (0.0_f32, 0.0_f32)],
            RGBColor(0x40, 0x40, 0x40).stroke_width(1),
        )))?;
    }
    Ok(())
}
