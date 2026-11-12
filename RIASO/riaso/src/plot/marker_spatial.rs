use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::SoData;
use crate::parallel::{par_bbox, par_min_max};

use super::heatmap::{densify, draw_colorbar};
use super::spatial::extract_position;

const COLORBAR_W: i32 = 72;

const PANEL_MARGIN: i32 = 12;
const PANEL_X_LABEL: i32 = 28;
const PANEL_Y_LABEL: i32 = 40;
const PANEL_CAPTION: i32 = 24;

const TARGET_PLOT_AREA_PX: f32 = 470.0 * 470.0;

fn grid_cols(n_panels: usize, n_batches: usize) -> usize {
    let pref = if n_batches > 1 { 3 } else { 2 };
    pref.min(n_panels.max(1)).max(1)
}

fn auto_canvas(xs: &[f32], ys: &[f32], n_panels: usize, n_batches: usize) -> (u32, u32) {
    let (x0, x1, y0, y1) = par_bbox(xs, ys);

    let ar = if (x1 - x0).is_finite() && (y1 - y0).is_finite() && y1 > y0 {
        ((x1 - x0) / (y1 - y0)).clamp(0.25, 4.0)
    } else {
        1.0
    };
    let pl_h = (TARGET_PLOT_AREA_PX / ar).sqrt();
    let pl_w = ar * pl_h;
    let cell_w = (pl_w.round() as i32 + PANEL_MARGIN * 2 + PANEL_Y_LABEL + COLORBAR_W).max(180);
    let cell_h = (pl_h.round() as i32 + PANEL_MARGIN * 2 + PANEL_X_LABEL + PANEL_CAPTION).max(140);
    let ncols = grid_cols(n_panels.max(1), n_batches) as u32;
    let nrows = (n_panels.max(1) as u32).div_ceil(ncols);
    (cell_w as u32 * ncols, cell_h as u32 * nrows)
}

const FIRE_ANCHORS: [(f32, (u8, u8, u8)); 5] = [
    (0.00, (0xff, 0xff, 0xff)),
    (0.25, (0xff, 0xd1, 0x2e)),
    (0.50, (0xf8, 0x7b, 0x00)),
    (0.75, (0x9e, 0x00, 0x1e)),
    (1.00, (0x00, 0x00, 0x00)),
];

fn fire_colormap(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    for w in FIRE_ANCHORS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
            return RGBColor(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    let (_, c) = FIRE_ANCHORS[FIRE_ANCHORS.len() - 1];
    RGBColor(c.0, c.1, c.2)
}

fn prepare_marker_spatial(
    sd: &SoData,
    genes: &[String],
) -> anyhow::Result<(Vec<(String, Vec<f32>)>, Vec<f32>, Vec<f32>)> {
    let exp = sd
        .exp_matrix
        .as_ref()
        .ok_or_else(|| anyhow!("exp_matrix is empty: normalization is required first"))?;
    let dense = densify(exp);
    let (xs, ys) = extract_position(sd)?;

    let syms = sd.symbol_names();
    let mut seen = std::collections::HashSet::<&str>::new();
    let mut out = Vec::new();
    for g in genes {
        if g.is_empty() || !seen.insert(g.as_str()) {
            continue;
        }
        let idx = syms.iter().position(|s| s == g).ok_or_else(|| {
            let shown: Vec<&str> = syms.iter().take(8).map(String::as_str).collect();
            anyhow!(
                "gene {g} is not in var(gene_name) (first 8: {shown:?}, {} genes in total)",
                syms.len()
            )
        })?;
        out.push((g.clone(), dense.column(idx).to_vec()));
    }
    if out.is_empty() {
        return Err(anyhow!("genes list is empty: no gene could be plotted"));
    }
    Ok((out, xs, ys))
}

#[allow(clippy::too_many_arguments)]
fn render_marker_spatial_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    name: &str,
    xs: &[f32],
    ys: &[f32],
    vals: &[f32],
    vmin: Option<f32>,
    vmax: Option<f32>,
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if vals.is_empty() {
        return Ok(());
    }

    let (lo, hi) = par_min_max(vals);
    let (mut vmin, mut vmax) = (vmin.unwrap_or(lo), vmax.unwrap_or(hi));
    if (vmax - vmin).abs() < 1e-12 {
        vmin -= 0.5;
        vmax += 0.5;
    }

    let (mut x0, mut x1, mut y0, mut y1) = par_bbox(xs, ys);
    if x0 == x1 {
        x0 -= 0.5;
        x1 += 0.5;
    }
    if y0 == y1 {
        y0 -= 0.5;
        y1 += 0.5;
    }

    let (w, _) = area.dim_in_pixel();
    let w = w as i32;
    let (plot_area, bar_area) = area.split_horizontally((w - COLORBAR_W).max(1));

    const MARGIN: i32 = PANEL_MARGIN;
    const X_LABEL: i32 = PANEL_X_LABEL;
    const Y_LABEL: i32 = PANEL_Y_LABEL;
    const CAPTION: i32 = PANEL_CAPTION;
    let (aw, ah) = plot_area.dim_in_pixel();
    let (aw, ah) = (aw as i32, ah as i32);
    let ar = (x1 - x0) / (y1 - y0);
    let plw_w = (aw - MARGIN * 2 - Y_LABEL).max(1) as f32;
    let mut pw = aw as f32;
    let mut ph = plw_w / ar + (MARGIN * 2 + X_LABEL + CAPTION) as f32;
    if ph > ah as f32 {
        ph = ah as f32;
        let plh = (ph - (MARGIN * 2 + X_LABEL + CAPTION) as f32).max(1.0);
        pw = plh * ar + (MARGIN * 2 + Y_LABEL) as f32;
    }
    let pw = (pw.max(1.0)).min(aw as f32) as i32;
    let ph = (ph.max(1.0)).min(ah as f32) as i32;
    let left_pad = ((aw - pw) / 2).max(0);
    let top_pad = ((ah - ph) / 2).max(0);
    let plot_area = plot_area.margin(top_pad, top_pad, left_pad, left_pad);

    let mut chart = ChartBuilder::on(&plot_area)
        .caption(name.replace('_', " "), ("sans-serif", 14.0).into_font())
        .margin(MARGIN)
        .x_label_area_size(X_LABEL)
        .y_label_area_size(Y_LABEL)
        .build_cartesian_2d(x0..x1, y1..y0)?;

    chart
        .configure_mesh()
        .light_line_style(WHITE)
        .disable_x_mesh()
        .disable_y_mesh()
        .disable_x_axis()
        .disable_y_axis()
        .draw()?;

    let (pw, ph) = chart.plotting_area().dim_in_pixel();
    let dot_size = dot_size.unwrap_or_else(|| {
        let area = (pw as f32) * (ph as f32);
        super::spatial::default_dot_size_px(xs.len(), area)
    });
    let hx = dot_size * 0.5 * (x1 - x0) / (pw as f32).max(1.0);
    let hy = dot_size * 0.5 * (y1 - y0) / (ph as f32).max(1.0);
    let inv = 1.0 / (vmax - vmin);

    let pts: Vec<(f32, f32, RGBColor)> = vals
        .par_iter()
        .zip(xs.par_iter().zip(ys.par_iter()))
        .map(|(&v, (&x, &y))| (x, y, fire_colormap((v - vmin) * inv)))
        .collect();
    chart.draw_series(
        pts.iter()
            .map(|&(x, y, c)| Rectangle::new([(x - hx, y - hy), (x + hx, y + hy)], c.filled())),
    )?;

    draw_colorbar(&bar_area, vmin, vmax, fire_colormap)?;
    Ok(())
}

struct GenePanels {
    title: String,
    xs: Vec<f32>,
    ys: Vec<f32>,
    vals: Vec<f32>,
}

struct GeneSegs {
    ends: Vec<usize>,
    n_batches: usize,
}

#[allow(clippy::too_many_arguments)]
fn render_marker_spatial<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    data: &[(String, Vec<f32>)],
    xs: &[f32],
    ys: &[f32],
    batch: Option<&[i64]>,
    title: Option<&str>,
    dot_size: Option<f32>,
    vmin: Option<f32>,
    vmax: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    let n = data.len();
    if n == 0 {
        root.present()?;
        return Ok(());
    }

    let (panels, segs) = build_panels(data, xs, ys, batch);
    let n_panels = panels.len();
    if n_panels == 0 {
        root.present()?;
        return Ok(());
    }

    let ncols = grid_cols(n_panels, segs.n_batches);
    let nrows = n_panels.div_ceil(ncols);
    let areas = root.split_evenly((nrows, ncols));

    let mut prev = 0usize;
    let mut area_i = 0usize;
    for (gi, &end) in segs.ends.iter().enumerate() {
        let (glo, ghi) = {
            let vals = &data[gi].1;
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &v in vals {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if lo <= hi {
                (lo, hi)
            } else {
                (0.0, 1.0)
            }
        };
        for p in &panels[prev..end] {
            render_marker_spatial_panel(
                &areas[area_i],
                &p.title,
                &p.xs,
                &p.ys,
                &p.vals,
                Some(vmin.unwrap_or(glo)),
                Some(vmax.unwrap_or(ghi)),
                dot_size,
            )?;
            area_i += 1;
        }
        prev = end;
    }
    areas[0].present()?;
    Ok(())
}

fn build_panels(
    data: &[(String, Vec<f32>)],
    xs: &[f32],
    ys: &[f32],
    batch: Option<&[i64]>,
) -> (Vec<GenePanels>, GeneSegs) {
    let (panels, segs): (Vec<GenePanels>, Option<GeneSegs>) = match batch {
        Some(b) if !b.is_empty() => {
            let mut bid: Vec<i64> = b.to_vec();
            bid.sort_unstable();
            bid.dedup();
            let n_batches = bid.len();

            let per_gene: Vec<Vec<GenePanels>> = data
                .par_iter()
                .map(|(name, vals)| {
                    let mut g = Vec::with_capacity(n_batches);
                    for &bb in &bid {
                        let mut pxs = Vec::new();
                        let mut pys = Vec::new();
                        let mut pvals = Vec::new();
                        for (((&x, &y), &v), &bl) in
                            xs.iter().zip(ys.iter()).zip(vals.iter()).zip(b.iter())
                        {
                            if bl == bb {
                                pxs.push(x);
                                pys.push(y);
                                pvals.push(v);
                            }
                        }
                        if pxs.is_empty() {
                            continue;
                        }
                        g.push(GenePanels {
                            title: format!("{name} · sample {bb}"),
                            xs: pxs,
                            ys: pys,
                            vals: pvals,
                        });
                    }
                    g
                })
                .collect();
            let mut out = Vec::with_capacity(data.len() * n_batches);
            let mut ends = Vec::with_capacity(data.len());
            for g in per_gene {
                out.extend(g);
                ends.push(out.len());
            }
            (out, Some(GeneSegs { ends, n_batches }))
        }
        _ => {
            let ends = (1..=data.len()).collect();
            (
                data.par_iter()
                    .map(|(name, vals)| GenePanels {
                        title: name.clone(),
                        xs: xs.to_vec(),
                        ys: ys.to_vec(),
                        vals: vals.clone(),
                    })
                    .collect(),
                Some(GeneSegs { ends, n_batches: 1 }),
            )
        }
    };

    (panels, segs.expect("segs is always non-empty"))
}

fn cell_batches(data: &SoData) -> Option<Vec<i64>> {
    let obs = data.obs.as_ref()?;
    let col = obs.column("batch").ok()?;
    let arr = col.i64().ok()?;
    Some(arr.into_no_null_iter().collect())
}

fn resolve_size(
    series: &[(String, Vec<f32>)],
    xs: &[f32],
    ys: &[f32],
    batch: Option<&[i64]>,
    width: Option<u32>,
    height: Option<u32>,
) -> (u32, u32) {
    if width.is_none() && height.is_none() {
        let (panels, segs) = build_panels(series, xs, ys, batch);
        return auto_canvas(xs, ys, panels.len(), segs.n_batches);
    }
    (width.unwrap_or(1200), height.unwrap_or(900))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_marker_spatial(
    sd: &SoData,
    genes: &[String],
    path: &str,
    title: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    dot_size: Option<f32>,
    vmin: Option<f32>,
    vmax: Option<f32>,
) -> anyhow::Result<()> {
    let (series, xs, ys) = prepare_marker_spatial(sd, genes)?;
    let batch = cell_batches(sd);
    let size = resolve_size(&series, &xs, &ys, batch.as_deref(), width, height);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_marker_spatial(
        root,
        &series,
        &xs,
        &ys,
        batch.as_deref(),
        title,
        dot_size,
        vmin,
        vmax,
    )
    .map_err(|e| anyhow!("failed to draw the marker gene spatial slices: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_marker_spatial_png(
    sd: &SoData,
    genes: &[String],
    title: Option<&str>,
    width: Option<u32>,
    height: Option<u32>,
    dot_size: Option<f32>,
    vmin: Option<f32>,
    vmax: Option<f32>,
) -> anyhow::Result<Vec<u8>> {
    let (series, xs, ys) = prepare_marker_spatial(sd, genes)?;
    let batch = cell_batches(sd);
    let size = resolve_size(&series, &xs, &ys, batch.as_deref(), width, height);

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_marker_spatial(
            root,
            &series,
            &xs,
            &ys,
            batch.as_deref(),
            title,
            dot_size,
            vmin,
            vmax,
        )
        .map_err(|e| anyhow!("failed to draw the marker gene spatial slices: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}
