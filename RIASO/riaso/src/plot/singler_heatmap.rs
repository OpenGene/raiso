use anyhow::{anyhow, bail};
use ndarray::Array2;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;

use super::heatmap::draw_colorbar;
use super::umap::{cluster_colors, unique_sorted, unique_sorted_str};
use crate::core::SoData;

const COLORBAR_W: i32 = 72;

const LEGEND_ROW_H: u32 = 20;

const LEGEND_TITLE_H: u32 = 22;

const LEGEND_PAD: i32 = 6;
const LEGEND_SWATCH: i32 = 12;
const LEGEND_GAP: i32 = 6;

const CHAR_W: i32 = 7;

const VIRIDIS_ANCHORS: [(f32, (u8, u8, u8)); 5] = [
    (0.00, (0x44, 0x01, 0x54)),
    (0.25, (0x3b, 0x52, 0x8b)),
    (0.50, (0x21, 0x91, 0x8c)),
    (0.75, (0x5e, 0xc9, 0x62)),
    (1.00, (0xfd, 0xe7, 0x25)),
];

fn viridis_colormap(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    for w in VIRIDIS_ANCHORS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
            return RGBColor(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    let (_, c) = VIRIDIS_ANCHORS[VIRIDIS_ANCHORS.len() - 1];
    RGBColor(c.0, c.1, c.2)
}

struct ScoreHeatmapData {
    mat: Array2<f32>,

    label_names: Vec<String>,

    cluster_names: Vec<String>,

    bar_cluster: Vec<RGBColor>,
    bar_label: Vec<RGBColor>,

    label_legend: Vec<(String, RGBColor)>,

    cluster_legend: Vec<(String, RGBColor)>,

    vmin: f32,
    vmax: f32,

    cmap: fn(f32) -> RGBColor,
}

fn assigned_labels(scores: &Array2<f32>, labels: &[String]) -> Vec<String> {
    (0..scores.nrows())
        .map(|c| {
            let mut best = 0usize;
            let mut bestv = f32::NEG_INFINITY;
            for l in 0..scores.ncols() {
                let v = scores[[c, l]];
                if v > bestv {
                    bestv = v;
                    best = l;
                }
            }
            labels[best].clone()
        })
        .collect()
}

fn pick_labels(
    scores: &Array2<f32>,
    assigned: &[String],
    labels: &[String],
    max_labels: usize,
) -> Vec<usize> {
    let n_labels = labels.len();
    let mut times_best = vec![0usize; n_labels];
    for a in assigned {
        if let Some(l) = labels.iter().position(|x| x == a) {
            times_best[l] += 1;
        }
    }
    let mut secondary = vec![f32::NEG_INFINITY; n_labels];
    for c in 0..scores.nrows() {
        for l in 0..n_labels {
            secondary[l] = secondary[l].max(scores[[c, l]]);
        }
    }
    let mut idx: Vec<usize> = (0..n_labels).collect();
    idx.sort_by(|&a, &b| {
        times_best[b]
            .cmp(&times_best[a])
            .then(
                secondary[b]
                    .partial_cmp(&secondary[a])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(labels[a].cmp(&labels[b]))
    });
    idx.truncate(max_labels);
    idx
}

fn normalize_by_cluster(mat: &mut Array2<f32>) {
    for mut col in mat.columns_mut() {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in col.iter() {
            if v.is_finite() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        if !mn.is_finite() || mx <= mn {
            for v in col.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let denom = (mx - mn).max(1e-8);
        for v in col.iter_mut() {
            *v = if v.is_finite() {
                let t = (*v - mn) / denom;
                t * t * t
            } else {
                0.0
            };
        }
    }
}

fn prepare(sd: &SoData, max_labels: usize, normalize: bool) -> anyhow::Result<ScoreHeatmapData> {
    let ann = sd
        .result
        .annotation
        .as_ref()
        .ok_or_else(|| anyhow!("SoData.result.annotation is empty: call singler() first"))?;
    if ann.clusters.is_empty() {
        bail!(
            "plot_singler_score_heatmap needs cluster-mode results from singler(cluster_labels=...); the current result is per-cell annotation (result.annotation.clusters is empty)"
        );
    }
    let scores = &ann.scores;
    let n_clusters = scores.nrows();
    if n_clusters != ann.clusters.len() {
        bail!(
            "scores has {} rows but there are {} clusters",
            n_clusters,
            ann.clusters.len()
        );
    }
    if scores.ncols() != ann.labels.len() {
        bail!(
            "scores has {} columns but there are {} labels",
            scores.ncols(),
            ann.labels.len()
        );
    }
    if n_clusters == 0 || ann.labels.is_empty() {
        bail!("the score matrix is empty, nothing to plot");
    }

    let assigned = assigned_labels(scores, &ann.labels);
    let shown = pick_labels(scores, &assigned, &ann.labels, max_labels.max(1));

    let mut order: Vec<usize> = (0..n_clusters).collect();
    order.sort_by(|&a, &b| assigned[a].cmp(&assigned[b]).then(a.cmp(&b)));

    let mut mat = Array2::<f32>::zeros((shown.len(), n_clusters));
    for (r, &l) in shown.iter().enumerate() {
        for (c, &oc) in order.iter().enumerate() {
            mat[[r, c]] = scores[[oc, l]];
        }
    }
    if normalize {
        normalize_by_cluster(&mut mat);
    }

    let cluster_ids = unique_sorted(&ann.clusters);
    let cluster_pal = cluster_colors(cluster_ids.len());
    let cid_color: std::collections::HashMap<i64, RGBColor> = cluster_ids
        .iter()
        .copied()
        .zip(cluster_pal.iter().copied())
        .collect();

    let assigned_sorted = unique_sorted_str(&assigned);
    let label_pal = cluster_colors(assigned_sorted.len());
    let lab_color: std::collections::HashMap<String, RGBColor> = assigned_sorted
        .iter()
        .cloned()
        .zip(label_pal.iter().copied())
        .collect();

    let fallback = RGBColor(0x88, 0x88, 0x88);
    let mut bar_cluster = Vec::with_capacity(n_clusters);
    let mut bar_label = Vec::with_capacity(n_clusters);
    let mut cluster_names = Vec::with_capacity(n_clusters);
    for &oc in &order {
        let cid = ann.clusters[oc];
        bar_cluster.push(*cid_color.get(&cid).unwrap_or(&fallback));
        bar_label.push(*lab_color.get(&assigned[oc]).unwrap_or(&fallback));
        cluster_names.push(cid.to_string());
    }

    let cluster_legend: Vec<(String, RGBColor)> = order
        .iter()
        .map(|&oc| {
            let cid = ann.clusters[oc];
            (cid.to_string(), *cid_color.get(&cid).unwrap_or(&fallback))
        })
        .collect();

    let label_names: Vec<String> = shown.iter().map(|&l| ann.labels[l].clone()).collect();

    let mut label_legend: Vec<(String, RGBColor)> = label_names
        .iter()
        .map(|n| (n.clone(), *lab_color.get(n).unwrap_or(&fallback)))
        .collect();

    for name in &assigned_sorted {
        if !label_names.iter().any(|n| n == name) {
            label_legend.push((name.clone(), *lab_color.get(name).unwrap_or(&fallback)));
        }
    }

    let (vmin, vmax, cmap) = if normalize {
        (0.0, 1.0, viridis_colormap as fn(f32) -> RGBColor)
    } else {
        let mut a = 0.0f32;
        for &v in mat.iter() {
            if v.is_finite() {
                a = a.max(v.abs());
            }
        }
        let a = if a > 0.0 { a } else { 1.0 };
        (
            -a,
            a,
            crate::plot::spatial::stereo_colormap as fn(f32) -> RGBColor,
        )
    };

    Ok(ScoreHeatmapData {
        mat,
        label_names,
        cluster_names,
        bar_cluster,
        bar_label,
        label_legend,
        cluster_legend,
        vmin,
        vmax,
        cmap,
    })
}

const BAR_H: i32 = 16;

struct Layout {
    x0: i32,
    y0: i32,
    cell_w: f32,
    cell_h: f32,

    row_label_x: i32,
    row_label_w: i32,

    bar1_y0: i32,
    bar2_y0: i32,
}

fn compute_layout(bw: i32, bh: i32, row_label_w: i32, n_cols: usize, n_rows: usize) -> Layout {
    let top_h = BAR_H * 2 + 6;
    let bottom_pad = 34;
    let right_pad = 6;
    let x0 = row_label_w + 6;
    let y0 = top_h;
    let x1 = (bw - right_pad).max(x0 + 1);
    let y1 = (bh - bottom_pad).max(y0 + 1);
    Layout {
        x0,
        y0,
        cell_w: (x1 - x0) as f32 / n_cols.max(1) as f32,
        cell_h: (y1 - y0) as f32 / n_rows.max(1) as f32,
        row_label_x: row_label_w,
        row_label_w,
        bar1_y0: 2,
        bar2_y0: 2 + BAR_H + 2,
    }
}

fn draw_annotation_bar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    colors: &[RGBColor],
    lay: &Layout,
    y0: i32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    for (c, color) in colors.iter().enumerate() {
        let cx0 = lay.x0 + (c as f32 * lay.cell_w).floor() as i32;
        let cx1 = (lay.x0 + ((c + 1) as f32 * lay.cell_w).ceil() as i32).max(cx0);
        area.draw(&Rectangle::new(
            [(cx0, y0), (cx1, y0 + BAR_H - 1)],
            color.filled(),
        ))?;
    }
    Ok(())
}

fn render_score_heatmap<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    d: &ScoreHeatmapData,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    let (w, _h) = root.dim_in_pixel();
    let w = w as i32;
    let n_rows = d.mat.nrows();
    let n_cols = d.mat.ncols();
    if n_rows == 0 || n_cols == 0 {
        root.present()?;
        return Ok(());
    }

    let side_w = side_panel_width(d);
    let (body_area, bar_area) = root.split_horizontally((w - side_w).max(1));
    let (bw, bh) = body_area.dim_in_pixel();
    let (bw, bh) = (bw as i32, bh as i32);

    let row_style = ("sans-serif", 12.0).into_font().color(&BLACK);
    let max_label_w = d
        .label_names
        .iter()
        .filter_map(|s| body_area.estimate_text_size(s, &row_style).ok())
        .map(|(tw, _)| tw as i32)
        .max()
        .unwrap_or(1)
        .max(1);
    let row_label_w = (max_label_w + 12).clamp(36, 260);
    let lay = compute_layout(bw, bh, row_label_w, n_cols, n_rows);

    for r in 0..n_rows {
        let cy0 = lay.y0 + (r as f32 * lay.cell_h).floor() as i32;
        let cy1 = (lay.y0 + ((r + 1) as f32 * lay.cell_h).ceil() as i32).max(cy0);
        for c in 0..n_cols {
            let cx0 = lay.x0 + (c as f32 * lay.cell_w).floor() as i32;
            let cx1 = (lay.x0 + ((c + 1) as f32 * lay.cell_w).ceil() as i32).max(cx0);
            let v = d.mat[[r, c]];
            let t = if d.vmax > d.vmin {
                (v - d.vmin) / (d.vmax - d.vmin)
            } else {
                0.5
            };
            body_area.draw(&Rectangle::new(
                [(cx0, cy0), (cx1, cy1)],
                (d.cmap)(t).filled(),
            ))?;
        }
    }

    draw_annotation_bar(&body_area, &d.bar_cluster, &lay, lay.bar1_y0)?;
    draw_annotation_bar(&body_area, &d.bar_label, &lay, lay.bar2_y0)?;

    let bar_name_style = ("sans-serif", 11.0).into_font().color(&BLACK);
    for (txt, y) in [("Clusters", lay.bar1_y0), ("Cell types", lay.bar2_y0)] {
        let w_px = text_w(&body_area, txt, &bar_name_style);
        body_area.draw_text(txt, &bar_name_style, (lay.row_label_x - 4 - w_px, y + 4))?;
    }

    let max_text_w = (lay.row_label_w - 8).max(12);
    let stride = ((13.0 / lay.cell_h).ceil() as usize).max(1);
    for (r, name) in d.label_names.iter().enumerate().step_by(stride) {
        let cy0 = lay.y0 + (r as f32 * lay.cell_h).floor() as i32;
        let cy1 = (lay.y0 + ((r + 1) as f32 * lay.cell_h).ceil() as i32 - 1).max(cy0);
        let cy_mid = cy0 + (cy1 - cy0) / 2;
        let shown = fit_to_width(&body_area, name, &row_style, max_text_w);
        let w_px = text_w(&body_area, &shown, &row_style);
        body_area.draw_text(
            shown.as_str(),
            &row_style,
            (lay.row_label_x - 4 - w_px, cy_mid - 6),
        )?;
    }

    let col_style = ("sans-serif", 12.0).into_font().color(&BLACK);
    let per_col = (26.0 / lay.cell_w).ceil() as usize;
    let cstride = per_col.max(1);
    for (c, name) in d.cluster_names.iter().enumerate().step_by(cstride) {
        let cx0 = lay.x0 + (c as f32 * lay.cell_w).floor() as i32;
        let cx1 = (lay.x0 + ((c + 1) as f32 * lay.cell_w).ceil() as i32 - 1).max(cx0);
        let cx_mid = cx0 + (cx1 - cx0) / 2;
        let w_px = text_w(&body_area, name, &col_style);
        body_area.draw_text(name, &col_style, (cx_mid - w_px / 2, bh - 16))?;
    }

    let (cb_area, legends_area) = bar_area.split_horizontally(COLORBAR_W);
    draw_colorbar(&cb_area, d.vmin, d.vmax, d.cmap)?;

    let (cl_area, ct_area) = legends_area.split_horizontally(legends_area.dim_in_pixel().0 / 2);
    draw_bars_legend(&cl_area, "Clusters", &d.cluster_legend)?;
    draw_bars_legend(&ct_area, "Cell types", &d.label_legend)?;

    root.present()?;
    Ok(())
}

fn side_panel_width(d: &ScoreHeatmapData) -> i32 {
    let max_len = d
        .cluster_legend
        .iter()
        .chain(d.label_legend.iter())
        .map(|(n, _)| n.chars().count())
        .max()
        .unwrap_or(8);
    let col_w =
        (max_len as i32 * CHAR_W + LEGEND_PAD * 2 + LEGEND_SWATCH + LEGEND_GAP).clamp(96, 320);
    COLORBAR_W + col_w * 2
}

fn draw_bars_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    title: &str,
    items: &[(String, RGBColor)],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    if w <= 0 || h <= 0 {
        return Ok(());
    }
    let pad = LEGEND_PAD;
    let row_h = LEGEND_ROW_H as i32;
    let swatch = LEGEND_SWATCH;
    let title_style = ("sans-serif", 13.0).into_font().color(&BLACK);
    let item_style = ("sans-serif", 11.0).into_font().color(&BLACK);

    area.draw_text(title, &title_style, (pad, pad))?;
    let mut y = pad + LEGEND_TITLE_H as i32;
    for (name, color) in items.iter() {
        if y + row_h > h {
            break;
        }
        area.draw(&Rectangle::new(
            [(pad, y + 3), (pad + swatch, y + 3 + swatch)],
            color.filled(),
        ))?;

        let text_w = w - (pad * 2 + swatch + LEGEND_GAP);
        let shown = truncate_to(name, (text_w / CHAR_W).max(3) as usize);
        area.draw_text(shown.as_str(), &item_style, (pad + swatch + 6, y))?;
        y += row_h;
    }
    Ok(())
}

fn truncate_to(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    let mut out: String = s.chars().take(max_chars - 1).collect();
    out.push('…');
    out
}

fn text_w<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    s: &str,
    style: &TextStyle<'static>,
) -> i32 {
    area.estimate_text_size(s, style)
        .map(|(w, _)| w as i32)
        .unwrap_or_else(|_| s.chars().count() as i32 * CHAR_W)
        .max(1)
}

fn fit_to_width<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    s: &str,
    style: &TextStyle<'static>,
    max_w: i32,
) -> String {
    if text_w(area, s, style) <= max_w {
        return s.to_string();
    }
    let n = s.chars().count();
    for k in (1..n).rev() {
        let cand = truncate_to(s, k);
        if text_w(area, &cand, style) <= max_w {
            return cand;
        }
    }
    String::new()
}

const MIN_ROW_H_PX: u32 = 14;

fn auto_size(size: (u32, u32), d: &ScoreHeatmapData) -> (u32, u32) {
    let (w, h) = size;
    let n_rows = d.label_names.len();
    let heatmap_need = 40 + n_rows as u32 * MIN_ROW_H_PX + 40;
    let legend_need = |n: usize| LEGEND_TITLE_H + LEGEND_ROW_H * n as u32 + LEGEND_ROW_H;
    let need = heatmap_need
        .max(legend_need(d.cluster_legend.len()).max(legend_need(d.label_legend.len())));
    if h < need {
        (w, need)
    } else {
        size
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plot_singler_score_heatmap(
    sd: &SoData,
    path: &str,
    max_labels: usize,
    normalize: bool,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let data = prepare(sd, max_labels, normalize)?;
    let size = auto_size(size, &data);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_score_heatmap(root, &data, title)
        .map_err(|e| anyhow!("failed to draw the SingleR score heatmap: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_singler_score_heatmap_png(
    sd: &SoData,
    max_labels: usize,
    normalize: bool,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let data = prepare(sd, max_labels, normalize)?;
    let size = auto_size(size, &data);

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_score_heatmap(root, &data, title)
            .map_err(|e| anyhow!("failed to draw the SingleR score heatmap: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}
