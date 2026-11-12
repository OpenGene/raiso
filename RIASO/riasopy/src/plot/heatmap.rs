use anyhow::{anyhow, bail};
use ndarray::Array2;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use super::umap::{extract_clusters, unique_sorted};
use crate::core::{ClusterMarkerResult, ExpMatrix, MarkerResult, SoData};
use crate::parallel::SendPtr;

const BAR_W: i32 = 72;

const GREENS_ANCHORS: [(f32, (u8, u8, u8)); 5] = [
    (0.00, (0xf7, 0xfc, 0xf5)),
    (0.25, (0xc7, 0xe9, 0xc0)),
    (0.50, (0x74, 0xc4, 0x76)),
    (0.75, (0x2f, 0x8f, 0x4c)),
    (1.00, (0x00, 0x44, 0x1b)),
];

fn greens_colormap(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    for w in GREENS_ANCHORS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
            return RGBColor(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    let (_, c) = GREENS_ANCHORS[GREENS_ANCHORS.len() - 1];
    RGBColor(c.0, c.1, c.2)
}

pub(crate) fn densify(exp: &ExpMatrix) -> Array2<f32> {
    match exp {
        ExpMatrix::Dense(d) => d.mapv(|v| v),
        ExpMatrix::Sparse(csr) => {
            let (nrows, ncols) = (csr.nrows(), csr.ncols());
            let mut dense = Array2::<f32>::zeros((nrows, ncols));
            let offsets = csr.row_offsets();
            let cols = csr.col_indices();
            let vals = csr.values();
            let ptr = SendPtr(dense.as_mut_ptr());
            (0..nrows).into_par_iter().for_each(|row| {
                let row_ptr = unsafe { ptr.add(row * ncols) };
                for idx in offsets[row]..offsets[row + 1] {
                    unsafe {
                        *row_ptr.add(cols[idx]) = vals[idx];
                    }
                }
            });
            dense
        }
    }
}

fn cluster_mean_matrix(dense: &Array2<f32>, labels: &[i64], cluster_ids: &[i64]) -> Array2<f32> {
    let n_genes = dense.ncols();
    let n_clusters = cluster_ids.len();
    let mut out = Array2::<f32>::zeros((n_clusters, n_genes));
    let ptr = SendPtr(out.as_mut_ptr());
    cluster_ids.par_iter().enumerate().for_each(|(r, &cid)| {
        let mut sums = vec![0.0f32; n_genes];
        let mut cnt = 0usize;
        for (i, &lab) in labels.iter().enumerate() {
            if lab == cid {
                cnt += 1;
                for g in 0..n_genes {
                    sums[g] += dense[[i, g]];
                }
            }
        }
        let cnt = cnt.max(1) as f32;
        let row_ptr = unsafe { ptr.add(r * n_genes) };

        #[allow(clippy::needless_range_loop)]
        for g in 0..n_genes {
            unsafe {
                *row_ptr.add(g) = sums[g] / cnt;
            }
        }
    });
    out
}

fn cluster_pct_matrix(dense: &Array2<f32>, labels: &[i64], cluster_ids: &[i64]) -> Array2<f32> {
    let n_genes = dense.ncols();
    let n_clusters = cluster_ids.len();
    let mut out = Array2::<f32>::zeros((n_clusters, n_genes));
    let ptr = SendPtr(out.as_mut_ptr());
    cluster_ids.par_iter().enumerate().for_each(|(r, &cid)| {
        let mut cnt_pos = vec![0usize; n_genes];
        let mut cnt = 0usize;
        for (i, &lab) in labels.iter().enumerate() {
            if lab == cid {
                cnt += 1;
                for g in 0..n_genes {
                    if dense[[i, g]] > 0.0 {
                        cnt_pos[g] += 1;
                    }
                }
            }
        }
        let cnt = cnt.max(1) as f32;
        let row_ptr = unsafe { ptr.add(r * n_genes) };

        #[allow(clippy::needless_range_loop)]
        for g in 0..n_genes {
            unsafe {
                *row_ptr.add(g) = cnt_pos[g] as f32 / cnt;
            }
        }
    });
    out
}

fn cluster_iter_order<'a>(
    marker: &'a MarkerResult,
    groups: Option<&[i64]>,
) -> Vec<&'a ClusterMarkerResult> {
    match groups {
        Some(gs) => {
            let by_id: std::collections::HashMap<i64, &'a ClusterMarkerResult> =
                marker.clusters.iter().map(|c| (c.cluster_id, c)).collect();
            gs.iter().filter_map(|id| by_id.get(id).copied()).collect()
        }
        None => marker.clusters.iter().collect(),
    }
}

fn select_marker_rows(
    marker: &MarkerResult,
    top_n: usize,
    groups: Option<&[i64]>,
) -> (Vec<usize>, Vec<(usize, usize, i64)>) {
    let mut sel = Vec::new();
    let mut blocks = Vec::new();
    for c in cluster_iter_order(marker, groups) {
        let mut block: Vec<usize> = c
            .gene_indices
            .iter()
            .zip(&c.log2fc)
            .filter(|(_, &lf)| lf > 0.0)
            .map(|(&gi, _)| gi)
            .collect();

        if block.len() < top_n {
            let pad: Vec<usize> = c
                .gene_indices
                .iter()
                .filter(|gi| !block.contains(gi))
                .take(top_n - block.len())
                .copied()
                .collect();
            block.extend(pad);
        }
        block.truncate(top_n);
        if block.is_empty() {
            continue;
        }
        let s = sel.len();
        sel.extend_from_slice(&block);
        blocks.push((s, sel.len(), c.cluster_id));
    }
    (sel, blocks)
}

fn standard_scale_gene(mat: &mut Array2<f32>) {
    for mut row in mat.rows_mut() {
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        let mut any_finite = false;
        for &v in row.iter() {
            if v.is_finite() {
                any_finite = true;
                mn = mn.min(v);
                mx = mx.max(v);
            }
        }
        if !any_finite || mx <= mn {
            for v in row.iter_mut() {
                *v = 0.0;
            }
            continue;
        }
        let denom = mx - mn;
        for v in row.iter_mut() {
            if v.is_finite() {
                *v = (*v - mn) / denom;
            } else {
                *v = 0.0;
            }
        }
    }
}

pub(crate) fn draw_colorbar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    vmin: f32,
    vmax: f32,
    cmap: fn(f32) -> RGBColor,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    let pad = 4;
    let x0 = 4;
    let x1 = (x0 + w - 32).max(x0 + 1);
    let y0 = pad;
    let y1 = (h - pad).max(y0 + 1);
    let n = 128;
    for i in 0..n {
        let t = 1.0 - i as f32 / (n as f32 - 1.0);
        let ya = y0 + (y1 - y0) * i / n;
        let yb = y0 + (y1 - y0) * (i + 1) / n;
        area.draw(&Rectangle::new([(x0, ya), (x1, yb)], cmap(t).filled()))?;
    }
    let style = ("sans-serif", 12.0).into_font().color(&BLACK);
    let mid = (vmin + vmax) * 0.5;
    area.draw_text(&fmt_short(vmax), &style, (x1 + 3, 0))?;
    area.draw_text(&fmt_short(mid), &style, (x1 + 3, h / 2 - 7))?;
    area.draw_text(&fmt_short(vmin), &style, (x1 + 3, (h - 14).max(0)))?;
    Ok(())
}

fn fmt_short(v: f32) -> String {
    let a = v.abs();
    if a >= 1000.0 {
        format!("{v:.0}")
    } else if a >= 100.0 {
        format!("{v:.1}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    }
}

struct Layout {
    x0: i32,
    y0: i32,
    cell_w: f32,
    cell_h: f32,
    grp_w: i32,
    bracket_x: i32,
}

fn compute_layout(
    bw: i32,
    bh: i32,
    max_gene_len: i32,
    n_clusters: usize,
    n_genes: usize,
) -> Layout {
    let gene_w = (max_gene_len * 7 + 10).clamp(36, 220);
    let grp_w = 26i32;
    let br_w = 12i32;
    let top_h = 24i32;
    let bottom_pad = 8i32;
    let right_pad = 6i32;
    let x0 = grp_w + br_w + gene_w + 6;
    let y0 = top_h;
    let x1 = (bw - right_pad).max(x0 + 1);
    let y1 = (bh - bottom_pad).max(y0 + 1);
    let cell_w = (x1 - x0) as f32 / n_clusters as f32;
    let cell_h = (y1 - y0) as f32 / n_genes as f32;
    let bracket_x = grp_w + br_w / 2;
    Layout {
        x0,
        y0,
        cell_w,
        cell_h,
        grp_w,
        bracket_x,
    }
}

fn draw_top_cluster_labels<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    labels: &[String],
    lay: &Layout,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let cl_style = ("sans-serif", 11.0).into_font().color(&BLACK);
    for (c, label) in labels.iter().enumerate() {
        let cx0 = lay.x0 + (c as f32 * lay.cell_w).floor() as i32;
        let cx1 = (lay.x0 + ((c + 1) as f32 * lay.cell_w).ceil() as i32 - 1).max(cx0);
        let cx_mid = cx0 + (cx1 - cx0) / 2;
        let w_px = (label.len() as i32 * 7).max(12);
        area.draw_text(label, &cl_style, (cx_mid - w_px / 2, 4))?;
    }
    Ok(())
}

fn text_w<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    s: &str,
    style: &TextStyle<'static>,
) -> i32 {
    area.estimate_text_size(s, style)
        .map(|(w, _)| w as i32)
        .unwrap_or_else(|_| s.chars().count() as i32 * 7)
        .max(1)
}

fn draw_left_group_labels<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    gene_blocks: &[(usize, usize, i64)],
    gene_labels: &[String],
    lay: &Layout,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let grp_style = ("sans-serif", 11.0).into_font().color(&BLACK);
    for &(s, e, cid) in gene_blocks {
        let gy0 = (lay.y0 as f32 + s as f32 * lay.cell_h).ceil() as i32;
        let gy1 = (lay.y0 as f32 + e as f32 * lay.cell_h).floor() as i32 - 1;
        if gy1 < gy0 {
            continue;
        }
        let gmid = gy0 + (gy1 - gy0) / 2;

        area.draw(&PathElement::new(
            vec![
                (lay.bracket_x - 4, gy0),
                (lay.bracket_x + 4, gy0),
                (lay.bracket_x, gy0),
                (lay.bracket_x, gy1),
                (lay.bracket_x - 4, gy1),
                (lay.bracket_x + 4, gy1),
            ],
            RGBColor(0x40, 0x40, 0x40).stroke_width(1),
        ))?;
        let id = cid.to_string();
        let w_px = text_w(area, &id, &grp_style);
        area.draw_text(&id, &grp_style, (lay.grp_w - 2 - w_px, gmid - 7))?;
    }

    let gene_style = ("sans-serif", 11.0).into_font().color(&BLACK);
    let stride = ((16.0 / lay.cell_h).ceil() as usize).max(1);
    for (g, name) in gene_labels.iter().enumerate().step_by(stride) {
        let gy0 = lay.y0 + (g as f32 * lay.cell_h).floor() as i32;
        let gy1 = (lay.y0 + ((g + 1) as f32 * lay.cell_h).ceil() as i32 - 1).max(gy0);
        let cy_mid = gy0 + (gy1 - gy0) / 2;
        let w_px = text_w(area, name, &gene_style);
        area.draw_text(name, &gene_style, (lay.x0 - 4 - w_px, cy_mid - 6))?;
    }
    Ok(())
}

fn render_marker_heatmap<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    d: &ClusterHeatmapData,
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
    let n_genes = d.mat.nrows();
    let n_clusters = d.mat.ncols();
    if n_genes == 0 || n_clusters == 0 {
        root.present()?;
        return Ok(());
    }

    let (body_area, bar_area) = root.split_horizontally((w - BAR_W).max(1));
    let (bw, bh) = body_area.dim_in_pixel();
    let (bw, bh) = (bw as i32, bh as i32);
    let max_gene_len = d
        .gene_labels
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(1) as i32;
    let lay = compute_layout(bw, bh, max_gene_len, n_clusters, n_genes);

    let cells: Vec<((i32, i32), (i32, i32), RGBColor)> = (0..n_genes * n_clusters)
        .into_par_iter()
        .map(|k| {
            let (g, c) = (k / n_clusters, k % n_clusters);
            let cy0 = lay.y0 + (g as f32 * lay.cell_h).floor() as i32;
            let cy1 = (lay.y0 + ((g + 1) as f32 * lay.cell_h).ceil() as i32).max(cy0);
            let cx0 = lay.x0 + (c as f32 * lay.cell_w).floor() as i32;
            let cx1 = (lay.x0 + ((c + 1) as f32 * lay.cell_w).ceil() as i32).max(cx0);
            ((cx0, cy0), (cx1, cy1), greens_colormap(d.mat[[g, c]]))
        })
        .collect();
    for &(a, b, col) in cells.iter() {
        body_area.draw(&Rectangle::new([a, b], col.filled()))?;
    }

    draw_top_cluster_labels(&body_area, &d.cluster_labels, &lay)?;
    draw_left_group_labels(&body_area, &d.gene_blocks, &d.gene_labels, &lay)?;

    draw_colorbar(&bar_area, 0.0, 1.0, greens_colormap)?;
    root.present()?;
    Ok(())
}

fn draw_size_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    let max_r = 12i32;
    let cx = w / 2;
    let step = 22i32;
    let base_y = (h - 14).max(step * 4);
    let color = RGBColor(0x2f, 0x8f, 0x4c);
    let frac_style = ("sans-serif", 10.0).into_font().color(&BLACK);
    for (k, f) in [0.2f32, 0.4, 0.6, 0.8, 1.0].iter().enumerate() {
        let r = (max_r as f32 * f.sqrt()).round() as i32;
        let cy = base_y - k as i32 * step;
        area.draw(&Circle::new((cx, cy), r.max(1), color.filled()))?;
        area.draw_text(&format!("{f:.1}"), &frac_style, (cx + max_r + 3, cy - 5))?;
    }
    area.draw_text("pct", &frac_style, (cx - max_r - 4, base_y - 4 * step - 18))?;
    Ok(())
}

fn render_marker_dotplot<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    d: &DotplotData,
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
    let n_genes = d.frac.nrows();
    let n_clusters = d.frac.ncols();
    if n_genes == 0 || n_clusters == 0 {
        root.present()?;
        return Ok(());
    }

    let (body_area, bar_area) = root.split_horizontally((w - BAR_W).max(1));
    let (bw, bh) = body_area.dim_in_pixel();
    let (bw, bh) = (bw as i32, bh as i32);
    let max_gene_len = d
        .gene_labels
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(1)
        .max(1) as i32;
    let lay = compute_layout(bw, bh, max_gene_len, n_clusters, n_genes);

    let max_r = ((lay.cell_h * 0.5 - 2.0).min(lay.cell_w * 0.5 - 2.0)).max(1.0) as i32;

    let dots: Vec<((i32, i32), i32, RGBColor)> = (0..n_genes * n_clusters)
        .into_par_iter()
        .filter_map(|k| {
            let (g, c) = (k / n_clusters, k % n_clusters);
            let cy0 = lay.y0 + (g as f32 * lay.cell_h).floor() as i32;
            let cy1 = (lay.y0 + ((g + 1) as f32 * lay.cell_h).ceil() as i32 - 1).max(cy0);
            let cy_mid = cy0 + (cy1 - cy0) / 2;
            let cx0 = lay.x0 + (c as f32 * lay.cell_w).floor() as i32;
            let cx1 = (lay.x0 + ((c + 1) as f32 * lay.cell_w).ceil() as i32 - 1).max(cx0);
            let cx_mid = cx0 + (cx1 - cx0) / 2;
            let p = d.frac[[g, c]];
            if p <= 0.0 {
                return None;
            }
            let r = (max_r as f32 * p.sqrt()).round() as i32;
            if r <= 0 {
                return None;
            }
            Some(((cx_mid, cy_mid), r, greens_colormap(d.color[[g, c]])))
        })
        .collect();
    for &(mid, r, col) in dots.iter() {
        body_area.draw(&Circle::new(mid, r, col.filled()))?;
    }

    draw_top_cluster_labels(&body_area, &d.cluster_labels, &lay)?;
    draw_left_group_labels(&body_area, &d.gene_blocks, &d.gene_labels, &lay)?;

    let (_bar_w, bar_h) = bar_area.dim_in_pixel();
    let bar_h = bar_h as i32;
    let legend_h = 130i32;
    let (cb_area, legend_area) = bar_area.split_vertically((bar_h - legend_h).max(1));
    draw_colorbar(&cb_area, 0.0, 1.0, greens_colormap)?;
    draw_size_legend(&legend_area)?;
    root.present()?;
    Ok(())
}

struct ClusterHeatmapData {
    mat: Array2<f32>,

    gene_labels: Vec<String>,

    cluster_labels: Vec<String>,

    gene_blocks: Vec<(usize, usize, i64)>,
}

struct CommonPrepare {
    dense: Array2<f32>,

    sel: Vec<usize>,

    gene_blocks: Vec<(usize, usize, i64)>,

    cluster_ids: Vec<i64>,

    labels: Vec<i64>,

    gene_labels: Vec<String>,

    cluster_labels: Vec<String>,
}

fn extract_marker(data: &SoData) -> anyhow::Result<MarkerResult> {
    match &data.result.marker {
        Some(m) => Ok(m.clone()),
        None => Err(anyhow!(
            "SoData.result.marker has no MarkerResult; call run_marker first"
        )),
    }
}

fn prepare_common(
    sd: &SoData,
    marker: &MarkerResult,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
) -> anyhow::Result<CommonPrepare> {
    let dense = crate::algorithm::marker::marker_input(sd)
        .ok_or_else(|| {
            anyhow!("no expression matrix (exp_matrix is missing): read the data first")
        })?
        .to_dense();
    let genes = sd.symbol_names();
    let labels = extract_clusters(sd, cluster_key)?;

    let (sel, gene_blocks) = select_marker_rows(marker, top_n, groups);
    if sel.is_empty() {
        bail!("marker results are empty: no gene to plot");
    }
    let all_ids = unique_sorted(&labels);
    if let Some(gs) = groups {
        let known: std::collections::HashSet<i64> = all_ids.iter().copied().collect();
        for &g in gs {
            if !known.contains(&g) {
                bail!(
                    "groups contains a non-existent cluster id {g} (existing clusters: {:?})",
                    all_ids
                );
            }
        }
    }

    let cluster_ids: Vec<i64> = gene_blocks.iter().map(|&(_, _, cid)| cid).collect();
    let gene_labels: Vec<String> = sel
        .iter()
        .map(|&gi| genes.get(gi).cloned().unwrap_or_else(|| format!("g{}", gi)))
        .collect();
    let cluster_labels: Vec<String> = cluster_ids.iter().map(|c| c.to_string()).collect();
    Ok(CommonPrepare {
        dense,
        sel,
        gene_blocks,
        cluster_ids,
        labels,
        gene_labels,
        cluster_labels,
    })
}

fn prepare(
    sd: &SoData,
    marker: &MarkerResult,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
) -> anyhow::Result<ClusterHeatmapData> {
    let p = prepare_common(sd, marker, cluster_key, top_n, groups)?;
    let means = cluster_mean_matrix(&p.dense, &p.labels, &p.cluster_ids);
    let mut mat = Array2::<f32>::zeros((p.sel.len(), p.cluster_ids.len()));
    for (r, &gi) in p.sel.iter().enumerate() {
        for c in 0..p.cluster_ids.len() {
            mat[[r, c]] = means[[c, gi]];
        }
    }
    standard_scale_gene(&mut mat);
    Ok(ClusterHeatmapData {
        mat,
        gene_labels: p.gene_labels,
        cluster_labels: p.cluster_labels,
        gene_blocks: p.gene_blocks,
    })
}

struct DotplotData {
    frac: Array2<f32>,

    color: Array2<f32>,

    gene_labels: Vec<String>,

    cluster_labels: Vec<String>,

    gene_blocks: Vec<(usize, usize, i64)>,
}

fn prepare_dotplot(
    sd: &SoData,
    marker: &MarkerResult,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
) -> anyhow::Result<DotplotData> {
    let p = prepare_common(sd, marker, cluster_key, top_n, groups)?;
    let means = cluster_mean_matrix(&p.dense, &p.labels, &p.cluster_ids);
    let pcts = cluster_pct_matrix(&p.dense, &p.labels, &p.cluster_ids);
    let mut color = Array2::<f32>::zeros((p.sel.len(), p.cluster_ids.len()));
    let mut frac = Array2::<f32>::zeros((p.sel.len(), p.cluster_ids.len()));
    for (r, &gi) in p.sel.iter().enumerate() {
        for c in 0..p.cluster_ids.len() {
            color[[r, c]] = means[[c, gi]];
            frac[[r, c]] = pcts[[c, gi]];
        }
    }
    standard_scale_gene(&mut color);
    Ok(DotplotData {
        frac,
        color,
        gene_labels: p.gene_labels,
        cluster_labels: p.cluster_labels,
        gene_blocks: p.gene_blocks,
    })
}

const MIN_ROW_H_PX: u32 = 16;

fn auto_heatmap_size(size: (u32, u32), n_genes: usize) -> (u32, u32) {
    let (w, h) = size;
    let need_h = 24 + 8 + n_genes as u32 * MIN_ROW_H_PX;
    if h < need_h {
        (w, need_h)
    } else {
        size
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plot_marker_heatmap(
    sd: &SoData,
    path: &str,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let marker = extract_marker(sd)?;
    let data = prepare(sd, &marker, cluster_key, top_n, groups)?;
    let size = auto_heatmap_size(size, data.gene_labels.len());
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_marker_heatmap(root, &data, title)
        .map_err(|e| anyhow!("failed to draw the marker heatmap: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_marker_heatmap_png(
    sd: &SoData,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let marker = extract_marker(sd)?;
    let data = prepare(sd, &marker, cluster_key, top_n, groups)?;
    let size = auto_heatmap_size(size, data.gene_labels.len());

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_marker_heatmap(root, &data, title)
            .map_err(|e| anyhow!("failed to draw the marker heatmap: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

#[allow(clippy::too_many_arguments)]
pub fn plot_marker_dotplot(
    sd: &SoData,
    path: &str,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let marker = extract_marker(sd)?;
    let data = prepare_dotplot(sd, &marker, cluster_key, top_n, groups)?;
    let size = auto_heatmap_size(size, data.gene_labels.len());
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_marker_dotplot(root, &data, title)
        .map_err(|e| anyhow!("failed to draw the marker dotplot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_marker_dotplot_png(
    sd: &SoData,
    cluster_key: &str,
    top_n: usize,
    groups: Option<&[i64]>,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let marker = extract_marker(sd)?;
    let data = prepare_dotplot(sd, &marker, cluster_key, top_n, groups)?;
    let size = auto_heatmap_size(size, data.gene_labels.len());

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_marker_dotplot(root, &data, title)
            .map_err(|e| anyhow!("failed to draw the marker dotplot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}
