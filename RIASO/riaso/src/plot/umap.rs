use std::collections::HashMap;

use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::SoData;
use crate::parallel::par_bbox;

pub(crate) const STEREO_30: [RGBColor; 30] = [
    RGBColor(0xE4, 0x1A, 0x1C),
    RGBColor(0x37, 0x7E, 0xB8),
    RGBColor(0x4D, 0xAF, 0x4A),
    RGBColor(0x98, 0x4E, 0xA3),
    RGBColor(0xFF, 0x7F, 0x00),
    RGBColor(0xA6, 0x56, 0x28),
    RGBColor(0xFF, 0xFF, 0x33),
    RGBColor(0xF7, 0x81, 0xBF),
    RGBColor(0x99, 0x99, 0x99),
    RGBColor(0xE5, 0xD8, 0xBD),
    RGBColor(0xB3, 0xCD, 0xE3),
    RGBColor(0xCC, 0xEB, 0xC5),
    RGBColor(0xFE, 0xD9, 0xA6),
    RGBColor(0xFB, 0xB4, 0xAE),
    RGBColor(0x8D, 0xD3, 0xC7),
    RGBColor(0xBE, 0xBA, 0xDA),
    RGBColor(0x80, 0xB1, 0xD3),
    RGBColor(0xB3, 0xDE, 0x69),
    RGBColor(0xFC, 0xCD, 0xE5),
    RGBColor(0xBC, 0x80, 0xBD),
    RGBColor(0xFF, 0xED, 0x6F),
    RGBColor(0x8D, 0xA0, 0xCB),
    RGBColor(0xE7, 0x9A, 0xD3),
    RGBColor(0xE5, 0xC4, 0x94),
    RGBColor(0xCC, 0xCC, 0xCC),
    RGBColor(0xFB, 0x9A, 0x99),
    RGBColor(0x10, 0xE0, 0x3C),
    RGBColor(0xDA, 0xB2, 0xD6),
    RGBColor(0x6A, 0x3D, 0x9A),
    RGBColor(0xD1, 0x59, 0x28),
];

fn stereo_30_interp(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let last = STEREO_30.len() - 1;
    let pos = t * last as f32;
    let i = (pos.floor() as usize).min(last);
    let j = (pos.ceil() as usize).min(last);
    if i == j {
        return STEREO_30[i];
    }
    let f = pos - i as f32;
    let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
    RGBColor(
        lerp(STEREO_30[i].0, STEREO_30[j].0),
        lerp(STEREO_30[i].1, STEREO_30[j].1),
        lerp(STEREO_30[i].2, STEREO_30[j].2),
    )
}

pub(crate) fn cluster_colors(n: usize) -> Vec<RGBColor> {
    if n <= STEREO_30.len() {
        STEREO_30[..n].to_vec()
    } else {
        (0..n)
            .map(|i| stereo_30_interp(i as f32 / (n as f32 - 1.0)))
            .collect()
    }
}

pub(crate) fn unique_sorted(labels: &[i64]) -> Vec<i64> {
    let mut v: Vec<i64> = labels.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

pub(crate) fn unique_sorted_str(labels: &[String]) -> Vec<String> {
    let mut v: Vec<String> = labels.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

pub(crate) fn extract_clusters(sd: &SoData, cluster_key: &str) -> anyhow::Result<Vec<i64>> {
    let obs = sd
        .obs
        .as_ref()
        .ok_or_else(|| anyhow!("obs is empty: run run_leiden() first"))?;
    let col = obs
        .column(cluster_key)
        .map_err(|_| anyhow!("obs is missing column {cluster_key:?}"))?;
    let ca = col
        .i64()
        .map_err(|_| anyhow!("obs column {cluster_key:?} is not Int64"))?;
    Ok(ca.into_no_null_iter().collect())
}

fn extract_umap(sd: &SoData, cluster_key: &str) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<i64>)> {
    let umap = match &sd.result.umap {
        Some(u) => &u.embedding,
        None => {
            return Err(anyhow!(
                "SoData.result.umap has no UmapResult; call run_umap first"
            ))
        }
    };
    if umap.ncols() < 2 {
        return Err(anyhow!(
            "UmapResult.embedding has {} columns < 2 (at least 2 are needed for umap1/umap2)",
            umap.ncols()
        ));
    }
    let x = umap.column(0).to_vec();
    let y = umap.column(1).to_vec();
    let labels = extract_clusters(sd, cluster_key)?;
    if labels.len() != x.len() {
        return Err(anyhow!(
            "{} cluster labels vs {} UMAP points",
            labels.len(),
            x.len()
        ));
    }
    Ok((x, y, labels))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_umap(
    data: &SoData,
    path: &str,
    cluster_key: &str,
    title: Option<&str>,
    x_label: &str,
    y_label: &str,
    dot_size: Option<f32>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let (x, y, labels) = extract_umap(data, cluster_key)?;
    let title = title.unwrap_or(cluster_key);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_umap(
        root,
        &x,
        &y,
        &labels,
        Some(title),
        x_label,
        y_label,
        dot_size,
    )
    .map_err(|e| anyhow!("failed to draw the UMAP scatter plot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_umap_png(
    data: &SoData,
    cluster_key: &str,
    title: Option<&str>,
    x_label: &str,
    y_label: &str,
    dot_size: Option<f32>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let (x, y, labels) = extract_umap(data, cluster_key)?;
    let title = title.unwrap_or(cluster_key);

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_umap(
            root,
            &x,
            &y,
            &labels,
            Some(title),
            x_label,
            y_label,
            dot_size,
        )
        .map_err(|e| anyhow!("failed to draw the UMAP scatter plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

#[allow(clippy::too_many_arguments)]
fn render_umap<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    x: &[f32],
    y: &[f32],
    labels: &[i64],
    title: Option<&str>,
    x_label: &str,
    y_label: &str,
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0, FontStyle::Bold).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    if x.is_empty() {
        root.present()?;
        return Ok(());
    }

    let sorted = unique_sorted(labels);
    let colors = cluster_colors(sorted.len());
    let color_of: HashMap<i64, RGBColor> =
        sorted.iter().copied().zip(colors.iter().copied()).collect();

    let (mut x0, mut x1, mut y0, mut y1) = par_bbox(x, y);
    if x0 == x1 {
        x0 -= 0.5;
        x1 += 0.5;
    }
    if y0 == y1 {
        y0 -= 0.5;
        y1 += 0.5;
    }

    const MARGIN: i32 = 12;
    const X_LABEL: i32 = 40;
    const Y_LABEL: i32 = 50;

    let (w, h) = root.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);

    let sorted_str: Vec<String> = sorted.iter().map(|l| l.to_string()).collect();
    let legend_w = legend_width(w, &sorted_str);
    let avail_w = (w - legend_w).max(200);

    let ar = (x1 - x0) / (y1 - y0);
    let plw_w = (avail_w - MARGIN * 2 - Y_LABEL).max(1) as f32;
    let mut pw = avail_w as f32;
    let mut ph = plw_w / ar + (MARGIN * 2 + X_LABEL) as f32;
    if ph > h as f32 {
        ph = h as f32;
        let plh = (ph - (MARGIN * 2 + X_LABEL) as f32).max(1.0);
        pw = plh * ar + (MARGIN * 2 + Y_LABEL) as f32;
    }
    let pw = (pw.max(200.0)).min(avail_w as f32) as i32;
    let ph = (ph.max(200.0)).min(h as f32) as i32;

    let (main_area, legend_area) = root.split_horizontally(avail_w.min(w));
    let (mw, mh) = main_area.dim_in_pixel();
    let (mw, mh) = (mw as i32, mh as i32);
    let left_pad = ((mw - pw) / 2).max(0);
    let top_pad = ((mh - ph) / 2).max(0);
    let plot_area = main_area.margin(top_pad, top_pad, left_pad, left_pad);

    let mut chart = ChartBuilder::on(&plot_area)
        .margin(MARGIN)
        .x_label_area_size(X_LABEL)
        .y_label_area_size(Y_LABEL)
        .build_cartesian_2d(x0..x1, y1..y0)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_label_style(("sans-serif", 15.0).into_font().color(&WHITE))
        .y_label_style(("sans-serif", 15.0).into_font().color(&WHITE))
        .axis_desc_style(("sans-serif", 15.0).into_font().color(&BLACK))
        .x_desc(x_label)
        .y_desc(y_label)
        .draw()?;

    let (pw_px, ph_px) = chart.plotting_area().dim_in_pixel();
    let dot_size = dot_size.unwrap_or_else(|| {
        let area = (pw_px as f32) * (ph_px as f32);
        super::spatial::default_dot_size_px(x.len(), area)
    });

    let hx = dot_size * 0.5 * (x1 - x0) / (pw_px as f32).max(1.0);
    let hy = dot_size * 0.5 * (y1 - y0) / (ph_px as f32).max(1.0);

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

    draw_legend(&legend_area, &sorted_str, &colors)?;
    plot_area.present()?;
    Ok(())
}

pub(crate) fn legend_width(w: i32, labels: &[String]) -> i32 {
    let max_chars = labels.iter().map(|l| l.chars().count()).max().unwrap_or(1);
    let want = 2 * (max_chars as i32 * 9 + 30) + 20;
    let cap = (((w as f32) * 0.30) as i32).max(140);
    want.clamp(140, cap)
}

fn truncate_label(lab: &str, max_chars: usize) -> String {
    let n = lab.chars().count();
    if n <= max_chars {
        return lab.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }
    let mut s: String = lab.chars().take(max_chars - 1).collect();
    s.push('…');
    s
}

const LEGEND_ROW_H: i32 = 26;

pub(crate) fn draw_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    labels: &[String],
    colors: &[RGBColor],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    if labels.is_empty() || w <= 0 || h <= 0 {
        return Ok(());
    }
    let pad = 10i32;
    let marker = 5i32;
    let gap = 10i32;
    let text_style = ("sans-serif", 14.0).into_font().color(&BLACK);

    let max_rows = ((h - 2 * pad) / LEGEND_ROW_H).max(1) as usize;
    let n = labels.len();
    let ncols = n.div_ceil(max_rows).max(1) as i32;
    let col_w = ((w - 2 * pad) / ncols).max(1);

    let avail_text = (col_w - marker * 2 - gap - 4).max(9);
    let max_chars = (avail_text / 9).max(1) as usize;

    for (i, (lab, color)) in labels.iter().zip(colors.iter()).enumerate() {
        let row = (i as i32) / ncols;
        let col = (i as i32) % ncols;
        let cx = pad + col * col_w + 6;
        let cy = pad + row * LEGEND_ROW_H + 7;
        if cy + LEGEND_ROW_H / 2 > h {
            break;
        }
        area.draw(&Circle::new((cx, cy), marker, color.filled()))?;
        let shown = truncate_label(lab, max_chars);
        area.draw_text(shown.as_str(), &text_style, (cx + gap + 2, cy - 7))?;
    }
    Ok(())
}
