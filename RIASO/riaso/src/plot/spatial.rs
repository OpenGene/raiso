use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::SoData;
use crate::parallel::{par_bbox, par_min_max};

const STEREO_ANCHORS: [(f32, (u8, u8, u8)); 5] = [
    (0.00, (0x0c, 0x33, 0x83)),
    (0.25, (0x0a, 0x88, 0xba)),
    (0.50, (0xf2, 0xd3, 0x38)),
    (0.75, (0xf2, 0x8f, 0x38)),
    (1.00, (0xd9, 0x1e, 0x1e)),
];

pub(crate) fn stereo_colormap(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    for w in STEREO_ANCHORS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
            return RGBColor(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    let (_, c) = STEREO_ANCHORS[STEREO_ANCHORS.len() - 1];
    RGBColor(c.0, c.1, c.2)
}

pub(crate) fn extract_position(data: &SoData) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let pos = data
        .position
        .as_ref()
        .ok_or_else(|| anyhow!("position is empty: no spatial coordinates"))?;
    Ok((pos.column(0).to_vec(), pos.column(1).to_vec()))
}

pub(crate) fn default_dot_size_px(n: usize, area_px: f32) -> f32 {
    let spacing = (area_px / n.max(1) as f32).sqrt();
    spacing * 1.3
}

pub(crate) fn spatial_scatter_panel<DB: DrawingBackend>(
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
    let bar_w = 72i32;
    let (plot_area, bar_area) = area.split_horizontally((w - bar_w).max(1));

    const MARGIN: i32 = 12;
    const X_LABEL: i32 = 28;
    const Y_LABEL: i32 = 40;
    const CAPTION: i32 = 24;
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
        default_dot_size_px(xs.len(), area)
    });
    let hx = dot_size * 0.5 * (x1 - x0) / (pw as f32).max(1.0);
    let hy = dot_size * 0.5 * (y1 - y0) / (ph as f32).max(1.0);
    let inv = 1.0 / (vmax - vmin);

    let pts: Vec<(f32, f32, RGBColor)> = vals
        .par_iter()
        .zip(xs.par_iter().zip(ys.par_iter()))
        .map(|(&v, (&x, &y))| (x, y, stereo_colormap((v - vmin) * inv)))
        .collect();
    chart.draw_series(
        pts.iter()
            .map(|&(x, y, c)| Rectangle::new([(x - hx, y - hy), (x + hx, y + hy)], c.filled())),
    )?;

    draw_colorbar(&bar_area, vmin, vmax)?;
    Ok(())
}

fn draw_colorbar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    vmin: f32,
    vmax: f32,
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
        area.draw(&Rectangle::new(
            [(x0, ya), (x1, yb)],
            stereo_colormap(t).filled(),
        ))?;
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
