use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use polars::frame::DataFrame;
use rayon::prelude::*;

use crate::parallel::{par_histogram_counts, par_min_max};

pub(crate) fn metric_vec(obs: &DataFrame, name: &str) -> anyhow::Result<Vec<f32>> {
    let col = obs
        .column(name)
        .map_err(|_| anyhow::anyhow!("obs is missing column {name:?}"))?;
    if let Ok(ca) = col.f32() {
        return Ok(ca.into_no_null_iter().collect());
    }
    if let Ok(ca) = col.i64() {
        return Ok(ca.into_no_null_iter().map(|v| v as f32).collect());
    }
    anyhow::bail!("column {name:?} is not numeric")
}

pub(crate) fn gaussian_kde(values: &[f32], n_grid: usize) -> (Vec<f32>, Vec<f32>) {
    let n = values.len().max(1);
    let mean = values.iter().sum::<f32>() / n as f32;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    let sigma = var.sqrt();
    let mut h = 1.06 * sigma * (n as f32).powf(-0.2);

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(h > 1e-9) {
        h = 1e-3;
    }
    let (lo, hi) = par_min_max(values);
    let (x_lo, x_hi) = (lo - 3.0 * h, hi + 3.0 * h);
    let xs: Vec<f32> = (0..n_grid)
        .map(|i| x_lo + (x_hi - x_lo) * i as f32 / (n_grid - 1) as f32)
        .collect();
    let inv_h = 1.0 / h;
    let norm = 1.0 / (n as f32 * h * (2.0 * std::f32::consts::PI).sqrt());

    let ds: Vec<f32> = xs
        .par_iter()
        .map(|&x| {
            let s: f32 = values
                .iter()
                .map(|&v| {
                    let z = (x - v) * inv_h;
                    (-0.5 * z * z).exp()
                })
                .sum();
            s * norm
        })
        .collect();
    (xs, ds)
}

pub(crate) fn histogram(values: &[f32], n_bins: usize) -> (Vec<usize>, Vec<f32>) {
    let n = values.len();
    if n == 0 {
        return (vec![0; n_bins], (0..=n_bins).map(|j| j as f32).collect());
    }
    let (mut lo, mut hi) = par_min_max(values);
    if (hi - lo).abs() < f32::EPSILON {
        lo -= 0.5;
        hi += 0.5;
    }
    let bin_w = (hi - lo) / n_bins as f32;

    let counts = par_histogram_counts(values, lo, bin_w, n_bins);
    let edges: Vec<f32> = (0..=n_bins).map(|j| lo + bin_w * j as f32).collect();
    (counts, edges)
}

pub(crate) fn plot_violin_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    name: &str,
    vals: &[f32],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if vals.is_empty() {
        return Ok(());
    }
    let (xs, ds) = gaussian_kde(vals, 256);
    let max_d = ds.iter().cloned().fold(0.0f32, f32::max);
    if max_d <= 0.0 {
        return Ok(());
    }
    let (x_lo, x_hi) = (xs[0], xs[xs.len() - 1]);
    let mut chart = ChartBuilder::on(area)
        .caption(name, ("sans-serif", 14.0).into_font())
        .margin(12)
        .x_label_area_size(28)
        .y_label_area_size(40)
        .build_cartesian_2d(x_lo..x_hi, -max_d..max_d)?;
    chart.configure_mesh().light_line_style(WHITE).draw()?;

    let mut pts: Vec<(f32, f32)> = xs.iter().zip(ds.iter()).map(|(&x, &d)| (x, d)).collect();
    pts.extend(xs.iter().zip(ds.iter()).rev().map(|(&x, &d)| (x, -d)));
    chart.draw_series(std::iter::once(Polygon::new(
        pts,
        RGBColor(46, 134, 222).mix(0.55).filled(),
    )))?;
    Ok(())
}

pub(crate) fn plot_hist_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    name: &str,
    vals: &[f32],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if vals.is_empty() {
        return Ok(());
    }
    let n_bins = 50;
    let (counts, edges) = histogram(vals, n_bins);
    let max_c = counts.iter().cloned().max().unwrap_or(0).max(1) as f32;
    let (x_lo, x_hi) = (edges[0], edges[edges.len() - 1]);
    let mut chart = ChartBuilder::on(area)
        .caption(name, ("sans-serif", 14.0).into_font())
        .margin(12)
        .x_label_area_size(28)
        .y_label_area_size(40)
        .build_cartesian_2d(x_lo..x_hi, 0.0..max_c)?;
    chart.configure_mesh().light_line_style(WHITE).draw()?;
    chart.draw_series(counts.iter().enumerate().map(|(j, &c)| {
        Rectangle::new(
            [(edges[j], 0.0), (edges[j + 1], c as f32)],
            RGBColor(46, 134, 222).mix(0.55).filled(),
        )
    }))?;
    Ok(())
}
