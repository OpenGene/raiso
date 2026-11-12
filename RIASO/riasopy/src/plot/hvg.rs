use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::SoData;
use crate::parallel::par_bbox_opt;

const OTHER_GRAY: RGBColor = RGBColor(0xcc, 0xcc, 0xcc);

fn extract_hvg(data: &SoData) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<bool>)> {
    let hvg = match &data.result.hvg {
        Some(h) => h,
        None => {
            return Err(anyhow!(
                "SoData.result.hvg has no HvgResult; call select_top_variable_genes first"
            ))
        }
    };
    Ok((
        hvg.means.clone(),
        hvg.dispersions.clone(),
        hvg.dispersions_norm.clone(),
        hvg.highly_variable.clone(),
    ))
}

pub fn plot_hvg(data: &SoData, path: &str, size: (u32, u32)) -> anyhow::Result<()> {
    let (means, dispersions, dispersions_norm, hv) = extract_hvg(data)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_hvg(root, &means, &dispersions, &dispersions_norm, &hv)
        .map_err(|e| anyhow!("failed to draw the highly variable gene plot: {e}"))
}

pub fn plot_hvg_png(data: &SoData, size: (u32, u32)) -> anyhow::Result<Vec<u8>> {
    let (means, dispersions, dispersions_norm, hv) = extract_hvg(data)?;

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_hvg(root, &means, &dispersions, &dispersions_norm, &hv)
            .map_err(|e| anyhow!("failed to draw the highly variable gene plot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn render_hvg<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    means: &[f32],
    dispersions: &[f32],
    dispersions_norm: &[f32],
    hv: &[bool],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));

    hvg_panel(
        &areas[0],
        "mean expression of genes",
        "dispersions of genes (normalized)",
        means,
        dispersions_norm,
        hv,
    )?;

    hvg_panel(
        &areas[1],
        "mean expression of genes",
        "dispersions of genes (not normalized)",
        means,
        dispersions,
        hv,
    )?;

    areas[0].present()?;
    Ok(())
}

fn hvg_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    x_label: &str,
    y_label: &str,
    means: &[f32],
    ys: &[f32],
    hv: &[bool],
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if means.is_empty() {
        return Ok(());
    }

    let (mut x0, mut x1, mut y0, mut y1) = par_bbox_opt(means, ys, true);
    if !x0.is_finite() {
        return Ok(());
    }
    if x0 == x1 {
        x0 -= 0.5;
        x1 += 0.5;
    }
    if y0 == y1 {
        y0 -= 0.5;
        y1 += 0.5;
    }

    let mut chart = ChartBuilder::on(area)
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x0..x1, y0..y1)?;

    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_label_style(("sans-serif", 15.0).into_font())
        .y_label_style(("sans-serif", 15.0).into_font())
        .draw()?;

    let (pw, _ph) = chart.plotting_area().dim_in_pixel();
    let r = 3.0 * (x1 - x0) / (pw as f32).max(1.0);

    let dots: Vec<((f32, f32), RGBColor)> = means
        .par_iter()
        .zip(ys.par_iter())
        .zip(hv.par_iter())
        .filter_map(|((&x, &y), &h)| {
            if !y.is_finite() {
                return None;
            }
            Some(((x, y), if h { BLACK } else { OTHER_GRAY }))
        })
        .collect();
    chart.draw_series(
        dots.iter()
            .map(|&((x, y), color)| Circle::new((x, y), r, color.filled())),
    )?;
    Ok(())
}
