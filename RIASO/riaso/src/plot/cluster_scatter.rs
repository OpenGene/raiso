use std::collections::HashMap;

use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use rayon::prelude::*;

use crate::core::SoData;
use crate::parallel::par_bbox;

use super::umap::{
    cluster_colors, draw_legend, extract_clusters, legend_width, unique_sorted, unique_sorted_str,
};

fn extract_positions_batch(sd: &SoData) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<i64>)> {
    let pos = sd
        .position
        .as_ref()
        .ok_or_else(|| anyhow!("position is empty: spatial scatter plots require coordinates"))?;

    let batch: Vec<i64> = match sd.obs.as_ref().and_then(|o| o.column("batch").ok()) {
        Some(col) => col
            .i64()
            .map_err(|_| anyhow!("obs[\"batch\"] is not Int64"))?
            .into_no_null_iter()
            .collect(),

        None => vec![0i64; pos.nrows()],
    };

    if pos.nrows() != batch.len() {
        return Err(anyhow!(
            "position has {} rows but there are {} batches",
            pos.nrows(),
            batch.len()
        ));
    }
    let xs = pos.column(0).to_vec();
    let ys = pos.column(1).to_vec();
    Ok((xs, ys, batch))
}

fn reorganize(batch: &[i64], xs: &mut [f32], ys: &mut [f32], ncols: usize, h_off: f32, v_off: f32) {
    if xs.is_empty() {
        return;
    }
    let mut batches: Vec<i64> = batch.to_vec();
    batches.sort_unstable();
    batches.dedup();
    let n_batches = batches.len();
    if n_batches == 0 {
        return;
    }
    let ncols = ncols.max(1).min(n_batches);
    let nrows = n_batches.div_ceil(ncols);

    let idx: Vec<Vec<usize>> = batches
        .par_iter()
        .map(|&bno| (0..batch.len()).filter(|&i| batch[i] == bno).collect())
        .collect();

    for ids in &idx {
        let (min_x, min_y) = ids
            .iter()
            .fold((f32::INFINITY, f32::INFINITY), |(mx, my), &i| {
                (mx.min(xs[i]), my.min(ys[i]))
            });
        for &i in ids {
            xs[i] -= min_x;
            ys[i] -= min_y;
        }
    }

    let mut max_xs = vec![0.0f32; ncols + 1];
    let mut max_ys = vec![0.0f32; nrows + 1];
    for (i, ids) in idx.iter().enumerate() {
        let row = i / ncols;
        let col = i % ncols;
        let (mut mx, mut mnx) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut my, mut mny) = (f32::NEG_INFINITY, f32::INFINITY);
        for &j in ids {
            mx = mx.max(xs[j]);
            mnx = mnx.min(xs[j]);
            my = my.max(ys[j]);
            mny = mny.min(ys[j]);
        }
        let w = mx - mnx + 1.0;
        let h = my - mny + 1.0;
        if w > max_xs[col + 1] {
            max_xs[col + 1] = w;
        }
        if h > max_ys[row + 1] {
            max_ys[row + 1] = h;
        }
    }

    for (i, ids) in idx.iter().enumerate() {
        let row = i / ncols;
        let col = i % ncols;
        let mut x_add = max_xs[col];
        let mut y_add = max_ys[row];
        if col > 0 {
            x_add += max_xs[..col].iter().sum::<f32>() + h_off * col as f32;
        }
        if row > 0 {
            y_add += max_ys[..row].iter().sum::<f32>() + v_off * row as f32;
        }
        for &j in ids {
            xs[j] += x_add;
            ys[j] += y_add;
        }
    }
}

fn resolution_nm(sd: &SoData) -> f32 {
    sd.uns
        .get("resolution")
        .and_then(|v| v.as_u64())
        .map(|v| v as f32)
        .unwrap_or(1.0)
}

fn fmt_real_length(nm: f32) -> String {
    let (v, unit) = if nm >= 1e9 {
        (nm / 1e9, "m")
    } else if nm >= 1e6 {
        (nm / 1e6, "mm")
    } else if nm >= 1e3 {
        (nm / 1e3, "um")
    } else {
        (nm, "nm")
    };
    if (v - v.round()).abs() < 1e-9 {
        format!("{:.0}{}", v, unit)
    } else {
        format!("{}{}", v, unit)
    }
}

fn extract_annotation_labels(data: &SoData) -> anyhow::Result<Vec<String>> {
    match &data.result.annotation {
        Some(a) => Ok(a.final_labels.clone()),
        None => Err(anyhow!(
            "SoData.result.annotation has no AnnotationResult; call run_singler first"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plot_cluster_scatter(
    sd: &SoData,
    path: &str,
    cluster_key: &str,
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    size: (u32, u32),
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    reorganize_coordinate: Option<usize>,
    horizontal_offset_additional: f32,
    vertical_offset_additional: f32,
    split_by_batch: bool,
) -> anyhow::Result<()> {
    let labels = extract_clusters(sd, cluster_key)?;

    let sorted = unique_sorted(&labels);
    let labels_str: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
    let sorted_str: Vec<String> = sorted.iter().map(|l| l.to_string()).collect();
    let title = title.unwrap_or(cluster_key);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    if split_by_batch {
        let samples = extract_categorical_by_batch(sd, &labels_str)?;
        let ncols = reorganize_coordinate.unwrap_or(3).max(1).min(samples.len());
        render_categorical_by_batch(root, &samples, &sorted_str, ncols, Some(title), dot_size)
            .map_err(|e| anyhow!("failed to draw the spatial cluster scatter plot: {e}"))
    } else {
        let (mut xs, mut ys, batch) = extract_positions_batch(sd)?;
        if let Some(ncols) = reorganize_coordinate {
            reorganize(
                &batch,
                &mut xs,
                &mut ys,
                ncols,
                horizontal_offset_additional,
                vertical_offset_additional,
            );
        }
        if labels.len() != xs.len() {
            return Err(anyhow!(
                "{} cluster labels vs {} spatial points",
                labels.len(),
                xs.len()
            ));
        }
        render_categorical_scatter(
            root,
            &xs,
            &ys,
            &labels_str,
            &sorted_str,
            Some(title),
            x_label,
            y_label,
            dot_size,
            show_plotting_scale,
            plotting_scale_width,
            resolution_nm(sd),
        )
        .map_err(|e| anyhow!("failed to draw the spatial cluster scatter plot: {e}"))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plot_cluster_scatter_png(
    sd: &SoData,
    cluster_key: &str,
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    size: (u32, u32),
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    reorganize_coordinate: Option<usize>,
    horizontal_offset_additional: f32,
    vertical_offset_additional: f32,
    split_by_batch: bool,
) -> anyhow::Result<Vec<u8>> {
    let labels = extract_clusters(sd, cluster_key)?;

    let sorted = unique_sorted(&labels);
    let labels_str: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
    let sorted_str: Vec<String> = sorted.iter().map(|l| l.to_string()).collect();
    let title = title.unwrap_or(cluster_key);

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        if split_by_batch {
            let samples = extract_categorical_by_batch(sd, &labels_str)?;
            let ncols = reorganize_coordinate.unwrap_or(3).max(1).min(samples.len());
            render_categorical_by_batch(root, &samples, &sorted_str, ncols, Some(title), dot_size)
                .map_err(|e| anyhow!("failed to draw the spatial cluster scatter plot: {e}"))?;
        } else {
            let (mut xs, mut ys, batch) = extract_positions_batch(sd)?;
            if let Some(ncols) = reorganize_coordinate {
                reorganize(
                    &batch,
                    &mut xs,
                    &mut ys,
                    ncols,
                    horizontal_offset_additional,
                    vertical_offset_additional,
                );
            }
            if labels.len() != xs.len() {
                return Err(anyhow!(
                    "{} cluster labels vs {} spatial points",
                    labels.len(),
                    xs.len()
                ));
            }
            render_categorical_scatter(
                root,
                &xs,
                &ys,
                &labels_str,
                &sorted_str,
                Some(title),
                x_label,
                y_label,
                dot_size,
                show_plotting_scale,
                plotting_scale_width,
                resolution_nm(sd),
            )
            .map_err(|e| anyhow!("failed to draw the spatial cluster scatter plot: {e}"))?;
        }
    }
    Ok(std::fs::read(&path)?)
}

#[allow(clippy::too_many_arguments)]
pub fn plot_annotation_scatter(
    sd: &SoData,
    path: &str,
    labels: &[String],
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    size: (u32, u32),
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    reorganize_coordinate: Option<usize>,
    horizontal_offset_additional: f32,
    vertical_offset_additional: f32,
    split_by_batch: bool,
) -> anyhow::Result<()> {
    let sorted = unique_sorted_str(labels);
    let title = title.unwrap_or("annotation");
    let root = BitMapBackend::new(path, size).into_drawing_area();
    if split_by_batch {
        let samples = extract_categorical_by_batch(sd, labels)?;
        let ncols = reorganize_coordinate.unwrap_or(3).max(1).min(samples.len());
        render_categorical_by_batch(root, &samples, &sorted, ncols, Some(title), dot_size)
            .map_err(|e| anyhow!("failed to draw the spatial annotation scatter plot: {e}"))
    } else {
        let (mut xs, mut ys, batch) = extract_positions_batch(sd)?;
        if let Some(ncols) = reorganize_coordinate {
            reorganize(
                &batch,
                &mut xs,
                &mut ys,
                ncols,
                horizontal_offset_additional,
                vertical_offset_additional,
            );
        }
        if labels.len() != xs.len() {
            return Err(anyhow!(
                "{} class labels vs {} spatial points",
                labels.len(),
                xs.len()
            ));
        }
        render_categorical_scatter(
            root,
            &xs,
            &ys,
            labels,
            &sorted,
            Some(title),
            x_label,
            y_label,
            dot_size,
            show_plotting_scale,
            plotting_scale_width,
            resolution_nm(sd),
        )
        .map_err(|e| anyhow!("failed to draw the spatial annotation scatter plot: {e}"))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn plot_annotation_scatter_png(
    sd: &SoData,
    labels: &[String],
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    size: (u32, u32),
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    reorganize_coordinate: Option<usize>,
    horizontal_offset_additional: f32,
    vertical_offset_additional: f32,
    split_by_batch: bool,
) -> anyhow::Result<Vec<u8>> {
    let sorted = unique_sorted_str(labels);
    let title = title.unwrap_or("annotation");

    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        if split_by_batch {
            let samples = extract_categorical_by_batch(sd, labels)?;
            let ncols = reorganize_coordinate.unwrap_or(3).max(1).min(samples.len());
            render_categorical_by_batch(root, &samples, &sorted, ncols, Some(title), dot_size)
                .map_err(|e| anyhow!("failed to draw the spatial annotation scatter plot: {e}"))?;
        } else {
            let (mut xs, mut ys, batch) = extract_positions_batch(sd)?;
            if let Some(ncols) = reorganize_coordinate {
                reorganize(
                    &batch,
                    &mut xs,
                    &mut ys,
                    ncols,
                    horizontal_offset_additional,
                    vertical_offset_additional,
                );
            }
            if labels.len() != xs.len() {
                return Err(anyhow!(
                    "{} class labels vs {} spatial points",
                    labels.len(),
                    xs.len()
                ));
            }
            render_categorical_scatter(
                root,
                &xs,
                &ys,
                labels,
                &sorted,
                Some(title),
                x_label,
                y_label,
                dot_size,
                show_plotting_scale,
                plotting_scale_width,
                resolution_nm(sd),
            )
            .map_err(|e| anyhow!("failed to draw the spatial annotation scatter plot: {e}"))?;
        }
    }
    Ok(std::fs::read(&path)?)
}

#[allow(clippy::too_many_arguments)]
pub fn plot_singler_scatter(
    data: &SoData,
    path: &str,
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    size: (u32, u32),
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    reorganize_coordinate: Option<usize>,
    horizontal_offset_additional: f32,
    vertical_offset_additional: f32,
    split_by_batch: bool,
) -> anyhow::Result<()> {
    let labels = extract_annotation_labels(data)?;
    plot_annotation_scatter(
        data,
        path,
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
}

#[allow(clippy::too_many_arguments)]
pub fn plot_singler_scatter_png(
    data: &SoData,
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    size: (u32, u32),
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    reorganize_coordinate: Option<usize>,
    horizontal_offset_additional: f32,
    vertical_offset_additional: f32,
    split_by_batch: bool,
) -> anyhow::Result<Vec<u8>> {
    let labels = extract_annotation_labels(data)?;
    plot_annotation_scatter_png(
        data,
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
}

#[allow(clippy::too_many_arguments)]
fn render_categorical_scatter<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    xs: &[f32],
    ys: &[f32],
    labels: &[String],
    sorted: &[String],
    title: Option<&str>,
    x_label: Option<&str>,
    y_label: Option<&str>,
    dot_size: Option<f32>,
    show_plotting_scale: bool,
    plotting_scale_width: Option<f32>,
    resolution_nm: f32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0, FontStyle::Bold).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    if xs.is_empty() {
        root.present()?;
        return Ok(());
    }

    let colors = cluster_colors(sorted.len());
    let color_of: HashMap<String, RGBColor> =
        sorted.iter().cloned().zip(colors.iter().copied()).collect();

    let (mut x0, mut x1, mut y0, mut y1) = par_bbox(xs, ys);
    if x0 == x1 {
        x0 -= 0.5;
        x1 += 0.5;
    }
    if y0 == y1 {
        y0 -= 0.5;
        y1 += 0.5;
    }

    let sw = plotting_scale_width.unwrap_or_else(|| ((x1 - x0).min(y1 - y0) / 5.0).ceil().max(1.0));
    let (cx0, cy1) = if show_plotting_scale {
        (x0 - sw * 0.6, y1 + sw * 0.5)
    } else {
        (x0, y1)
    };

    const MARGIN: i32 = 12;
    const X_LABEL: i32 = 40;
    const Y_LABEL: i32 = 50;

    let (w, h) = root.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);

    let legend_w = legend_width(w, sorted);
    let avail_w = (w - legend_w).max(200);

    let ar = (x1 - cx0) / (cy1 - y0);
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
        .build_cartesian_2d(cx0..x1, cy1..y0)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_label_style(("sans-serif", 15.0).into_font().color(&WHITE))
        .y_label_style(("sans-serif", 15.0).into_font().color(&WHITE))
        .axis_desc_style(("sans-serif", 15.0).into_font().color(&BLACK))
        .x_desc(x_label.unwrap_or(""))
        .y_desc(y_label.unwrap_or(""))
        .draw()?;

    let (pw_px, ph_px) = chart.plotting_area().dim_in_pixel();
    let dot_size = dot_size.unwrap_or_else(|| {
        let area = (pw_px as f32) * (ph_px as f32);
        super::spatial::default_dot_size_px(xs.len(), area)
    });
    let hx = dot_size * 0.5 * (x1 - cx0) / (pw_px as f32).max(1.0);
    let hy = dot_size * 0.5 * (cy1 - y0) / (ph_px as f32).max(1.0);

    let pts: Vec<(f32, f32, RGBColor)> = xs
        .par_iter()
        .zip(ys.par_iter())
        .zip(labels.par_iter())
        .map(|((&px, &py), lab)| {
            let c = *color_of.get(lab).unwrap_or(&RGBColor(0x88, 0x88, 0x88));
            (px, py, c)
        })
        .collect();
    chart.draw_series(pts.iter().map(|&(px, py, c)| {
        Rectangle::new([(px - hx, py - hy), (px + hx, py + hy)], c.filled())
    }))?;

    if show_plotting_scale {
        let thick = (sw / 10.0).max(1e-9);
        let black = RGBColor(0, 0, 0).filled();
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y1), (x0 + sw, y1 + thick)],
            black,
        )))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0 - thick, y1 - sw), (x0, y1)],
            black,
        )))?;
        let label = fmt_real_length(resolution_nm * sw);
        let style = ("sans-serif", 14.0).into_font().color(&BLACK);
        chart.draw_series(std::iter::once(Text::new(
            label,
            (x0 + sw / 2.0, y1 + thick * 3.0),
            &style,
        )))?;
    }

    draw_legend(&legend_area, sorted, &colors)?;
    plot_area.present()?;
    Ok(())
}

fn extract_categorical_by_batch(
    sd: &SoData,
    labels: &[String],
) -> anyhow::Result<Vec<(String, Vec<f32>, Vec<f32>, Vec<String>)>> {
    let (xs, ys, batch) = extract_positions_batch(sd)?;
    if labels.len() != xs.len() {
        return Err(anyhow!(
            "{} class labels vs {} spatial points",
            labels.len(),
            xs.len()
        ));
    }
    let mut batches: Vec<i64> = batch.clone();
    batches.sort_unstable();
    batches.dedup();

    let out: Vec<(String, Vec<f32>, Vec<f32>, Vec<String>)> = batches
        .par_iter()
        .map(|&b| {
            let name = sd
                .samples_names
                .get(b as usize)
                .cloned()
                .unwrap_or_else(|| format!("{b}"));
            let mut bxs = Vec::new();
            let mut bys = Vec::new();
            let mut blabs = Vec::new();
            for (((&bb, &x), &y), lab) in batch
                .iter()
                .zip(xs.iter())
                .zip(ys.iter())
                .zip(labels.iter())
            {
                if bb == b {
                    bxs.push(x);
                    bys.push(y);
                    blabs.push(lab.clone());
                }
            }
            (name, bxs, bys, blabs)
        })
        .collect();
    for (name, bxs, _, blabs) in &out {
        if bxs.len() != blabs.len() {
            return Err(anyhow!(
                "sample {name} has {} class labels but {} spatial points",
                blabs.len(),
                bxs.len()
            ));
        }
    }
    Ok(out)
}

fn categorical_scatter_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    name: &str,
    xs: &[f32],
    ys: &[f32],
    labels: &[String],
    color_of: &HashMap<String, RGBColor>,
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    if xs.is_empty() {
        return Ok(());
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

    const MARGIN: i32 = 12;
    const X_LABEL: i32 = 28;
    const Y_LABEL: i32 = 40;
    const CAPTION: i32 = 24;
    let (aw, ah) = area.dim_in_pixel();
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
    let plot_area = area.margin(top_pad, top_pad, left_pad, left_pad);

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

    let (pw_px, ph_px) = chart.plotting_area().dim_in_pixel();
    let dot_size = dot_size.unwrap_or_else(|| {
        let area = (pw_px as f32) * (ph_px as f32);
        super::spatial::default_dot_size_px(xs.len(), area)
    });
    let hx = dot_size * 0.5 * (x1 - x0) / (pw_px as f32).max(1.0);
    let hy = dot_size * 0.5 * (y1 - y0) / (ph_px as f32).max(1.0);
    let pts: Vec<(f32, f32, RGBColor)> = xs
        .par_iter()
        .zip(ys.par_iter())
        .zip(labels.par_iter())
        .map(|((&x, &y), lab)| {
            let c = *color_of.get(lab).unwrap_or(&RGBColor(0x88, 0x88, 0x88));
            (x, y, c)
        })
        .collect();
    chart.draw_series(
        pts.iter()
            .map(|&(x, y, c)| Rectangle::new([(x - hx, y - hy), (x + hx, y + hy)], c.filled())),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_categorical_by_batch<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    samples: &[(String, Vec<f32>, Vec<f32>, Vec<String>)],
    sorted: &[String],
    ncols: usize,
    title: Option<&str>,
    dot_size: Option<f32>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0, FontStyle::Bold).into_font())?,
        None => root,
    };
    root.fill(&WHITE)?;
    let n = samples.len();
    if n == 0 {
        root.present()?;
        return Ok(());
    }
    let ncols = ncols.min(n).max(1);
    let nrows = n.div_ceil(ncols);

    let colors = cluster_colors(sorted.len());
    let color_of: HashMap<String, RGBColor> =
        sorted.iter().cloned().zip(colors.iter().copied()).collect();

    let (w, _) = root.dim_in_pixel();
    let w = w as i32;
    let legend_w = legend_width(w, sorted);
    let avail_w = (w - legend_w).max(200).min(w);
    let (main_area, legend_area) = root.split_horizontally(avail_w);

    let areas = main_area.split_evenly((nrows, ncols));
    for (i, (name, xs, ys, labels)) in samples.iter().enumerate() {
        categorical_scatter_panel(
            &areas[i],
            &format!("sample {name}"),
            xs,
            ys,
            labels,
            &color_of,
            dot_size,
        )?;
    }
    draw_legend(&legend_area, sorted, &colors)?;
    areas[0].present()?;
    Ok(())
}
