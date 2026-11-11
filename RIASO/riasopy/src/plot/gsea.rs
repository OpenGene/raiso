use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::core::{GseaGroupResult, GseaResult, GseaTerm, SoData};

const DOT_RIGHT_W: i32 = 104;

const AXIS_H: i32 = 44;

const TITLE_BAND: usize = 38;

const SIZE_LEGEND_H: i32 = 150;

const ES_AXIS_H: i32 = 38;

const SUBTITLE_H: i32 = 24;

const RUG_H: i32 = 46;

const PANEL_GAP: i32 = 12;

const MAX_ROW_H: f32 = 22.0;

const LABEL_MAX_CHARS: usize = 52;

const SIG_ANCHORS: [(f32, (u8, u8, u8)); 5] = [
    (0.00, (0xf8, 0xf8, 0xf8)),
    (0.25, (0xfd, 0xd9, 0xd9)),
    (0.50, (0xfb, 0x90, 0x90)),
    (0.75, (0xd6, 0x35, 0x35)),
    (1.00, (0x7a, 0x00, 0x00)),
];

const LINE_BLUE: RGBColor = RGBColor(0x1f, 0x6f, 0xe0);

const BAND: RGBColor = RGBColor(0xf7, 0xf7, 0xf7);

fn sig_colormap(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    for w in SIG_ANCHORS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            let lerp = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * f).round() as u8;
            return RGBColor(lerp(c0.0, c1.0), lerp(c0.1, c1.1), lerp(c0.2, c1.2));
        }
    }
    let (_, c) = SIG_ANCHORS[SIG_ANCHORS.len() - 1];
    RGBColor(c.0, c.1, c.2)
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

fn fmt_signed(v: f32) -> String {
    format!("{}{}", if v >= 0.0 { "+" } else { "" }, fmt_short(v))
}

fn fmt_tick(v: f32, step: f32) -> String {
    let mut d = 0usize;
    loop {
        let scaled = step * 10f32.powi(d as i32);
        if (scaled - scaled.round()).abs() < 1e-6 || d >= 4 {
            break;
        }
        d += 1;
    }
    format!("{:.*}", d, v)
}

fn nice_step(max_v: f32, target_ticks: usize) -> f32 {
    if max_v <= 0.0 {
        return 1.0;
    }
    let raw = max_v / target_ticks.max(1) as f32;
    let mag = 10f32.powf(raw.log10().floor());
    for m in [1.0, 2.0, 5.0, 10.0] {
        let cand = m * mag;
        if cand >= raw {
            return cand;
        }
    }
    10.0 * mag
}

fn truncate_label(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max_chars).collect();
        t.push('…');
        t
    }
}

fn right_center(size: f32) -> TextStyle<'static> {
    ("sans-serif", size)
        .into_font()
        .color(&BLACK)
        .pos(Pos::new(HPos::Right, VPos::Center))
}

fn left_center(size: f32) -> TextStyle<'static> {
    ("sans-serif", size)
        .into_font()
        .color(&BLACK)
        .pos(Pos::new(HPos::Left, VPos::Center))
}

fn center_mid(size: f32) -> TextStyle<'static> {
    ("sans-serif", size)
        .into_font()
        .color(&BLACK)
        .pos(Pos::new(HPos::Center, VPos::Center))
}

fn with_title<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    title: Option<&str>,
) -> Result<DrawingArea<DB, Shift>, DrawingAreaErrorKind<DB::ErrorType>> {
    root.fill(&WHITE)?;
    let root = match title {
        Some(t) => root.titled(t, ("sans-serif", 18.0).into_font().color(&BLACK))?,
        None => root,
    };
    Ok(root)
}

fn label_zone_w<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    style: &TextStyle<'static>,
    labels: &[String],
    lo: i32,
    hi: i32,
) -> i32 {
    let fallback = labels
        .iter()
        .map(|s| s.chars().count() * 7)
        .max()
        .unwrap_or(0);
    let max_tw = labels
        .iter()
        .filter_map(|s| area.estimate_text_size(s, style).ok())
        .map(|(w, _)| w as i32)
        .max()
        .unwrap_or(fallback as i32);
    (max_tw + 16).clamp(lo, hi)
}

fn draw_colorbar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    vmin: f32,
    vmax: f32,
    cmap: fn(f32) -> RGBColor,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, h) = area.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    let x0 = 4;
    let x1 = (x0 + w - 32).max(x0 + 1);
    let y0 = 4;
    let y1 = (h - 4).max(y0 + 1);
    let n = 128;
    for i in 0..n {
        let t = 1.0 - i as f32 / (n as f32 - 1.0);
        let ya = y0 + (y1 - y0) * i / n;
        let yb = y0 + (y1 - y0) * (i + 1) / n;
        area.draw(&Rectangle::new([(x0, ya), (x1, yb)], cmap(t).filled()))?;
    }
    let style = ("sans-serif", 11.0).into_font().color(&BLACK);
    let mid = (vmin + vmax) * 0.5;
    area.draw_text(&fmt_short(vmax), &style, (x1 + 3, 0))?;
    area.draw_text(&fmt_short(mid), &style, (x1 + 3, h / 2 - 7))?;
    area.draw_text(&fmt_short(vmin), &style, (x1 + 3, (h - 14).max(0)))?;
    Ok(())
}

fn draw_size_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    max_size: f32,
    max_r: i32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, _h) = area.dim_in_pixel();
    let cx = w as i32 / 2;
    let step = 22i32;
    let color = sig_colormap(0.85);
    let style = ("sans-serif", 10.0).into_font().color(&BLACK);
    area.draw_text("size (K)", &style, (cx - 10, 4))?;
    for (k, f) in [1.0f32, 0.8, 0.6, 0.4, 0.2].iter().enumerate() {
        let r = (max_r as f32 * f.sqrt()).round() as i32;
        let cy = 26 + k as i32 * step;
        area.draw(&Circle::new((cx, cy), r.max(1), color.filled()))?;
        area.draw_text(&fmt_short(max_size * f), &style, (cx + max_r + 3, cy - 5))?;
    }
    Ok(())
}

fn draw_right_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    content_h: f32,
    max_sig: f32,
    max_size: f32,
    max_r: i32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (_rw, rh) = area.dim_in_pixel();
    let rh = rh as i32;

    let legend_h = SIZE_LEGEND_H.min((rh - 110).max(40));
    let cb_block = (18 + content_h.round().clamp(90.0, 320.0) as i32)
        .min(rh - legend_h)
        .max(40);
    let (cb_area, size_area) = area.split_vertically(cb_block);
    let (cap_area, cb_body) = cb_area.split_vertically(16);
    let cap_style = ("sans-serif", 11.0).into_font().color(&BLACK);
    cap_area.draw_text("-log10(p.adj)", &cap_style, (4, 2))?;
    draw_colorbar(&cb_body, 0.0, max_sig, sig_colormap)?;
    draw_size_legend(&size_area, max_size, max_r)?;
    Ok(())
}

fn extract_gsea<'a>(sd: &'a SoData, db_name: &str) -> anyhow::Result<&'a GseaResult> {
    let enrich =
        sd.result.enrichment.as_ref().ok_or_else(|| {
            anyhow::anyhow!("SoData.result.enrichment is empty; call run_gsea first")
        })?;
    enrich.gsea.get(db_name).ok_or_else(|| {
        let mut avail: Vec<&str> = enrich.gsea.keys().map(|s| s.as_str()).collect();
        avail.sort_unstable();
        anyhow::anyhow!(
            "enrichment.gsea has no database \"{db_name}\" (available: {avail:?}); run run_gsea with that database first"
        )
    })
}

fn pick_gsea_group(gsea: &GseaResult, cluster_id: Option<i64>) -> anyhow::Result<&GseaGroupResult> {
    match cluster_id {
        Some(id) => gsea
            .groups
            .iter()
            .find(|g| g.group_id == id)
            .ok_or_else(|| {
                let avail: Vec<i64> = gsea.groups.iter().map(|g| g.group_id).collect();
                anyhow::anyhow!("GSEA results have no cluster {id} (available group_id: {avail:?})")
            }),
        None => gsea
            .groups
            .iter()
            .find(|g| !g.terms.is_empty())
            .ok_or_else(|| anyhow::anyhow!("GSEA results are empty: no cluster to plot")),
    }
}

fn pick_terms(g: &GseaGroupResult, top_n: usize) -> anyhow::Result<Vec<&GseaTerm>> {
    if g.terms.is_empty() {
        anyhow::bail!("cluster {} has no significant gene set to plot", g.group_id);
    }
    let cap = if top_n == 0 {
        g.terms.len()
    } else {
        g.terms.len().min(top_n)
    };
    let mut v: Vec<&GseaTerm> = g.terms[..cap].iter().collect();
    v.sort_by(|a, b| {
        b.nes
            .partial_cmp(&a.nes)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(v)
}

fn sig_of(t: &GseaTerm) -> f32 {
    (-t.pvalue_adj.log10()).max(0.0)
}

fn symmetric_axis(half: f32, target_ticks: usize) -> (f32, f32) {
    let half = half.max(1e-6);
    let step = nice_step(half, target_ticks);
    let h = (half / step).ceil() * step;
    (h, step)
}

struct Plan {
    cys: Vec<i32>,

    row_h: f32,

    content_bot: f32,
}

fn plan_rows(n: usize, y0: f32, y1: f32, min_row_h: f32) -> Plan {
    if n == 0 {
        return Plan {
            cys: Vec::new(),
            row_h: 1.0,
            content_bot: y0,
        };
    }
    let row_h = ((y1 - y0) / n as f32).clamp(min_row_h, MAX_ROW_H);
    let cys: Vec<i32> = (0..n)
        .map(|r| (y0 + (r as f32 + 0.5) * row_h).round() as i32)
        .collect();
    let content_bot = (y0 + n as f32 * row_h).min(y1);
    Plan {
        cys,
        row_h,
        content_bot,
    }
}

fn render_enrichment<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    term: &GseaTerm,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let root = with_title(root, title)?;
    let (bw, bh) = root.dim_in_pixel();
    let (bw, bh) = (bw as i32, bh as i32);
    if bh < SUBTITLE_H + RUG_H + ES_AXIS_H + PANEL_GAP + 40 || bw < 200 {
        root.present()?;
        return Ok(());
    }

    let n_total = term
        .running_sum
        .last()
        .map(|&(p, _)| p + 1)
        .or_else(|| term.hits.last().map(|&h| h + 1))
        .unwrap_or(1)
        .max(1);

    let n_max = ((n_total - 1) as f32).max(1.0);

    let (sub_area, body) = root.split_vertically(SUBTITLE_H as u32);
    let sub_style = ("sans-serif", 12.0).into_font().color(&BLACK);
    sub_area.draw_text(
        &format!(
            "K = {}   ES = {}   NES = {}   p = {}   p.adj = {}   rank at ES = {}",
            term.size,
            fmt_signed(term.es),
            fmt_signed(term.nes),
            fmt_short(term.pvalue),
            fmt_short(term.pvalue_adj),
            term.rank_at_es
        ),
        &sub_style,
        (6, SUBTITLE_H / 2),
    )?;

    let (lab_area, plot_area) = body.split_horizontally(60u32);
    let (_lw, ph) = plot_area.dim_in_pixel();
    let ph = ph as i32;
    let es_h = (ph - RUG_H - ES_AXIS_H - PANEL_GAP).max(20) as u32;
    let (es_area, rest) = plot_area.split_vertically(es_h);
    let (_gap_area, rest) = rest.split_vertically(PANEL_GAP as u32);
    let (rug_area, ax_area) = rest.split_vertically(RUG_H as u32);

    let (ew, eh) = es_area.dim_in_pixel();
    let (ew, eh) = (ew as i32, eh as i32);
    let (px0, px1) = (2i32, (ew - 8).max(3));
    let (py0, py1) = (6i32, (eh - 6).max(7));
    let mut vmin = 0.0f32;
    let mut vmax = 0.0f32;
    for &(_, v) in &term.running_sum {
        vmin = vmin.min(v);
        vmax = vmax.max(v);
    }
    let pad = ((vmax - vmin) * 0.12).max(0.02);
    let (vlo, vhi) = (vmin - pad, vmax + pad);
    let xm = |rank: f32| -> i32 {
        let t = (rank / n_max).clamp(0.0, 1.0);
        px0 + ((px1 - px0) as f32 * t).round() as i32
    };
    let ym = |v: f32| -> i32 {
        let t = ((v - vlo) / (vhi - vlo)).clamp(0.0, 1.0);
        py1 - ((py1 - py0) as f32 * t).round() as i32
    };

    let tick_style = right_center(10.0);
    let zero_y = ym(0.0);
    lab_area.draw_text(&fmt_short(vhi), &tick_style, (58, py0))?;
    lab_area.draw_text(&fmt_short(vlo), &tick_style, (58, py1))?;
    lab_area.draw_text("0", &tick_style, (58, zero_y))?;
    let ylab = ("sans-serif", 11.0)
        .into_font()
        .color(&BLACK)
        .pos(Pos::new(HPos::Left, VPos::Center));
    lab_area.draw_text("ES", &ylab, (6, (py0 + py1) / 2))?;

    let peak_x = xm(term.rank_at_es as f32);
    es_area.draw(&PathElement::new(
        vec![(peak_x, py0), (peak_x, py1)],
        RGBColor(0x99, 0x99, 0x99).stroke_width(1),
    ))?;

    es_area.draw(&PathElement::new(
        vec![(px0, zero_y), (px1, zero_y)],
        RGBColor(0x55, 0x55, 0x55).stroke_width(1),
    ))?;

    if term.running_sum.len() >= 2 {
        let pts: Vec<(i32, i32)> = term
            .running_sum
            .iter()
            .map(|&(r, v)| (xm(r as f32), ym(v)))
            .collect();
        es_area.draw(&PathElement::new(pts, LINE_BLUE.stroke_width(2)))?;
    }

    es_area.draw(&Circle::new((peak_x, ym(term.es)), 3, LINE_BLUE.filled()))?;
    es_area.draw_text(
        &format!("ES {}", fmt_signed(term.es)),
        &("sans-serif", 11.0).into_font().color(&LINE_BLUE),
        ((peak_x + 4).min(px1 - 60), (ym(term.es) - 10).max(py0)),
    )?;

    let (rw, rh) = rug_area.dim_in_pixel();
    let (rw, rh) = (rw as i32, rh as i32);
    let rx1 = (rw - 8).max(3);
    let rxm = |rank: f32| -> i32 {
        let t = (rank / n_max).clamp(0.0, 1.0);
        2 + ((rx1 - 2) as f32 * t).round() as i32
    };
    rug_area.draw_text("hits", &left_center(10.0), (6, 8))?;
    for &h in &term.hits {
        let x = rxm(h as f32);
        rug_area.draw(&PathElement::new(
            vec![(x, (rh - 4).max(0)), (x, (rh - 18).max(0))],
            LINE_BLUE.stroke_width(1),
        ))?;
    }

    let (aw, ah) = ax_area.dim_in_pixel();
    let (aw, ah) = (aw as i32, ah as i32);
    let ax1 = (aw - 8).max(3);
    let axm = |rank: f32| -> i32 {
        let t = (rank / n_max).clamp(0.0, 1.0);
        2 + ((ax1 - 2) as f32 * t).round() as i32
    };
    let ax_style = center_mid(10.0);
    for k in 0..=4 {
        let rank = n_max * k as f32 / 4.0;
        ax_area.draw_text(
            &format!("{}", rank.round() as i64),
            &ax_style,
            (axm(rank), 5),
        )?;
    }

    ax_area.draw_text(
        "rank in ordered gene list  (0 = most up-regulated in this cluster)",
        &center_mid(11.0),
        ((2 + ax1) / 2, (ah - 7).max(8)),
    )?;

    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_gsea_enrichment(
    sd: &SoData,
    db_name: &str,
    path: &str,
    cluster_id: Option<i64>,
    term_index: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let (term, size, owned_title) =
        prepare_enrichment(sd, db_name, cluster_id, term_index, title, size)?;
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_enrichment(root, term, owned_title.as_deref())
        .map_err(|e| anyhow::anyhow!("failed to draw the GSEA enrichment curve: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_gsea_enrichment_png(
    sd: &SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    term_index: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let (term, size, owned_title) =
        prepare_enrichment(sd, db_name, cluster_id, term_index, title, size)?;
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_enrichment(root, term, owned_title.as_deref())
            .map_err(|e| anyhow::anyhow!("failed to draw the GSEA enrichment curve: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn prepare_enrichment<'a>(
    sd: &'a SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    term_index: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<(&'a GseaTerm, (u32, u32), Option<String>)> {
    let g = pick_gsea_group(extract_gsea(sd, db_name)?, cluster_id)?;
    let term = g.terms.get(term_index).ok_or_else(|| {
        anyhow::anyhow!(
            "cluster {} has only {} GSEA results and no term_index = {term_index}",
            g.group_id,
            g.terms.len()
        )
    })?;
    if term.running_sum.len() < 2 {
        anyhow::bail!(
            "term {} has an empty running_sum trajectory (only {} points) -- plot the results produced by run_gsea",
            term.id,
            term.running_sum.len()
        );
    }
    let (w, h) = size;
    let need = (SUBTITLE_H + RUG_H + ES_AXIS_H + PANEL_GAP + 120) as u32;
    let size = (w, h.max(need));
    let owned_title = title.map(|t| t.to_string());
    Ok((term, size, owned_title))
}

struct Panel<'a> {
    terms: Vec<&'a GseaTerm>,
    max_sig: f32,
    max_size: f32,
    step: f32,
    panel_max: f32,
}

fn prepare_panel<'a>(
    sd: &'a SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    top_n: usize,
) -> anyhow::Result<Panel<'a>> {
    let g = pick_gsea_group(extract_gsea(sd, db_name)?, cluster_id)?;
    let terms = pick_terms(g, top_n)?;
    let mut max_sig = 0.0f32;
    let mut max_size = 0.0f32;
    let mut half = 0.0f32;
    for t in &terms {
        max_sig = max_sig.max(sig_of(t));
        max_size = max_size.max(t.size as f32);
        half = half.max(t.nes.abs());
    }
    let (panel_max, step) = symmetric_axis(half, 5);
    Ok(Panel {
        terms,
        max_sig: max_sig.max(1e-9),
        max_size: max_size.max(1.0),
        step,
        panel_max,
    })
}

fn panel_geometry<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    p: &Panel,
    w: i32,
    h: i32,
) -> (i32, i32, f32, Plan) {
    let style = right_center(11.0);
    let labels: Vec<String> = p
        .terms
        .iter()
        .map(|t| truncate_label(&t.name, LABEL_MAX_CHARS))
        .collect();
    let x0 = label_zone_w(area, &style, &labels, 120, 620);
    let x1 = (w - DOT_RIGHT_W - 12).max(x0 + 40);
    let y0 = 4.0f32;
    let y1 = (h - AXIS_H) as f32;
    let plan = plan_rows(p.terms.len(), y0, y1, 14.0);
    (x0, x1, y1, plan)
}

#[allow(clippy::too_many_arguments)]
pub fn plot_gsea_dotplot(
    sd: &SoData,
    db_name: &str,
    path: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let p = prepare_panel(sd, db_name, cluster_id, top_n)?;
    let size = auto_panel_size(size, p.terms.len(), 18, true);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_gsea_dotplot(root, &p, title)
        .map_err(|e| anyhow::anyhow!("failed to draw the GSEA dotplot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_gsea_dotplot_png(
    sd: &SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let p = prepare_panel(sd, db_name, cluster_id, top_n)?;
    let size = auto_panel_size(size, p.terms.len(), 18, true);
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_gsea_dotplot(root, &p, title)
            .map_err(|e| anyhow::anyhow!("failed to draw the GSEA dotplot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn render_gsea_dotplot<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    p: &Panel,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let root = with_title(root, title)?;
    let (w, h) = root.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    if p.terms.is_empty() {
        root.present()?;
        return Ok(());
    }
    let (x0, x1, _y1, plan) = panel_geometry(&root, p, w, h);
    let row_h = plan.row_h;
    let max_r = (row_h * 0.5 - 2.0).clamp(4.0, 12.0) as i32;
    let axis_max = p.panel_max;
    let ax = |v: f32| x0 + ((x1 - x0) as f32 * ((v + axis_max) / (2.0 * axis_max))).round() as i32;
    let n_ticks = (2.0 * axis_max / p.step).round() as usize;
    let zero_x = ax(0.0);

    let (_body_area, right_area) = root.split_horizontally(x1.max(1));

    for (i, cy) in plan.cys.iter().enumerate() {
        if i % 2 == 1 {
            let ya = (*cy as f32 - row_h * 0.5).round() as i32;
            let yb = (*cy as f32 + row_h * 0.5).round() as i32;
            root.draw(&Rectangle::new(
                [(0, ya.max(0)), (x1, yb.min(h))],
                BAND.filled(),
            ))?;
        }
    }

    let grid_style = RGBColor(0xea, 0xea, 0xea).stroke_width(1);
    let grid_bot = plan.content_bot.round() as i32;
    for i in 0..=n_ticks {
        let v = -axis_max + i as f32 * p.step;
        let px = ax(v);
        root.draw(&PathElement::new(vec![(px, 4), (px, grid_bot)], grid_style))?;
    }

    root.draw(&PathElement::new(
        vec![(zero_x, 4), (zero_x, grid_bot)],
        RGBColor(0x66, 0x66, 0x66).stroke_width(2),
    ))?;

    let label_style = right_center(11.0);
    for (i, t) in p.terms.iter().enumerate() {
        let cy = plan.cys[i];
        root.draw_text(
            &truncate_label(&t.name, LABEL_MAX_CHARS),
            &label_style,
            (x0 - 6, cy),
        )?;
        let r = ((max_r as f32) * (t.size as f32 / p.max_size).sqrt())
            .round()
            .max(1.0) as i32;
        root.draw(&Circle::new(
            (ax(t.nes), cy),
            r,
            sig_colormap(sig_of(t) / p.max_sig).filled(),
        ))?;
    }

    let axis_y = (plan.content_bot + 12.0).round() as i32;
    let stride = tick_stride(x1 - x0, n_ticks);
    let ax_style = center_mid(11.0);
    for i in 0..=n_ticks {
        if !label_tick(i, n_ticks, stride) {
            continue;
        }
        let v = -axis_max + i as f32 * p.step;
        root.draw_text(&fmt_tick(v, p.step), &ax_style, (ax(v), axis_y))?;
    }

    root.draw_text(
        "NES   (left = down-regulated in this cluster / right = up-regulated)",
        &center_mid(12.0),
        ((x0 + x1) / 2, axis_y + 24),
    )?;

    draw_right_legend(
        &right_area,
        plan.content_bot - 4.0,
        p.max_sig,
        p.max_size,
        max_r,
    )?;

    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_gsea_barplot(
    sd: &SoData,
    db_name: &str,
    path: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let p = prepare_panel(sd, db_name, cluster_id, top_n)?;
    let size = auto_panel_size(size, p.terms.len(), 20, false);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_gsea_barplot(root, &p, title)
        .map_err(|e| anyhow::anyhow!("failed to draw the GSEA barplot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_gsea_barplot_png(
    sd: &SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let p = prepare_panel(sd, db_name, cluster_id, top_n)?;
    let size = auto_panel_size(size, p.terms.len(), 20, false);
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_gsea_barplot(root, &p, title)
            .map_err(|e| anyhow::anyhow!("failed to draw the GSEA barplot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn render_gsea_barplot<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    p: &Panel,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let root = with_title(root, title)?;
    let (w, h) = root.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    if p.terms.is_empty() {
        root.present()?;
        return Ok(());
    }
    let (x0, x1, _y1, plan) = panel_geometry(&root, p, w, h);
    let row_h = plan.row_h;
    let axis_max = p.panel_max;
    let ax = |v: f32| x0 + ((x1 - x0) as f32 * ((v + axis_max) / (2.0 * axis_max))).round() as i32;
    let n_ticks = (2.0 * axis_max / p.step).round() as usize;
    let zero_x = ax(0.0);

    let axis_y = (plan.content_bot + 2.0).round() as i32;

    for (i, cy) in plan.cys.iter().enumerate() {
        if i % 2 == 1 {
            let ya = (*cy as f32 - row_h * 0.5).round() as i32;
            let yb = (*cy as f32 + row_h * 0.5).round() as i32;
            root.draw(&Rectangle::new(
                [(0, ya.max(0)), (x1, yb.min(h))],
                BAND.filled(),
            ))?;
        }
    }

    let grid_style = RGBColor(0xea, 0xea, 0xea).stroke_width(1);
    let grid_bot = plan.content_bot.round() as i32;
    for i in 0..=n_ticks {
        let px = ax(-axis_max + i as f32 * p.step);
        root.draw(&PathElement::new(vec![(px, 4), (px, grid_bot)], grid_style))?;
    }
    root.draw(&PathElement::new(
        vec![(x0, axis_y), (x1, axis_y)],
        RGBColor(0x88, 0x88, 0x88).stroke_width(1),
    ))?;

    let label_style = right_center(11.0);
    for (i, t) in p.terms.iter().enumerate() {
        let cy = plan.cys[i];
        root.draw_text(
            &truncate_label(&t.name, LABEL_MAX_CHARS),
            &label_style,
            (x0 - 6, cy),
        )?;
        let px = ax(t.nes);
        let y_top = (cy as f32 - row_h * 0.5).round() as i32;
        let y_bot = (cy as f32 + row_h * 0.5).round() as i32 - 1;
        if px != zero_x {
            root.draw(&Rectangle::new(
                [(px.min(zero_x), y_top), (px.max(zero_x), y_bot.max(y_top))],
                sig_colormap(sig_of(t) / p.max_sig).filled(),
            ))?;
        }
    }

    root.draw(&PathElement::new(
        vec![(zero_x, 4), (zero_x, axis_y)],
        RGBColor(0x44, 0x44, 0x44).stroke_width(2),
    ))?;

    let stride = tick_stride(x1 - x0, n_ticks);
    let ax_style = center_mid(11.0);
    for i in 0..=n_ticks {
        if !label_tick(i, n_ticks, stride) {
            continue;
        }
        let v = -axis_max + i as f32 * p.step;
        root.draw_text(&fmt_tick(v, p.step), &ax_style, (ax(v), axis_y + 12))?;
    }
    root.draw_text(
        "NES   (left = down-regulated in this cluster / right = up-regulated)",
        &center_mid(12.0),
        ((x0 + x1) / 2, axis_y + 34),
    )?;

    root.present()?;
    Ok(())
}

fn auto_panel_size(
    size: (u32, u32),
    n_term: usize,
    per_row: usize,
    with_legend: bool,
) -> (u32, u32) {
    let (w, _) = size;

    let rows_need = TITLE_BAND + 4 + n_term * per_row + AXIS_H as usize + 6;
    if !with_legend {
        return (w, rows_need as u32);
    }

    let cbar_h = 18 + (n_term * per_row).clamp(90, 320);
    let legend_need = TITLE_BAND + 4 + cbar_h + SIZE_LEGEND_H as usize + 6;
    (w, rows_need.max(legend_need) as u32)
}

fn tick_stride(plot_w: i32, n_ticks: usize) -> usize {
    if n_ticks == 0 {
        return 1;
    }
    if (plot_w as f32) / (n_ticks as f32) < 46.0 {
        2
    } else {
        1
    }
}

fn label_tick(i: usize, n_ticks: usize, stride: usize) -> bool {
    i.is_multiple_of(stride) || i == n_ticks / 2
}
