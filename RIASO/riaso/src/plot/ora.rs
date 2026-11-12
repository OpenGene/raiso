use std::collections::HashMap;

use anyhow::anyhow;
use plotters::coord::Shift;
use plotters::drawing::DrawingAreaErrorKind;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::core::{EnrichmentResult, OraGroupResult, OraTerm, SoData};

const DOT_RIGHT_W: i32 = 104;

const HEADER_H: i32 = 18;

const AXIS_H: i32 = 28;

const MAX_ROW_H: f32 = 22.0;

const TITLE_BAND: usize = 38;

const SIZE_LEGEND_H: usize = 150;

const LABEL_MAX_CHARS: usize = 52;

const PREF_ONT: [&str; 3] = ["BP", "CC", "MF"];

const SIG_ANCHORS: [(f32, (u8, u8, u8)); 5] = [
    (0.00, (0xf8, 0xf8, 0xf8)),
    (0.25, (0xfd, 0xd9, 0xd9)),
    (0.50, (0xfb, 0x90, 0x90)),
    (0.75, (0xd6, 0x35, 0x35)),
    (1.00, (0x7a, 0x00, 0x00)),
];

const HEADER_BAND: RGBColor = RGBColor(0xef, 0xef, 0xef);

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

fn extract_enrichment(sd: &SoData) -> anyhow::Result<&EnrichmentResult> {
    match &sd.result.enrichment {
        Some(e) => Ok(e),
        None => Err(anyhow!(
            "SoData.result.enrichment has no EnrichmentResult; call run_ora first"
        )),
    }
}

fn pick_group<'a>(
    enrich: &'a EnrichmentResult,
    db_name: &str,
    cluster_id: Option<i64>,
) -> anyhow::Result<&'a OraGroupResult> {
    let groups = enrich.groups.get(db_name).map(|v| v.as_slice()).ok_or_else(|| {
        let mut avail: Vec<&str> = enrich.groups.keys().map(|s| s.as_str()).collect();
        avail.sort_unstable();
        anyhow!(
            "enrichment.groups has no database \"{db_name}\" (available: {avail:?}); run run_ora with that database first"
        )
    })?;
    match cluster_id {
        Some(id) => groups.iter().find(|g| g.group_id == id).ok_or_else(|| {
            let avail: Vec<i64> = groups.iter().map(|g| g.group_id).collect();
            anyhow!("no cluster {id} (available group_id: {avail:?})")
        }),
        None => groups
            .iter()
            .find(|g| !g.terms.is_empty())
            .ok_or_else(|| anyhow!("enrichment results are empty: no cluster to plot")),
    }
}

struct Row {
    label: String,

    count: f32,

    gr: f32,

    sig: f32,
}

struct Block {
    cat: String,
    rows: Vec<Row>,
}

fn ont_key(cat: &Option<String>) -> String {
    match cat {
        Some(c) => {
            let u = c.trim().to_ascii_uppercase();
            if u.is_empty() {
                "OTHER".to_string()
            } else {
                u
            }
        }
        None => "OTHER".to_string(),
    }
}

struct Prepared {
    blocks: Vec<Block>,

    n_term: usize,

    sectional: bool,
    max_count: f32,
    max_sig: f32,
}

fn prepare(
    enrich: &EnrichmentResult,
    db_name: &str,
    cluster_id: Option<i64>,
    top_n: usize,
) -> anyhow::Result<Prepared> {
    let g = pick_group(enrich, db_name, cluster_id)?;
    let cap = if top_n == 0 {
        g.terms.len()
    } else {
        g.terms.len().min(top_n)
    };
    let terms = &g.terms[..cap];
    if terms.is_empty() {
        anyhow::bail!("cluster {} has no significant term to plot", g.group_id);
    }

    let sectional = terms.iter().any(|t: &OraTerm| {
        t.category
            .as_deref()
            .map(|c| !c.trim().is_empty())
            .unwrap_or(false)
    });

    let mut blocks: Vec<Block> = Vec::new();
    if sectional {
        let mut seen: Vec<String> = Vec::new();
        let mut buckets: HashMap<String, Vec<Row>> = HashMap::new();
        for t in terms {
            let cat = ont_key(&t.category);
            if !buckets.contains_key(&cat) {
                seen.push(cat.clone());
            }
            buckets.entry(cat).or_default().push(to_row(t));
        }
        seen.sort_by_cached_key(|c| {
            let rank = PREF_ONT
                .iter()
                .position(|&x| x == c)
                .unwrap_or(PREF_ONT.len());
            (rank, c.clone())
        });
        for cat in seen {
            let mut rows = buckets.remove(&cat).unwrap_or_default();
            rows.sort_by(|a, b| {
                b.sig
                    .partial_cmp(&a.sig)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.label.cmp(&b.label))
            });
            if !rows.is_empty() {
                blocks.push(Block { cat, rows });
            }
        }
    } else {
        blocks.push(Block {
            cat: String::new(),
            rows: terms.iter().map(to_row).collect(),
        });
    }

    let n_term: usize = blocks.iter().map(|b| b.rows.len()).sum();
    let mut max_count = 0.0f32;
    let mut max_sig = 0.0f32;
    for b in &blocks {
        for r in &b.rows {
            max_count = max_count.max(r.count);
            max_sig = max_sig.max(r.sig);
        }
    }
    Ok(Prepared {
        blocks,
        n_term,
        sectional,
        max_count: max_count.max(1.0),
        max_sig: max_sig.max(1e-9),
    })
}

fn to_row(t: &OraTerm) -> Row {
    Row {
        label: truncate_label(&t.name, LABEL_MAX_CHARS),
        count: t.count as f32,
        gr: t.gene_ratio,
        sig: (-t.pvalue_adj.log10()).max(0.0),
    }
}

struct Plan {
    cys: Vec<i32>,

    row_h: f32,

    content_bot: f32,

    headers: Vec<(i32, i32, String)>,

    zebra: bool,
}

fn plan_layout(p: &Prepared, y0: f32, y1: f32, min_row_h: f32) -> Plan {
    if p.n_term == 0 {
        return Plan {
            cys: Vec::new(),
            row_h: 1.0,
            content_bot: y0,
            headers: Vec::new(),
            zebra: !p.sectional,
        };
    }
    if p.sectional {
        let avail = (y1 - y0 - p.blocks.len() as f32 * HEADER_H as f32).max(0.0);
        let row_h = (avail / p.n_term as f32).max(min_row_h);
        let mut cys = Vec::with_capacity(p.n_term);
        let mut headers = Vec::with_capacity(p.blocks.len());
        let mut topf = y0;
        for b in &p.blocks {
            let hy0 = topf.round() as i32;
            topf += HEADER_H as f32;
            headers.push((hy0, topf.round() as i32, b.cat.clone()));
            for r in 0..b.rows.len() {
                cys.push((topf + (r as f32 + 0.5) * row_h).round() as i32);
            }
            topf += b.rows.len() as f32 * row_h;
        }
        Plan {
            cys,
            row_h,
            content_bot: y1,
            headers,
            zebra: false,
        }
    } else {
        let row_h = ((y1 - y0) / p.n_term as f32).clamp(min_row_h, MAX_ROW_H);
        let cys: Vec<i32> = (0..p.n_term)
            .map(|r| (y0 + (r as f32 + 0.5) * row_h).round() as i32)
            .collect();
        let content_bot = (y0 + p.n_term as f32 * row_h).min(y1);
        Plan {
            cys,
            row_h,
            content_bot,
            headers: Vec::new(),
            zebra: true,
        }
    }
}

fn label_zone_w<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    style: &TextStyle<'static>,
    labels: &[String],
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
    (max_tw + 16).clamp(70, 560)
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
    max_count: f32,
    max_r: i32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (w, _h) = area.dim_in_pixel();
    let cx = w as i32 / 2;
    let step = 22i32;
    let color = sig_colormap(0.85);
    let style = ("sans-serif", 10.0).into_font().color(&BLACK);
    area.draw_text("count", &style, (cx - 4, 4))?;
    for (k, f) in [1.0f32, 0.8, 0.6, 0.4, 0.2].iter().enumerate() {
        let r = (max_r as f32 * f.sqrt()).round() as i32;
        let cy = 26 + k as i32 * step;
        area.draw(&Circle::new((cx, cy), r.max(1), color.filled()))?;
        let lbl = fmt_short(max_count * f);
        area.draw_text(&lbl, &style, (cx + max_r + 3, cy - 5))?;
    }
    Ok(())
}

fn draw_right_legend<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    content_h: f32,
    max_sig: f32,
    max_count: f32,
    max_r: i32,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let (_rw, rh) = area.dim_in_pixel();
    let rh = rh as i32;

    let legend_h = SIZE_LEGEND_H as i32;
    let cb_block = (18 + content_h.round().clamp(90.0, 320.0) as i32).min((rh - legend_h).max(40));
    let (cb_area, size_area) = area.split_vertically(cb_block);
    let (cap_area, cb_body) = cb_area.split_vertically(16);
    let cap_style = ("sans-serif", 11.0).into_font().color(&BLACK);
    cap_area.draw_text("-log10(p.adj)", &cap_style, (4, 2))?;
    draw_colorbar(&cb_body, 0.0, max_sig, sig_colormap)?;
    draw_size_legend(&size_area, max_count, max_r)?;
    Ok(())
}

fn render_dotplot<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    p: &Prepared,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let root = with_title(root, title)?;
    let (w, _h) = root.dim_in_pixel();
    let w = w as i32;
    if p.n_term == 0 {
        root.present()?;
        return Ok(());
    }

    let (body_area, right_area) = root.split_horizontally((w - DOT_RIGHT_W).max(1));
    let (bw, bh) = body_area.dim_in_pixel();
    let (bw, bh) = (bw as i32, bh as i32);

    let lb_style = right_center(11.0);
    let labels: Vec<String> = p
        .blocks
        .iter()
        .flat_map(|b| b.rows.iter().map(|r| r.label.clone()))
        .collect();
    let label_w = label_zone_w(&body_area, &lb_style, &labels);
    let x0 = label_w;
    let x1 = (bw - 10).max(x0 + 1);
    let y0 = 4.0f32;
    let y1 = (bh - AXIS_H) as f32;

    let lay = plan_layout(p, y0, y1, 12.0);
    let row_h = lay.row_h;
    let max_r = (row_h * 0.5 - 2.0).clamp(4.0, 12.0) as i32;

    let axis_y = if lay.zebra {
        (lay.content_bot + 4.0).round() as i32
    } else {
        y1 as i32
    };

    let max_gr = p
        .blocks
        .iter()
        .flat_map(|b| b.rows.iter())
        .fold(0.0f32, |m, r| m.max(r.gr))
        .max(1e-9);
    let step = nice_step(max_gr, 5);
    let axis_max = (max_gr / step).ceil() * step;
    let ax = |v: f32| x0 + ((x1 - x0) as f32 * (v / axis_max)).round() as i32;
    let n_ticks = (axis_max / step).round() as usize;

    if lay.zebra {
        for (i, cy) in lay.cys.iter().enumerate() {
            if i % 2 == 1 {
                let ya = (*cy as f32 - row_h * 0.5).round() as i32;
                let yb = (*cy as f32 + row_h * 0.5).round() as i32;
                body_area.draw(&Rectangle::new(
                    [(0, ya.max(y0 as i32)), (x1, yb.min(y1 as i32))],
                    BAND.filled(),
                ))?;
            }
        }
    }

    let grid_style = RGBColor(0xea, 0xea, 0xea).stroke_width(1);
    let grid_bot = lay.content_bot.round() as i32;
    for i in 0..=n_ticks {
        let px = ax(i as f32 * step);
        body_area.draw(&PathElement::new(
            vec![(px, y0 as i32), (px, grid_bot)],
            grid_style,
        ))?;
    }

    let header_text = left_center(12.0);
    for (hy0, hy1, cat) in &lay.headers {
        body_area.draw(&Rectangle::new(
            [(0, *hy0), (x1, *hy1)],
            HEADER_BAND.filled(),
        ))?;
        body_area.draw_text(
            &format!("[{cat}]"),
            &header_text,
            (x0 + 6, (*hy0 + *hy1) / 2),
        )?;
    }

    let term_style = right_center(11.0);
    let mut i = 0usize;
    for b in p.blocks.iter() {
        for r in b.rows.iter() {
            let cy = lay.cys[i];
            i += 1;
            body_area.draw_text(&r.label, &term_style, (x0 - 6, cy))?;
            let t = (r.sig / p.max_sig).clamp(0.0, 1.0);
            let rr = ((max_r as f32) * (r.count / p.max_count).sqrt())
                .round()
                .max(1.0) as i32;
            let cx = ax(r.gr);
            body_area.draw(&Circle::new((cx, cy), rr, sig_colormap(t).filled()))?;
        }
    }

    if !lay.zebra {
        for (k, (hy0, _, _)) in lay.headers.iter().enumerate() {
            if k > 0 {
                body_area.draw(&PathElement::new(
                    vec![(x0, *hy0), (x1, *hy0)],
                    RGBColor(0xcc, 0xcc, 0xcc).stroke_width(1),
                ))?;
            }
        }
    }

    let ax_style = center_mid(11.0);
    for i in 0..=n_ticks {
        let px = ax(i as f32 * step);
        body_area.draw_text(
            &fmt_tick(i as f32 * step, step),
            &ax_style,
            (px, axis_y + 6),
        )?;
    }
    let axis_lab = ("sans-serif", 12.0).into_font().color(&BLACK);
    body_area.draw_text("GeneRatio", &axis_lab, ((x0 + x1) / 2 - 22, axis_y + 16))?;

    draw_right_legend(
        &right_area,
        lay.content_bot - y0,
        p.max_sig,
        p.max_count,
        max_r,
    )?;

    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_ora_dotplot(
    sd: &SoData,
    db_name: &str,
    path: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let enrich = extract_enrichment(sd)?;
    let p = prepare(enrich, db_name, cluster_id, top_n)?;
    let size = auto_dotplot_size(size, &p);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_dotplot(root, &p, title)
        .map_err(|e| anyhow!("failed to draw the enrichment dotplot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_ora_dotplot_png(
    sd: &SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let enrich = extract_enrichment(sd)?;
    let p = prepare(enrich, db_name, cluster_id, top_n)?;
    let size = auto_dotplot_size(size, &p);
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_dotplot(root, &p, title)
            .map_err(|e| anyhow!("failed to draw the enrichment dotplot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn fit_rows_size(size: (u32, u32), p: &Prepared, per_row: usize, with_legend: bool) -> (u32, u32) {
    let (w, _) = size;
    let headers = if p.sectional {
        p.blocks.len() * HEADER_H as usize
    } else {
        0
    };
    let rows_need = TITLE_BAND + 4 + headers + p.n_term * per_row + AXIS_H as usize + 8;
    if !with_legend {
        return (w, rows_need as u32);
    }

    let cbar_h = 18 + clamp_content_h(p.n_term * per_row);
    let legend_need = TITLE_BAND + 4 + cbar_h + SIZE_LEGEND_H + 8;
    (w, rows_need.max(legend_need) as u32)
}

fn clamp_content_h(content_h: usize) -> usize {
    content_h.clamp(90, 320)
}

fn auto_dotplot_size(size: (u32, u32), p: &Prepared) -> (u32, u32) {
    fit_rows_size(size, p, 18, true)
}

fn render_barplot<DB: DrawingBackend>(
    root: DrawingArea<DB, Shift>,
    p: &Prepared,
    title: Option<&str>,
) -> Result<(), DrawingAreaErrorKind<DB::ErrorType>> {
    let root = with_title(root, title)?;
    let (w, h) = root.dim_in_pixel();
    let (w, h) = (w as i32, h as i32);
    if p.n_term == 0 {
        root.present()?;
        return Ok(());
    }

    let lb_style = right_center(11.0);
    let labels: Vec<String> = p
        .blocks
        .iter()
        .flat_map(|b| b.rows.iter().map(|r| r.label.clone()))
        .collect();
    let label_w = label_zone_w(&root, &lb_style, &labels);
    let x0 = label_w;
    let x1 = (w - 14).max(x0 + 1);
    let y0 = 4.0f32;
    let y1 = (h - AXIS_H) as f32;

    let lay = plan_layout(p, y0, y1, 14.0);
    let row_h = lay.row_h;
    let axis_y = if lay.zebra {
        (lay.content_bot + 2.0).round() as i32
    } else {
        y1 as i32
    };

    let max_v = p
        .blocks
        .iter()
        .flat_map(|b| b.rows.iter())
        .fold(0.0f32, |m, r| m.max(r.count))
        .max(1.0);
    let step = nice_step(max_v, 5);
    let axis_max = (max_v / step).ceil() * step;
    let ax = |v: f32| x0 + ((x1 - x0) as f32 * (v / axis_max)).round() as i32;
    let n_ticks = (axis_max / step).round() as usize;

    if lay.zebra {
        for (i, cy) in lay.cys.iter().enumerate() {
            if i % 2 == 1 {
                let ya = (*cy as f32 - row_h * 0.5).round() as i32;
                let yb = (*cy as f32 + row_h * 0.5).round() as i32;
                root.draw(&Rectangle::new(
                    [(0, ya.max(0)), (x1, yb.min(h))],
                    BAND.filled(),
                ))?;
            }
        }
    }

    let grid_style = RGBColor(0xea, 0xea, 0xea).stroke_width(1);
    let grid_bot = lay.content_bot.round() as i32;
    for i in 0..=n_ticks {
        let px = ax(i as f32 * step);
        root.draw(&PathElement::new(
            vec![(px, y0 as i32), (px, grid_bot)],
            grid_style,
        ))?;
    }
    root.draw(&PathElement::new(
        vec![(x0, axis_y), (x1, axis_y)],
        RGBColor(0x88, 0x88, 0x88).stroke_width(1),
    ))?;
    let ax_style = center_mid(11.0);
    for i in 0..=n_ticks {
        let px = ax(i as f32 * step);
        root.draw_text(
            &fmt_tick(i as f32 * step, step),
            &ax_style,
            (px, axis_y + 6),
        )?;
    }

    let header_text = left_center(12.0);
    for (hy0, hy1, cat) in &lay.headers {
        root.draw(&Rectangle::new(
            [(0, *hy0), (x1, *hy1)],
            HEADER_BAND.filled(),
        ))?;
        root.draw_text(
            &format!("[{cat}]"),
            &header_text,
            (x0 + 6, (*hy0 + *hy1) / 2),
        )?;
    }

    let label_style = right_center(11.0);
    let mut i = 0usize;
    for b in p.blocks.iter() {
        for r in b.rows.iter() {
            let cy = lay.cys[i];
            i += 1;
            root.draw_text(&r.label, &label_style, (x0 - 6, cy))?;
            if r.count > 0.0 {
                let px = ax(r.count).max(x0);
                let t = (r.sig / p.max_sig).clamp(0.0, 1.0);
                let y_top = (cy as f32 - row_h * 0.5).round() as i32;
                let y_bot = (cy as f32 + row_h * 0.5).round() as i32 - 1;
                root.draw(&Rectangle::new(
                    [(x0, y_top), (px, y_bot.max(y_top))],
                    sig_colormap(t).filled(),
                ))?;
            }
        }
    }

    if !lay.zebra {
        for (k, (hy0, _, _)) in lay.headers.iter().enumerate() {
            if k > 0 {
                root.draw(&PathElement::new(
                    vec![(x0, *hy0), (x1, *hy0)],
                    RGBColor(0xcc, 0xcc, 0xcc).stroke_width(1),
                ))?;
            }
        }
    }
    let axis_lab = ("sans-serif", 12.0).into_font().color(&BLACK);
    root.draw_text("count", &axis_lab, ((x0 + x1) / 2 - 14, axis_y + 16))?;

    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn plot_ora_barplot(
    sd: &SoData,
    db_name: &str,
    path: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<()> {
    let enrich = extract_enrichment(sd)?;
    let p = prepare(enrich, db_name, cluster_id, top_n)?;
    let size = auto_barplot_size(size, &p);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    render_barplot(root, &p, title)
        .map_err(|e| anyhow!("failed to draw the enrichment barplot: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn plot_ora_barplot_png(
    sd: &SoData,
    db_name: &str,
    cluster_id: Option<i64>,
    top_n: usize,
    title: Option<&str>,
    size: (u32, u32),
) -> anyhow::Result<Vec<u8>> {
    let enrich = extract_enrichment(sd)?;
    let p = prepare(enrich, db_name, cluster_id, top_n)?;
    let size = auto_barplot_size(size, &p);
    let tmp = tempfile::Builder::new().suffix(".png").tempfile()?;
    let tmp_path = tmp.into_temp_path();
    let path = tmp_path.to_path_buf();
    {
        let root = BitMapBackend::new(&path, size).into_drawing_area();
        render_barplot(root, &p, title)
            .map_err(|e| anyhow!("failed to draw the enrichment barplot: {e}"))?;
    }
    Ok(std::fs::read(&path)?)
}

fn auto_barplot_size(size: (u32, u32), p: &Prepared) -> (u32, u32) {
    fit_rows_size(size, p, 20, false)
}
