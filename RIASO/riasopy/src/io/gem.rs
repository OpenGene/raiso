use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use flate2::read::MultiGzDecoder;
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use polars::prelude::*;
use rayon::prelude::*;
use serde_json::Value;

use crate::core::{ExpMatrix, SoData};
use crate::parallel::csr_to_dense;

const CHIP_RESOLUTION: [(&str, i64); 26] = [
    ("CL1", 900),
    ("N1", 900),
    ("V3", 715),
    ("K2", 715),
    ("S2", 715),
    ("S1", 900),
    ("F3", 715),
    ("F1", 800),
    ("V1", 800),
    ("DP84", 715),
    ("DP8", 850),
    ("FP2", 500),
    ("SS2", 500),
    ("FP1", 600),
    ("E1", 700),
    ("DP40", 700),
    ("G1", 700),
    ("A", 500),
    ("B", 500),
    ("C", 500),
    ("D", 500),
    ("U", 715),
    ("V", 715),
    ("W", 715),
    ("X", 715),
    ("Y", 500),
];

#[derive(Debug, Clone)]
struct GemHeader {
    gene_id_col: usize,
    gene_name_col: Option<usize>,
    x_col: usize,
    y_col: usize,
    count_col: usize,

    exon_col: Option<usize>,
}

struct GemRow<'a> {
    gene_id: &'a [u8],
    gene_name: &'a [u8],
    x: u32,
    y: u32,
    count: u32,
    exon_count: u32,
}

pub fn read_gem(
    path: &str,
    bin_size: u32,
    is_sparse: bool,
    dedup_symbols: bool,
    save_counts_layer: bool,
    save_exon_layer: bool,
) -> Result<SoData> {
    crate::parallel::init_rayon(0);
    if bin_size == 0 {
        bail!("read_gem: bin_size must be >= 1");
    }
    let h = read_gem_header_or_bail(path)?;

    let t0 = std::time::Instant::now();
    let _t = || {
        if std::env::var("RIASO_TIMING").is_ok() {
            println!("  [read_gem] +{:?}", t0.elapsed());
        }
    };

    let mut gene_lookup: HashMap<Vec<u8>, u32> = HashMap::new();
    let mut gene_table: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut bins: HashSet<u64> = HashSet::new();
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut n_rows = 0usize;
    for_each_row(path, &h, |row| {
        let (bx, by) = bin_of(row.x, row.y, bin_size);
        bins.insert(pack_bin(bx, by));
        if !gene_lookup.contains_key(row.gene_id) {
            gene_lookup.insert(row.gene_id.to_vec(), gene_table.len() as u32);
            gene_table.push((row.gene_id.to_vec(), row.gene_name.to_vec()));
        }
        min_x = min_x.min(row.x);
        min_y = min_y.min(row.y);
        max_x = max_x.max(row.x);
        max_y = max_y.max(row.y);
        n_rows += 1;
    })?;
    if n_rows == 0 {
        bail!("read_gem: the file has no data row ({path})");
    }
    _t();

    let n_genes_file = gene_table.len();
    let mut order: Vec<u32> = (0..n_genes_file as u32).collect();
    order.par_sort_unstable_by(|&a, &b| gene_table[a as usize].0.cmp(&gene_table[b as usize].0));
    let mut file_to_sorted = vec![0u32; n_genes_file];
    for (s, &f) in order.iter().enumerate() {
        file_to_sorted[f as usize] = s as u32;
    }

    let sorted: Vec<(&[u8], &[u8])> = order
        .iter()
        .map(|&i| {
            let (id, name) = &gene_table[i as usize];
            (id.as_slice(), name.as_slice())
        })
        .collect();
    let (final_genes, sorted_to_final): (Vec<(&[u8], &[u8])>, Vec<u32>) = if dedup_symbols {
        let mut seen: HashMap<&[u8], u32> = HashMap::new();
        let mut tab: Vec<(&[u8], &[u8])> = Vec::new();
        let mut map = vec![0u32; sorted.len()];
        for (s, &(id, name)) in sorted.iter().enumerate() {
            let idx = *seen.entry(name).or_insert_with(|| {
                tab.push((id, name));
                (tab.len() - 1) as u32
            });
            map[s] = idx;
        }
        (tab, map)
    } else {
        (sorted, (0..n_genes_file as u32).collect())
    };
    let n_genes = final_genes.len();

    let file_to_final: Vec<u32> = (0..n_genes_file)
        .map(|i| sorted_to_final[file_to_sorted[i] as usize])
        .collect();
    _t();

    let mut sorted_bins: Vec<u64> = bins.into_iter().collect();
    sorted_bins.par_sort_unstable();
    let n_cells = sorted_bins.len();
    let bin_row: HashMap<u64, u32> = sorted_bins
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i as u32))
        .collect();

    let mut counts: HashMap<u64, u32> = HashMap::with_capacity(n_cells * 8);

    let mut exon_counts: Option<HashMap<u64, u32>> = if save_exon_layer && h.exon_col.is_some() {
        Some(HashMap::with_capacity(n_cells * 8))
    } else {
        None
    };
    for_each_row(path, &h, |row| {
        let g = file_to_final[gene_lookup[row.gene_id] as usize] as u64;
        let (bx, by) = bin_of(row.x, row.y, bin_size);
        let r = bin_row[&pack_bin(bx, by)] as u64;
        let key = (r << 32) | g;
        *counts.entry(key).or_insert(0) += row.count;
        if let Some(ec) = exon_counts.as_mut() {
            *ec.entry(key).or_insert(0) += row.exon_count;
        }
    })?;
    _t();

    let mut keys: Vec<u64> = counts.keys().copied().collect();
    keys.par_sort_unstable();
    let nnz = keys.len();
    let mut row_offsets = vec![0usize; n_cells + 1];
    for &k in &keys {
        row_offsets[(k >> 32) as usize + 1] += 1;
    }
    for i in 0..n_cells {
        row_offsets[i + 1] += row_offsets[i];
    }
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![0f32; nnz];
    let mut max_exp = 0u32;
    for (i, &k) in keys.iter().enumerate() {
        let c = counts[&k];
        max_exp = max_exp.max(c);
        col_indices[i] = (k & 0xFFFF_FFFF) as usize;
        values[i] = c as f32;
    }
    drop(counts);

    let exon_csr = exon_counts.as_ref().map(|ec| {
        let ev: Vec<f32> = keys
            .iter()
            .map(|k| ec.get(k).copied().unwrap_or(0) as f32)
            .collect();
        CsrMatrix::try_from_csr_data(
            n_cells,
            n_genes,
            row_offsets.clone(),
            col_indices.clone(),
            ev,
        )
    });
    let exon_csr = match exon_csr {
        Some(c) => {
            Some(c.map_err(|e| anyhow::anyhow!("read_gem: failed to build the exon CSR: {e:?}"))?)
        }
        None => None,
    };
    let csr = CsrMatrix::try_from_csr_data(n_cells, n_genes, row_offsets, col_indices, values)
        .map_err(|e| anyhow::anyhow!("read_gem: failed to build the CSR: {e:?}"))?;
    _t();

    let mut data = SoData::new();
    data.exp_matrix = Some(if is_sparse {
        ExpMatrix::Sparse(csr)
    } else {
        ExpMatrix::Dense(csr_to_dense(&csr))
    });

    let gene_ids: Vec<String> = final_genes
        .iter()
        .map(|(id, _)| String::from_utf8_lossy(id).into_owned())
        .collect();
    let gene_names: Vec<String> = final_genes
        .iter()
        .map(|(_, name)| String::from_utf8_lossy(name).into_owned())
        .collect();
    data.var = Some(
        DataFrame::new(vec![
            Column::new("gene".into(), gene_ids),
            Column::new("gene_name".into(), gene_names),
        ])
        .context("read_gem: failed to build the var DataFrame")?,
    );

    let cell_names: Vec<String> = sorted_bins
        .iter()
        .map(|&b| format!("{}_{}", bin_x(b), bin_y(b)))
        .collect();
    data.obs = Some(
        DataFrame::new(vec![Column::new("cell".into(), cell_names)])
            .context("read_gem: failed to build the obs DataFrame")?,
    );

    let mut pos_flat = Vec::with_capacity(n_cells * 2);
    for &b in &sorted_bins {
        pos_flat.push(bin_x(b) as f32);
        pos_flat.push(bin_y(b) as f32);
    }
    data.position = Some(
        Array2::from_shape_vec((n_cells, 2), pos_flat)
            .context("read_gem: failed to build position")?,
    );

    let sn = match read_header_sn(path) {
        Some(s) => Some(s),
        None => sn_from_path(path),
    };
    data.file_path = Some(PathBuf::from(path));
    data.file_format = Some("gem".into());
    data.samples_names = sn.iter().cloned().collect();

    let mut uns = HashMap::new();
    uns.insert(
        "resolution".into(),
        Value::from(resolution_of(sn.as_deref())),
    );
    uns.insert("minX".into(), Value::from(min_x));
    uns.insert("minY".into(), Value::from(min_y));
    uns.insert("maxX".into(), Value::from(max_x));
    uns.insert("maxY".into(), Value::from(max_y));
    uns.insert("maxExp".into(), Value::from(max_exp));
    uns.insert("n_rows".into(), Value::from(n_rows as u64));
    data.uns = uns;

    if save_counts_layer {
        if let Some(ref exp) = data.exp_matrix {
            data.layers.insert("counts".into(), exp.clone());
        }
    }

    if let Some(exon_csr) = exon_csr {
        data.layers
            .insert("exon".into(), ExpMatrix::Sparse(exon_csr));
    }

    Ok(data)
}

#[inline]
fn bin_of(x: u32, y: u32, bin_size: u32) -> (u32, u32) {
    ((x / bin_size) * bin_size, (y / bin_size) * bin_size)
}

#[inline]
fn pack_bin(bx: u32, by: u32) -> u64 {
    ((bx as u64) << 32) | by as u64
}

#[inline]
fn bin_x(b: u64) -> u32 {
    (b >> 32) as u32
}

#[inline]
fn bin_y(b: u64) -> u32 {
    (b & 0xFFFF_FFFF) as u32
}

fn resolution_of(sn: Option<&str>) -> i64 {
    let sn = match sn {
        Some(s) => s,
        None => return 500,
    };
    for (prefix, res) in CHIP_RESOLUTION {
        if sn.starts_with(prefix) {
            return res;
        }
    }
    500
}

fn sn_from_path(path: &str) -> Option<String> {
    let base = std::path::Path::new(path).file_name()?.to_str()?;
    let stem = base.strip_suffix(".gz").unwrap_or(base);
    let stem = stem.strip_suffix(".gem").unwrap_or(stem);
    let sn = stem.split('.').next().unwrap_or(stem);
    (!sn.is_empty()).then(|| sn.to_string())
}

fn open_lines(path: &str) -> Result<Box<dyn BufRead>> {
    let mut f = File::open(path).with_context(|| format!("read_gem: cannot open {path}"))?;
    let mut magic = [0u8; 2];
    let n = f.read(&mut magic)?;
    f.rewind()?;
    let gz = n == 2 && magic == [0x1f, 0x8b];
    let r: Box<dyn BufRead> = if gz {
        Box::new(BufReader::with_capacity(1 << 20, MultiGzDecoder::new(f)))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, f))
    };
    Ok(r)
}

fn trim_eol(line: &[u8]) -> &[u8] {
    let mut s = line;
    while let Some((&last, rest)) = s.split_last() {
        if last == b'\n' || last == b'\r' {
            s = rest;
        } else {
            break;
        }
    }
    s
}

fn eq_ignore_ascii_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.eq_ignore_ascii_case(y))
}

fn column_of(cols: &[&[u8]], names: &[&str]) -> Option<usize> {
    names.iter().find_map(|n| {
        cols.iter()
            .position(|c| eq_ignore_ascii_case(c, n.as_bytes()))
    })
}

fn read_gem_header_or_bail(path: &str) -> Result<GemHeader> {
    let mut r = open_lines(path)?;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        if r.read_until(b'\n', &mut buf)? == 0 {
            bail!("read_gem: no header row found in {path} (all `#` comments or an empty file)");
        }
        let line = trim_eol(&buf);
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        let cols: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
        let gene_id_col = column_of(&cols, &["geneID", "gene"]).ok_or_else(|| {
            anyhow::anyhow!(
                "read_gem: header is missing the geneID/gene column: {:?}",
                String::from_utf8_lossy(line)
            )
        })?;
        let x_col = column_of(&cols, &["x"]).ok_or_else(|| {
            anyhow::anyhow!(
                "read_gem: header is missing the x column: {:?}",
                String::from_utf8_lossy(line)
            )
        })?;
        let y_col = column_of(&cols, &["y"]).ok_or_else(|| {
            anyhow::anyhow!(
                "read_gem: header is missing the y column: {:?}",
                String::from_utf8_lossy(line)
            )
        })?;
        let count_col = column_of(&cols, &["MIDCount", "MIDCounts", "UMICount", "UMI"])
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "read_gem: header is missing the MIDCount/UMICount column: {:?}",
                    String::from_utf8_lossy(line)
                )
            })?;
        if column_of(&cols, &["CellID", "label"]).is_some() {
            bail!("read_gem: this GEM is cell-bin (it has CellID/label columns); only square bins are currently supported");
        }
        return Ok(GemHeader {
            gene_id_col,
            gene_name_col: column_of(&cols, &["geneName"]),
            x_col,
            y_col,
            count_col,
            exon_col: column_of(&cols, &["ExonCount"]),
        });
    }
}

fn read_header_sn(path: &str) -> Option<String> {
    let mut r = open_lines(path).ok()?;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        if r.read_until(b'\n', &mut buf).ok()? == 0 {
            return None;
        }
        let line = trim_eol(&buf);
        if line.first() != Some(&b'#') {
            return None;
        }
        let body = &line[1..];
        if let Some(eq) = body.iter().position(|&b| b == b'=') {
            let key = trim_eol(&body[..eq]);
            if eq_ignore_ascii_case(key, b"Stereo-seqChip") {
                let val = String::from_utf8_lossy(trim_eol(&body[eq + 1..]))
                    .trim()
                    .to_string();
                return (!val.is_empty()).then_some(val);
            }
        }
    }
}

fn parse_u32(f: &[u8]) -> Option<u32> {
    if f.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &b in f {
        if !b.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (b - b'0') as u64;
        if v > u32::MAX as u64 {
            return None;
        }
    }
    Some(v as u32)
}

fn parse_row<'a>(line: &'a [u8], h: &GemHeader) -> Option<GemRow<'a>> {
    let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
    let max_col = h
        .count_col
        .max(h.x_col)
        .max(h.y_col)
        .max(h.gene_name_col.unwrap_or(0))
        .max(h.gene_id_col);
    if fields.len() <= max_col {
        return None;
    }
    Some(GemRow {
        gene_id: fields[h.gene_id_col],
        gene_name: h
            .gene_name_col
            .map(|c| fields[c])
            .unwrap_or(fields[h.gene_id_col]),
        x: parse_u32(fields[h.x_col])?,
        y: parse_u32(fields[h.y_col])?,
        count: parse_u32(fields[h.count_col])?,

        exon_count: h
            .exon_col
            .and_then(|c| fields.get(c))
            .and_then(|f| parse_u32(f))
            .unwrap_or(0),
    })
}

fn for_each_row<F>(path: &str, h: &GemHeader, mut cb: F) -> Result<()>
where
    F: FnMut(GemRow<'_>),
{
    let mut r = open_lines(path)?;
    let mut buf: Vec<u8> = Vec::new();
    let mut header_seen = false;
    let mut lineno = 0usize;
    let mut n_skipped = 0usize;
    loop {
        buf.clear();
        if r.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        lineno += 1;
        let line = trim_eol(&buf);
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        if !header_seen {
            header_seen = true;
            continue;
        }
        match parse_row(line, h) {
            Some(row) => cb(row),
            None => n_skipped += 1,
        }
    }
    if n_skipped > 0 {
        eprintln!("[read_gem] skipped {n_skipped} rows (missing fields or unparsable values) out of {lineno} rows");
    }
    Ok(())
}
