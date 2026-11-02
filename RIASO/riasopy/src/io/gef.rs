use std::collections::HashMap;
use std::path::PathBuf;

use rayon::prelude::*;

use anyhow::{anyhow, bail, Context, Result};
use hdf5::types::{FixedAscii, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{Dataset, File, Group, Location};
use nalgebra_sparse::csr::CsrMatrix;
use ndarray::Array2;
use polars::prelude::*;
use serde_json::Value;

use crate::core::{ExpMatrix, SoData};
use crate::parallel::{csr_to_dense, SendPtr};

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct ExprU16 {
    x: i32,
    y: i32,
    count: u16,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct ExprU8 {
    x: i32,
    y: i32,
    count: u8,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct GeneV4 {
    #[hdf5(rename = "geneID")]
    gene_id: FixedAscii<64>,
    #[hdf5(rename = "geneName")]
    gene_name: FixedAscii<64>,
    offset: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct GeneV3 {
    #[hdf5(rename = "geneName")]
    gene_name: FixedAscii<64>,
    offset: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct GeneV2 {
    #[hdf5(rename = "gene")]
    gene_name: FixedAscii<64>,
    offset: u32,
    count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct CellExpRec {
    #[hdf5(rename = "geneID")]
    gene_id: u32,
    count: u16,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct CellBinCell {
    #[hdf5(rename = "id")]
    id: u32,
    #[hdf5(rename = "x")]
    x: i32,
    #[hdf5(rename = "y")]
    y: i32,
    #[hdf5(rename = "offset")]
    offset: u32,
    #[hdf5(rename = "geneCount")]
    gene_count: u16,
    #[hdf5(rename = "expCount")]
    _exp_count: u16,
    #[hdf5(rename = "dnbCount")]
    dnb_count: u16,
    #[hdf5(rename = "area")]
    area: u16,
    #[hdf5(rename = "cellTypeID")]
    _cell_type_id: u16,
    #[hdf5(rename = "clusterID")]
    _cluster_id: u16,
}

#[repr(C)]
#[derive(Clone, Copy, hdf5::H5Type)]
struct CellGene {
    #[hdf5(rename = "geneID")]
    gene_id: FixedAscii<64>,
    #[hdf5(rename = "geneName")]
    gene_name: FixedAscii<64>,
    offset: u32,
    #[hdf5(rename = "cellCount")]
    _cell_count: u32,
    #[hdf5(rename = "expCount")]
    _exp_count: u32,
    #[hdf5(rename = "maxMIDcount")]
    _max_mid_count: u16,
}

pub fn read_gef(
    path: &str,
    bin_type: &str,
    bin_size: u32,
    is_sparse: bool,
    dedup_symbols: bool,
    save_counts_layer: bool,
    save_exon_layer: bool,
) -> Result<SoData> {
    crate::parallel::init_rayon(0);
    let file = File::open(path)?;

    if bin_type == "cell_bins" || is_cellbin_file(&file) {
        return read_gef_cellbin(
            &file,
            path,
            is_sparse,
            dedup_symbols,
            save_counts_layer,
            save_exon_layer,
        );
    }
    if bin_type != "bins" {
        bail!(
            "read_gef: only bin_type=\"bins\" (square bin) / \"cell_bins\" (cell bin) are supported, got {bin_type:?}"
        );
    }

    let gene_exp = file
        .group("geneExp")
        .context("read_gef: missing the geneExp group (not a square-bin GEF?)")?;
    let bin_name = format!("bin{bin_size}");
    if !gene_exp.link_exists(&bin_name) {
        let available: Vec<String> = gene_exp.member_names().unwrap_or_default();
        bail!("read_gef: missing the geneExp/{bin_name} bin level; available bin levels: {available:?}");
    }
    let bin_group = gene_exp.group(&bin_name)?;

    let t0 = std::time::Instant::now();
    let _t = || {
        if std::env::var("RIASO_TIMING").is_ok() {
            println!("  [read_gef] +{:?}", t0.elapsed());
        }
    };

    let mut genes = read_gene_table(&bin_group)?;
    genes.sort_by(|a, b| a.key().cmp(b.key()));
    _t();

    let expr_ds = bin_group
        .dataset("expression")
        .context("read_gef: missing the expression dataset")?;
    let (min_x, min_y, max_x, max_y, max_exp, resolution) = read_expression_attrs(&expr_ds)?;
    let count_u1 = expression_count_u1(&expr_ds)?;
    let expr_ds_path = format!("geneExp/{bin_name}/expression");

    let expr_full = read_expression(path, &expr_ds_path, expr_ds.size(), count_u1)?;

    let exon_full = if save_exon_layer && bin_group.link_exists("exon") {
        let exon_ds_path = format!("geneExp/{bin_name}/exon");
        Some(read_exon(path, &exon_ds_path, expr_ds.size())?)
    } else {
        None
    };
    _t();

    let (genes, expr, exon) = if dedup_symbols {
        merge_duplicate_symbols(genes, expr_full, exon_full)
    } else {
        (genes, expr_full, exon_full)
    };
    let n_genes = genes.len();
    _t();

    let n_threads = rayon::current_num_threads();
    let chunk = expr.len() / n_threads;
    let mut shards: Vec<HashMap<u64, u32>> = (0..n_threads).map(|_| HashMap::new()).collect();
    shards.par_iter_mut().enumerate().for_each(|(tid, shard)| {
        let r0 = tid * chunk;
        let r1 = if tid == n_threads - 1 {
            expr.len()
        } else {
            (tid + 1) * chunk
        };
        for &packed in &expr[r0..r1] {
            *shard.entry(packed >> 16).or_insert(0) += 1;
        }
    });
    let mut bin_counts: HashMap<u64, u32> =
        HashMap::with_capacity(shards.iter().map(|s| s.len()).sum());
    for shard in shards {
        for (k, v) in shard {
            *bin_counts.entry(k).or_insert(0) += v;
        }
    }
    let mut bins: Vec<u64> = bin_counts.keys().copied().collect();
    bins.sort_unstable();
    let n_cells = bins.len();
    let mut bin_index: HashMap<u64, usize> = HashMap::with_capacity(n_cells);
    let mut row_offsets = vec![0usize; n_cells + 1];
    for (i, &b) in bins.iter().enumerate() {
        bin_index.insert(b, i);
        row_offsets[i + 1] = row_offsets[i] + bin_counts[&b] as usize;
    }
    let nnz = row_offsets[n_cells];
    _t();

    const PAR_FILL_MIN: usize = 1024;
    const FILL_NTHREADS: usize = 4;
    let fill_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(FILL_NTHREADS)
        .build()
        .context("read_gef: failed to build the CSR fill thread pool")?;
    let mut cursor = row_offsets[..n_cells].to_vec();
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![0f64; nnz];

    let mut exon_values: Option<Vec<f64>> = exon.as_ref().map(|_| vec![0f64; nnz]);
    let bin_index = &bin_index;
    let cursor_ptr = SendPtr(cursor.as_mut_ptr());
    let col_ptr = SendPtr(col_indices.as_mut_ptr());
    let val_ptr = SendPtr(values.as_mut_ptr());
    let exon_ptr = exon_values.as_mut().map(|v| SendPtr(v.as_mut_ptr()));
    for (g, gene) in genes.iter().enumerate() {
        let s = gene.offset as usize;
        let e = s + gene.count as usize;
        if e > expr.len() {
            bail!(
                "read_gef: gene #{} offset/count out of range (offset={}, count={}, expression length={})",
                g,
                gene.offset,
                gene.count,
                expr.len()
            );
        }
        let run = &expr[s..e];
        let exon_run = exon.as_ref().map(|ex| &ex[s..e]);
        if run.len() < PAR_FILL_MIN {
            for (j, &packed) in run.iter().enumerate() {
                let cell = bin_index[&(packed >> 16)];
                let pos = cursor[cell];
                col_indices[pos] = g;
                values[pos] = (packed & 0xFFFF) as f64;
                if let (Some(ep), Some(er)) = (exon_ptr.as_ref(), exon_run) {
                    unsafe {
                        *ep.add(pos) = er[j];
                    }
                }
                cursor[cell] += 1;
            }
        } else {
            fill_pool.install(|| {
                run.par_iter().enumerate().for_each(|(j, &packed)| {
                    let cell = bin_index[&(packed >> 16)];
                    let pos = unsafe { *cursor_ptr.add(cell) };
                    unsafe {
                        *col_ptr.add(pos) = g;
                        *val_ptr.add(pos) = (packed & 0xFFFF) as f64;
                        *cursor_ptr.add(cell) = pos + 1;
                    }
                    if let (Some(ep), Some(er)) = (exon_ptr.as_ref(), exon_run) {
                        unsafe {
                            *ep.add(pos) = er[j];
                        }
                    }
                });
            });
        }
    }

    let exon_csr = match exon_values {
        Some(ev) => Some(
            CsrMatrix::try_from_csr_data(
                n_cells,
                n_genes,
                row_offsets.clone(),
                col_indices.clone(),
                ev.into_iter().map(|v| v as f32).collect(),
            )
            .map_err(|err| anyhow!("read_gef: failed to build the exon CSR: {err:?}"))?,
        ),
        None => None,
    };
    let csr = CsrMatrix::try_from_csr_data(
        n_cells,
        n_genes,
        row_offsets,
        col_indices,
        values.into_iter().map(|v| v as f32).collect(),
    )
    .map_err(|e| anyhow!("read_gef: failed to build the CSR: {e:?}"))?;
    _t();

    let mut data = SoData::new();
    data.exp_matrix = Some(if is_sparse {
        ExpMatrix::Sparse(csr)
    } else {
        ExpMatrix::Dense(csr_to_dense(&csr))
    });

    if let Some(exon_csr) = exon_csr {
        data.layers
            .insert("exon".into(), ExpMatrix::Sparse(exon_csr));
    }

    let gene_ids: Vec<String> = genes.iter().map(|g| g.key().to_string()).collect();
    let gene_names: Vec<String> = genes.iter().map(|g| g.gene_name.clone()).collect();
    data.var = Some(
        DataFrame::new(vec![
            Column::new("gene".into(), gene_ids),
            Column::new("gene_name".into(), gene_names),
        ])
        .context("read_gef: failed to build the var DataFrame")?,
    );

    let cell_names: Vec<String> = bins
        .iter()
        .map(|&b| format!("{}_{}", bin_x(b), bin_y(b)))
        .collect();
    data.obs = Some(
        DataFrame::new(vec![Column::new("cell".into(), cell_names)])
            .context("read_gef: failed to build the obs DataFrame")?,
    );

    let mut pos_flat = Vec::with_capacity(n_cells * 2);
    for &b in &bins {
        pos_flat.push(bin_x(b) as f64);
        pos_flat.push(bin_y(b) as f64);
    }
    data.position = Some(
        Array2::from_shape_vec((n_cells, 2), pos_flat)
            .context("read_gef: failed to build position")?,
    );

    data.file_path = Some(PathBuf::from(path));
    data.file_format = Some("gef".into());
    data.samples_names = read_sn(&file).into_iter().collect();

    let bin_list: Vec<Value> = bin_names(&gene_exp)
        .into_iter()
        .map(Value::String)
        .collect();
    let mut uns = HashMap::new();
    uns.insert("resolution".into(), Value::from(resolution));
    uns.insert("minX".into(), Value::from(min_x));
    uns.insert("minY".into(), Value::from(min_y));
    uns.insert("maxX".into(), Value::from(max_x));
    uns.insert("maxY".into(), Value::from(max_y));
    uns.insert("maxExp".into(), Value::from(max_exp));
    uns.insert("bin_list".into(), Value::Array(bin_list));
    data.uns = uns;

    if save_counts_layer {
        if let Some(ref exp) = data.exp_matrix {
            data.layers.insert("counts".into(), exp.clone());
        }
    }

    Ok(data)
}

fn read_gef_cellbin(
    file: &File,
    path: &str,
    is_sparse: bool,
    dedup_symbols: bool,
    save_counts_layer: bool,
    save_exon_layer: bool,
) -> Result<SoData> {
    let cellbin = file
        .group("cellBin")
        .context("read_gef: missing the cellBin group (not a cellbin GEF?)")?;

    let genes_raw = read_cellbin_gene_table(&cellbin)?;

    let (genes, old_to_new) = if dedup_symbols {
        let (g, m) = merge_cellbin_gene_rows(&genes_raw);
        (g, Some(m))
    } else {
        (genes_raw, None)
    };
    let n_genes = genes.len();

    let cells = read_cellbin_cell_table(&cellbin)?;
    let n_cells = cells.len();
    let mut row_offsets = Vec::with_capacity(n_cells + 1);
    let mut pos_flat = Vec::with_capacity(n_cells * 2);
    for c in &cells {
        row_offsets.push(c.offset as usize);
        pos_flat.push(c.x as f64);
        pos_flat.push(c.y as f64);
    }

    let cell_exp = cellbin
        .dataset("cellExp")
        .context("read_gef: missing the cellBin/cellExp dataset")?;
    let nnz = cell_exp.size();
    row_offsets.push(nnz);
    let mut col_indices = vec![0usize; nnz];
    let mut values = vec![0f64; nnz];
    read_cell_exp(path, "cellBin/cellExp", nnz, &mut col_indices, &mut values)?;

    let mut exon_values: Option<Vec<f64>> = if save_exon_layer && cellbin.link_exists("cellExpExon")
    {
        Some(read_exon(path, "cellBin/cellExpExon", nnz)?)
    } else {
        None
    };

    let row_offsets = if let Some(map) = old_to_new.as_ref() {
        let (new_offsets, unsorted_rows) = merge_csr_cols_inplace(
            &row_offsets,
            &mut col_indices,
            &mut values,
            exon_values.as_mut(),
            map,
        );
        if unsorted_rows > 0 {
            eprintln!(
                "read_gef: cellbin cellExp has {unsorted_rows} rows whose in-row column indices are not descending; those rows fell back to sort-merge (results are still correct, but the data layout differs from the expected one)"
            );
        }
        new_offsets
    } else {
        row_offsets
    };

    let exon_csr = match exon_values {
        Some(ev) => Some(
            CsrMatrix::try_from_csr_data(
                n_cells,
                n_genes,
                row_offsets.clone(),
                col_indices.clone(),
                ev.into_iter().map(|v| v as f32).collect(),
            )
            .map_err(|err| anyhow!("read_gef: failed to build the cellbin exon CSR: {err:?}"))?,
        ),
        None => None,
    };

    let csr = CsrMatrix::try_from_csr_data(
        n_cells,
        n_genes,
        row_offsets,
        col_indices,
        values.into_iter().map(|v| v as f32).collect(),
    )
    .map_err(|e| anyhow!("read_gef: failed to build the cellbin CSR: {e:?}"))?;

    let mut data = SoData::new();

    data.exp_matrix = Some(if is_sparse {
        ExpMatrix::Sparse(csr)
    } else {
        ExpMatrix::Dense(csr_to_dense(&csr))
    });

    if let Some(exon_csr) = exon_csr {
        data.layers
            .insert("exon".into(), ExpMatrix::Sparse(exon_csr));
    }

    let gene_ids: Vec<String> = genes.iter().map(|g| g.key().to_string()).collect();
    let gene_names: Vec<String> = genes.iter().map(|g| g.gene_name.clone()).collect();
    data.var = Some(
        DataFrame::new(vec![
            Column::new("gene".into(), gene_ids),
            Column::new("gene_name".into(), gene_names),
        ])
        .context("read_gef: failed to build the var DataFrame")?,
    );

    let cell_names: Vec<String> = cells.iter().map(|c| format!("{}", c.id)).collect();
    let area: Vec<i64> = cells.iter().map(|c| c.area as i64).collect();
    let dnb: Vec<i64> = cells.iter().map(|c| c.dnb_count as i64).collect();
    data.obs = Some(
        DataFrame::new(vec![
            Column::new("cell".into(), cell_names),
            Column::new("area".into(), area),
            Column::new("dnbCount".into(), dnb),
        ])
        .context("read_gef: failed to build the obs DataFrame")?,
    );

    data.position = Some(
        Array2::from_shape_vec((n_cells, 2), pos_flat)
            .context("read_gef: failed to build position")?,
    );

    data.file_path = Some(PathBuf::from(path));
    data.file_format = Some("gef".into());
    data.samples_names = read_sn(file).into_iter().collect();

    let resolution = read_attr_i64(file, "resolution").unwrap_or(0) as u32;
    let offset_x = read_attr_i64(file, "offsetX").unwrap_or(0);
    let offset_y = read_attr_i64(file, "offsetY").unwrap_or(0);
    let max_x = read_attr_i64(file, "maxX").unwrap_or(0);
    let max_y = read_attr_i64(file, "maxY").unwrap_or(0);
    let mut uns = HashMap::new();
    uns.insert("resolution".into(), Value::from(resolution));
    uns.insert("minX".into(), Value::from(offset_x));
    uns.insert("minY".into(), Value::from(offset_y));
    uns.insert("maxX".into(), Value::from(max_x));
    uns.insert("maxY".into(), Value::from(max_y));
    uns.insert("maxExp".into(), Value::from(0));
    data.uns = uns;

    if save_counts_layer {
        if let Some(ref exp) = data.exp_matrix {
            data.layers.insert("counts".into(), exp.clone());
        }
    }

    Ok(data)
}

fn merge_duplicate_symbols(
    genes: Vec<GeneRow>,
    expr: Vec<u64>,
    exon: Option<Vec<f64>>,
) -> (Vec<GeneRow>, Vec<u64>, Option<Vec<f64>>) {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, g) in genes.iter().enumerate() {
        groups.entry(g.gene_name.clone()).or_default().push(i);
    }

    let mut new_genes: Vec<GeneRow> = Vec::with_capacity(groups.len());
    let mut new_expr: Vec<u64> = Vec::with_capacity(expr.len());

    let mut new_exon: Option<Vec<f64>> = exon.as_ref().map(|_| Vec::with_capacity(expr.len()));
    let mut acc: HashMap<u64, (u64, f64)> = HashMap::new();
    for (i, g) in genes.iter().enumerate() {
        let members = &groups[g.gene_name.as_str()];
        if members[0] != i {
            continue;
        }
        acc.clear();
        for &m in members {
            let s = genes[m].offset as usize;
            let e = s + genes[m].count as usize;
            for (k, &packed) in expr[s..e].iter().enumerate() {
                let cell = packed >> 16;
                let cnt = packed & 0xFFFF;
                let ex = exon.as_ref().map_or(0.0, |ex| ex[s + k]);
                let slot = acc.entry(cell).or_insert((0, 0.0));
                slot.0 += cnt;
                slot.1 += ex;
            }
        }
        let offset = new_expr.len() as u32;
        for (cell, (cnt, ex)) in acc.drain() {
            new_expr.push((cell << 16) | (cnt & 0xFFFF));
            if let Some(ref mut exv) = new_exon {
                exv.push(ex);
            }
        }
        new_genes.push(GeneRow {
            gene_id: g.gene_id.clone(),
            gene_name: g.gene_name.clone(),
            offset,
            count: (new_expr.len() - offset as usize) as u32,
        });
    }
    (new_genes, new_expr, new_exon)
}

fn merge_cellbin_gene_rows(genes: &[GeneRow]) -> (Vec<GeneRow>, Vec<u32>) {
    let mut seen: HashMap<&str, u32> = HashMap::with_capacity(genes.len());
    let mut new_genes: Vec<GeneRow> = Vec::with_capacity(genes.len());
    let mut map = vec![0u32; genes.len()];
    for (i, g) in genes.iter().enumerate() {
        let idx = *seen.entry(g.gene_name.as_str()).or_insert_with(|| {
            new_genes.push(GeneRow {
                gene_id: g.gene_id.clone(),
                gene_name: g.gene_name.clone(),
                offset: 0,
                count: 0,
            });
            (new_genes.len() - 1) as u32
        });
        map[i] = idx;
    }
    (new_genes, map)
}

fn merge_csr_cols_inplace(
    row_offsets: &[usize],
    col_indices: &mut Vec<usize>,
    values: &mut Vec<f64>,
    mut exon_values: Option<&mut Vec<f64>>,
    old_to_new: &[u32],
) -> (Vec<usize>, usize) {
    let n_cells = row_offsets.len() - 1;
    let mut new_offsets = Vec::with_capacity(row_offsets.len());
    let mut scratch: Vec<(usize, f64, f64)> = Vec::new();
    let mut write = 0usize;
    let mut unsorted_rows = 0usize;

    for r in 0..n_cells {
        let s = row_offsets[r];
        let e = row_offsets[r + 1];
        new_offsets.push(write);
        if s == e {
            continue;
        }

        let mut prev: Option<u32> = None;
        let mut sorted = true;
        for k in s..e {
            let nc = old_to_new[col_indices[k]];
            if matches!(prev, Some(p) if nc < p) {
                sorted = false;
            }
            prev = Some(nc);
            col_indices[k] = nc as usize;
        }

        if sorted {
            let mut w = write;
            let mut cur = col_indices[s];
            let mut acc = values[s];
            let mut ex_acc = exon_values.as_deref().map(|ex| ex[s]).unwrap_or(0.0);
            for k in (s + 1)..e {
                let c = col_indices[k];
                let v = values[k];
                let ev = exon_values.as_deref().map(|ex| ex[k]).unwrap_or(0.0);
                if c == cur {
                    acc += v;
                    ex_acc += ev;
                } else {
                    col_indices[w] = cur;
                    values[w] = acc;
                    if let Some(ex) = exon_values.as_deref_mut() {
                        ex[w] = ex_acc;
                    }
                    w += 1;
                    cur = c;
                    acc = v;
                    ex_acc = ev;
                }
            }
            col_indices[w] = cur;
            values[w] = acc;
            if let Some(ex) = exon_values.as_deref_mut() {
                ex[w] = ex_acc;
            }
            write = w + 1;
        } else {
            unsorted_rows += 1;
            scratch.clear();
            match exon_values.as_deref() {
                Some(ex) => scratch.extend((s..e).map(|k| (col_indices[k], values[k], ex[k]))),
                None => scratch.extend((s..e).map(|k| (col_indices[k], values[k], 0.0))),
            }
            scratch.sort_unstable_by_key(|a| a.0);
            let mut w = write;
            let mut i = 0;
            while i < scratch.len() {
                let c = scratch[i].0;
                let mut acc = scratch[i].1;
                let mut ex_acc = scratch[i].2;
                let mut j = i + 1;
                while j < scratch.len() && scratch[j].0 == c {
                    acc += scratch[j].1;
                    ex_acc += scratch[j].2;
                    j += 1;
                }
                col_indices[w] = c;
                values[w] = acc;
                if let Some(ex) = exon_values.as_deref_mut() {
                    ex[w] = ex_acc;
                }
                w += 1;
                i = j;
            }
            write = w;
        }
    }

    new_offsets.push(write);
    col_indices.truncate(write);
    values.truncate(write);
    if let Some(ex) = exon_values {
        ex.truncate(write);
    }
    (new_offsets, unsorted_rows)
}

#[derive(Debug, Clone)]
pub struct GefInfo {
    pub bin_list: Vec<String>,
    pub resolution: u32,
    pub gene_count: usize,
    pub offset_x: i64,
    pub offset_y: i64,
    pub width: i64,
    pub height: i64,
    pub max_exp: i64,
}

pub fn read_gef_info(path: &str) -> Result<GefInfo> {
    let file = File::open(path)?;

    if is_cellbin_file(&file) {
        let resolution = read_attr_i64(&file, "resolution").unwrap_or(0) as u32;
        let offset_x = read_attr_i64(&file, "offsetX").unwrap_or(0);
        let offset_y = read_attr_i64(&file, "offsetY").unwrap_or(0);
        let max_x = read_attr_i64(&file, "maxX").unwrap_or(0);
        let max_y = read_attr_i64(&file, "maxY").unwrap_or(0);
        let gene_count = file.group("cellBin")?.dataset("gene")?.size();
        return Ok(GefInfo {
            bin_list: vec![],
            resolution,
            gene_count,
            offset_x,
            offset_y,
            width: max_x - offset_x,
            height: max_y - offset_y,
            max_exp: 0,
        });
    }

    let gene_exp = file
        .group("geneExp")
        .context("read_gef_info: missing the geneExp group")?;

    let bin_list = bin_names(&gene_exp);
    if bin_list.is_empty() {
        bail!("read_gef_info: no bin level under geneExp");
    }

    let first_bin = gene_exp
        .group("bin1")
        .or_else(|_| gene_exp.group(&bin_list[0]))?;
    let expr_ds = first_bin.dataset("expression")?;
    let (min_x, min_y, max_x, max_y, max_exp, resolution) = read_expression_attrs(&expr_ds)?;
    let gene_count = first_bin.dataset("gene")?.size();

    Ok(GefInfo {
        bin_list,
        resolution: resolution as u32,
        gene_count,
        offset_x: min_x,
        offset_y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
        max_exp,
    })
}

struct GeneRow {
    gene_id: String,
    gene_name: String,
    offset: u32,
    count: u32,
}

impl GeneRow {
    fn key(&self) -> &str {
        if self.gene_id.is_empty() {
            &self.gene_name
        } else {
            &self.gene_id
        }
    }
}

fn is_cellbin_file(file: &File) -> bool {
    file.link_exists("cellBin")
}

fn read_cellbin_gene_table(cellbin: &Group) -> Result<Vec<GeneRow>> {
    let ds = cellbin
        .dataset("gene")
        .context("read_gef: missing the cellBin/gene dataset")?;
    let arr = ds.read_1d::<CellGene>()?;
    Ok(arr
        .iter()
        .map(|g| GeneRow {
            gene_id: g.gene_id.as_str().to_string(),
            gene_name: g.gene_name.as_str().to_string(),
            offset: 0,
            count: 0,
        })
        .collect())
}

fn read_cellbin_cell_table(cellbin: &Group) -> Result<Vec<CellBinCell>> {
    let ds = cellbin
        .dataset("cell")
        .context("read_gef: missing the cellBin/cell dataset")?;
    Ok(ds.read_1d::<CellBinCell>()?.to_vec())
}

fn read_cell_exp(
    path: &str,
    dataset_path: &str,
    n_elems: usize,
    col_indices: &mut [usize],
    values: &mut [f64],
) -> Result<()> {
    const IO_NTHREADS: usize = 4;
    let n_threads = IO_NTHREADS.min(n_elems.max(1));
    let chunk = n_elems.div_ceil(n_threads);
    let col_ptr = SendPtr(col_indices.as_mut_ptr());
    let val_ptr = SendPtr(values.as_mut_ptr());
    (0..n_threads)
        .into_par_iter()
        .try_for_each(|tid| -> Result<()> {
            let r0 = tid * chunk;
            let r1 = (r0 + chunk).min(n_elems);
            if r0 >= n_elems {
                return Ok(());
            }
            let file = File::open(path).context("read_cell_exp: failed to reopen the GEF file")?;
            let ds = file
                .dataset(dataset_path)
                .context("read_cell_exp: failed to open the cellExp dataset")?;
            let arr = ds.read_slice_1d::<CellExpRec, _>(r0..r1)?;
            for (k, rec) in arr.iter().enumerate() {
                let pos = r0 + k;
                unsafe {
                    *col_ptr.add(pos) = rec.gene_id as usize;
                    *val_ptr.add(pos) = rec.count as f64;
                }
            }
            Ok(())
        })
}

fn read_gene_table(bin_group: &Group) -> Result<Vec<GeneRow>> {
    let ds = bin_group
        .dataset("gene")
        .context("read_gef: missing the gene dataset")?;
    let td = ds.dtype()?.to_descriptor()?;

    let field_names = match &td {
        TypeDescriptor::Compound(ct) => ct
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };

    if field_names.contains(&"geneID") {
        let arr = ds.read_1d::<GeneV4>()?;
        Ok(arr
            .iter()
            .map(|g| GeneRow {
                gene_id: g.gene_id.as_str().to_string(),
                gene_name: g.gene_name.as_str().to_string(),
                offset: g.offset,
                count: g.count,
            })
            .collect())
    } else if field_names.contains(&"geneName") {
        let arr = ds.read_1d::<GeneV3>()?;
        Ok(arr
            .iter()
            .map(|g| GeneRow {
                gene_id: String::new(),
                gene_name: g.gene_name.as_str().to_string(),
                offset: g.offset,
                count: g.count,
            })
            .collect())
    } else if field_names.contains(&"gene") {
        let arr = ds.read_1d::<GeneV2>()?;
        Ok(arr
            .iter()
            .map(|g| GeneRow {
                gene_id: String::new(),
                gene_name: g.gene_name.as_str().to_string(),
                offset: g.offset,
                count: g.count,
            })
            .collect())
    } else {
        bail!("read_gef: unrecognized gene compound layout (fields: {field_names:?})")
    }
}

fn expression_count_u1(ds: &Dataset) -> Result<bool> {
    let td = ds.dtype()?.to_descriptor()?;
    match &td {
        TypeDescriptor::Compound(ct) => Ok(ct
            .fields
            .iter()
            .any(|f| f.name == "count" && f.ty == TypeDescriptor::Unsigned(IntSize::U1))),
        _ => bail!("read_gef: expression is not a compound type"),
    }
}

fn read_expression(
    path: &str,
    dataset_path: &str,
    n_elems: usize,
    count_u1: bool,
) -> Result<Vec<u64>> {
    const IO_NTHREADS: usize = 4;
    let n_threads = IO_NTHREADS.min(n_elems.max(1));
    let chunk = n_elems.div_ceil(n_threads);
    let shards: Vec<Result<Vec<u64>>> = (0..n_threads)
        .into_par_iter()
        .map(|tid| {
            let r0 = tid * chunk;
            let r1 = (r0 + chunk).min(n_elems);
            if r0 >= n_elems {
                return Ok(Vec::new());
            }
            let file =
                File::open(path).context("read_expression: failed to reopen the GEF file")?;
            let ds = file
                .dataset(dataset_path)
                .context("read_expression: failed to open the expression dataset")?;
            if count_u1 {
                let arr = ds.read_slice_1d::<ExprU8, _>(r0..r1)?;
                arr.iter()
                    .map(|e| pack_bin(e.x, e.y, e.count as u32))
                    .collect()
            } else {
                let arr = ds.read_slice_1d::<ExprU16, _>(r0..r1)?;
                arr.iter()
                    .map(|e| pack_bin(e.x, e.y, e.count as u32))
                    .collect()
            }
        })
        .collect();
    let mut expr = Vec::with_capacity(n_elems);
    for shard in shards {
        expr.extend(shard?);
    }
    Ok(expr)
}

fn read_exon(path: &str, dataset_path: &str, n_elems: usize) -> Result<Vec<f64>> {
    let td = {
        let file = File::open(path).context("read_exon: failed to reopen the GEF file")?;
        let ds = file
            .dataset(dataset_path)
            .context("read_exon: failed to open the exon dataset")?;
        ds.dtype()?.to_descriptor()?
    };
    let is_u1 = matches!(td, TypeDescriptor::Unsigned(IntSize::U1));
    let is_u2 = matches!(td, TypeDescriptor::Unsigned(IntSize::U2));
    let is_u4 = matches!(td, TypeDescriptor::Unsigned(IntSize::U4));
    if !(is_u1 || is_u2 || is_u4) {
        bail!("read_exon: exon dataset dtype is not u8/u16/u32: {td:?}");
    }

    const IO_NTHREADS: usize = 4;
    let n_threads = IO_NTHREADS.min(n_elems.max(1));
    let chunk = n_elems.div_ceil(n_threads);
    let shards: Vec<Result<Vec<f64>>> = (0..n_threads)
        .into_par_iter()
        .map(|tid| {
            let r0 = tid * chunk;
            let r1 = (r0 + chunk).min(n_elems);
            if r0 >= n_elems {
                return Ok(Vec::new());
            }
            let file = File::open(path).context("read_exon: failed to reopen the GEF file")?;
            let ds = file
                .dataset(dataset_path)
                .context("read_exon: failed to open the exon dataset")?;
            let mut out = Vec::with_capacity(r1 - r0);
            if is_u1 {
                out.extend(ds.read_slice_1d::<u8, _>(r0..r1)?.iter().map(|&v| v as f64));
            } else if is_u2 {
                out.extend(
                    ds.read_slice_1d::<u16, _>(r0..r1)?
                        .iter()
                        .map(|&v| v as f64),
                );
            } else {
                out.extend(
                    ds.read_slice_1d::<u32, _>(r0..r1)?
                        .iter()
                        .map(|&v| v as f64),
                );
            }
            Ok(out)
        })
        .collect();
    let mut exon = Vec::with_capacity(n_elems);
    for shard in shards {
        exon.extend(shard?);
    }
    Ok(exon)
}

#[inline]
fn pack_bin(x: i32, y: i32, count: u32) -> Result<u64> {
    if !(0..=0xFFFF).contains(&x) || y < 0 {
        bail!(
            "read_gef: coordinates out of range, cannot pack the bin (x={x}, y={y}); square bins currently support only 0 <= x < 65536 and y >= 0"
        );
    }
    let bin_id = ((x as u32 as u64) << 32) | (y as u32 as u64);
    Ok((bin_id << 16) | (count & 0xFFFF) as u64)
}

#[inline]
fn bin_x(b: u64) -> i32 {
    (b >> 32) as u32 as i32
}

#[inline]
fn bin_y(b: u64) -> i32 {
    (b & 0xFFFF_FFFF) as u32 as i32
}

fn read_expression_attrs(ds: &Dataset) -> Result<(i64, i64, i64, i64, i64, i64)> {
    Ok((
        read_attr_i64(ds, "minX").unwrap_or(0),
        read_attr_i64(ds, "minY").unwrap_or(0),
        read_attr_i64(ds, "maxX").unwrap_or(0),
        read_attr_i64(ds, "maxY").unwrap_or(0),
        read_attr_i64(ds, "maxExp").unwrap_or(0),
        read_attr_i64(ds, "resolution").unwrap_or(0),
    ))
}

fn read_attr_i64(loc: &Location, name: &str) -> Option<i64> {
    let attr = loc.attr(name).ok()?;
    if let Ok(v) = attr.read_scalar::<i64>() {
        return Some(v);
    }
    if let Ok(v) = attr.read_scalar::<i32>() {
        return Some(v as i64);
    }
    if let Ok(v) = attr.read_scalar::<u32>() {
        return Some(v as i64);
    }
    if let Ok(v) = attr.read_1d::<i32>() {
        if let Some(&x) = v.first() {
            return Some(x as i64);
        }
    }
    if let Ok(v) = attr.read_1d::<u32>() {
        if let Some(&x) = v.first() {
            return Some(x as i64);
        }
    }
    if let Ok(v) = attr.read_1d::<i64>() {
        if let Some(&x) = v.first() {
            return Some(x);
        }
    }
    None
}

fn read_sn(file: &File) -> Option<String> {
    let attr = file.attr("sn").ok()?;
    if let Ok(v) = attr.read_scalar::<VarLenUnicode>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<VarLenAscii>() {
        return Some(v.to_string());
    }
    if let Ok(v) = attr.read_scalar::<FixedAscii<256>>() {
        return Some(v.to_string());
    }
    None
}

fn bin_names(gene_exp: &Group) -> Vec<String> {
    let mut names: Vec<String> = gene_exp
        .member_names()
        .unwrap_or_default()
        .into_iter()
        .filter(|n| n.starts_with("bin"))
        .collect();
    names.sort_by_key(|s| bin_suffix(s));
    names
}

fn bin_suffix(name: &str) -> u32 {
    name.trim_start_matches("bin").parse().unwrap_or(0)
}
