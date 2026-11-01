use polars::prelude::*;
use rayon::prelude::*;

use crate::core::{ExpMatrix, SoData};

pub fn cal_qc(data: &mut SoData) {
    let exp = match &data.exp_matrix {
        Some(exp) => exp,
        None => return,
    };

    let n_rows = exp.n_rows();
    let n_cols = exp.n_cols();

    let mt_names = data.mt_gene_names();
    let mt_mask: Vec<bool> = if mt_names.len() == n_cols {
        mt_names
            .iter()
            .map(|g| {
                let l = g.to_lowercase();
                l.starts_with("mt-") || l.starts_with("mt.")
            })
            .collect()
    } else {
        vec![false; n_cols]
    };

    let (row_sums, row_nnz, col_sums, col_nnz, mt_counts) = exp_metrics(exp, &mt_mask);

    if let Some(obs) = data.obs.as_mut() {
        let pct_mt: Vec<f64> = (0..n_rows)
            .map(|r| {
                if row_sums[r] != 0.0 {
                    mt_counts[r] / row_sums[r] * 100.0
                } else {
                    0.0
                }
            })
            .collect();
        upsert_column(obs, Column::new("total_counts".into(), row_sums));
        upsert_column(obs, Column::new("n_genes_by_counts".into(), row_nnz));
        upsert_column(obs, Column::new("pct_counts_mt".into(), pct_mt));
    }

    if let Some(var) = data.var.as_mut() {
        let mean_umi: Vec<f64> = (0..n_cols)
            .map(|c| {
                if col_nnz[c] != 0 {
                    col_sums[c] / col_nnz[c] as f64
                } else {
                    0.0
                }
            })
            .collect();
        upsert_column(var, Column::new("n_counts".into(), col_sums));
        upsert_column(var, Column::new("n_cells".into(), col_nnz));
        upsert_column(var, Column::new("mean_umi".into(), mean_umi));
    }
}

fn exp_metrics(
    exp: &ExpMatrix,
    mt_mask: &[bool],
) -> (Vec<f64>, Vec<i64>, Vec<f64>, Vec<i64>, Vec<f64>) {
    let n_rows = exp.n_rows();
    let n_cols = exp.n_cols();

    type RowAcc = (Vec<f64>, Vec<i64>, Vec<f64>, Vec<i64>, Vec<f64>);
    let zero = || -> RowAcc {
        (
            vec![0f64; n_rows],
            vec![0i64; n_rows],
            vec![0f64; n_cols],
            vec![0i64; n_cols],
            vec![0f64; n_rows],
        )
    };

    let acc = match exp {
        ExpMatrix::Dense(arr) => (0..n_rows)
            .into_par_iter()
            .fold(zero, |mut a, r| {
                for c in 0..n_cols {
                    let v = arr[[r, c]] as f64;
                    if v != 0.0 {
                        a.0[r] += v;
                        a.1[r] += 1;
                        a.2[c] += v;
                        a.3[c] += 1;
                        if mt_mask[c] {
                            a.4[r] += v;
                        }
                    }
                }
                a
            })
            .reduce(zero, |mut a, b| {
                for i in 0..n_rows {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                    a.4[i] += b.4[i];
                }
                for i in 0..n_cols {
                    a.2[i] += b.2[i];
                    a.3[i] += b.3[i];
                }
                a
            }),
        ExpMatrix::Sparse(csr) => {
            let offsets = csr.row_offsets();
            let cols = csr.col_indices();
            let vals = csr.values();
            (0..n_rows)
                .into_par_iter()
                .fold(zero, |mut a, r| {
                    for i in offsets[r]..offsets[r + 1] {
                        let c = cols[i];
                        let v = vals[i] as f64;
                        if v != 0.0 {
                            a.0[r] += v;
                            a.1[r] += 1;
                            a.2[c] += v;
                            a.3[c] += 1;
                            if mt_mask[c] {
                                a.4[r] += v;
                            }
                        }
                    }
                    a
                })
                .reduce(zero, |mut a, b| {
                    for i in 0..n_rows {
                        a.0[i] += b.0[i];
                        a.1[i] += b.1[i];
                        a.4[i] += b.4[i];
                    }
                    for i in 0..n_cols {
                        a.2[i] += b.2[i];
                        a.3[i] += b.3[i];
                    }
                    a
                })
        }
    };

    acc
}

pub(crate) fn upsert_column(df: &mut DataFrame, col: Column) {
    let name = col.name().to_string();
    if df.column(&name).is_ok() {
        let _ = df.drop(&name);
    }
    let _ = df.with_column(col);
}
