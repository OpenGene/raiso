use anyhow::{bail, Result};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct SingleRRef {
    ref_x: Array2<f32>,
    ref_genes: Vec<String>,
    ref_labels: Vec<String>,
}

impl SingleRRef {
    pub fn new(
        ref_x: Array2<f32>,
        ref_genes: Vec<String>,
        ref_labels: Vec<String>,
    ) -> Result<Self> {
        if ref_x.nrows() != ref_labels.len() {
            bail!(
                "SingleRRef: ref_x rows ({}) != ref_labels len ({})",
                ref_x.nrows(),
                ref_labels.len()
            );
        }
        if ref_x.ncols() != ref_genes.len() {
            bail!(
                "SingleRRef: ref_x cols ({}) != ref_genes len ({})",
                ref_x.ncols(),
                ref_genes.len()
            );
        }
        Ok(Self {
            ref_x,
            ref_genes,
            ref_labels,
        })
    }

    pub fn ref_x(&self) -> &Array2<f32> {
        &self.ref_x
    }

    pub fn ref_genes(&self) -> &[String] {
        &self.ref_genes
    }

    pub fn ref_labels(&self) -> &[String] {
        &self.ref_labels
    }

    pub fn n_cells(&self) -> usize {
        self.ref_x.nrows()
    }

    pub fn n_genes(&self) -> usize {
        self.ref_x.ncols()
    }

    pub fn shape(&self) -> (usize, usize) {
        self.ref_x.dim()
    }
}
