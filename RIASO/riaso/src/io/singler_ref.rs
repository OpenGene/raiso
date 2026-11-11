use anyhow::{bail, Result};
use hdf5::File;
use ndarray::Array2;

use crate::core::SingleRRef;
use crate::parallel::csr_to_dense;

use super::h5ad::{read_dense_x, read_index_strings, read_obs_column_strings, read_sparse_matrix};

pub fn read_ref(path: &str, label_key: &str, gene_key: Option<&str>) -> Result<SingleRRef> {
    let file = File::open(path)?;

    let ref_x: Array2<f32> = if let Ok(ds) = file.dataset("X") {
        read_dense_x(&ds)?
    } else if let Ok(group) = file.group("X") {
        csr_to_dense(&read_sparse_matrix(&group)?)
    } else {
        bail!("the reference h5ad is missing the X expression matrix")
    };

    let var_group = file.group("var")?;
    let ref_genes = match gene_key {
        Some(gk) => read_obs_column_strings(&var_group, gk)?,
        None => read_index_strings(&var_group)?,
    };

    let obs_group = file.group("obs")?;
    let ref_labels = read_obs_column_strings(&obs_group, label_key)?;

    SingleRRef::new(ref_x, ref_genes, ref_labels)
}
