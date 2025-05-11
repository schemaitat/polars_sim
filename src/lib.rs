mod cossim;
mod csr;
mod helper;

use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PolarsAllocator, PyDataFrame};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pyfunction]
#[pyo3(signature=(pydf_left, pydf_right, col_left, col_right, ntop, threads=None, normalize=None, parallelize_left=None))]
fn awesome_cossim(
    pydf_left: PyDataFrame,
    pydf_right: PyDataFrame,
    col_left: &str,
    col_right: &str,
    ntop: usize,
    threads: Option<usize>,
    normalize: Option<bool>,
    parallelize_left: Option<bool>,
) -> PyResult<PyDataFrame> {
    let df_left = pydf_left.into();
    let df_right = pydf_right.into();

    let res = cossim::awesome_cossim(
        df_left,
        df_right,
        col_left,
        col_right,
        ntop,
        threads,
        normalize,
        parallelize_left,
    )
    .map_err(PyPolarsErr::from)?;

    Ok(PyDataFrame(res))
}

#[pymodule]
fn _polars_simed(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(awesome_cossim, m)?)?;
    Ok(())
}
