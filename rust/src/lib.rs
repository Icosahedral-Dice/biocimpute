//! Python bindings for the Rust kernels: `biocimpute._kernels._rust`.
//!
//! The functions have the same names and in-place signatures as
//! `biocimpute._kernels._numba`. Matrices must be Fortran-contiguous and 1-D
//! arrays contiguous; `biocimpute._kernels` prepares them that way.

mod kernels;

use numpy::{
    Element, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn fortran<'a, T: Element>(a: &'a PyReadonlyArray2<'_, T>, name: &str) -> PyResult<&'a [T]> {
    if !a.is_fortran_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be Fortran-contiguous"
        )));
    }
    a.as_slice()
        .map_err(|e| PyValueError::new_err(format!("{name}: {e}")))
}

fn fortran_mut<'a, T: Element>(
    a: &'a mut PyReadwriteArray2<'_, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    if !a.is_fortran_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be Fortran-contiguous"
        )));
    }
    a.as_slice_mut()
        .map_err(|e| PyValueError::new_err(format!("{name}: {e}")))
}

fn contiguous<'a, T: Element>(a: &'a PyReadonlyArray1<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|e| PyValueError::new_err(format!("{name}: {e}")))
}

fn contiguous_mut<'a, T: Element>(
    a: &'a mut PyReadwriteArray1<'_, T>,
    name: &str,
) -> PyResult<&'a mut [T]> {
    a.as_slice_mut()
        .map_err(|e| PyValueError::new_err(format!("{name}: {e}")))
}

fn check_len(name: &str, len: usize, needed: usize) -> PyResult<()> {
    if len < needed {
        return Err(PyValueError::new_err(format!(
            "{name} has length {len}, expected at least {needed}"
        )));
    }
    Ok(())
}

#[pyfunction]
fn misdis(
    x0: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    imiss0: PyReadonlyArray1<'_, i32>,
    imiss: PyReadonlyArray2<'_, i32>,
    mut dis: PyReadwriteArray1<'_, f64>,
    mut iworkp: PyReadwriteArray1<'_, i32>,
) -> PyResult<()> {
    let (p, n) = (x.shape()[0], x.shape()[1]);
    let x0 = contiguous(&x0, "x0")?;
    let imiss0 = contiguous(&imiss0, "imiss0")?;
    let dis = contiguous_mut(&mut dis, "dis")?;
    let iworkp = contiguous_mut(&mut iworkp, "iworkp")?;
    check_len("x0", x0.len(), n)?;
    check_len("imiss0", imiss0.len(), n)?;
    check_len("dis", dis.len(), p)?;
    check_len("iworkp", iworkp.len(), p)?;
    kernels::misdis(
        x0,
        fortran(&x, "x")?,
        p,
        n,
        imiss0,
        fortran(&imiss, "imiss")?,
        dis,
        iworkp,
    );
    Ok(())
}

#[pyfunction]
fn misave(
    x: PyReadonlyArray2<'_, f64>,
    mut x0: PyReadwriteArray1<'_, f64>,
    mut imiss0: PyReadwriteArray1<'_, i32>,
    imiss: PyReadonlyArray2<'_, i32>,
    index: PyReadonlyArray1<'_, i64>,
    m: usize,
) -> PyResult<()> {
    let (p, n) = (x.shape()[0], x.shape()[1]);
    let x0 = contiguous_mut(&mut x0, "x0")?;
    let imiss0 = contiguous_mut(&mut imiss0, "imiss0")?;
    let index = contiguous(&index, "index")?;
    check_len("x0", x0.len(), n)?;
    check_len("imiss0", imiss0.len(), n)?;
    check_len("index", index.len(), m)?;
    if index[..m].iter().any(|&j| j < 0 || j as usize >= p) {
        return Err(PyValueError::new_err("index out of range"));
    }
    kernels::misave(
        fortran(&x, "x")?,
        p,
        n,
        x0,
        imiss0,
        fortran(&imiss, "imiss")?,
        index,
        m,
    );
    Ok(())
}

#[pyfunction]
fn porder(
    kn: usize,
    dist: PyReadonlyArray1<'_, f64>,
    ntr: usize,
    mut pos: PyReadwriteArray1<'_, i64>,
    mut nndist: PyReadwriteArray1<'_, f64>,
) -> PyResult<()> {
    let dist = contiguous(&dist, "dist")?;
    let pos = contiguous_mut(&mut pos, "pos")?;
    let nndist = contiguous_mut(&mut nndist, "nndist")?;
    check_len("dist", dist.len(), ntr)?;
    check_len("pos", pos.len(), kn)?;
    check_len("nndist", nndist.len(), kn)?;
    kernels::porder(kn, dist, ntr, pos, nndist);
    Ok(())
}

#[pyfunction]
fn knnimp(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    mut ximp: PyReadwriteArray2<'_, f64>,
    mut imiss: PyReadwriteArray2<'_, i32>,
    irmiss: PyReadonlyArray1<'_, i32>,
    kn: usize,
) -> PyResult<()> {
    let (p, n) = (x.shape()[0], x.shape()[1]);
    if ximp.shape() != x.shape() || imiss.shape() != x.shape() {
        return Err(PyValueError::new_err(
            "x, ximp and imiss must have the same shape",
        ));
    }
    let x = fortran(&x, "x")?;
    let ximp = fortran_mut(&mut ximp, "ximp")?;
    let imiss = fortran_mut(&mut imiss, "imiss")?;
    let irmiss = contiguous(&irmiss, "irmiss")?;
    check_len("irmiss", irmiss.len(), p)?;
    py.detach(|| kernels::knnimp(x, ximp, p, n, imiss, irmiss, kn));
    Ok(())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn twomis(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    imiss: PyReadonlyArray2<'_, i32>,
    maxit: i64,
    eps: f64,
    istart: PyReadonlyArray1<'_, i64>,
    mut clust: PyReadwriteArray2<'_, i64>,
    mut nsize: PyReadwriteArray1<'_, i64>,
) -> PyResult<(f64, i64)> {
    let (p, n) = (x.shape()[0], x.shape()[1]);
    if imiss.shape() != x.shape() || clust.shape() != [p, 2] {
        return Err(PyValueError::new_err(
            "imiss must match x, and clust must be (p, 2)",
        ));
    }
    let istart = contiguous(&istart, "istart")?;
    check_len("istart", istart.len(), 2)?;
    if istart[..2].iter().any(|&j| j < 0 || j as usize >= p) {
        return Err(PyValueError::new_err("istart out of range"));
    }
    let istart = [istart[0] as usize, istart[1] as usize];
    let x = fortran(&x, "x")?;
    let imiss = fortran(&imiss, "imiss")?;
    let clust = fortran_mut(&mut clust, "clust")?;
    let nsize = contiguous_mut(&mut nsize, "nsize")?;
    check_len("nsize", nsize.len(), 2)?;
    Ok(py.detach(|| kernels::twomis(x, p, n, imiss, maxit, eps, istart, clust, nsize)))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DISMAX", kernels::DISMAX)?;
    m.add_function(wrap_pyfunction!(misdis, m)?)?;
    m.add_function(wrap_pyfunction!(misave, m)?)?;
    m.add_function(wrap_pyfunction!(porder, m)?)?;
    m.add_function(wrap_pyfunction!(knnimp, m)?)?;
    m.add_function(wrap_pyfunction!(twomis, m)?)?;
    Ok(())
}
