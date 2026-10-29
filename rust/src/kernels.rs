//! Rust port of impute's src/knnimpute.f.
//!
//! Each function is a literal, 0-based translation of the Fortran subroutine
//! of the same name, keeping the loop and accumulation order so that floating
//! point results match. Matrices are column-major slices (element `(j, k)` of
//! a `p x n` matrix is at `j + k * p`), as in Fortran. Missing-value flags:
//! 0 means present, 1 means missing, 2 means "could not be imputed".

pub const DISMAX: f64 = 1e10;

/// Mean squared distance from `x0` to every row of `x` over the columns
/// observed in both.
#[allow(clippy::too_many_arguments)]
pub fn misdis(
    x0: &[f64],
    x: &[f64],
    p: usize,
    n: usize,
    imiss0: &[i32],
    imiss: &[i32],
    dis: &mut [f64],
    iworkp: &mut [i32],
) {
    let dis = &mut dis[..p];
    let iworkp = &mut iworkp[..p];
    dis.fill(0.0);
    iworkp.fill(0);
    for k in 0..n {
        if imiss0[k] > 0 {
            continue;
        }
        let x0k = x0[k];
        let xk = &x[k * p..(k + 1) * p];
        let mk = &imiss[k * p..(k + 1) * p];
        for (((d, c), &xjk), &mjk) in dis.iter_mut().zip(iworkp.iter_mut()).zip(xk).zip(mk) {
            if mjk > 0 {
                continue;
            }
            let diff = x0k - xjk;
            *d += diff * diff;
            *c += 1;
        }
    }
    for (d, &c) in dis.iter_mut().zip(iworkp.iter()) {
        *d = if c > 0 { *d / f64::from(c) } else { DISMAX };
    }
}

/// Column means of the rows `index[..m]` of `x` over observed entries,
/// computed only where `imiss0` is set; sets `imiss0[k] = 2` when none of
/// those rows is observed in column `k`.
#[allow(clippy::too_many_arguments)]
pub fn misave(
    x: &[f64],
    p: usize,
    n: usize,
    x0: &mut [f64],
    imiss0: &mut [i32],
    imiss: &[i32],
    index: &[i64],
    m: usize,
) {
    for k in 0..n {
        x0[k] = 0.0;
        if imiss0[k] == 0 {
            continue;
        }
        let mut ktot = 0i32;
        for &jj in &index[..m] {
            let jk = jj as usize + k * p;
            if imiss[jk] > 0 {
                continue;
            }
            x0[k] += x[jk];
            ktot += 1;
        }
        if ktot > 0 {
            x0[k] /= f64::from(ktot);
        } else {
            imiss0[k] = 2;
        }
    }
}

/// Positions of the `kn` smallest of `dist[..ntr]` in increasing order, ties
/// going to the lower index. Only `min(kn, ntr)` slots are filled.
pub fn porder(kn: usize, dist: &[f64], ntr: usize, pos: &mut [i64], nndist: &mut [f64]) {
    for j in 0..ntr {
        let dj = dist[j];
        if j < kn {
            let mut inserted = false;
            for k in 0..j {
                if dj < nndist[k] {
                    for k1 in (k..j).rev() {
                        nndist[k1 + 1] = nndist[k1];
                        pos[k1 + 1] = pos[k1];
                    }
                    nndist[k] = dj;
                    pos[k] = j as i64;
                    inserted = true;
                    break;
                }
            }
            if !inserted {
                nndist[j] = dj;
                pos[j] = j as i64;
            }
        } else {
            if dj >= nndist[kn - 1] {
                continue;
            }
            for k in 0..kn {
                if dj < nndist[k] {
                    for k1 in (k..kn - 1).rev() {
                        nndist[k1 + 1] = nndist[k1];
                        pos[k1 + 1] = pos[k1];
                    }
                    nndist[k] = dj;
                    pos[k] = j as i64;
                    break;
                }
            }
        }
    }
}

/// KNN-impute every row with missing values. Writes into `ximp` and sets
/// `imiss[i, k] = 2` where no neighbor was observed.
pub fn knnimp(
    x: &[f64],
    ximp: &mut [f64],
    p: usize,
    n: usize,
    imiss: &mut [i32],
    irmiss: &[i32],
    kn: usize,
) {
    let m = kn + 1;
    let mut workp = vec![0.0; p];
    let mut iworkp = vec![0i32; p];
    let mut workn = vec![0.0; n];
    let mut iworkn = vec![0i32; n];
    let mut pos = vec![0i64; m];
    let mut nndist = vec![0.0; m];
    // Neighbors are positions 2..kn+1 of the ordering. Fortran reads past the
    // filled slots when p <= kn; we use only the p - 1 that exist.
    let nnb = m.min(p).saturating_sub(1);
    for i in 0..p {
        if irmiss[i] == 0 {
            continue;
        }
        for j in 0..n {
            workn[j] = x[i + j * p];
            iworkn[j] = imiss[i + j * p];
        }
        misdis(&workn, x, p, n, &iworkn, imiss, &mut workp, &mut iworkp);
        porder(m, &workp, p, &mut pos, &mut nndist);
        misave(x, p, n, &mut workn, &mut iworkn, imiss, &pos[1..], nnb);
        for k in 0..n {
            if iworkn[k] == 0 {
                continue;
            }
            ximp[i + k * p] = workn[k];
            if iworkn[k] == 2 {
                imiss[i + k * p] = 2;
            }
        }
    }
}

/// Two-means clustering of the rows of `x` with missing entries, started
/// from rows `istart`. Fills `clust[..nsize[i]]` of column `i` (a `p x 2`
/// column-major matrix) with the members of cluster `i`; returns
/// `(ratio, iter)`.
#[allow(clippy::too_many_arguments)]
pub fn twomis(
    x: &[f64],
    p: usize,
    n: usize,
    imiss: &[i32],
    mut maxit: i64,
    eps: f64,
    istart: [usize; 2],
    clust: &mut [i64],
    nsize: &mut [i64],
) -> (f64, i64) {
    if maxit < 1 {
        maxit = 5;
    }
    let mut x0 = vec![0.0; n * 2];
    let mut imiss0 = vec![0i32; n * 2];
    let mut dist = vec![0.0; p * 2];
    let mut iworkp = vec![0i32; p];
    let mut iworkn = vec![0i32; n];
    for i in 0..2 {
        for j in 0..n {
            x0[j + i * n] = x[istart[i] + j * p];
            imiss0[j + i * n] = imiss[istart[i] + j * p];
        }
    }
    let mut iter = 0i64;
    let mut ratio = 10.0;
    let mut dold = 0.0;
    while iter < maxit && ratio > eps {
        iter += 1;
        for i in 0..2 {
            misdis(
                &x0[i * n..(i + 1) * n],
                x,
                p,
                n,
                &imiss0[i * n..(i + 1) * n],
                imiss,
                &mut dist[i * p..(i + 1) * p],
                &mut iworkp,
            );
            nsize[i] = 0;
        }
        let mut dnew = 0.0;
        for j in 0..p {
            let imax = if dist[j] < dist[j + p] { 0 } else { 1 };
            clust[nsize[imax] as usize + imax * p] = j as i64;
            nsize[imax] += 1;
            dnew += dist[j + imax * p];
        }
        if dnew == 0.0 {
            break;
        }
        if iter == 1 {
            dold = dnew * 10.0;
        }
        ratio = (dnew - dold).abs() / dold;
        dold = dnew;
        for i in 0..2 {
            iworkn.fill(1);
            misave(
                x,
                p,
                n,
                &mut x0[i * n..(i + 1) * n],
                &mut iworkn,
                imiss,
                &clust[i * p..(i + 1) * p],
                nsize[i] as usize,
            );
            for j in 0..n {
                imiss0[j + i * n] = if iworkn[j] == 2 { 1 } else { 0 };
            }
        }
    }
    (ratio, iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn porder_is_stable_partial_sort() {
        let dist = [3.0, 1.0, 2.0, 1.0, 0.0, 2.0];
        let mut pos = [-1i64; 4];
        let mut nndist = [0.0; 4];
        porder(4, &dist, dist.len(), &mut pos, &mut nndist);
        assert_eq!(pos, [4, 1, 3, 2]);
        assert_eq!(nndist, [0.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn misdis_averages_over_shared_columns() {
        // rows: [1, 2, -], [3, -, 5], [-, -, -]
        let x = [1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 5.0, 0.0];
        let imiss = [0, 0, 1, 0, 1, 1, 1, 0, 1];
        let mut dis = [0.0; 3];
        let mut cnt = [0i32; 3];
        misdis(
            &[1.0, 2.0, 0.0],
            &x,
            3,
            3,
            &[0, 0, 1],
            &imiss,
            &mut dis,
            &mut cnt,
        );
        assert_eq!(cnt, [2, 1, 0]);
        assert_eq!(dis, [0.0, 4.0, DISMAX]);
    }
}
