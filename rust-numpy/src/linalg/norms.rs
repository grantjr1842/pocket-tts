use crate::array::Array;
use crate::error::NumPyError;
use crate::linalg::LinalgScalar;
use num_traits::{Float, One, Zero};

/// Compute the determinant of an array.
pub fn det<T>(a: &Array<T>) -> Result<T, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error("det requires 2D array", "linalg"));
    }
    let n = a.shape()[0];
    if n != a.shape()[1] {
        return Err(NumPyError::value_error(
            "det requires square matrix",
            "linalg",
        ));
    }

    let strides = a.strides();
    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    let mut data = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let a_idx = idx(i, j, strides);
            data[i * n + j] = *a
                .get(a_idx)
                .ok_or_else(|| NumPyError::invalid_operation("det index out of bounds"))?;
        }
    }

    let mut det = T::one();
    let mut sign = T::one();

    let eps = <T::Real as num_traits::Float>::epsilon();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = data[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = data[row * n + col].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = row;
            }
        }

        if pivot_val <= eps {
            return Ok(T::zero());
        }

        if pivot_row != col {
            for j in 0..n {
                data.swap(col * n + j, pivot_row * n + j);
            }
            sign = -sign;
        }

        let pivot = data[col * n + col];
        det = det * pivot;

        for row in (col + 1)..n {
            let factor = data[row * n + col] / pivot;
            if factor.abs() <= eps {
                continue;
            }
            for j in col..n {
                data[row * n + j] = data[row * n + j] - factor * data[col * n + j];
            }
        }
    }

    Ok(det * sign)
}

/// Matrix rank
pub fn matrix_rank<T>(a: &Array<T>) -> Result<usize, NumPyError>
where
    T: LinalgScalar,
{
    if a.ndim() != 2 {
        return Err(NumPyError::value_error(
            "matrix_rank requires 2D array",
            "linalg",
        ));
    }

    let rows = a.shape()[0];
    let cols = a.shape()[1];
    let strides = a.strides();

    let idx = |row: usize, col: usize, strides: &[isize]| -> usize {
        (row as isize * strides[0] + col as isize * strides[1]) as usize
    };

    let mut data = vec![T::zero(); rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let a_idx = idx(i, j, strides);
            data[i * cols + j] = *a
                .get(a_idx)
                .ok_or_else(|| NumPyError::invalid_operation("matrix_rank index out of bounds"))?;
        }
    }

    let mut rank = 0;
    let mut row = 0;
    let eps = <T::Real as num_traits::Float>::epsilon();
    let real_zero = T::Real::zero();

    for col in 0..cols {
        let mut pivot_row = row;
        let mut pivot_val = if row < rows {
            data[row * cols + col].abs()
        } else {
            real_zero
        };

        for r in (row + 1)..rows {
            let candidate = data[r * cols + col].abs();
            if candidate > pivot_val {
                pivot_val = candidate;
                pivot_row = r;
            }
        }

        if pivot_val <= eps {
            continue;
        }

        if pivot_row != row {
            for j in 0..cols {
                data.swap(row * cols + j, pivot_row * cols + j);
            }
        }

        let pivot = data[row * cols + col];
        for r in (row + 1)..rows {
            let factor = data[r * cols + col] / pivot;
            if factor.abs() <= eps {
                continue;
            }
            for j in col..cols {
                data[r * cols + j] = data[r * cols + j] - factor * data[row * cols + j];
            }
        }

        rank += 1;
        row += 1;
        if row >= rows {
            break;
        }
    }

    Ok(rank)
}

/// Normalize axis parameter, handling negative indices
fn normalize_axis(axis: isize, ndim: usize) -> Result<usize, NumPyError> {
    let normalized = if axis < 0 { axis + ndim as isize } else { axis };

    if normalized < 0 || normalized >= ndim as isize {
        return Err(NumPyError::index_error(axis as usize, ndim));
    }

    Ok(normalized as usize)
}

/// Normalize multiple axes, handling negative indices and duplicates
fn normalize_axes(axes: &[isize], ndim: usize) -> Result<Vec<usize>, NumPyError> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(axes.len());

    for &axis in axes {
        let normalized = normalize_axis(axis, ndim)?;
        if !seen.insert(normalized) {
            return Err(NumPyError::value_error(
                format!("duplicate axis in normalization: {}", axis),
                "linalg",
            ));
        }
        result.push(normalized);
    }

    Ok(result)
}

/// Compute matrix or vector norm.
pub fn norm<T>(
    x: &Array<T>,
    ord: Option<&str>,
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    // Determine norm type
    let norm_type = match ord {
        None | Some("fro") => NormType::Frobenius,
        Some("nuc") => NormType::Nuclear,
        Some("1") => NormType::L1,
        Some("2") => NormType::L2,
        Some("inf") => NormType::Linf,
        Some("-inf") => NormType::LNegInf,
        Some(s) => {
            if let Ok(p) = s.parse::<i32>() {
                if p > 0 {
                    NormType::Lp(p as u32)
                } else if p < 0 {
                    NormType::LNegP(p.abs() as u32)
                } else {
                    return Err(NumPyError::value_error(
                        "ord must be non-zero for Lp norms",
                        "linalg",
                    ));
                }
            } else {
                return Err(NumPyError::value_error(
                    format!("Invalid norm order: {}", s),
                    "linalg",
                ));
            }
        }
    };

    // Handle axis parameter
    let axes = match axis {
        None => None,
        Some(axes_slice) => Some(normalize_axes(axes_slice, x.ndim())?),
    };

    match norm_type {
        NormType::Nuclear => {
            if axes.is_some() {
                return Err(NumPyError::value_error(
                    "nuclear norm does not support axis parameter",
                    "linalg",
                ));
            }
            compute_nuclear_norm(x)
        }
        NormType::Frobenius => compute_norm_with_axis(x, 2, axes.as_deref(), keepdims),
        NormType::L1 => compute_norm_with_axis(x, 1, axes.as_deref(), keepdims),
        NormType::L2 => compute_norm_with_axis(x, 2, axes.as_deref(), keepdims),
        NormType::Linf => compute_norm_inf_with_axis(x, true, axes.as_deref(), keepdims),
        NormType::LNegInf => compute_norm_inf_with_axis(x, false, axes.as_deref(), keepdims),
        NormType::Lp(p) => compute_norm_with_axis(x, p, axes.as_deref(), keepdims),
        NormType::LNegP(p) => compute_norm_neg_p_with_axis(x, p, axes.as_deref(), keepdims),
    }
}

/// Norm type enumeration
enum NormType {
    Nuclear,
    Frobenius,
    L1,
    L2,
    Lp(u32),
    Linf,
    LNegInf,
    LNegP(u32),
}

/// Helper function to compute output shape after axis reduction
fn compute_output_shape(shape: &[usize], axes: Option<&[usize]>, keepdims: bool) -> Vec<usize> {
    match axes {
        None => {
            if keepdims {
                vec![1; shape.len()]
            } else {
                vec![]
            }
        }
        Some(axes_to_reduce) => {
            let mut result = if keepdims {
                shape.to_vec()
            } else {
                let mut temp = shape.to_vec();
                let mut sorted_axes: Vec<usize> = axes_to_reduce.to_vec();
                sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
                for &ax in &sorted_axes {
                    temp.remove(ax);
                }
                temp
            };

            if keepdims {
                for &ax in axes_to_reduce {
                    result[ax] = 1;
                }
            }

            result
        }
    }
}

/// Compute Lp norm along specified axis/axes
fn compute_norm_with_axis<T>(
    x: &Array<T>,
    p: u32,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let output_shape = compute_output_shape(x.shape(), axes, keepdims);
    let output_size: usize = output_shape.iter().product();

    // Handle scalar output case
    if output_shape.is_empty() || (output_size == 1 && axes.is_none()) {
        return compute_lp_norm(x, p);
    }

    let mut result = vec![T::Real::zero(); output_size];
    let shape = x.shape();
    let ndim = shape.len();

    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
        let mut sum_abs_p = T::Real::zero();
        let mut input_base = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_ax_ptr = 0;
            for dim in 0..ndim {
                if axes_to_reduce.contains(&dim) {
                    if keepdims {
                        input_base[dim] = 0;
                        out_ax_ptr += 1;
                    } else {
                        input_base[dim] = 0;
                    }
                } else {
                    input_base[dim] = output_idx[out_ax_ptr];
                    out_ax_ptr += 1;
                }
            }
        }

        if let Some(axes_to_reduce) = axes {
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

            let mut reduced_iter = vec![0usize; reduced_dims.len()];
            loop {
                let mut input_idx = input_base.clone();
                for (i, &(dim, _)) in reduced_dims.iter().enumerate() {
                    input_idx[dim] = reduced_iter[i];
                }

                if let Ok(val) = x.get_multi(&input_idx) {
                    let abs_val = LinalgScalar::abs(val);
                    sum_abs_p = sum_abs_p + num_traits::Float::powi(abs_val, p as i32);
                }

                let mut carry = true;
                for (i, &(_, max_val)) in reduced_dims.iter().enumerate() {
                    if carry {
                        if reduced_iter[i] + 1 < max_val {
                            reduced_iter[i] += 1;
                            carry = false;
                        } else {
                            reduced_iter[i] = 0;
                        }
                    }
                }
                if carry { break; }
            }
        }

        result[output_flat] = if p == 1 {
            sum_abs_p
        } else if p == 2 {
            num_traits::Float::sqrt(sum_abs_p)
        } else {
            let log_sum = num_traits::Float::ln(sum_abs_p);
            let inv_p = T::Real::one() / num_traits::cast(p as f64).unwrap();
            num_traits::Float::exp(log_sum * inv_p)
        };

        let mut carry = true;
        for dim in (0..output_shape.len()).rev() {
            if carry {
                if output_idx[dim] + 1 < output_shape[dim] {
                    output_idx[dim] += 1;
                    carry = false;
                } else {
                    output_idx[dim] = 0;
                }
            }
        }

        output_flat += 1;
        if carry || output_flat >= output_size {
            break;
        }
    }

    let result_t: Vec<T> = result.into_iter().map(|r| T::from(r).unwrap()).collect();
    Ok(Array::from_shape_vec(output_shape, result_t))
}

/// Compute L-infinity or L-negative-infinity norm along specified axis/axes
fn compute_norm_inf_with_axis<T>(
    x: &Array<T>,
    max_norm: bool,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let output_shape = compute_output_shape(x.shape(), axes, keepdims);
    let output_size: usize = output_shape.iter().product();

    let mut result = vec![
        if max_norm {
            T::Real::neg_infinity()
        } else {
            T::Real::infinity()
        };
        output_size
    ];
    let shape = x.shape();
    let ndim = shape.len();

    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
        let mut current_result = if max_norm {
            T::Real::neg_infinity()
        } else {
            T::Real::infinity()
        };

        let mut input_base = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_ax_ptr = 0;
            for dim in 0..ndim {
                if axes_to_reduce.contains(&dim) {
                    if keepdims {
                        input_base[dim] = 0;
                        out_ax_ptr += 1;
                    }
                } else {
                    input_base[dim] = output_idx[out_ax_ptr];
                    out_ax_ptr += 1;
                }
            }
        }

        if let Some(axes_to_reduce) = axes {
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

            let mut reduced_iter = vec![0usize; reduced_dims.len()];
            loop {
                let mut input_idx = input_base.clone();
                for (i, &(dim, _)) in reduced_dims.iter().enumerate() {
                    input_idx[dim] = reduced_iter[i];
                }

                if let Ok(val) = x.get_multi(&input_idx) {
                    let abs_val = LinalgScalar::abs(val);
                    current_result = if max_norm {
                        current_result.max(abs_val)
                    } else {
                        current_result.min(abs_val)
                    };
                }

                let mut carry = true;
                for (i, &(_, max_val)) in reduced_dims.iter().enumerate() {
                    if carry {
                        if reduced_iter[i] + 1 < max_val {
                            reduced_iter[i] += 1;
                            carry = false;
                        } else {
                            reduced_iter[i] = 0;
                        }
                    }
                }
                if carry { break; }
            }
        } else {
            for i in 0..x.size() {
                if let Some(val) = x.get_linear(i) {
                    let abs_val = LinalgScalar::abs(*val);
                    current_result = if max_norm {
                        current_result.max(abs_val)
                    } else {
                        current_result.min(abs_val)
                    };
                }
            }
        }

        result[output_flat] = current_result;

        let mut carry = true;
        for dim in (0..output_shape.len()).rev() {
            if carry {
                if output_idx[dim] + 1 < output_shape[dim] {
                    output_idx[dim] += 1;
                    carry = false;
                } else {
                    output_idx[dim] = 0;
                }
            }
        }

        output_flat += 1;
        if carry || output_flat >= output_size {
            break;
        }
    }

    let result_t: Vec<T> = result.into_iter().map(|r| T::from(r).unwrap()).collect();
    Ok(Array::from_shape_vec(output_shape, result_t))
}

/// Compute negative Lp norm along specified axis/axes
fn compute_norm_neg_p_with_axis<T>(
    x: &Array<T>,
    p: u32,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let output_shape = compute_output_shape(x.shape(), axes, keepdims);
    let output_size: usize = output_shape.iter().product();

    let mut result = vec![T::Real::zero(); output_size];
    let shape = x.shape();
    let ndim = shape.len();

    let mut output_idx = vec![0usize; output_shape.len()];
    let mut output_flat = 0usize;

    loop {
        let mut sum_abs_neg_p = T::Real::zero();
        let mut input_base = vec![0usize; ndim];
        if let Some(axes_to_reduce) = axes {
            let mut out_ax_ptr = 0;
            for dim in 0..ndim {
                if axes_to_reduce.contains(&dim) {
                    if keepdims {
                        input_base[dim] = 0;
                        out_ax_ptr += 1;
                    }
                } else {
                    input_base[dim] = output_idx[out_ax_ptr];
                    out_ax_ptr += 1;
                }
            }
        }

        if let Some(axes_to_reduce) = axes {
            let reduced_dims: Vec<(usize, usize)> =
                axes_to_reduce.iter().map(|&ax| (ax, shape[ax])).collect();

            let mut reduced_iter = vec![0usize; reduced_dims.len()];
            loop {
                let mut input_idx = input_base.clone();
                for (i, &(dim, _)) in reduced_dims.iter().enumerate() {
                    input_idx[dim] = reduced_iter[i];
                }

                if let Ok(val) = x.get_multi(&input_idx) {
                    let abs_val = LinalgScalar::abs(val);
                    if abs_val > T::Real::zero() {
                        sum_abs_neg_p = sum_abs_neg_p + num_traits::Float::powi(abs_val, -(p as i32));
                    }
                }

                let mut carry = true;
                for (i, &(_, max_val)) in reduced_dims.iter().enumerate() {
                    if carry {
                        if reduced_iter[i] + 1 < max_val {
                            reduced_iter[i] += 1;
                            carry = false;
                        } else {
                            reduced_iter[i] = 0;
                        }
                    }
                }
                if carry { break; }
            }
        }

        result[output_flat] = if sum_abs_neg_p > T::Real::zero() {
            num_traits::Float::powi(sum_abs_neg_p, -(1 as i32))
        } else {
            T::Real::infinity()
        };

        let mut carry = true;
        for dim in (0..output_shape.len()).rev() {
            if carry {
                if output_idx[dim] + 1 < output_shape[dim] {
                    output_idx[dim] += 1;
                    carry = false;
                } else {
                    output_idx[dim] = 0;
                }
            }
        }

        output_flat += 1;
        if carry || output_flat >= output_size {
            break;
        }
    }

    let result_t: Vec<T> = result.into_iter().map(|r| T::from(r).unwrap()).collect();
    Ok(Array::from_shape_vec(output_shape, result_t))
}

/// Compute nuclear norm (sum of singular values)
fn compute_nuclear_norm<T>(x: &Array<T>) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if x.ndim() != 2 {
        return Err(NumPyError::value_error("nuclear norm requires 2D array", "linalg"));
    }
    let singular_values = compute_singular_values(x)?;
    let mut sum = T::Real::zero();
    for sv in &singular_values { sum = sum + *sv; }
    Ok(Array::from_vec(vec![T::from(sum).unwrap()]))
}

fn compute_singular_values<T>(a: &Array<T>) -> Result<Vec<T::Real>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    let m = a.shape()[0];
    let n = a.shape()[1];
    let mut ata = vec![T::Real::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = T::Real::zero();
            for k in 0..m {
                let val_ki = LinalgScalar::abs(a.get_multi(&[k, i])?);
                let val_kj = LinalgScalar::abs(a.get_multi(&[k, j])?);
                sum = sum + val_ki * val_kj;
            }
            ata[i * n + j] = sum;
        }
    }
    compute_eigenvalues_symmetric_real(&ata, n)
        .map(|evs| evs.into_iter().map(|ev| if ev > T::Real::zero() { ev.sqrt() } else { T::Real::zero() }).collect())
}

fn compute_eigenvalues_symmetric_real<R>(a: &[R], n: usize) -> Result<Vec<R>, NumPyError>
where
    R: num_traits::Float + Clone,
{
    if n == 0 { return Ok(vec![]); }
    if n == 1 { return Ok(vec![a[0].clone()]); }
    let mut h: Vec<R> = a.to_vec();
    let eps = R::epsilon() * num_traits::cast(1000.0).unwrap();
    let mut m = n;
    let mut iter = 0;
    while m > 1 && iter < 100 * n {
        let mut k = m - 1;
        while k > 0 {
            let off_diag = h[k * n + (k - 1)].abs();
            let diag_sum = h[(k - 1) * n + (k - 1)].abs() + h[k * n + k].abs();
            if off_diag <= eps * diag_sum {
                h[k * n + (k - 1)] = R::zero();
                h[(k - 1) * n + k] = R::zero();
                break;
            }
            k -= 1;
        }
        if k == m - 1 { m -= 1; continue; }
        let d = (h[(m - 2) * n + (m - 2)] - h[(m - 1) * n + (m - 1)]) / num_traits::cast(2.0).unwrap();
        let t_sq = h[(m - 1) * n + (m - 2)] * h[(m - 1) * n + (m - 2)];
        let shift = h[(m - 1) * n + (m - 1)] - t_sq / (d + if d >= R::zero() { R::one() } else { -R::one() } * (d * d + t_sq).sqrt());
        let mut x = h[0 * n + 0] - shift;
        let mut z = h[1 * n + 0];
        for i in 0..m - 1 {
            let r = (x * x + z * z).sqrt();
            if r > eps {
                let c = x / r; let s = z / r;
                for j in i..n {
                    let t1 = h[i * n + j]; let t2 = h[(i + 1) * n + j];
                    h[i * n + j] = c * t1 + s * t2; h[(i + 1) * n + j] = -s * t1 + c * t2;
                }
                for j in 0..std::cmp::min(i + 3, n) {
                    let t1 = h[j * n + i]; let t2 = h[j * n + (i + 1)];
                    h[j * n + i] = c * t1 + s * t2; h[j * n + (i + 1)] = -s * t1 + c * t2;
                }
            }
            if i < m - 2 { x = h[(i + 1) * n + i]; z = h[(i + 2) * n + i]; }
        }
        iter += 1;
    }
    Ok((0..n).map(|i| h[i * n + i]).collect())
}

fn compute_lp_norm<T>(x: &Array<T>, p: u32) -> Result<Array<T>, NumPyError>
where
    T: LinalgScalar + num_traits::Float,
{
    if p == 0 { return Err(NumPyError::value_error("p must be positive", "linalg")); }
    let mut sum_abs_p = T::Real::zero();
    for i in 0..x.size() {
        if let Some(val) = x.get_linear(i) {
            sum_abs_p = sum_abs_p + LinalgScalar::abs(*val).powi(p as i32);
        }
    }
    let res = if p == 1 { sum_abs_p } else if p == 2 { sum_abs_p.sqrt() } else { sum_abs_p.powf(T::Real::one() / num_traits::cast(p).unwrap()) };
    Ok(Array::from_vec(vec![T::from(res).unwrap()]))
}
