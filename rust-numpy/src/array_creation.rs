//! Array creation and manipulation functions

use crate::array::Array;
use crate::dtype::Dtype;
use crate::error::{NumPyError, Result};

/// Create an array from a Python-like list of values.
/// This is a simplified version of NumPy's np.array() that creates a 1D array from a slice of f32 values.
pub fn array<T>(data: Vec<T>, _dtype: Option<Dtype>) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let arr = Array::from_vec(data);
    Ok(arr)
}

/// Create an array from a range of values (similar to np.arange).
pub fn arange(start: f32, stop: f32, step: Option<f32>) -> Result<Array<f32>> {
    let step_val = step.unwrap_or(1.0f32);

    if step_val == 0.0 {
        return Err(NumPyError::invalid_value("step cannot be zero"));
    }

    let num_elements = if step_val > 0.0 {
        if stop <= start {
            return Ok(Array::from_vec(vec![]));
        }
        ((stop - start) / step_val).ceil() as usize
    } else {
        if stop >= start {
            return Ok(Array::from_vec(vec![]));
        }
        ((start - stop) / step_val.abs()).ceil() as usize
    };

    let data: Vec<f32> = (0..num_elements)
        .map(|i| start + (i as f32) * step_val)
        .collect();

    Ok(Array::from_vec(data))
}

/// Clip values to be within a specified range (similar to np.clip).
pub fn clip<T>(array: &Array<T>, a_min: Option<T>, a_max: Option<T>) -> Result<Array<T>>
where
    T: PartialOrd + Clone + num_traits::Bounded + Default + 'static,
{
    let clipped: Vec<T> = array
        .data()
        .iter()
        .map(|x| {
            let mut val = x.clone();
            if let Some(min_v) = &a_min {
                if val < *min_v {
                    val = min_v.clone();
                }
            }
            if let Some(max_v) = &a_max {
                if val > *max_v {
                    val = max_v.clone();
                }
            }
            val
        })
        .collect();

    Ok(Array::from_vec(clipped))
}

/// Create evenly spaced numbers over a specified interval (similar to np.linspace).
pub fn linspace(
    start: f32,
    stop: f32,
    num: Option<usize>,
    endpoint: Option<bool>,
) -> Result<Array<f32>> {
    let num_val = num.unwrap_or(50);

    if num_val == 0 {
        return Ok(Array::from_vec(vec![]));
    }

    if num_val == 1 {
        return Ok(Array::from_vec(vec![start]));
    }

    let endpoint_val = endpoint.unwrap_or(true);
    let div = if endpoint_val {
        (num_val - 1) as f32
    } else {
        num_val as f32
    };

    let step = (stop - start) / div;
    let data: Vec<f32> = (0..num_val).map(|i| start + (i as f32) * step).collect();

    Ok(Array::from_vec(data))
}

/// Create evenly spaced numbers on a log scale (similar to np.logspace).
pub fn logspace(
    start: f32,
    stop: f32,
    num: Option<usize>,
    endpoint: Option<bool>,
    base: Option<f32>,
) -> Result<Array<f32>> {
    let base_val = base.unwrap_or(10.0);
    let lin = linspace(start, stop, num, endpoint)?;
    let logged: Vec<f32> = lin.data().iter().map(|&x| base_val.powf(x)).collect();

    Ok(Array::from_vec(logged))
}

/// Create evenly spaced numbers on a geometric progression (similar to np.geomspace).
pub fn geomspace(
    start: f32,
    stop: f32,
    num: Option<usize>,
    endpoint: Option<bool>,
) -> Result<Array<f32>> {
    if start == 0.0 || stop == 0.0 {
        return Err(NumPyError::invalid_value(
            "geomspace: start and stop must not be zero",
        ));
    }

    let start_log = start.ln();
    let stop_log = stop.ln();
    let lin = linspace(start_log, stop_log, num, endpoint)?;
    let geo: Vec<f32> = lin.data().iter().map(|&x| x.exp()).collect();

    Ok(Array::from_vec(geo))
}

pub fn full<T>(shape: &[usize], fill_value: T) -> Result<Array<T>>
where
    T: Clone + Default + 'static,
{
    let size: usize = shape.iter().product();
    let data: Vec<T> = (0..size).map(|_| fill_value.clone()).collect();
    Ok(Array::from_data(data, shape.to_vec()))
}

/// Generate a Vandermonde matrix.
pub fn vander<T>(x: &Array<T>, n: Option<usize>, increasing: bool) -> Result<Array<T>>
where
    T: Clone + Default + num_traits::Num + num_traits::Pow<usize, Output = T> + 'static,
{
    if x.ndim() != 1 {
        return Err(NumPyError::invalid_value("vander: x must be 1-D"));
    }

    let m = x.size();
    let n_val = n.unwrap_or(m);
    let mut data = Vec::with_capacity(m * n_val);

    for i in 0..m {
        let val = x.get(i).ok_or_else(|| {
            NumPyError::invalid_value(format!("vander: failed to get element at {}", i))
        })?;

        if increasing {
            for j in 0..n_val {
                data.push(val.clone().pow(j));
            }
        } else {
            for j in (0..n_val).rev() {
                data.push(val.clone().pow(j));
            }
        }
    }

    Ok(Array::from_shape_vec(vec![m, n_val], data))
}

pub fn min<T>(array: &Array<T>) -> Result<T>
where
    T: PartialOrd + Copy + num_traits::Bounded + 'static,
{
    if array.size() == 0 {
        return Err(NumPyError::invalid_value("min() arg is an empty sequence"));
    }

    Ok(array
        .data()
        .iter()
        .cloned()
        .fold(T::max_value(), |a, b| if a < b { a } else { b }))
}

pub fn log<T>(array: &Array<T>) -> Result<Array<T>>
where
    T: num_traits::Float + Default + 'static,
{
    let logged: Vec<T> = array
        .data()
        .iter()
        .map(|&x| {
            let x_f64 = x.to_f64().unwrap_or(0.0);
            if x_f64 <= 0.0 {
                T::from(f64::NEG_INFINITY).unwrap_or(T::zero())
            } else {
                T::from(x_f64.ln()).unwrap_or(T::zero())
            }
        })
        .collect();

    Ok(Array::from_vec(logged))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arange_basic() {
        let arr = arange(0.0, 10.0, None).unwrap();
        assert_eq!(arr.shape(), &[10]);
        let data = arr.data();
        for i in 0..10 {
            assert_eq!(data[i], i as f32);
        }
    }

    #[test]
    fn test_arange_with_step() {
        let arr = arange(0.0, 10.0, Some(2.0)).unwrap();
        assert_eq!(arr.shape(), &[5]);
        let data = arr.data();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 4.0);
        assert_eq!(data[3], 6.0);
        assert_eq!(data[4], 8.0);
    }

    #[test]
    fn test_arange_negative_step() {
        let arr = arange(10.0, 0.0, Some(-1.0)).unwrap();
        assert_eq!(arr.shape(), &[10]);
        let data = arr.data();
        for i in 0..10 {
            assert_eq!(data[i], (10.0 - i as f32));
        }
    }

    #[test]
    fn test_clip_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::from_vec(input);
        let clipped = clip(&arr, Some(2.0), Some(4.0)).unwrap();
        let data = clipped.data();
        assert_eq!(data[0], 2.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
        assert_eq!(data[3], 4.0);
        assert_eq!(data[4], 4.0);
    }

    #[test]
    fn test_clip_no_min() {
        let input = vec![1.0, 2.0, 3.0];
        let arr = Array::from_vec(input);
        let clipped = clip(&arr, None, Some(2.0)).unwrap();
        let data = clipped.data();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 2.0);
    }

    #[test]
    fn test_clip_no_max() {
        let input = vec![1.0, 2.0, 3.0];
        let arr = Array::from_vec(input);
        let clipped = clip(&arr, Some(2.0), None).unwrap();
        let data = clipped.data();
        assert_eq!(data[0], 2.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
    }

    #[test]
    fn test_min() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let arr = Array::from_vec(input);
        let min_val = min(&arr).unwrap();
        assert_eq!(min_val, 1.0);
    }

    #[test]
    fn test_min_empty() {
        let input: Vec<f32> = vec![];
        let arr = Array::from_vec(input);
        let result = min(&arr);
        assert!(result.is_err());
    }

    #[test]
    fn test_log() {
        let input = vec![1.0, 2.0, 10.0, 0.5, std::f32::consts::E];
        let arr = Array::from_vec(input);
        let logged = log(&arr).unwrap();
        let data = logged.data();
        assert!((data[0] - 1.0f32.ln()).abs() < 1e-6);
        assert!((data[1] - 2.0f32.ln()).abs() < 1e-6);
        assert!((data[2] - 10.0f32.ln()).abs() < 1e-6);
        assert!((data[3] - 0.5f32.ln()).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_negative() {
        let input = vec![-1.0, 0.0, 1.0];
        let arr = Array::from_vec(input);
        let logged = log(&arr).unwrap();
        let data = logged.data();
        let v0: f32 = data[0];
        let v1: f32 = data[1];
        assert!(v0.is_infinite() && v0 < 0.0);
        assert!(v1.is_infinite() && v1 < 0.0);
        assert!((data[2] - 1.0f32.ln()).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_basic() {
        let arr = linspace(0.0, 10.0, Some(5), Some(true)).unwrap();
        assert_eq!(arr.shape(), &[5]);
        let data = arr.data();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 2.5).abs() < 1e-6);
        assert!((data[2] - 5.0).abs() < 1e-6);
        assert!((data[3] - 7.5).abs() < 1e-6);
        assert!((data[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_no_endpoint() {
        let arr = linspace(0.0, 10.0, Some(5), Some(false)).unwrap();
        assert_eq!(arr.shape(), &[5]);
        let data = arr.data();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[4] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_default_num() {
        let arr = linspace(0.0, 10.0, None, Some(true)).unwrap();
        assert_eq!(arr.shape(), &[50]);
    }

    #[test]
    fn test_linspace_single_value() {
        let arr = linspace(5.0, 10.0, Some(1), Some(true)).unwrap();
        assert_eq!(arr.shape(), &[1]);
        assert_eq!(arr.data()[0], 5.0);
    }

    #[test]
    fn test_logspace_basic() {
        let arr = logspace(0.0, 3.0, Some(4), Some(true), Some(10.0)).unwrap();
        assert_eq!(arr.shape(), &[4]);
        let data = arr.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        assert!((data[2] - 100.0).abs() < 1e-6);
        assert!((data[3] - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_logspace_default_base() {
        let arr = logspace(0.0, 2.0, Some(3), Some(true), None).unwrap();
        assert_eq!(arr.shape(), &[3]);
        let data = arr.data();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);
        assert!((data[2] - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_geomspace_basic() {
        let arr = geomspace(1.0, 1000.0, Some(4), Some(true)).unwrap();
        assert_eq!(arr.shape(), &[4]);
        let data = arr.data();
        assert!((data[0] - 1.0).abs() < 1e-4);
        assert!((data[1] - 10.0).abs() < 1e-3);
        assert!((data[2] - 100.0).abs() < 1e-2); // Allow more tolerance for larger values
        assert!((data[3] - 1000.0).abs() < 1e-1);
    }

    #[test]
    fn test_geomspace_zero_error() {
        let result = geomspace(0.0, 100.0, Some(5), Some(true));
        assert!(result.is_err());
    }

    #[test]
    fn test_full_1d() {
        let arr = full(&[5], 7.0_f32).unwrap();
        assert_eq!(arr.shape(), &[5]);
        let data = arr.data();
        for &v in data.iter() {
            assert_eq!(v, 7.0);
        }
    }

    #[test]
    fn test_full_2d() {
        let arr = full(&[2, 3], 5.0_f32).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.size(), 6);
        let data = arr.data();
        for &v in data.iter() {
            assert_eq!(v, 5.0);
        }
    }

    #[test]
    fn test_full_empty() {
        let arr = full(&[0], 5.0_f32).unwrap();
        assert_eq!(arr.shape(), &[0]);
        assert_eq!(arr.size(), 0);
    }

    #[test]
    fn test_vander_basic() {
        let x = Array::from_vec(vec![1, 2, 3]);
        let result = vander(&x, None, false).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result.data(), &[1, 1, 1, 4, 2, 1, 9, 3, 1]);
    }

    #[test]
    fn test_vander_increasing() {
        let x = Array::from_vec(vec![1, 2, 3]);
        let result = vander(&x, None, true).unwrap();
        assert_eq!(result.data(), &[1, 1, 1, 1, 2, 4, 1, 3, 9]);
    }

    #[test]
    fn test_vander_n() {
        let x = Array::from_vec(vec![1, 2, 3]);
        let result = vander(&x, Some(2), false).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data(), &[1, 1, 2, 1, 3, 1]);
    }
}
