use crate::dtype::DType;
use crate::error::NumPyError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    RowMajor,
    ColumnMajor,
}

impl MemoryOrder {
    pub fn from_strides(shape: &[usize], strides: &[isize]) -> Self {
        if shape.len() != strides.len() {
            return MemoryOrder::RowMajor;
        }

        let mut c_order_score = 0;
        let mut f_order_score = 0;

        for i in 0..shape.len() {
            if strides[i] > 0 {
                if i == 0 || strides[i] == strides[i - 1] * shape[i - 1] as isize {
                    c_order_score += 1;
                }
                if i == 0 || strides[i] == strides[i - 1] / shape[i - 1] as isize {
                    f_order_score += 1;
                }
            }
        }

        if c_order_score >= f_order_score {
            MemoryOrder::RowMajor
        } else {
            MemoryOrder::ColumnMajor
        }
    }
}

pub fn optimize_memory_layout<T>(
    data: &mut [T],
    shape: &[usize],
    from_order: MemoryOrder,
    to_order: MemoryOrder,
) -> Result<(), NumPyError> {
    if from_order == to_order {
        return Ok(());
    }

    match (from_order, to_order) {
        (MemoryOrder::RowMajor, MemoryOrder::ColumnMajor) => {
            transpose_in_place(data, shape)?;
        }
        (MemoryOrder::ColumnMajor, MemoryOrder::RowMajor) => {
            transpose_in_place(data, shape)?;
        }
        _ => {}
    }

    Ok(())
}

fn transpose_in_place<T>(data: &mut [T], shape: &[usize]) -> Result<(), NumPyError> {
    if shape.len() != 2 {
        return Err(NumPyError::invalid_value(
            "Transpose only supported for 2D arrays",
        ));
    }

    let (rows, cols) = (shape[0], shape[1]);

    for i in 0..rows {
        for j in i + 1..cols {
            let idx_i = i * cols + j;
            let idx_j = j * rows + i;
            data.swap(idx_i, idx_j);
        }
    }

    Ok(())
}

pub fn calculate_strides(shape: &[usize], dtype: &DType, order: MemoryOrder) -> Vec<isize> {
    let element_size = dtype.item_size();
    let mut strides = Vec::with_capacity(shape.len());

    match order {
        MemoryOrder::RowMajor => {
            let mut stride = element_size as isize;
            for &dim in shape.iter().rev() {
                strides.insert(0, stride);
                stride *= dim as isize;
            }
        }
        MemoryOrder::ColumnMajor => {
            let mut stride = element_size as isize;
            for &dim in shape.iter() {
                strides.push(stride);
                stride *= dim as isize;
            }
        }
    }

    strides
}

pub fn is_contiguous(shape: &[usize], strides: &[isize], order: MemoryOrder) -> bool {
    if shape.len() != strides.len() {
        return false;
    }

    let expected_strides = calculate_strides(shape, &crate::dtype::DType::Float32, order);

    strides
        .iter()
        .zip(expected_strides.iter())
        .all(|(a, b)| *a == *b)
}
