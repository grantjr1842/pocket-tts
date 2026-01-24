use crate::dtype::Dtype;
use crate::error::NumPyError;

pub struct CoalescingInfo {
    pub coalesced_dims: Vec<usize>,
    pub coalesced_strides: Vec<isize>,
    pub coalesced_shape: Vec<usize>,
}

pub fn coalesce_dimensions(
    shape: &[usize],
    strides: &[isize],
    dtype: &Dtype,
) -> Result<CoalescingInfo, NumPyError> {
    if shape.len() != strides.len() {
        return Err(NumPyError::invalid_value(
            "Shape and strides must have same length",
        ));
    }

    if shape.is_empty() {
        return Ok(CoalescingInfo {
            coalesced_dims: vec![],
            coalesced_strides: vec![],
            coalesced_shape: vec![],
        });
    }

    let element_size = dtype.item_size() as isize;
    let mut coalesced_dims = Vec::new();
    let mut coalesced_strides = Vec::new();
    let mut coalesced_shape = Vec::new();

    let mut current_contiguous_block = Vec::new();
    let mut current_size = 1;

    for i in 0..shape.len() {
        current_contiguous_block.push(i);
        current_size *= shape[i];

        if i == shape.len() - 1 || !is_contiguous_dimension(shape, strides, i) {
            let block_shape: usize = current_contiguous_block
                .iter()
                .map(|&dim| shape[dim])
                .product();
            let block_stride = if current_contiguous_block.is_empty() {
                element_size
            } else {
                strides[*current_contiguous_block.first().unwrap()]
            };

            coalesced_dims.extend_from_slice(&current_contiguous_block);
            coalesced_shape.push(block_shape);
            coalesced_strides.push(block_stride);

            current_contiguous_block.clear();
            current_size = 1;
        }
    }

    Ok(CoalescingInfo {
        coalesced_dims,
        coalesced_strides,
        coalesced_shape,
    })
}

fn is_contiguous_dimension(shape: &[usize], strides: &[isize], dim: usize) -> bool {
    if dim == shape.len() - 1 {
        return true;
    }

    let expected_stride = if dim == 0 {
        strides[dim]
    } else if strides[dim] > 0 {
        strides[dim - 1] * shape[dim - 1] as isize
    } else {
        0
    };

    strides[dim] == expected_stride
}

pub fn optimize_for_cache<T>(
    data: &[T],
    shape: &[usize],
    strides: &[isize],
) -> Result<Vec<usize>, NumPyError> {
    let cache_line_size = 64;
    let element_size = std::mem::size_of::<T>();

    let cache_optimized_order = if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        let row_major_stride = strides[0] > 0 && strides[1] == strides[0] * rows as isize;
        let col_major_stride = strides[1] > 0 && strides[0] == strides[1] * cols as isize;

        if row_major_stride {
            let row_size = cols * element_size;
            if row_size <= cache_line_size {
                vec![0, 1]
            } else {
                let optimal_block_size = cache_line_size / element_size;
                vec![0, 1]
            }
        } else if col_major_stride {
            let col_size = rows * element_size;
            if col_size <= cache_line_size {
                vec![1, 0]
            } else {
                vec![1, 0]
            }
        } else {
            (0..shape.len()).collect()
        }
    } else {
        (0..shape.len()).collect()
    };

    Ok(cache_optimized_order)
}

pub fn should_coalesce_for_operation(operation_type: OperationType) -> bool {
    matches!(
        operation_type,
        OperationType::Reduction | OperationType::Ufunc
    )
}

#[derive(Debug, Clone, Copy)]
pub enum OperationType {
    Reduction,
    Ufunc,
    Indexing,
    Transform,
}
