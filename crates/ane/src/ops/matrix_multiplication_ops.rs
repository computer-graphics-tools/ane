use crate::DataType;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, ensure};
use crate::ir::{BuiltinOp, Operator, Parameter, Value};

impl Graph {
    pub fn matrix_multiplication(
        &self,
        left_hand_side: &Tensor,
        right_hand_side: &Tensor,
        transpose_x: bool,
        transpose_y: bool,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*left_hand_side)?;
        self.numeric(*right_hand_side)?;
        let (left_vector, right_vector) = (left_hand_side.rank() == 1, right_hand_side.rank() == 1);
        if left_vector || right_vector {
            ensure(
                !(left_vector && transpose_x) && !(right_vector && transpose_y),
                GraphError::InvalidArgument("vectors cannot be transposed"),
            )?;
            let left = if left_vector {
                self.reshape(left_hand_side, [1, left_hand_side.shape()[0]])?
            } else {
                *left_hand_side
            };
            let right = if right_vector {
                self.reshape(right_hand_side, [right_hand_side.shape()[0], 1])?
            } else {
                *right_hand_side
            };
            let product = self.matrix_multiplication(&left, &right, transpose_x, transpose_y)?;
            let mut shape = product.shape().to_vec();
            if right_vector {
                shape.pop();
            }
            if left_vector {
                shape.remove(shape.len() - if right_vector { 1 } else { 2 });
            }
            return self.reshape_to(product, &shape);
        }
        let shape =
            BuiltinOp::matmul_shape(*left_hand_side, *right_hand_side, transpose_x, transpose_y)?;
        let rank = left_hand_side.rank().max(right_hand_side.rank());
        self.builtin(
            Operator::Matmul,
            &[
                (Parameter::X, *left_hand_side),
                (Parameter::Y, *right_hand_side),
            ],
            &[
                (Parameter::TransposeX, Value::Bool(transpose_x)),
                (Parameter::TransposeY, Value::Bool(transpose_y)),
            ],
            &shape[4 - rank..],
            DataType::Float16,
        )
    }

    pub fn band_part(&self, input: &Tensor, lower: i64, upper: i64) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        ensure(
            input.rank() >= 2,
            GraphError::ShapeMismatch("band part requires a matrix"),
        )?;
        let [.., rows, columns] = input.physical_shape();
        let keep: Vec<f32> = (0..rows * columns)
            .map(|i| {
                let (row, column) = ((i / columns) as i64, (i % columns) as i64);
                let inside =
                    (lower < 0 || row - column <= lower) && (upper < 0 || column - row <= upper);
                f32::from(u8::from(inside))
            })
            .collect();
        let keep = self.constant(&keep, [rows, columns])?;
        let keep = self.cast(&keep, DataType::Bool)?;
        let zero = self.constant_scalar(0.0, &[])?;
        let zero = if input.data_type() == DataType::Float16 {
            zero
        } else {
            self.cast(&zero, input.data_type())?
        };
        self.select(&keep, input, &zero)
    }

    pub fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, GraphError> {
        let query = *query;
        let key = *key;
        let value = *value;
        let mask = mask.copied();
        self.numeric(query)?;
        self.numeric(key)?;
        self.numeric(value)?;
        ensure(
            query.rank() >= 2
                && key.rank() >= 2
                && value.rank() >= 2
                && query.physical_shape()[3] == key.physical_shape()[3]
                && key.physical_shape()[2] == value.physical_shape()[2],
            GraphError::ShapeMismatch("attention dimensions differ"),
        )?;
        ensure(
            (0..2).all(|a| {
                (query.physical_shape()[a] == key.physical_shape()[a]
                    || query.physical_shape()[a] == 1
                    || key.physical_shape()[a] == 1)
                    && (query.physical_shape()[a].max(key.physical_shape()[a])
                        == value.physical_shape()[a]
                        || value.physical_shape()[a] == 1)
            }),
            GraphError::ShapeMismatch("attention batch dimensions differ"),
        )?;
        let shape = [
            query.physical_shape()[0].max(key.physical_shape()[0]),
            query.physical_shape()[1].max(key.physical_shape()[1]),
            query.physical_shape()[2],
            value.physical_shape()[3],
        ];
        let mut inputs = vec![
            (Parameter::Query, query),
            (Parameter::Key, key),
            (Parameter::Value, value),
        ];
        if let Some(mask) = mask {
            self.check_tensor(mask)?;
            let scores = [shape[0], shape[1], shape[2], key.physical_shape()[2]];
            ensure(
                matches!(mask.data_type(), DataType::Float16 | DataType::Bool)
                    && (0..4).all(|a| {
                        mask.physical_shape()[a] == 1 || mask.physical_shape()[a] == scores[a]
                    }),
                GraphError::ShapeMismatch("attention mask cannot broadcast to scores"),
            )?;
            let mask = if mask.data_type() == DataType::Bool {
                let mask = self.cast(&mask, DataType::Float16)?;
                let mask = self.add_scalar(&mask, -1.0)?;
                self.multiply_scalar(&mask, 65504.0)?
            } else {
                mask
            };
            inputs.push((Parameter::AttnMask, mask));
        }
        let rank = query.rank().max(key.rank()).max(value.rank());
        self.builtin(
            Operator::ScaledDotProductAttention,
            &inputs,
            &[],
            &shape[4 - rank..],
            DataType::Float16,
        )
    }

    pub fn rotary_embedding(
        &self,
        input: &Tensor,
        cosines: &Tensor,
        sines: &Tensor,
        rotary_width: usize,
    ) -> Result<Tensor, GraphError> {
        self.numeric(*input)?;
        ensure(
            input.rank() > 0
                && rotary_width > 0
                && rotary_width.is_multiple_of(2)
                && rotary_width <= input.physical_shape()[3],
            GraphError::InvalidArgument("invalid rotary embedding width"),
        )?;
        let mut begin = vec![0; input.rank()];
        let mut shape = input.shape().to_vec();
        let axis = input.rank() - 1;
        shape[axis] = rotary_width / 2;
        let first = self.slice(input, &begin, &shape)?;
        begin[axis] = rotary_width / 2;
        let second = self.slice(input, &begin, &shape)?;
        let negative = self.negative(&second)?;
        let rotated = self.concat(&[&negative, &first], axis)?;
        let original = self.concat(&[&first, &second], axis)?;
        let real = self.multiplication(&original, cosines)?;
        let imaginary = self.multiplication(&rotated, sines)?;
        let result = self.addition(&real, &imaginary)?;
        if rotary_width == input.physical_shape()[3] {
            return Ok(result);
        }
        begin[axis] = rotary_width;
        shape[axis] = input.physical_shape()[3] - rotary_width;
        let tail = self.slice(input, &begin, &shape)?;
        self.concat(&[&result, &tail], axis)
    }

    fn einsum_operand(
        &self,
        input: Tensor,
        labels: &[u8],
        order: &[u8],
    ) -> Result<Tensor, GraphError> {
        let mut tensor = input;
        let mut labels = labels.to_vec();
        for axis in (0..labels.len()).rev() {
            if !order.contains(&labels[axis]) {
                tensor = self.reduction_sum(&tensor, axis as i64)?;
                tensor = self.squeeze(&tensor, &[axis])?;
                labels.remove(axis);
            }
        }
        let permutation: Vec<_> = order
            .iter()
            .map(|label| labels.iter().position(|l| l == label).unwrap())
            .collect();
        self.transpose(&tensor, &permutation)
    }

    fn einsum_pair(
        &self,
        left: (Tensor, &[u8]),
        right: (Tensor, &[u8]),
        output: &[u8],
        dimensions: &std::collections::HashMap<u8, usize>,
    ) -> Result<Tensor, GraphError> {
        let (a, al) = left;
        let (b, bl) = right;
        let batch: Vec<_> = al
            .iter()
            .copied()
            .filter(|l| bl.contains(l) && output.contains(l))
            .collect();
        let free_a: Vec<_> = al
            .iter()
            .copied()
            .filter(|l| !bl.contains(l) && output.contains(l))
            .collect();
        let free_b: Vec<_> = bl
            .iter()
            .copied()
            .filter(|l| !al.contains(l) && output.contains(l))
            .collect();
        let contracted: Vec<_> = al
            .iter()
            .copied()
            .filter(|l| bl.contains(l) && !output.contains(l))
            .collect();
        let order_a: Vec<_> = batch
            .iter()
            .chain(&free_a)
            .chain(&contracted)
            .copied()
            .collect();
        let order_b: Vec<_> = batch
            .iter()
            .chain(&contracted)
            .chain(&free_b)
            .copied()
            .collect();
        let a = self.einsum_operand(a, al, &order_a)?;
        let b = self.einsum_operand(b, bl, &order_b)?;
        let product = |labels: &[u8]| {
            labels.iter().try_fold(1usize, |n, label| {
                n.checked_mul(dimensions[label]).ok_or(GraphError::Overflow)
            })
        };
        let batch_size = product(&batch)?;
        let m = product(&free_a)?;
        let k = product(&contracted)?;
        let n = product(&free_b)?;
        let a = self.reshape_to(a, &[batch_size, 1, m, k])?;
        let b = self.reshape_to(b, &[batch_size, 1, k, n])?;
        let value = if n == 1 && m > 1 {
            self.matrix_multiplication(&b, &a, true, true)?
        } else {
            self.matrix_multiplication(&a, &b, false, false)?
        };
        let ordered: Vec<_> = batch
            .iter()
            .chain(&free_a)
            .chain(&free_b)
            .copied()
            .collect();
        let shape: Vec<_> = ordered.iter().map(|l| dimensions[l]).collect();
        let value = self.reshape_to(value, &shape)?;
        let permutation: Vec<_> = output
            .iter()
            .map(|label| ordered.iter().position(|l| l == label).unwrap())
            .collect();
        self.transpose(&value, permutation)
    }

    pub fn einsum(&self, inputs: &[&Tensor], equation: &str) -> Result<Tensor, GraphError> {
        let inputs: Vec<Tensor> = inputs.iter().map(|&&tensor| tensor).collect();
        let inputs = &inputs[..];
        let (operands, output) = equation
            .split_once("->")
            .ok_or(GraphError::InvalidArgument(
                "einsum requires explicit output labels",
            ))?;
        let operands: Vec<_> = operands.split(',').collect();
        ensure(
            !inputs.is_empty()
                && operands.len() == inputs.len()
                && output.len() <= 4
                && output.bytes().all(|c| c.is_ascii_alphabetic()),
            GraphError::InvalidArgument("invalid einsum equation"),
        )?;
        let output = output.as_bytes();
        ensure(
            output
                .iter()
                .enumerate()
                .all(|(i, l)| !output[..i].contains(l)),
            GraphError::InvalidArgument("duplicate einsum output label"),
        )?;
        let mut dimensions = std::collections::HashMap::new();
        let mut work = Vec::new();
        for (&input, labels) in inputs.iter().zip(operands) {
            self.numeric(input)?;
            let labels = labels.as_bytes();
            ensure(
                labels.len() == input.rank() && labels.iter().all(u8::is_ascii_alphabetic),
                GraphError::InvalidArgument("einsum requires explicit labels matching tensor rank"),
            )?;
            ensure(
                labels
                    .iter()
                    .enumerate()
                    .all(|(i, l)| !labels[..i].contains(l)),
                GraphError::UnsupportedComposition(
                    "diagonal einsum labels require an explicit gather",
                ),
            )?;
            for (&label, &size) in labels.iter().zip(input.shape()) {
                if let Some(old) = dimensions.insert(label, size) {
                    ensure(
                        old == size,
                        GraphError::ShapeMismatch("einsum label dimensions differ"),
                    )?;
                }
            }
            work.push((input, labels.to_vec()));
        }
        ensure(
            output.iter().all(|l| dimensions.contains_key(l)),
            GraphError::InvalidArgument("einsum output label has no input"),
        )?;
        while work.len() > 1 {
            let mut best = None;
            for a in 0..work.len() {
                for b in a + 1..work.len() {
                    let mut labels = Vec::new();
                    for &label in work[a].1.iter().chain(&work[b].1) {
                        if !labels.contains(&label)
                            && (output.contains(&label)
                                || work
                                    .iter()
                                    .enumerate()
                                    .any(|(i, (_, ls))| i != a && i != b && ls.contains(&label)))
                        {
                            labels.push(label);
                        }
                    }
                    if labels.len() > 4 {
                        continue;
                    }
                    let size = labels
                        .iter()
                        .try_fold(1usize, |n, l| n.checked_mul(dimensions[l]));
                    if let Some(size) = size
                        && best.as_ref().is_none_or(|(_, _, _, old)| size < *old)
                    {
                        best = Some((a, b, labels, size));
                    }
                }
            }
            let (a, b, labels, _) = best.ok_or(GraphError::UnsupportedComposition(
                "einsum intermediate rank or extent exceeds the ANE layout",
            ))?;
            let value = self.einsum_pair(
                (work[a].0, &work[a].1),
                (work[b].0, &work[b].1),
                &labels,
                &dimensions,
            )?;
            work.remove(b);
            work.remove(a);
            work.push((value, labels));
        }
        self.einsum_operand(work[0].0, &work[0].1, output)
    }
}
