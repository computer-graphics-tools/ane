use crate::DataType;
use crate::PadFillMode;
use crate::graph::Graph;
use crate::graph::GraphBuilder;
use crate::graph::Tensor;
use crate::graph::TensorHandle;
use crate::graph::{GraphError, checked_shape, ensure};
use crate::ir::{Operator, Parameter, Value};
use crate::logical_shape;

impl Graph {
    /// `length` elements along `axis` from the runtime scalar Int32 `start`. The ANE takes dynamic
    /// offsets only as scalar parameters, so the begin vector is assembled in the program.
    /// MIL `slice_by_size`.
    pub fn slice_dynamic(
        &self,
        input: &Tensor,
        start: &Tensor,
        axis: i64,
        length: usize,
    ) -> Result<Tensor, GraphError> {
        let axis = self.axis(*input, axis)?;
        self.check_tensor(*start)?;
        ensure(
            start.data_type() == DataType::Int32
                && start.physical_shape().iter().product::<usize>() == 1,
            GraphError::UnsupportedDataType(
                "dynamic slice start requires a scalar Int32 parameter",
            ),
        )?;
        let mut shape = input.physical_shape();
        ensure(
            length > 0 && length <= shape[axis],
            GraphError::OutOfBounds("dynamic slice length exceeds its axis"),
        )?;
        shape[axis] = length;
        self.builtin(
            Operator::DynamicSlice,
            &[(Parameter::X, *input), (Parameter::Begin, *start)],
            &[
                (Parameter::Axis, Value::Int32(axis)),
                (Parameter::Size, Value::Int32(length)),
            ],
            &shape[4 - input.rank()..],
            input.data_type(),
        )
    }
    /// Pads the last two axes by `[top, bottom, left, right]` in constant, reflect or replicate mode;
    /// MIL has no symmetric mode. MIL `pad`.
    pub fn pad(
        &self,
        input: &Tensor,
        padding: [usize; 4],
        mode: PadFillMode,
        value: f32,
    ) -> Result<Tensor, GraphError> {
        let [top, bottom, left, right] = padding;
        let input = *input;
        self.numeric(input)?;
        ensure(
            value.is_finite(),
            GraphError::InvalidArgument("padding value must be finite"),
        )?;
        let height = input.physical_shape()[2]
            .checked_add(top)
            .and_then(|n| n.checked_add(bottom))
            .ok_or(GraphError::Overflow)?;
        let width = input.physical_shape()[3]
            .checked_add(left)
            .and_then(|n| n.checked_add(right))
            .ok_or(GraphError::Overflow)?;
        ensure(
            (input.rank() >= 2 || (top == 0 && bottom == 0))
                && (input.rank() >= 1 || (left == 0 && right == 0)),
            GraphError::ShapeMismatch(
                "padding cannot add dimensions; expand the tensor rank first",
            ),
        )?;
        if [top, bottom, left, right] == [0; 4] {
            return Ok(input);
        }
        let mode = match mode {
            PadFillMode::Constant => "constant",
            PadFillMode::Reflect => "reflect",
            PadFillMode::Replicate => "replicate",
            PadFillMode::Symmetric => {
                return Err(GraphError::InvalidArgument(
                    "MIL pad supports constant, reflect and replicate modes",
                ));
            }
        };
        if mode == "reflect" {
            let [_, _, rows, columns] = input.physical_shape();
            ensure(
                top < rows && bottom < rows && left < columns && right < columns,
                GraphError::OutOfBounds("mirrored padding exceeds input"),
            )?;
        }
        let shape = [
            input.physical_shape()[0],
            input.physical_shape()[1],
            height,
            width,
        ];
        self.builtin(
            Operator::Pad,
            &[(Parameter::X, input)],
            &[
                (
                    Parameter::Pad,
                    Value::int32_list(&[top, bottom, left, right]),
                ),
                (Parameter::Mode, Value::String(mode)),
                (Parameter::ConstantVal, Value::Fp16(value)),
            ],
            &shape[4 - input.rank()..],
            DataType::Float16,
        )
    }

    /// Converts to `dtype`; float to integer rounds half away from zero. MIL `cast`.
    pub fn cast(&self, x: &Tensor, dtype: DataType) -> Result<Tensor, GraphError> {
        ensure(
            dtype != DataType::Int32 && dtype != DataType::Float32,
            GraphError::UnsupportedDataType(
                "general Int32 and FP32 graph arithmetic is unavailable",
            ),
        )?;
        self.builtin(
            Operator::Cast,
            &[(Parameter::X, *x)],
            &[(Parameter::Dtype, Value::String(dtype.as_str()))],
            x.shape(),
            dtype,
        )
    }

    /// Reshapes without moving data. MIL `reshape`.
    pub fn reshape<const RANK: usize>(
        &self,
        input: &Tensor,
        shape: [usize; RANK],
    ) -> Result<Tensor, GraphError> {
        self.reshape_to(*input, logical_shape(&shape))
    }

    /// Permutes the axes. MIL `transpose`.
    pub fn transpose(
        &self,
        input: &Tensor,
        permutation: impl AsRef<[usize]>,
    ) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        let permutation = permutation.as_ref();
        ensure(
            permutation.len() == input.rank(),
            GraphError::InvalidAxes("permutation rank differs"),
        )?;
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        ensure(
            sorted.iter().copied().eq(0..input.rank()),
            GraphError::InvalidAxes("invalid permutation"),
        )?;
        let offset = 4 - input.rank();
        let mut physical = [0, 1, 2, 3];
        for (i, &axis) in permutation.iter().enumerate() {
            physical[offset + i] = offset + axis;
        }
        let shape = physical.map(|axis| input.physical_shape()[axis]);
        self.builtin(
            Operator::Transpose,
            &[(Parameter::X, *input)],
            &[(Parameter::Perm, Value::int32_list(&physical))],
            &shape[offset..],
            input.data_type(),
        )
    }

    /// Static slice of `size` elements from `begin`. MIL `slice_by_size`.
    pub fn slice(
        &self,
        input: &Tensor,
        begin: impl AsRef<[usize]>,
        size: impl AsRef<[usize]>,
    ) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        let begin = begin.as_ref();
        let size = size.as_ref();
        ensure(
            begin.len() == input.rank() && size.len() == input.rank(),
            GraphError::ShapeMismatch("slice rank differs"),
        )?;
        let shape = checked_shape(size)?;
        let mut start = [0; 4];
        start[4 - input.rank()..].copy_from_slice(begin);
        ensure(
            (0..4).all(|axis| {
                start[axis] <= input.physical_shape()[axis]
                    && shape[axis] <= input.physical_shape()[axis] - start[axis]
            }),
            GraphError::OutOfBounds("slice exceeds tensor"),
        )?;
        self.builtin(
            Operator::SliceBySize,
            &[(Parameter::X, *input)],
            &[
                (Parameter::Begin, Value::int32_list(&start)),
                (Parameter::Size, Value::int32_list(&shape)),
            ],
            size,
            input.data_type(),
        )
    }

    /// Static slice of `[begin, end)` with `stride`. MIL `slice_by_index`.
    pub fn strided_slice(
        &self,
        input: &Tensor,
        begin: &[usize],
        end: &[usize],
        stride: &[usize],
    ) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        let rank = input.rank();
        ensure(
            begin.len() == rank && end.len() == rank && stride.len() == rank,
            GraphError::ShapeMismatch("slice_by_index rank differs"),
        )?;
        let (mut first, mut last, mut step) = ([0; 4], input.physical_shape(), [1; 4]);
        first[4 - rank..].copy_from_slice(begin);
        last[4 - rank..].copy_from_slice(end);
        step[4 - rank..].copy_from_slice(stride);
        ensure(
            (0..4)
                .all(|a| first[a] < last[a] && last[a] <= input.physical_shape()[a] && step[a] > 0),
            GraphError::OutOfBounds("slice_by_index exceeds tensor"),
        )?;
        let shape: Vec<_> = (0..4)
            .map(|a| (last[a] - first[a]).div_ceil(step[a]))
            .collect();
        self.builtin(
            Operator::SliceByIndex,
            &[(Parameter::X, *input)],
            &[
                (Parameter::Begin, Value::int32_list(&first)),
                (Parameter::End, Value::int32_list(&last)),
                (Parameter::Stride, Value::int32_list(&step)),
                (Parameter::BeginMask, Value::BoolList([false; 4].into())),
                (Parameter::EndMask, Value::BoolList([false; 4].into())),
                (Parameter::SqueezeMask, Value::BoolList([false; 4].into())),
            ],
            &shape[4 - rank..],
            input.data_type(),
        )
    }

    /// Concatenates along `axis`, or interleaves elements when `interleave` is set. MIL `concat`.
    pub fn concat(
        &self,
        inputs: &[&Tensor],
        axis: usize,
        interleave: bool,
    ) -> Result<Tensor, GraphError> {
        let inputs: Vec<Tensor> = inputs.iter().map(|&&tensor| tensor).collect();
        let inputs = &inputs[..];
        ensure(
            !inputs.is_empty(),
            GraphError::InvalidArgument("concat requires an input"),
        )?;
        let first = inputs[0];
        let physical_axis = self.axis(first, axis as i64)?;
        let mut shape = first.physical_shape();
        shape[physical_axis] = 0;
        for &input in inputs {
            self.check_tensor(input)?;
            ensure(
                input.rank() == first.rank()
                    && input.data_type() == first.data_type()
                    && (0..4).all(|a| {
                        a == physical_axis || input.physical_shape()[a] == first.physical_shape()[a]
                    }),
                GraphError::ShapeMismatch("concat shapes or types differ"),
            )?;
            shape[physical_axis] = shape[physical_axis]
                .checked_add(input.physical_shape()[physical_axis])
                .ok_or(GraphError::Overflow)?;
        }
        let values: Vec<_> = inputs.iter().map(|&t| (Parameter::Values, t)).collect();
        self.builtin(
            Operator::Concat,
            &values,
            &[
                (Parameter::Axis, Value::Int32(physical_axis)),
                (Parameter::Interleave, Value::Bool(interleave)),
            ],
            &shape[4 - first.rank()..],
            first.data_type(),
        )
    }

    /// Splits along `axis` into pieces of `sizes`. MIL `split`.
    pub fn split(
        &self,
        input: &Tensor,
        sizes: &[usize],
        axis: usize,
    ) -> Result<Vec<Tensor>, GraphError> {
        let physical = self.axis(*input, axis as i64)?;
        ensure(
            !sizes.is_empty() && sizes.iter().all(|&x| x > 0),
            GraphError::InvalidArgument("split sizes must be positive"),
        )?;
        let sum = sizes.iter().try_fold(0usize, |sum, &n| {
            sum.checked_add(n).ok_or(GraphError::Overflow)
        })?;
        ensure(
            sum == input.physical_shape()[physical],
            GraphError::ShapeMismatch("split sizes differ from axis length"),
        )?;
        let shapes: Vec<Vec<usize>> = sizes
            .iter()
            .map(|&size| {
                let mut shape = input.shape().to_vec();
                shape[axis] = size;
                shape
            })
            .collect();
        let outputs: Vec<_> = shapes
            .iter()
            .map(|shape| (input.data_type(), &shape[..]))
            .collect();
        self.builtin_many(
            Operator::Split,
            &[(Parameter::X, *input)],
            &[
                (Parameter::SplitSizes, Value::int32_list(sizes)),
                (Parameter::Axis, Value::Int32(physical)),
            ],
            &outputs,
        )
    }

    /// Inserts size-1 axes at `axes`. MIL `reshape`.
    pub fn expand_dims(&self, input: &Tensor, axes: &[usize]) -> Result<Tensor, GraphError> {
        self.check_tensor(*input)?;
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        let rank = input.rank() + axes.len();
        ensure(
            rank <= 4 && axes.iter().all(|&a| a < rank) && axes.windows(2).all(|p| p[0] != p[1]),
            GraphError::InvalidAxes("invalid expanded axes"),
        )?;
        let mut shape = input.shape().to_vec();
        for axis in axes {
            shape.insert(axis, 1);
        }
        self.reshape_to(*input, &shape)
    }

    /// Removes the size-1 axes at `axes`. MIL `reshape`.
    pub fn squeeze(&self, input: &Tensor, axes: &[usize]) -> Result<Tensor, GraphError> {
        let input = *input;
        self.check_tensor(input)?;
        let mut axes = axes.to_vec();
        if axes.is_empty() {
            axes = input
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, &n)| (n == 1).then_some(i))
                .collect();
        }
        axes.sort_unstable();
        ensure(
            axes.iter()
                .all(|&a| a < input.rank() && input.shape()[a] == 1)
                && axes.windows(2).all(|p| p[0] != p[1]),
            GraphError::InvalidAxes("squeeze requires distinct size-one axes"),
        )?;
        let mut shape = input.shape().to_vec();
        for &axis in axes.iter().rev() {
            shape.remove(axis);
        }
        self.reshape_to(input, &shape)
    }

    /// Repeats the tensor `repeats` times along every axis. MIL `tile`.
    pub fn tile(&self, input: &Tensor, repeats: &[usize]) -> Result<Tensor, GraphError> {
        let input = *input;
        self.check_tensor(input)?;
        ensure(
            repeats.len() == input.rank() && repeats.iter().all(|&r| r > 0),
            GraphError::InvalidArgument("tile repeat count differs or is zero"),
        )?;
        let shape = input
            .shape()
            .iter()
            .zip(repeats)
            .map(|(&n, &r)| n.checked_mul(r).ok_or(GraphError::Overflow))
            .collect::<Result<Vec<_>, _>>()?;
        let reps = checked_shape(repeats)?;
        self.builtin(
            Operator::Tile,
            &[(Parameter::X, input)],
            &[(Parameter::Reps, Value::int32_list(&reps))],
            &shape,
            input.data_type(),
        )
    }

    /// Reverses the order along `axes`. MIL `reverse`.
    pub fn reverse(&self, input: &Tensor, axes: &[i64]) -> Result<Tensor, GraphError> {
        let input = *input;
        let axes = axes
            .iter()
            .map(|&a| self.axis(input, a))
            .collect::<Result<Vec<_>, _>>()?;
        ensure(
            axes.iter().enumerate().all(|(i, a)| !axes[..i].contains(a)),
            GraphError::InvalidAxes("duplicate reverse axis"),
        )?;
        self.builtin(
            Operator::Reverse,
            &[(Parameter::X, input)],
            &[(Parameter::Axes, Value::int32_list(&axes))],
            input.shape(),
            input.data_type(),
        )
    }

    /// Writes `update` into `input` at `begin`. The ANE compiles it only when the update spans whole
    /// planes along the channel axis. MIL `slice_update`.
    pub fn slice_update(
        &self,
        input: &Tensor,
        update: &Tensor,
        begin: &[usize],
    ) -> Result<Tensor, GraphError> {
        let input = *input;
        let update = *update;
        self.check_tensor(input)?;
        self.check_tensor(update)?;
        ensure(
            input.rank() == update.rank()
                && begin.len() == input.rank()
                && input.data_type() == update.data_type(),
            GraphError::ShapeMismatch("slice update rank or type differs"),
        )?;
        ensure(
            (0..input.rank()).all(|a| {
                begin[a] <= input.shape()[a] && update.shape()[a] <= input.shape()[a] - begin[a]
            }),
            GraphError::OutOfBounds("slice update exceeds tensor"),
        )?;
        let mut start = [0; 4];
        start[4 - input.rank()..].copy_from_slice(begin);
        let end: Vec<_> = (0..4)
            .map(|a| start[a] + update.physical_shape()[a])
            .collect();
        self.builtin(
            Operator::SliceUpdate,
            &[(Parameter::X, input), (Parameter::Update, update)],
            &[
                (Parameter::Begin, Value::int32_list(&start)),
                (Parameter::End, Value::int32_list(&end)),
            ],
            input.shape(),
            input.data_type(),
        )
    }

    fn spatial_shuffle(
        &self,
        input: Tensor,
        factor: usize,
        inverse: bool,
        pixel: bool,
    ) -> Result<Tensor, GraphError> {
        self.numeric(input)?;
        ensure(
            input.rank() == 4 && factor > 0,
            GraphError::ShapeMismatch("spatial shuffle requires NCHW and a positive factor"),
        )?;
        let square = factor.checked_mul(factor).ok_or(GraphError::Overflow)?;
        let mut shape = input.physical_shape();
        if inverse {
            ensure(
                shape[2].is_multiple_of(factor) && shape[3].is_multiple_of(factor),
                GraphError::ShapeMismatch("spatial dimensions must divide by the factor"),
            )?;
            shape[1] = shape[1].checked_mul(square).ok_or(GraphError::Overflow)?;
            shape[2] /= factor;
            shape[3] /= factor;
        } else {
            ensure(
                shape[1].is_multiple_of(square),
                GraphError::ShapeMismatch("channel count must divide by the squared factor"),
            )?;
            shape[1] /= square;
            shape[2] = shape[2].checked_mul(factor).ok_or(GraphError::Overflow)?;
            shape[3] = shape[3].checked_mul(factor).ok_or(GraphError::Overflow)?;
        }
        let (op, key) = match (inverse, pixel) {
            (false, false) => (Operator::DepthToSpace, Parameter::BlockSize),
            (true, _) => (Operator::SpaceToDepth, Parameter::BlockSize),
            (false, true) => (Operator::PixelShuffle, Parameter::UpscaleFactor),
        };
        self.builtin(
            op,
            &[(Parameter::X, input)],
            &[(key, Value::Int32(factor))],
            &shape,
            DataType::Float16,
        )
    }

    /// Moves channel blocks into `factor × factor` spatial blocks in TensorFlow order: output channel
    /// `c` at offset `(dy, dx)` comes from input channel `(dy · factor + dx) · C + c`. MIL `depth_to_space`.
    pub fn depth_to_space(&self, input: &Tensor, factor: usize) -> Result<Tensor, GraphError> {
        self.spatial_shuffle(*input, factor, false, false)
    }

    /// Inverse of `depth_to_space`, in the same TensorFlow channel order. MIL `space_to_depth`.
    pub fn space_to_depth(&self, input: &Tensor, factor: usize) -> Result<Tensor, GraphError> {
        self.spatial_shuffle(*input, factor, true, false)
    }

    /// `depth_to_space` in PyTorch order: output channel `c` at offset `(dy, dx)` comes from input
    /// channel `c · factor² + dy · factor + dx`. MIL `pixel_shuffle`.
    pub fn pixel_shuffle(&self, input: &Tensor, factor: usize) -> Result<Tensor, GraphError> {
        self.spatial_shuffle(*input, factor, false, true)
    }

    /// Pads the last two axes by `padding` and moves `block` spatial blocks into the batch axis.
    /// MIL `space_to_batch`.
    pub fn space_to_batch(
        &self,
        input: &Tensor,
        block: [usize; 2],
        padding: [usize; 4],
    ) -> Result<Tensor, GraphError> {
        let input = *input;
        self.numeric(input)?;
        ensure(
            input.rank() == 4 && block.iter().all(|&n| n > 0),
            GraphError::ShapeMismatch("space-to-batch requires NCHW and positive blocks"),
        )?;
        let h = input.physical_shape()[2]
            .checked_add(padding[0])
            .and_then(|n| n.checked_add(padding[1]))
            .ok_or(GraphError::Overflow)?;
        let w = input.physical_shape()[3]
            .checked_add(padding[2])
            .and_then(|n| n.checked_add(padding[3]))
            .ok_or(GraphError::Overflow)?;
        ensure(
            h.is_multiple_of(block[0]) && w.is_multiple_of(block[1]),
            GraphError::ShapeMismatch("padded spatial dimensions must divide by block size"),
        )?;
        let [batches, channels, ..] = input.physical_shape();
        let factor = block[0].checked_mul(block[1]).ok_or(GraphError::Overflow)?;
        self.builtin(
            Operator::SpaceToBatch,
            &[(Parameter::X, input)],
            &[
                (Parameter::BlockShape, Value::int32_list(&block)),
                (
                    Parameter::Paddings,
                    Value::Int32Matrix([[padding[0], padding[1]], [padding[2], padding[3]]]),
                ),
            ],
            &[
                batches.checked_mul(factor).ok_or(GraphError::Overflow)?,
                channels,
                h / block[0],
                w / block[1],
            ],
            input.data_type(),
        )
    }

    /// Inverse of `space_to_batch`, cropping by `crops`. MIL `batch_to_space`.
    pub fn batch_to_space(
        &self,
        input: &Tensor,
        block: [usize; 2],
        crops: [usize; 4],
    ) -> Result<Tensor, GraphError> {
        let input = *input;
        self.numeric(input)?;
        ensure(
            input.rank() == 4 && block.iter().all(|&n| n > 0),
            GraphError::ShapeMismatch("batch-to-space requires NCHW and positive blocks"),
        )?;
        let factor = block[0].checked_mul(block[1]).ok_or(GraphError::Overflow)?;
        ensure(
            input.physical_shape()[0].is_multiple_of(factor),
            GraphError::ShapeMismatch("batch count must divide by block size"),
        )?;
        let h = input.physical_shape()[2]
            .checked_mul(block[0])
            .and_then(|n| n.checked_sub(crops[0]))
            .and_then(|n| n.checked_sub(crops[1]))
            .ok_or(GraphError::OutOfBounds("invalid crop or height overflow"))?;
        let w = input.physical_shape()[3]
            .checked_mul(block[1])
            .and_then(|n| n.checked_sub(crops[2]))
            .and_then(|n| n.checked_sub(crops[3]))
            .ok_or(GraphError::OutOfBounds("invalid crop or width overflow"))?;
        self.builtin(
            Operator::BatchToSpace,
            &[(Parameter::X, input)],
            &[
                (Parameter::BlockShape, Value::int32_list(&block)),
                (
                    Parameter::Crops,
                    Value::Int32Matrix([[crops[0], crops[1]], [crops[2], crops[3]]]),
                ),
            ],
            &[
                input.physical_shape()[0] / factor,
                input.physical_shape()[1],
                h,
                w,
            ],
            input.data_type(),
        )
    }
}
