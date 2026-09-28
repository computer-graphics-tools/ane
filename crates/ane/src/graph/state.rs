use super::{Graph, Tensor};
use crate::{DataType, Op, ops::StateUpdateOp};

pub struct State {
    tensor: Tensor,
}

impl State {
    pub fn new(graph: &mut Graph, shape: &[usize]) -> Self {
        let shape = crate::dimensions(shape);
        assert!(
            shape[3] >= 64 && shape[3].is_multiple_of(32),
            "state rows must be aligned"
        );
        Self {
            tensor: graph.placeholder_with_type(&shape, DataType::Float16),
        }
    }

    pub fn update_rows(&self, graph: &mut Graph, update: Tensor, position: Tensor) -> Tensor {
        self.update_rows_at_channel(graph, update, position, 0)
    }

    pub fn update_rows_at_channel(
        &self,
        graph: &mut Graph,
        update: Tensor,
        position: Tensor,
        channel: usize,
    ) -> Tensor {
        let shape = self.tensor.shape;
        assert!(
            graph.inputs.iter().any(|(t, d)| t.id == position.id
                && *d == DataType::Int32
                && t.shape.iter().product::<usize>() == 1),
            "state position must be an integer parameter"
        );
        assert_eq!((update.shape[0], update.shape[3]), (shape[0], shape[3]));
        assert!(update.shape[2] > 0 && update.shape[2] <= shape[2]);
        let state = Graph::tensor_name(self.tensor);
        assert!(
            channel
                .checked_add(update.shape[1])
                .is_some_and(|end| end <= shape[1])
        );
        let result = graph.alloc(shape);
        graph.ops.push((
            Op::StateUpdate(StateUpdateOp {
                name: Graph::op_name(result),
                top: Graph::tensor_name(result),
                bottom: Graph::tensor_name(update),
                state,
                position: Graph::tensor_name(position),
                rows: update.shape[2],
                channel,
                channels: update.shape[1],
            }),
            result,
        ));
        result
    }
}
