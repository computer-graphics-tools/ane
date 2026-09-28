use std::collections::HashMap;

use super::Tensor;
use crate::{DataType, Op, ops::WeightBlob};

#[path = "compile.rs"]
mod compile;
#[path = "ops.rs"]
mod ops;
#[path = "state.rs"]
mod state;
pub use compile::MIN_SPATIAL_WIDTH;
pub use state::State;

pub struct Graph {
    inputs: Vec<(Tensor, DataType)>,
    constants: HashMap<usize, (WeightBlob, [usize; 4])>,
    ops: Vec<(Op, Tensor)>,
    counter: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            constants: HashMap::new(),
            ops: Vec::new(),
            counter: 0,
        }
    }

    fn alloc(&mut self, shape: [usize; 4]) -> Tensor {
        let id = self.counter;
        self.counter += 1;
        Tensor { id, shape }
    }

    fn tensor_name(tensor: Tensor) -> String {
        format!("t{}", tensor.id)
    }

    fn op_name(tensor: Tensor) -> String {
        format!("op{}", tensor.id)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
