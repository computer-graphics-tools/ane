//! W8A8 programs of the ANE for a Metal host: a dense FFN and a single linear layer whose int4 (RHT,
//! group 32) weights the GPU stages as the int8 rows the ANE reads.
//!
//! A [`Session`] lazily creates a [`Host`] and owns its program inventory. A bound [`Layer`] starts
//! and finishes its calls through one [`Queue`] adapter, which executes the required [`Step`]s.

mod gpu;
mod graph;
mod kernels;
mod program;
mod signs;

pub use gpu::{Event, Residency};
pub use kernels::{
    Launch, OutputBasis, Region, Rotation, W4, W4Layout, WeightCodeLayout, WeightLayer,
    WeightSource,
};
pub use program::{Call, ExecutionError, Host, Layer, Queue, QueueScope, Session, Step, TOKENS};

const INPUT_CHANNELS: usize = 4096;
const FFN_SHAPE_ALIGNMENT: usize = 512;
