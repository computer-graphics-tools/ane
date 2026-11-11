use ane::{DataType, Graph, GraphError, TensorDataError, TensorSpec};

#[test]
fn graph_buffer_rejects_byte_stride_overflow() {
    let graph = Graph::new();
    assert!(matches!(
        graph.placeholder_buffer([1], DataType::Float16, usize::MAX),
        Err(GraphError::Overflow)
    ));
}

#[test]
fn tensor_spec_rejects_byte_stride_overflow() {
    let spec = TensorSpec::new("x", &[1], DataType::Float16).unwrap();
    assert!(matches!(
        spec.with_row_stride(usize::MAX),
        Err(TensorDataError::LayoutOverflow)
    ));
}

#[test]
fn graph_buffer_rejects_plane_overflow_after_aligned_row_bytes() {
    let graph = Graph::new();
    let row_stride = usize::MAX - 63;
    assert!(matches!(
        graph.placeholder_buffer([1, 2, 1], DataType::Int8, row_stride),
        Err(GraphError::Overflow)
    ));
}

#[test]
fn aligned_layout_retains_strides_and_allocation_size() {
    let spec = TensorSpec::new("x", &[2, 3], DataType::Float16)
        .unwrap()
        .with_row_stride(32)
        .unwrap();
    assert_eq!(spec.strides(), &[128, 128, 64, 2]);
    assert_eq!(spec.allocation_size(), 128);

    let graph = Graph::new();
    let input = graph
        .placeholder_buffer([2, 3], DataType::Float16, 32)
        .unwrap();
    let mil = graph
        .program(&[input], &[DataType::Float16])
        .unwrap()
        .mil()
        .unwrap();
    assert!(
        mil.text
            .contains("tensor_buffer<fp16, shape=[1, 1, 2, 3], strides=[64, 64, 32, 1]"),
        "unexpected buffer strides in MIL:\n{}",
        mil.text
    );
}
