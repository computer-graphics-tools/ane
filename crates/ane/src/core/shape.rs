pub fn padded_shape(shape: &[usize]) -> Option<[usize; 4]> {
    let size = shape.iter().try_fold(8usize, |size, &dimension| {
        size.checked_mul(dimension).filter(|&size| {
            (1..=i32::MAX as usize).contains(&dimension) && size <= isize::MAX as usize
        })
    });
    (shape.len() <= 4 && size.is_some()).then(|| {
        let mut padded = [1; 4];
        padded[4 - shape.len()..].copy_from_slice(shape);
        padded
    })
}

pub fn logical_shape<const RANK: usize>(shape: &[usize; RANK]) -> &[usize] {
    const { assert!(RANK <= 4, "ANE tensors support at most four dimensions") };
    shape
}
