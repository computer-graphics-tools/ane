pub fn ane_available() -> bool {
    match ane::DeviceInfo::current() {
        Ok(info) if info.has_ane => {}
        _ => {
            eprintln!("skipped: no Apple Neural Engine on this machine");
            return false;
        }
    }
    let graph = ane::Graph::new();
    let compiled = graph
        .placeholder([64, 64], ane::DataType::Float16)
        .and_then(|input| graph.relu(&input))
        .map_err(ane::Error::from)
        .and_then(|output| graph.compile(&[output], &[], None).map(|_| ()));
    match compiled {
        Ok(()) => true,
        Err(error) => {
            eprintln!("skipped: the Neural Engine is not usable here: {error}");
            false
        }
    }
}

#[macro_export]
macro_rules! require_ane {
    () => {
        if !$crate::common::ane_available() {
            return Ok(());
        }
    };
}
