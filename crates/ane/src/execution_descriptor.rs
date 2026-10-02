use crate::SharedEvent;

#[derive(Clone, Copy, Default)]
pub struct ExecutionDescriptor<'a> {
    pub wait_events: &'a [(&'a SharedEvent, u64)],
    pub signal_events: &'a [(&'a SharedEvent, u64)],
}
