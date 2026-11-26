use std::{
    ffi::c_void,
    ops::{Range, RangeInclusive},
    ptr,
    sync::{
        Arc, Mutex, MutexGuard, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use bytemuck::NoUninit;
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLDevice;

use super::{
    INPUT_CHANNELS,
    gpu::{Buffer, Event, Residency},
    graph::{
        MAX_INTERMEDIATE, MAX_LINEAR_OUTPUT, MAX_TOKENS, TOKEN_COUNTS, UP_ROTATION_BLOCK,
        compile_ffn, compile_linear,
    },
    kernels::{DownRegions, Kernels, Launch, LayerRegions, Region, Surface, W4, W4Layout},
    signs::{LayerSigns, RhtW4, intermediate_signs, pack_signs, sign_bits},
};
use crate::{
    BoundExecution, DataType, Error, EventSequence, EventValues, Executable, SharedEvent,
    TensorData, TensorSpec,
};

mod session;

pub use session::Session;

/// The batch sizes a program runs: it picks the smallest compiled function that holds a batch.
pub const TOKENS: RangeInclusive<u32> = 1..=TOKEN_COUNTS[TOKEN_COUNTS.len() - 1];

fn fits_ffn(hidden: usize, intermediate: usize) -> bool {
    hidden == INPUT_CHANNELS
        && intermediate > 0
        && intermediate.is_multiple_of(INPUT_CHANNELS)
        && intermediate <= MAX_INTERMEDIATE
}

fn fits_linear(input: usize, output: usize) -> bool {
    input == INPUT_CHANNELS
        && output > 0
        && output <= MAX_LINEAR_OUTPUT
        && output.is_multiple_of(UP_ROTATION_BLOCK)
}

fn valid_w4(w4: W4, rows: usize, columns: usize) -> bool {
    if rows == 0 || columns == 0 || !columns.is_multiple_of(32) || !columns.is_multiple_of(8) {
        return false;
    }
    if (w4.scale_stride as usize) < rows || !w4.scale_stride.is_multiple_of(2) {
        return false;
    }
    if w4.layout != W4Layout::RowMajor && !rows.is_multiple_of(32) {
        return false;
    }

    let Some(code_bytes) = rows.checked_mul(columns / 2) else {
        return false;
    };
    let Some(scale_bytes) = (columns / 32)
        .checked_mul(w4.scale_stride as usize)
        .and_then(|values| values.checked_mul(2))
    else {
        return false;
    };
    let Some(zero_point_bytes) = (columns / 32).checked_mul(w4.scale_stride as usize / 2) else {
        return false;
    };

    valid_region(w4.codes, code_bytes, 8)
        && valid_region(w4.scales, scale_bytes, 2)
        && valid_region(w4.zero_points, zero_point_bytes, 1)
}

fn valid_region(region: Region, bytes: usize, alignment: u64) -> bool {
    region.address != 0
        && region.address.is_multiple_of(alignment)
        && region.len >= bytes
        && u64::try_from(bytes)
            .ok()
            .and_then(|bytes| region.address.checked_add(bytes))
            .is_some()
}

/// The Metal device, kernels, and residency policy shared by programs.
#[derive(Clone)]
pub struct Host {
    kernels: Kernels,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    residency: Arc<Residency>,
}

impl Host {
    /// Compiles the kernels for `device`.
    ///
    /// # Safety
    ///
    /// `device` is a live `id<MTLDevice>`.
    pub unsafe fn new(device: *mut c_void, residency: Residency) -> Result<Self, Error> {
        // SAFETY: forwarded to the caller.
        let device = unsafe { &*device.cast::<ProtocolObject<dyn MTLDevice>>() };
        Ok(Self {
            kernels: Kernels::new(device).map_err(Error::Metal)?,
            device: device.retain(),
            residency: Arc::new(residency),
        })
    }

    unsafe fn alias(&self, data: &TensorData) -> Result<Buffer, Error> {
        // SAFETY: `surface` stores the data after the alias, so it drops the alias first.
        unsafe { Buffer::alias(&self.device, &self.residency, data) }
    }

    fn upload(&self, bytes: &[u8]) -> Result<Buffer, Error> {
        Buffer::upload(&self.device, &self.residency, bytes)
    }

    fn event(&self) -> Result<(Event, SharedEvent), Error> {
        Event::new(&self.device)
    }
}

fn surface(host: &Host, rows: usize, width: usize, data_type: DataType) -> Result<Surface, Error> {
    let row_stride = data_type.aligned_row_stride(width);
    let spec = TensorSpec::new("", &[rows, width], data_type)?;
    let data = spec.with_row_stride(row_stride)?.allocate()?;
    // SAFETY: `Surface` retains `data` and declares `_alias` before it for drop order.
    let alias = unsafe { host.alias(&data)? };
    Ok(Surface {
        region: alias.region(),
        _alias: alias,
        row_stride: row_stride as u32,
        data,
    })
}

struct Rows {
    weights: Surface,
    scales: Surface,
}

impl Rows {
    fn new(host: &Host, rows: usize, width: usize) -> Result<Self, Error> {
        Ok(Self {
            weights: surface(host, rows, width, DataType::Int8)?,
            scales: surface(host, rows, 1, DataType::Float16)?,
        })
    }
}

/// One lane's surfaces, sized for the widest function of its kind.
struct Surfaces {
    x: Surface,
    tx: Surface,
    y: Surface,
    up: Vec<Rows>,
    down: Option<(Surface, Rows)>,
    /// Held by a call from start through finish: two calls on one set of surfaces would overwrite
    /// each other.
    in_use: Mutex<Option<Weak<Scope>>>,
}

impl Surfaces {
    fn ffn(host: &Host, hidden: usize, intermediate: usize) -> Result<Arc<Self>, Error> {
        let (int8, fp16) = (DataType::Int8, DataType::Float16);
        Ok(Arc::new(Self {
            x: surface(host, hidden, MAX_TOKENS, int8)?,
            tx: surface(host, 1, MAX_TOKENS, fp16)?,
            y: surface(host, hidden + 1, MAX_TOKENS, fp16)?,
            up: vec![
                Rows::new(host, intermediate, hidden)?,
                Rows::new(host, intermediate, hidden)?,
            ],
            down: Some((
                surface(host, intermediate, 1, fp16)?,
                Rows::new(host, hidden, intermediate)?,
            )),
            in_use: Mutex::new(None),
        }))
    }

    fn linear(host: &Host) -> Result<Arc<Self>, Error> {
        let (int8, fp16) = (DataType::Int8, DataType::Float16);
        Ok(Arc::new(Self {
            x: surface(host, INPUT_CHANNELS, MAX_TOKENS, int8)?,
            tx: surface(host, 1, MAX_TOKENS, fp16)?,
            y: surface(host, MAX_LINEAR_OUTPUT, MAX_TOKENS, fp16)?,
            up: vec![Rows::new(host, MAX_LINEAR_OUTPUT, INPUT_CHANNELS)?],
            down: None,
            in_use: Mutex::new(None),
        }))
    }

    /// Must follow the feed tensor order in `graph`.
    fn feeds(&self) -> Vec<&TensorData> {
        let up = &self.up;
        if let Some((gate_signs, down)) = &self.down {
            // `up` is the value (`wu`, `su`), then the gate (`wg`, `sg`).
            vec![
                &self.x.data,
                &self.tx.data,
                &up[1].weights.data,
                &up[0].weights.data,
                &up[1].scales.data,
                &up[0].scales.data,
                &gate_signs.data,
                &down.weights.data,
                &down.scales.data,
            ]
        } else {
            vec![&self.x.data, &up[0].weights.data, &up[0].scales.data]
        }
    }

    fn bind(&self, executables: &[Executable]) -> Result<Vec<BoundExecution>, Error> {
        let feeds = self.feeds();
        executables
            .iter()
            .map(|executable| {
                let inputs = executable
                    .feed_tensors()
                    .iter()
                    .zip(&feeds)
                    .map(|(tensor, data)| {
                        // SAFETY: the views alias the surfaces, and every access to them is ordered
                        // through events (see `Program`) as `view_as` and `submit` require.
                        Ok(unsafe { data.view_as(executable.input(*tensor)?)? })
                    })
                    .collect::<Result<Vec<TensorData>, Error>>()?;
                let output = executable.output(executable.target_tensors()[0])?;
                // SAFETY: as above.
                let result = unsafe { self.y.data.view_as(output)? };
                let inputs: Vec<_> = inputs.iter().collect();
                executable.bind(&inputs, &[&result])
            })
            .collect()
    }
}

/// The two chunks of an overlapped pair run on lanes of their own: round trips that overlap cannot
/// share surfaces or a sequence.
struct Lane {
    surfaces: Arc<Surfaces>,
    /// One per compiled row count, in [`TOKEN_COUNTS`] order.
    executions: Vec<BoundExecution>,
    sequence: EventSequence,
    event: Event,
}

impl Lane {
    fn new(
        surfaces: Arc<Surfaces>,
        executables: &[Executable],
        host: &Host,
    ) -> Result<Self, Error> {
        let executions = surfaces.bind(executables)?;
        let (event, shared) = host.event()?;
        Ok(Self {
            surfaces,
            executions,
            sequence: EventSequence::new(shared),
            event,
        })
    }
}

/// Bound layers retain their program independently of the session.
struct Program {
    host: Host,
    /// The width of the result (an FFN's is also that of its input).
    width: u32,
    /// Rows of each up matrix: the FFN's intermediate channels, a linear's output.
    channels: u32,
    _signs: Buffer,
    pack_signs: Region,
    intermediate_bits: Region,
    lanes: [Lane; 2],
    /// Counts the calls started: they alternate between the lanes.
    calls: AtomicUsize,
    retired: Arc<AtomicBool>,
}

impl Program {
    fn ffn(
        hidden: usize,
        intermediate: usize,
        host: &Host,
        retired: Arc<AtomicBool>,
    ) -> Result<Self, Error> {
        assert!(fits_ffn(hidden, intermediate));
        let executables = compile_ffn(hidden, intermediate)?;
        let surfaces = [
            Surfaces::ffn(host, hidden, intermediate)?,
            Surfaces::ffn(host, hidden, intermediate)?,
        ];
        let signs = intermediate_signs(intermediate);
        Self::new(
            hidden,
            intermediate,
            &signs,
            &executables,
            surfaces,
            host,
            retired,
        )
    }

    fn linear(
        output: usize,
        shared: &mut Option<[Arc<Surfaces>; 2]>,
        host: &Host,
        retired: Arc<AtomicBool>,
    ) -> Result<Self, Error> {
        assert!(fits_linear(INPUT_CHANNELS, output));
        let executables = compile_linear(output)?;
        let surfaces = if let Some(lanes) = shared.as_ref() {
            lanes.clone()
        } else {
            let lanes = [Surfaces::linear(host)?, Surfaces::linear(host)?];
            *shared = Some(lanes.clone());
            lanes
        };
        Self::new(output, output, &[], &executables, surfaces, host, retired)
    }

    fn new(
        width: usize,
        channels: usize,
        intermediate_signs: &[f32],
        executables: &[Executable],
        surfaces: [Arc<Surfaces>; 2],
        host: &Host,
        retired: Arc<AtomicBool>,
    ) -> Result<Self, Error> {
        let [first, second] = surfaces;
        let lanes = [
            Lane::new(first, executables, host)?,
            Lane::new(second, executables, host)?,
        ];
        let mut bytes = Vec::new();
        let pack = append(&mut bytes, &pack_signs());
        let bits = append(&mut bytes, &sign_bits(intermediate_signs));
        let signs = host.upload(&bytes)?;
        let base = signs.region();
        Ok(Self {
            host: host.clone(),
            width: width as u32,
            channels: channels as u32,
            pack_signs: slice(base, pack),
            intermediate_bits: slice(base, bits),
            _signs: signs,
            lanes,
            calls: AtomicUsize::new(0),
            retired,
        })
    }

    fn bind_layer(
        self: &Arc<Self>,
        up: &RhtW4,
        down: Option<&RhtW4>,
        owner: Arc<dyn Send + Sync>,
    ) -> Result<Layer, Error> {
        assert_eq!(down.is_some(), self.lanes[0].surfaces.down.is_some());
        let signs = LayerSigns::new(up, down);
        // The kernels read the first vectors in float4s: each is a multiple of 16 bytes.
        let mut bytes = Vec::new();
        let pre = append(&mut bytes, &signs.pre);
        let output = append(&mut bytes, &signs.output);
        let gate = append(&mut bytes, &signs.gate);
        let bits = append(&mut bytes, &signs.pre_bits);
        let signs = self.host.upload(&bytes)?;
        let base = signs.region();
        let mut down_regions = None;
        if let Some(down) = down {
            down_regions = Some(DownRegions {
                w4: down.w4,
                gate_signs: slice(base, gate),
                pre_bits: slice(base, bits),
                intermediate_bits: self.intermediate_bits,
            });
        }
        Ok(Layer {
            program: self.clone(),
            _owner: owner,
            regions: LayerRegions {
                up: up.w4,
                pre_signs: slice(base, pre),
                output_signs: slice(base, output),
                channels: self.channels,
                width: self.width,
                down: down_regions,
            },
            _signs: Arc::new(signs),
        })
    }

    fn stage_layer<Q: Queue>(
        &self,
        layer: &LayerRegions,
        surfaces: &Surfaces,
        queue: &mut Q,
    ) -> Result<(), ExecutionError<Q::Error>> {
        let kernels = &self.host.kernels;
        for (index, rows) in surfaces.up.iter().enumerate() {
            let launch = kernels.stage_up(
                layer.up,
                layer.pre_signs,
                &rows.weights,
                &rows.scales,
                index as u32 * layer.channels,
                layer.channels,
            );
            // SAFETY: the layer and lane keep the regions live through this encoding.
            unsafe { queue.execute(Step::Dispatch(launch)) }.map_err(ExecutionError::Queue)?;
        }
        if let (Some(down), Some((gate_signs, rows))) = (&layer.down, &surfaces.down) {
            let signs = kernels.stage_gate_signs(down.gate_signs, gate_signs, layer.channels);
            // SAFETY: the layer and lane keep the regions live through this encoding.
            unsafe { queue.execute(Step::Dispatch(signs)) }.map_err(ExecutionError::Queue)?;
            let rows = kernels.stage_down(
                down,
                &rows.weights,
                &rows.scales,
                layer.width,
                layer.channels,
            );
            // SAFETY: the layer and lane keep the regions live through this encoding.
            unsafe { queue.execute(Step::Dispatch(rows)) }.map_err(ExecutionError::Queue)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Layer {
    program: Arc<Program>,
    _owner: Arc<dyn Send + Sync>,
    regions: LayerRegions,
    _signs: Arc<Buffer>,
}

/// What the host does on its queue, in order.
pub enum Step<'a> {
    /// Encodes a compute dispatch.
    Dispatch(Launch<'a>),
    /// Makes the queue signal the event to the value once the steps before it are done.
    Signal(&'a Event, u64),
    /// Makes the queue wait for the event to reach the value before the steps after it.
    Wait(&'a Event, u64),
}

/// An in-flight call holds its lane until all finish work is accepted by the host queue.
/// Abandoning an accepted call waits for the request before releasing that lane.
pub struct Call<'a> {
    layer: &'a Layer,
    queue: QueueScope,
    lane: &'a Lane,
    event_values: EventValues,
    tokens: u32,
    submitted: bool,
    finished: bool,
    _guard: MutexGuard<'a, Option<Weak<Scope>>>,
}

impl Layer {
    /// Whether this compiled layer can run `tokens` rows before the caller's placement policy.
    pub fn accepts(&self, tokens: u32) -> bool {
        TOKENS.contains(&tokens) && !self.program.retired.load(Ordering::Acquire)
    }

    pub fn start<Q: Queue>(
        &self,
        input: &Q::Buffer,
        tokens: u32,
        queue: &mut Q,
    ) -> Result<Call<'_>, ExecutionError<Q::Error>> {
        let program = &self.program;
        if program.retired.load(Ordering::Acquire) {
            return Err(ExecutionError::Retired);
        }
        if !TOKENS.contains(&tokens) {
            return Err(
                Error::Metadata("token count is outside this program's compiled range").into(),
            );
        }
        if queue.device() != ptr::from_ref(&*program.host.device).cast_mut().cast() {
            return Err(Error::Metadata("queue belongs to another Metal device").into());
        }
        let input = checked_region(queue, input, tokens as usize * INPUT_CHANNELS * 2)?;
        let lane = &program.lanes[program.calls.fetch_add(1, Ordering::Relaxed) % 2];
        let mut guard = lane
            .surfaces
            .in_use
            .try_lock()
            .map_err(|_| ExecutionError::Program(Error::BufferBusy))?;
        let identity = queue.identity();
        if guard
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|previous| {
                !Arc::ptr_eq(&previous, &identity.0) && !previous.complete.load(Ordering::Acquire)
            })
        {
            return Err(Error::BufferBusy.into());
        }
        *guard = Some(Arc::downgrade(&identity.0));
        let index = TOKEN_COUNTS
            .iter()
            .position(|rows| *rows >= tokens)
            .expect("token count is inside TOKENS");
        let codes_len = tokens as usize * INPUT_CHANNELS;
        let codes = queue.allocate(codes_len).map_err(ExecutionError::Queue)?;
        let packed = checked_region(queue, &codes, codes_len)?;
        let mut call = Call {
            layer: self,
            queue: identity,
            lane,
            event_values: lane.sequence.reserve(),
            tokens,
            submitted: false,
            finished: false,
            _guard: guard,
        };

        call.queue.retain(self.clone());
        program.stage_layer(&self.regions, &lane.surfaces, queue)?;
        for launch in program.host.kernels.pack(
            input,
            program.pack_signs,
            packed,
            &lane.surfaces.x,
            &lane.surfaces.tx,
            tokens,
        ) {
            // SAFETY: validated buffers and the lane keep the regions live through this encoding.
            unsafe { queue.execute(Step::Dispatch(launch)) }.map_err(ExecutionError::Queue)?;
        }
        // SAFETY: the queue preserves the order of the dispatches above and this signal.
        unsafe { queue.execute(Step::Signal(&lane.event, call.event_values.inputs_ready)) }
            .map_err(ExecutionError::Queue)?;

        if program.retired.load(Ordering::Acquire) {
            let error = Error::Metadata("ANE session retired before request submission");
            call.queue.fail(&error);
            return Ok(call);
        }

        let retired = program.retired.clone();
        // The completion callback must not keep the queue scope and its layer alive.
        let scope = Arc::downgrade(&call.queue.0);
        // SAFETY: the lane lease and finish enforce the inputs_ready/completed ordering on this encoding.
        let submitted = unsafe {
            lane.sequence
                .submit(call.event_values, &lane.executions[index], move |error| {
                    retired.store(true, Ordering::Release);
                    if let Some(scope) = scope.upgrade() {
                        QueueScope(scope).fail(error);
                    }
                })
        };
        if let Err(error) = &submitted {
            program.retired.store(true, Ordering::Release);
            call.queue.fail(error);
        }
        call.submitted = submitted.is_ok();
        Ok(call)
    }
}

/// A host adapter for ordered GPU execution.
///
/// # Safety
/// `identity` must identify one ordered encoding and stay alive until its GPU work completes.
/// `device` identifies its Metal device. `region` must
/// validate buffer ownership on that device. Accepted work must retain its resources, preserve
/// the declared hazards and event order, and keep buffer storage valid through GPU completion.
pub unsafe trait Queue {
    type Buffer;
    type Error;

    fn identity(&self) -> QueueScope;
    fn device(&self) -> *mut c_void;
    fn region(&self, buffer: &Self::Buffer) -> Result<Region, Self::Error>;
    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, Self::Error>;

    /// # Safety
    /// The step's regions and resources must belong to this device and cover its accesses.
    unsafe fn execute(&mut self, step: Step<'_>) -> Result<(), Self::Error>;
}

#[derive(Clone, Default)]
pub struct QueueScope(Arc<Scope>);

#[derive(Default)]
struct Scope {
    complete: AtomicBool,
    layers: Mutex<Vec<Layer>>,
    error: Mutex<Option<String>>,
}

impl QueueScope {
    fn retain(&self, layer: Layer) {
        self.0
            .layers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(layer);
    }

    pub fn error(&self) -> Option<String> {
        self.0
            .error
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    fn fail(&self, error: &Error) {
        let message = error.to_string();
        self.0
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get_or_insert(message);
    }

    /// # Safety
    /// All work in this encoding has completed and no more will be submitted.
    pub unsafe fn complete(&self) {
        self.0.complete.store(true, Ordering::Release);
        let layers = std::mem::take(
            &mut *self
                .0
                .layers
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        );
        drop(layers);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExecutionError<E> {
    #[error("ANE session is retired")]
    Retired,
    #[error(transparent)]
    Program(#[from] Error),
    #[error("host queue rejected an execution step: {0}")]
    Queue(E),
}

fn checked_region<Q: Queue>(
    queue: &Q,
    buffer: &Q::Buffer,
    bytes: usize,
) -> Result<Region, ExecutionError<Q::Error>> {
    let region = queue.region(buffer).map_err(ExecutionError::Queue)?;
    if region.address == 0
        || !region.address.is_multiple_of(16)
        || region.len < bytes
        || region.address.checked_add(bytes as u64).is_none()
    {
        return Err(Error::Metadata("buffer is too small or has an invalid GPU address").into());
    }
    Ok(region)
}

impl Call<'_> {
    pub fn finish<Q: Queue>(
        mut self,
        queue: &mut Q,
    ) -> Result<Q::Buffer, ExecutionError<Q::Error>> {
        if !Arc::ptr_eq(&self.queue.0, &queue.identity().0) {
            return Err(
                Error::Metadata("finish must use the encoding that started the call").into(),
            );
        }
        let program = &self.layer.program;
        let output_len = self.tokens as usize * program.width as usize * 2;
        let output = queue.allocate(output_len).map_err(ExecutionError::Queue)?;
        let region = checked_region(queue, &output, output_len)?;
        let surfaces = &self.lane.surfaces;
        if !self.submitted {
            // SAFETY: the encoding identity was checked above.
            unsafe { queue.execute(Step::Signal(&self.lane.event, self.event_values.completed)) }
                .map_err(ExecutionError::Queue)?;
        }
        // SAFETY: validated output and the lane lease cover this dispatch.
        unsafe { queue.execute(Step::Wait(&self.lane.event, self.event_values.completed)) }
            .map_err(ExecutionError::Queue)?;
        let unpack = program.host.kernels.unpack(
            &self.layer.regions,
            &surfaces.y,
            &surfaces.tx,
            region,
            self.tokens,
        );
        // SAFETY: validated output and the lane lease cover this dispatch.
        unsafe { queue.execute(Step::Dispatch(unpack)) }.map_err(ExecutionError::Queue)?;
        self.finished = true;
        Ok(output)
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        if self.submitted && !self.finished {
            self.lane
                .sequence
                .wait_until_signaled(self.event_values.completed);
        }
    }
}

fn append(bytes: &mut Vec<u8>, values: &[impl NoUninit]) -> Range<usize> {
    let start = bytes.len();
    bytes.extend_from_slice(bytemuck::cast_slice(values));
    start..bytes.len()
}

fn slice(base: Region, range: Range<usize>) -> Region {
    Region {
        address: base.address + range.start as u64,
        len: range.len(),
    }
}
