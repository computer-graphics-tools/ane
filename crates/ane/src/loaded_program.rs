use crate::apple_neural_engine::{ANEClient, ANEModel, ANERequest, ModelAttributes};
use crate::graph::TensorHandle;
use crate::ir::{GraphCompiler, catch_native};
use crate::{DataType, DeviceError, Error, GraphError, Program};
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSDictionary, NSQualityOfService, NSString, NSURL};
use objc2_metal::MTLCreateSystemDefaultDevice;
use objc2_metal_performance_shaders_graph::{
    MPSGraphDevice, MPSGraphExecutableSerializationDescriptor, MPSGraphShapedType,
};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

static MODEL_LIFECYCLE: Mutex<()> = Mutex::new(());

pub struct LoadedProgram {
    model: Retained<ANEModel>,
    client: Retained<ANEClient>,
    qos: NSQualityOfService,
    dispatch: Mutex<()>,
    directory: PathBuf,
    pub input_symbols: Box<[String]>,
    pub output_symbols: Box<[String]>,
    pub state_outputs: Box<[(String, usize)]>,
    pub discarded_outputs: Box<[(String, [usize; 4], DataType)]>,
    pub anec_bytes: usize,
}
unsafe impl Send for LoadedProgram {}
unsafe impl Sync for LoadedProgram {}

fn dictionary<'a>(
    value: &'a plist::Value,
    key: &'static str,
) -> Result<&'a plist::Dictionary, Error> {
    value
        .as_dictionary()
        .and_then(|v| v.get(key))
        .and_then(plist::Value::as_dictionary)
        .ok_or(Error::Metadata(key))
}
fn module_attributes(package: &plist::Value) -> Result<(&str, &[plist::Value]), Error> {
    let versions = dictionary(package, "Package Version")?;
    if versions.len() != 1 {
        return Err(Error::Metadata("expected one compiler package version"));
    }
    let version = versions.values().next().unwrap();
    let architectures = dictionary(version, "ANERegionsHash")?;
    if architectures.len() != 1 {
        return Err(Error::Metadata("expected one ANE target architecture"));
    }
    let architecture = architectures.keys().next().unwrap();
    let modules = dictionary(version, "Optimized Modules")?;
    if modules.len() != 1 {
        return Err(Error::Metadata("expected one compiled graph module"));
    }
    let functions = dictionary(
        modules.values().next().unwrap(),
        "Entry Function Attributes",
    )?;
    let attrs = functions
        .get("main")
        .and_then(plist::Value::as_array)
        .ok_or(Error::Metadata("main entrypoint attributes"))?;
    Ok((architecture, attrs))
}
fn attribute<'a>(
    attributes: &'a [plist::Value],
    name: &'static str,
) -> Result<&'a plist::Value, Error> {
    attributes
        .iter()
        .filter_map(plist::Value::as_dictionary)
        .find_map(|d| d.get(name))
        .ok_or(Error::Metadata(name))
}
fn indices(attributes: &[plist::Value], name: &'static str) -> Result<Vec<usize>, Error> {
    attribute(attributes, name)?
        .as_array()
        .ok_or(Error::Metadata(name))?
        .iter()
        .map(|v| {
            v.as_unsigned_integer()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(Error::Metadata(name))
        })
        .collect()
}

impl LoadedProgram {
    pub fn new(program: &Program, qos: NSQualityOfService) -> Result<Self, Error> {
        let _lifecycle = MODEL_LIFECYCLE.lock().map_err(|_| Error::Synchronization)?;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ane-compile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory)?;
        let result = objc2::rc::autoreleasepool(|_| {
            catch_native(|| Self::compile(program, qos, &directory))
        })
        .map_err(Error::from)
        .and_then(std::convert::identity);
        if result.is_err() {
            let _ = std::fs::remove_dir_all(&directory);
        }
        result
    }
    fn compile(
        program: &Program,
        qos: NSQualityOfService,
        directory: &Path,
    ) -> Result<Self, Error> {
        let compiler = GraphCompiler::new(program)?;
        let (executable, descriptor) = compiler.executable()?;
        let feed_names = compiler.feed_names(&executable)?;
        let device = MTLCreateSystemDefaultDevice().ok_or(DeviceError::NoMetalDevice)?;
        let graph_device = unsafe { MPSGraphDevice::deviceWithMTLDevice(&device) };
        let input_types = unsafe {
            Retained::retain_autoreleased(
                raw_message!(&*executable,c"getInputShapes"; *mut NSArray<MPSGraphShapedType>),
            )
        }
        .ok_or(Error::Metadata("compiled input types"))?;
        unsafe {
            raw_message!(&*executable,c"specializeWithDevice:inputShapes:compilationDescriptor:",&*graph_device => &MPSGraphDevice,&*input_types => &NSArray<MPSGraphShapedType>,&*descriptor => &objc2_metal_performance_shaders_graph::MPSGraphCompilationDescriptor; ())
        };
        let package_path = directory.join("graph.mpsgraphpackage");
        let package_url =
            NSURL::fileURLWithPath(&NSString::from_str(&package_path.to_string_lossy()));
        let serialization = unsafe { MPSGraphExecutableSerializationDescriptor::new() };
        unsafe {
            executable
                .serializeToMPSGraphPackageAtURL_descriptor(&package_url, Some(&serialization))
        };
        let package = plist::Value::from_file(package_path.join("manifest.plist"))?;
        let (architecture, attrs) = module_attributes(&package)?;
        let key = attribute(attrs, "mps.ane.calleeName")?
            .as_string()
            .ok_or(Error::Metadata("ANE region name"))?;
        let fully_placed = attrs
            .iter()
            .any(|v| v.as_string() == Some("mps.fullyPlacedOnANE"));
        if !fully_placed && !program.integer_io_adapters() {
            return Err(GraphError::UnsupportedComposition(
                "Apple could not place the complete graph on ANE",
            )
            .into());
        }
        let input_map = indices(attrs, "mps.ane.regionCallArgToEntrypointArgIndex")?;
        let output_map = attribute(attrs, "mps.ane.regionCallResultToEntrypointReturnIndex")
            .ok()
            .map(|_| indices(attrs, "mps.ane.regionCallResultToEntrypointReturnIndex"))
            .transpose()?;
        let symbols = |mapping: &[usize],
                       count: usize,
                       names: &[String],
                       output: bool|
         -> Result<Box<[String]>, Error> {
            if !output && mapping.len() < names.len() {
                return Err(GraphError::UnsupportedComposition(
                    "Apple's compiler removed graph inputs that do not affect the outputs; remove those inputs",
                )
                .into());
            }
            if mapping.len() != names.len() || count != names.len() {
                return Err(Error::Metadata("compiled entrypoint mapping is incomplete"));
            }
            let mut symbols = vec![String::new(); count];
            for (region, &entry) in mapping.iter().enumerate() {
                if entry >= count || !symbols[entry].is_empty() {
                    return Err(Error::Metadata(
                        "compiled entrypoint mapping is not a permutation",
                    ));
                }
                symbols[entry] = if output {
                    format!("{key}__out:{region}")
                } else {
                    format!("{key}__arg{region}")
                };
            }
            Ok(symbols.into())
        };
        let native_inputs = symbols(&input_map, program.inputs().len(), &feed_names, false)?;
        let input_symbols = program
            .inputs()
            .iter()
            .map(|(name, _, _)| {
                feed_names
                    .iter()
                    .position(|n| n == name)
                    .map(|i| native_inputs[i].clone())
                    .ok_or(Error::Metadata(
                        "logical feed is absent from native entrypoint",
                    ))
            })
            .collect::<Result<_, _>>()?;
        let source=unsafe {Retained::retain_autoreleased(raw_message!(&*executable,c"valueForKey:",&*NSString::from_str(obfstr::obfstr!("modelFileArchivePath")) => &NSString; *mut NSString))}.ok_or(Error::Metadata("executable has no compiler archive"))?;
        let source = PathBuf::from(source.to_string());
        if !source.join(format!("{key}.bc.mlir")).is_file() {
            return Err(Error::Metadata(
                "compiler archive does not contain the selected ANEC region",
            ));
        }
        for entry in std::fs::read_dir(&source)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(Error::Metadata("unexpected directory in ANEC product"));
            }
            std::fs::copy(entry.path(), directory.join(entry.file_name()))?;
        }
        let filename = format!("{key}.bc.mlir");
        let options_name = format!("compiler_options_{key}.plist");
        let keys = [
            "kANEFModelType",
            "kANEFCompilerOptionsFilenameKey",
            "kANEFNetPlistFilenameKey",
            "kANEFTargetArchitectureKey",
        ]
        .map(NSString::from_str);
        let values =
            ["kANEFModelANECIR", &options_name, &filename, architecture].map(NSString::from_str);
        let options = NSDictionary::from_slices(
            &keys.iter().map(|k| &**k).collect::<Vec<_>>(),
            &values.iter().map(|v| &**v).collect::<Vec<_>>(),
        );
        let model = ANEModel::new(directory, key)?;
        let anec_bytes = std::fs::metadata(directory.join(filename))?.len() as usize;
        let client = ANEClient::shared()?;
        client.load(
            &model,
            unsafe {
                &*(&*options as *const NSDictionary<NSString, NSString>).cast::<NSDictionary>()
            },
            qos,
        )?;
        let binding = catch_native(|| -> Result<_, Error> {
            let count = program.outputs().len()
                + compiler.state_outputs.len()
                + compiler.discarded_outputs.len();
            let native_outputs = if let Some(mapping) = output_map {
                symbols(&mapping, count, &vec![String::new(); count], true)?
            } else {
                let attributes = model.model_attributes()?;
                let layouts = &attributes
                    .networks
                    .first()
                    .ok_or(Error::Metadata("compiled output layouts"))?
                    .outputs;
                if layouts.len() != count {
                    return Err(Error::Metadata(
                        "unmapped ranking outputs differ from selected targets",
                    ));
                }
                let specs = program.outputs().iter().map(|(_, s, d)| (*s, *d)).chain(
                    compiler
                        .discarded_outputs
                        .iter()
                        .map(|t| (t.physical_shape(), t.data_type())),
                );
                let mut used = HashSet::new();
                specs
                    .map(|(shape, dtype)| {
                        let matches = layouts
                            .iter()
                            .filter(|l| {
                                l.storage_type == dtype.storage_name()
                                    && [l.batches, l.channels, l.height, l.width] == shape.map(Some)
                            })
                            .collect::<Vec<_>>();
                        if matches.len() != 1 || !used.insert(matches[0].name.clone()) {
                            return Err(Error::Metadata("ranking output layout is ambiguous"));
                        }
                        Ok(matches[0].name.clone())
                    })
                    .collect::<Result<Box<[_]>, Error>>()?
            };
            let output_symbols = native_outputs[..program.outputs().len()].into();
            let state_end = program.outputs().len() + compiler.state_outputs.len();
            let state_outputs = native_outputs[program.outputs().len()..state_end]
                .iter()
                .cloned()
                .zip(compiler.state_outputs.iter().copied())
                .collect();
            let discarded_outputs = native_outputs[state_end..]
                .iter()
                .zip(&compiler.discarded_outputs)
                .map(|(symbol, tensor)| {
                    (symbol.clone(), tensor.physical_shape(), tensor.data_type())
                })
                .collect();
            Ok((output_symbols, state_outputs, discarded_outputs))
        })
        .map_err(Error::from)
        .and_then(std::convert::identity);
        let (output_symbols, state_outputs, discarded_outputs) = match binding {
            Ok(binding) => binding,
            Err(error) => {
                let _ = client.unload(&model, qos);
                return Err(error);
            }
        };

        Ok(Self {
            model,
            client,
            qos,
            dispatch: Mutex::new(()),
            directory: directory.into(),
            input_symbols,
            output_symbols,
            state_outputs,
            discarded_outputs,
            anec_bytes,
        })
    }
    pub fn model_attributes(&self) -> Result<ModelAttributes, Error> {
        Ok(self.model.model_attributes()?)
    }
    pub fn evaluate(&self, request: &ANERequest) -> Result<(), Error> {
        let _dispatch = self.dispatch.lock().map_err(|_| Error::Synchronization)?;
        Ok(self.client.evaluate(&self.model, self.qos, request)?)
    }
    pub fn map(&self, request: &ANERequest) -> Result<(), Error> {
        let _dispatch = self.dispatch.lock().map_err(|_| Error::Synchronization)?;
        Ok(self.client.map(&self.model, request)?)
    }
    pub fn unmap(&self, request: &ANERequest) {
        let _dispatch = self.dispatch.lock().unwrap_or_else(PoisonError::into_inner);
        self.client.unmap(&self.model, request);
    }
}
impl Drop for LoadedProgram {
    fn drop(&mut self) {
        let _lifecycle = MODEL_LIFECYCLE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let _ = self.client.unload(&self.model, self.qos);
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
