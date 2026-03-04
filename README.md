# ane

Rust bindings for Apple Neural Engine (ANE) via the private `AppleNeuralEngine.framework`.

Provides a symbolic graph builder and a compile-then-run lifecycle through `_ANEInMemoryModel`, using IOSurface-backed zero-copy I/O.

## Example

```rust
use ane::{Graph, Shape, TensorData, NSQualityOfService};

let mut graph = Graph::new();

let input   = graph.placeholder(Shape::channels(64));
let weights = graph.constant(&weight_data, Shape { channels: 64, height: 1, width: 1, batch: 1 });
let output  = graph.convolution_2d_1x1(input, weights, None);
let output  = graph.relu(output);

let executable = graph.compile(NSQualityOfService::Default)?;

let input_tensor  = TensorData::with_f32(&data, Shape::channels(64));
let output_tensor = TensorData::new(Shape::channels(64));
executable.run(&[&input_tensor], &[&output_tensor])?;

let result = output_tensor.read_f32();
```

## GPT-2 forward pass

The included `gpt2_forward` example downloads GPT-2 124M from Hugging Face, compiles the transformer layers to ANE, and runs autoregressive text generation with KV-cache:

```
cargo run --release --example gpt2_forward
```

## Research

The ANE internals research behind this crate was inspired by Mohamed Ghannam's [weightBufs exploit chain](https://cra.sh/public_html/strlcpy3/iosmacos-exploit-chain-cve-2022-32845-32948-42805-32899-weightbufs) writeup, which documents the ANE architecture, the `aned` / `ANECompilerService` pipeline, and the kernel interface (`AppleH11ANEInterface`).

## License

[MIT](LICENSE)
