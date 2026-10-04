# ane

Rust bindings for Apple Neural Engine (ANE) via the private `AppleNeuralEngine.framework`.

Provides a typed graph builder that emits MIL, compiles it with Apple's ANE compiler through `_ANEInMemoryModel`, and runs it on IOSurface-backed zero-copy buffers. A graph that cannot run entirely on ANE returns an error; there is no CPU or GPU fallback.

## Example

```rust
use ane::{DataType, Graph};

fn main() -> Result<(), ane::Error> {
    let graph = Graph::new();
    let x = graph.placeholder([64, 64], DataType::Float32)?;
    let w = graph.constant(&[0.5; 64 * 64], [64, 64])?;
    let y = graph.matrix_multiplication(&x, &w, false, false)?;
    let y = graph.relu(&y)?;
    let executable = graph.compile(&[y], &[], None)?;

    let input = executable.input(x)?.allocate()?;
    input.copy_from_f32(&[1.0; 64 * 64])?;
    let results = executable.run(&[&input], None, None)?;
    assert!(results[0].read_f32()?.iter().all(|&v| v == 32.0));
    Ok(())
}
```

The API follows MPSGraph (`placeholder`, `matrix_multiplication`, `read_variable`, `compile`, `run`), but every graph method is one operation the ANE runs natively; an operation the ANE cannot run fails to compile. `run` takes inputs in `executable.feed_tensors()` order and returns results in target order; passing your own result buffers reuses their mapped request. `compile_shared` compiles several target sets into one ANE model so they share weights and variables. Shapes have up to four dimensions; spatial operations use NCHW.

## Mutable weights

```sh
cargo run --release --example mutable_matmul
```

The example binds an FP16 weight IOSurface as a variable, rewrites it from the CPU between executions without recompiling, and checks every output exactly.

## Research

The ANE internals research behind this crate was inspired by Mohamed Ghannam's [weightBufs exploit chain](https://cra.sh/public_html/strlcpy3/iosmacos-exploit-chain-cve-2022-32845-32948-42805-32899-weightbufs) writeup, which documents the ANE architecture, the `aned` / `ANECompilerService` pipeline, and the kernel interface (`AppleH11ANEInterface`).

## License

[MIT](https://github.com/computer-graphics-tools/ane/blob/main/LICENSE)
