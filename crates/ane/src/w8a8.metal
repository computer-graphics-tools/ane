// GPU side of the W8A8 programs (see w8a8/kernels.rs): activation pack/unpack and weight staging.
//
// Activations are token-major [tokens, channels] on the GPU and channel-major [channels, row_stride]
// in the programs' surfaces (tokens contiguous within a channel row). Every kernel runs 256 threads
// per threadgroup; the 4096-value Hadamard chunks are register-resident, thread t owning values
// [16 t, 16 t + 16).
#include <metal_stdlib>

using namespace metal;

#define THREADS 256
#define CHUNK 4096
#define BLOCK 512
#define TILE 32
#define VALUES 16
#define DOWN_BLOCK 512
// round-to-nearest-even int8 codes by adding 1.5 * 2^23: the sum's low byte is the code in two's complement.
#define UNROLL _Pragma("clang loop unroll(full)")
#define ROUNDING_MAGIC 12582912.0f

// Unnormalized Walsh-Hadamard transform of `count` values.
template <ushort count>
static void butterfly(thread float* values) {
  UNROLL
  for (ushort span = 1; span < count; span <<= 1) {
    UNROLL
    for (ushort index = 0; index < count; ++index) {
      if (!(index & span)) {
        const float left = values[index];
        const float right = values[index + span];
        values[index] = left + right;
        values[index + span] = left - right;
      }
    }
  }
}

// Spans 16 * lane_stride for lane_stride in [1, lane_strides).
static void hadamard_across_lanes(thread float* values, ushort lane, ushort lane_strides) {
  for (ushort stride = 1; stride < lane_strides; stride <<= 1) {
    const float sign = (lane & stride) ? -1.0f : 1.0f;
    UNROLL
    for (ushort index = 0; index < VALUES; ++index) {
      values[index] = fma(values[index], sign, simd_shuffle_xor(values[index], stride));
    }
  }
}

// Unnormalized transform of the 4096-value chunk. Spans 1..8 run within a thread, 16..256 across lanes,
// 512..2048 across simdgroups as one threadgroup-memory transpose (value v of thread t at v * 256 + t,
// bank-conflict-free both ways). Thread (simdgroup s, lane l) then owns the 8-point columns
// {g * 512 + 16 l + 2 s + j}, g in [0, 8), j in [0, 2), returned as values[2 * g + j]: the chunk stays in
// this layout (see store_codes_columns), so no transpose back is needed.
static void block_hadamard_columns(
    thread float* values,
    ushort lane,
    ushort simdgroup,
    uint thread_id,
    threadgroup float* shared
) {
  butterfly<VALUES>(values);
  hadamard_across_lanes(values, lane, 512 / VALUES);
  UNROLL
  for (ushort index = 0; index < VALUES; ++index) {
    shared[index * THREADS + thread_id] = values[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  UNROLL
  for (ushort j = 0; j < 2; ++j) {
    float column[8];
    UNROLL
    for (ushort group = 0; group < 8; ++group) {
      column[group] = shared[(2 * simdgroup + j) * THREADS + group * 32 + lane];
    }
    butterfly<8>(column);
    UNROLL
    for (ushort group = 0; group < 8; ++group) {
      values[2 * group + j] = column[group];
    }
  }
}

// Orthonormal basis block for widths that are not one 4096-value Hadamard chunk.
static void block_hadamard_512(thread float* values, ushort lane) {
  butterfly<VALUES>(values);
  hadamard_across_lanes(values, lane, BLOCK / VALUES);
}

// `value` negated where bit `index` of `bits` is set (the +-1 sign vectors are stored as bit masks).
static float flip_sign(float value, uint bits, ushort index) {
  return as_type<float>(as_type<uint>(value) ^ ((bits << (31 - index)) & 0x80000000u));
}

static float threadgroup_peak(float value, threadgroup float* partials, ushort lane, ushort simdgroup) {
  const float local = simd_max(value);
  if (lane == 0) {
    partials[simdgroup] = local;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return simd_max(lane < THREADS / 32 ? partials[lane] : 0.0f);
}

static float values_peak(const thread float* values) {
  float peak = 0.0f;
  UNROLL
  for (ushort index = 0; index < VALUES; ++index) {
    peak = max(peak, fabs(values[index]));
  }
  return peak;
}

// Codes are clamped to +-127 (symmetric, as the programs expect).
static void store_codes(device char* destination, const thread float* values, float inverse) {
  UNROLL
  for (ushort index = 0; index < VALUES; index += 4) {
    const uint4 bits = as_type<uint4>(clamp(
        fma(float4(values[index], values[index + 1], values[index + 2], values[index + 3]),
            float4(inverse),
            float4(ROUNDING_MAGIC)),
        float4(ROUNDING_MAGIC - 127.0f),
        float4(ROUNDING_MAGIC + 127.0f)
    ));
    uint packed = bits.w << 24;
    packed = insert_bits(packed, bits.z, 16, 8);
    packed = insert_bits(packed, bits.y, 8, 8);
    packed = insert_bits(packed, bits.x, 0, 8);
    *reinterpret_cast<device uint*>(destination + index) = packed;
  }
}

static void store_codes_columns(
    device char* destination,
    const thread float* values,
    float inverse,
    ushort lane,
    ushort simdgroup
) {
  UNROLL
  for (ushort group = 0; group < 8; ++group) {
    const uint2 bits = as_type<uint2>(clamp(
        fma(float2(values[2 * group], values[2 * group + 1]), float2(inverse), float2(ROUNDING_MAGIC)),
        float2(ROUNDING_MAGIC - 127.0f),
        float2(ROUNDING_MAGIC + 127.0f)
    ));
    *reinterpret_cast<device ushort*>(destination + group * 512 + lane * VALUES + simdgroup * 2) =
        ushort(insert_bits(bits.x, bits.y, 8, 8));
  }
}

// Orthonormal 32-point transform across the lanes of a simdgroup.
static float simdgroup_hadamard(ushort lane, float value) {
  UNROLL
  for (ushort stride = 1; stride < 32; stride <<= 1) {
    const float partner = simd_shuffle_xor(value, stride);
    value = (lane & stride) ? partner - value : partner + value;
  }
  return value * rsqrt(32.0f);
}

// Per token: a = H_4096(signs * x), int8 codes with one scale per token. Writes token-major codes (see
// transpose_codes) and the token scale (x128, the programs' dequantize convention).
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void pack_activations(
    const device bfloat* input [[buffer(0)]],
    const device float* signs [[buffer(1)]],
    device char* codes [[buffer(2)]],
    device half* token_scales [[buffer(3)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float shared[CHUNK];
  threadgroup float partials[32];
  const uint base = thread_id * VALUES;
  const device bfloat* row = input + ulong(token) * CHUNK + base;
  float values[VALUES];
  UNROLL
  for (ushort index = 0; index < VALUES; ++index) {
    values[index] = float(row[index]) * signs[base + index];
  }
  block_hadamard_columns(values, lane, simdgroup, thread_id, shared);
  const float normalization = rsqrt(float(CHUNK));
  const float peak = max(threadgroup_peak(values_peak(values), partials, lane, simdgroup) * normalization, 1e-20f);
  store_codes_columns(codes + ulong(token) * CHUNK, values, 127.0f * normalization / peak, lane, simdgroup);
  if (thread_id == 0) {
    token_scales[token] = half(peak / 127.0f * 128.0f);
  }
}

struct BlockPackArguments {
  uint width;
  uint block_count;
};

// The shape uses an independent H512 per block so any multiple-of-512 width has an orthogonal basis.
template <ushort BLOCKS_PER_SIMDGROUP>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void pack_activations_blocks(
    const device bfloat* input [[buffer(0)]],
    const device float* signs [[buffer(1)]],
    device char* codes [[buffer(2)]],
    device half* token_scales [[buffer(3)]],
    constant BlockPackArguments& arguments [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  float values[BLOCKS_PER_SIMDGROUP][VALUES];
  float peak = 0.0f;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < arguments.block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      thread float* block_values = values[slot];
      UNROLL
      for (ushort index = 0; index < VALUES; ++index) {
        block_values[index] = float(input[ulong(token) * arguments.width + base + index]) * signs[base + index];
      }
      block_hadamard_512(block_values, lane);
      peak = max(peak, values_peak(block_values));
    }
  }

  const float normalization = rsqrt(float(BLOCK));
  const float row_peak = max(threadgroup_peak(peak, partials, lane, simdgroup) * normalization, 1e-20f);
  const float inverse = 127.0f * normalization / row_peak;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < arguments.block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      store_codes(codes + ulong(token) * arguments.width + base, values[slot], inverse);
    }
  }
  if (thread_id == 0) {
    token_scales[token] = half(row_peak / 127.0f * 128.0f);
  }
}

template [[host_name("pack_activations_blocks_1")]] kernel decltype(pack_activations_blocks<1>) pack_activations_blocks<1>;
template [[host_name("pack_activations_blocks_2")]] kernel decltype(pack_activations_blocks<2>) pack_activations_blocks<2>;

// Token-major int8 [tokens, 4096] -> channel-major surface rows [4096, row_stride], `row_stride` in
// elements.
struct TransposeArguments {
  uint tokens;
  uint row_stride;
};

[[kernel, max_total_threads_per_threadgroup(THREADS)]] void transpose_codes(
    const device char* codes [[buffer(0)]],
    device char* surface [[buffer(1)]],
    constant TransposeArguments& arguments [[buffer(2)]],
    uint3 tile_index [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]]
) {
  threadgroup ushort tile[TILE * (TILE + 1)];
  const uint thread_id = thread_index.x;
  const uint channel_base = tile_index.x * TILE;
  const uint token_base = tile_index.y * TILE;
  for (uint index = thread_id; index < TILE * TILE; index += THREADS) {
    const uint token = index / TILE;
    const uint channel = index % TILE;
    ushort value = 0;
    if (token_base + token < arguments.tokens) {
      value = ushort(uchar(codes[ulong(token_base + token) * CHUNK + channel_base + channel]));
    }
    tile[channel * (TILE + 1) + token] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = thread_id; index < TILE * TILE; index += THREADS) {
    const uint channel = index / TILE;
    const uint token = index % TILE;
    if (token_base + token < arguments.tokens) {
      surface[ulong(channel_base + channel) * arguments.row_stride + token_base + token] =
          char(tile[channel * (TILE + 1) + token]);
    }
  }
}

struct BlockTransposeArguments {
  uint tokens;
  uint row_stride;
  uint width;
};

[[kernel, max_total_threads_per_threadgroup(THREADS)]] void transpose_codes_blocks(
    const device char* codes [[buffer(0)]],
    device char* surface [[buffer(1)]],
    constant BlockTransposeArguments& arguments [[buffer(2)]],
    uint3 tile_index [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]]) {
  threadgroup ushort tile[TILE * (TILE + 1)];
  const uint thread_id = thread_index.x;
  const uint channel_base = tile_index.x * TILE;
  const uint token_base = tile_index.y * TILE;
  for (uint index = thread_id; index < TILE * TILE; index += THREADS) {
    const uint token = index / TILE;
    const uint channel = index % TILE;
    ushort value = 0;
    if (token_base + token < arguments.tokens) {
      value = ushort(uchar(codes[ulong(token_base + token) * arguments.width + channel_base + channel]));
    }
    tile[channel * (TILE + 1) + token] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = thread_id; index < TILE * TILE; index += THREADS) {
    const uint channel = index / TILE;
    const uint token = index % TILE;
    if (token_base + token < arguments.tokens) {
      surface[ulong(channel_base + channel) * arguments.row_stride + token_base + token] =
          char(tile[channel * (TILE + 1) + token]);
    }
  }
}

// Surface [channels (+ 1), row_stride] fp16 -> token-major bf16 [tokens, channels]:
// y = signs * H32(y * token_scale * intermediate_scale), the projection's output rotation per 32-channel
// block. With `scale_row` the surface carries a per-token intermediate scale in the row after its
// channels (the FFN); without it (a linear) that factor is 1.
struct UnpackArguments {
  uint channels;
  uint tokens;
  uint row_stride;
  uint scale_row;
};

[[kernel, max_total_threads_per_threadgroup(THREADS)]] void unpack_activations(
    const device half* surface [[buffer(0)]],
    const device half* token_scales [[buffer(1)]],
    const device float* signs [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant UnpackArguments& arguments [[buffer(4)]],
    uint3 tile_index [[threadgroup_position_in_grid]],
    uint3 thread_index [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float tile[TILE * (TILE + 1)];
  const uint thread_id = thread_index.x;
  const uint channel_base = tile_index.x * TILE;
  const uint token_base = tile_index.y * TILE;
  for (uint index = thread_id; index < TILE * TILE; index += THREADS) {
    const uint channel = index / TILE;
    const uint token = index % TILE;
    float value = 0.0f;
    if (token_base + token < arguments.tokens) {
      value = float(surface[ulong(channel_base + channel) * arguments.row_stride + token_base + token]);
    }
    tile[channel * (TILE + 1) + token] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float sign = signs[channel_base + lane];
  for (uint token = simdgroup; token < TILE; token += THREADS / 32) {
    const uint global_token = token_base + token;
    if (global_token >= arguments.tokens) {
      break;
    }
    float scale = float(token_scales[global_token]);
    if (arguments.scale_row) {
      scale *= float(surface[ulong(arguments.channels) * arguments.row_stride + global_token]);
    }
    const float value = simdgroup_hadamard(lane, tile[lane * (TILE + 1) + token] * scale) * sign;
    output[ulong(global_token) * arguments.channels + channel_base + lane] = bfloat(value);
  }
}

// The FFN's `zg` surface: the gate's output signs times 1/2 (the `g/2` of SiLU), one per row of a [rows, 1]
// fp16 surface.
struct GateSignsArguments {
  uint rows;
  uint row_stride;
};

[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_gate_signs(
    const device float* signs [[buffer(0)]],
    device half* surface [[buffer(1)]],
    constant GateSignsArguments& arguments [[buffer(2)]],
    uint block [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]]
) {
  const uint row = block * THREADS + thread_id;
  if (row < arguments.rows) {
    surface[ulong(row) * arguments.row_stride] = half(0.5f * signs[row]);
  }
}

// Dequantize and transform one thread's 16 values of an int4 asymmetric group-32 row chunk:
// H32(scale * (code - zero)) up to the 1/sqrt(32) normalization, which is returned as the factor
// (scale / sqrt(32)) the caller folds into its sign multiply. The zero point only moves the DC term, so
// the transform runs on the raw (exact) integer codes.
static float dequantized_h32(thread float* values, uint2 packed, float scale, float zero, ushort lane) {
  UNROLL
  for (ushort index = 0; index < VALUES; ++index) {
    const uint word = index < 8 ? packed.x : packed.y;
    values[index] = float((word >> ((index % 8) * 4)) & 0xF);
  }
  butterfly<VALUES>(values);
  hadamard_across_lanes(values, lane, 2);
  values[0] -= (lane & 1) ? 0.0f : 32.0f * zero;
  return scale * rsqrt(32.0f);
}

// Staged weight rows: dequantize an int4 group-32 row (codes [rows, columns/2] low nibble first, scales
// [groups, scale_stride] bf16, zero points [groups, scale_stride/2] low nibble = even row), then
//   r' = post_signs * H_block(pre_signs * H32(r))
// and quantize to int8 with one scale per row (written x128 into a [rows, 1] fp16 surface). The second
// transform's normalization is folded into the quantization (peak and inverse scale).
struct StageArguments {
  uint source_row_offset;
  uint scale_stride;
  uint staged_row_stride;
  uint row_scale_stride;
  uint code_row_words;
  uint code_layout;
};

static uint code_word_index(uint row, uint word, uint row_words, uint layout) {
  if (layout == 1) {
    return (row / 32) * 32 * row_words + word * 32 + (row % 8) * 4 + (row / 8) % 4;
  }
  if (layout == 2) {
    const uint y = row % 8;
    const uint x = word % 4;
    const uint lane = ((y & 3) << 1) | ((y & 4) << 2) | ((x & 2) << 2) | (x & 1);
    const uint row_slot = 2 * (row / 16 % 2) + (row % 16) / 8;
    return (row / 32) * 32 * row_words + (word / 4) * 128 + lane * 4 + row_slot;
  }
  return row * row_words + word;
}

static uint2 load_code_words(
    const device uchar* codes,
    uint row,
    uint word,
    uint row_words,
    uint layout
) {
  if (layout == 0) {
    return *reinterpret_cast<const device uint2*>(codes + (ulong(row) * row_words + word) * sizeof(uint));
  }
  return uint2(
      *reinterpret_cast<const device uint*>(codes + ulong(code_word_index(row, word, row_words, layout)) * sizeof(uint)),
      *reinterpret_cast<const device uint*>(codes + ulong(code_word_index(row, word + 1, row_words, layout)) * sizeof(uint))
  );
}

// Up projection: a 4096-column row, block 4096, no post signs. Float pre signs (their 16 KB per row stay
// cache resident, and the float multiply is what keeps this kernel ALU-light).
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_up_rows(
    const device uchar* weight_codes [[buffer(0)]],
    const device bfloat* weight_scales [[buffer(1)]],
    const device uchar* zero_points [[buffer(2)]],
    const device float* pre_signs [[buffer(3)]],
    device char* staged [[buffer(4)]],
    device half* row_scales [[buffer(5)]],
    constant StageArguments& arguments [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float shared[CHUNK];
  threadgroup float partials[32];
  const uint source_row = arguments.source_row_offset + row;
  const uint base = thread_id * VALUES;
  const uint group = base / 32;
  const float scale = float(weight_scales[ulong(group) * arguments.scale_stride + source_row]);
  const float zero = float(
      (zero_points[ulong(group) * (arguments.scale_stride / 2) + source_row / 2] >> ((source_row & 1) * 4)) & 0xF
  );
  const uint2 packed = load_code_words(weight_codes, source_row, base / 8, arguments.code_row_words, arguments.code_layout);

  float values[VALUES];
  const float group_scale = dequantized_h32(values, packed, scale, zero, lane);
  UNROLL
  for (ushort index = 0; index < VALUES; index += 4) {
    const float4 signs = *reinterpret_cast<const device float4*>(pre_signs + base + index) * group_scale;
    values[index] *= signs.x;
    values[index + 1] *= signs.y;
    values[index + 2] *= signs.z;
    values[index + 3] *= signs.w;
  }
  block_hadamard_columns(values, lane, simdgroup, thread_id, shared);

  const float normalization = rsqrt(float(CHUNK));
  const float row_peak = max(threadgroup_peak(values_peak(values), partials, lane, simdgroup) * normalization, 1e-20f);
  store_codes_columns(
      staged + ulong(row) * arguments.staged_row_stride,
      values,
      127.0f * normalization / row_peak,
      lane,
      simdgroup
  );
  if (thread_id == 0) {
    row_scales[ulong(row) * arguments.row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}

struct BlockStageArguments {
  uint source_row_offset;
  uint scale_stride;
  uint staged_row_stride;
  uint row_scale_stride;
  uint code_row_words;
  uint code_layout;
  uint block_count;
};

template <ushort BLOCKS_PER_SIMDGROUP>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_up_rows_blocks(
    const device uchar* weight_codes [[buffer(0)]],
    const device bfloat* weight_scales [[buffer(1)]],
    const device uchar* zero_points [[buffer(2)]],
    const device float* pre_signs [[buffer(3)]],
    device char* staged [[buffer(4)]],
    device half* row_scales [[buffer(5)]],
    constant BlockStageArguments& arguments [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  const uint source_row = arguments.source_row_offset + row;
  float values[BLOCKS_PER_SIMDGROUP][VALUES];
  float peak = 0.0f;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < arguments.block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      const uint group = base / 32;
      const float scale = float(weight_scales[ulong(group) * arguments.scale_stride + source_row]);
      const float zero = float(
          (zero_points[ulong(group) * (arguments.scale_stride / 2) + source_row / 2] >> ((source_row & 1) * 4)) & 0xF
      );
      const uint2 packed = load_code_words(weight_codes, source_row, base / 8, arguments.code_row_words, arguments.code_layout);
      thread float* block_values = values[slot];
      const float group_scale = dequantized_h32(block_values, packed, scale, zero, lane);
      UNROLL
      for (ushort index = 0; index < VALUES; index += 4) {
        const float4 signs = *reinterpret_cast<const device float4*>(pre_signs + base + index) * group_scale;
        block_values[index] *= signs.x;
        block_values[index + 1] *= signs.y;
        block_values[index + 2] *= signs.z;
        block_values[index + 3] *= signs.w;
      }
      block_hadamard_512(block_values, lane);
      peak = max(peak, values_peak(block_values));
    }
  }

  const float normalization = rsqrt(float(BLOCK));
  const float row_peak = max(threadgroup_peak(peak, partials, lane, simdgroup) * normalization, 1e-20f);
  const float inverse = 127.0f * normalization / row_peak;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < arguments.block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      store_codes(staged + ulong(row) * arguments.staged_row_stride + base, values[slot], inverse);
    }
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * arguments.row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}

template [[host_name("stage_up_rows_blocks_1")]] kernel decltype(stage_up_rows_blocks<1>) stage_up_rows_blocks<1>;
template [[host_name("stage_up_rows_blocks_2")]] kernel decltype(stage_up_rows_blocks<2>) stage_up_rows_blocks<2>;

// Down projection: `CHUNKS` * 4096 columns (at most 3 chunks, held in registers) of source rows
// addressed through `code_row_words`, block 512, with post signs. The +-1 sign vectors are bit masks: the float signs of a 12288-column
// row (96 KB with both) did not stay cache resident and dominated the kernel.
struct StageDownArguments {
  uint scale_stride;
  uint staged_row_stride;
  uint row_scale_stride;
  uint code_row_words;
  uint code_layout;
};

template <ushort CHUNKS>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_down_rows(
    const device uchar* weight_codes [[buffer(0)]],
    const device bfloat* weight_scales [[buffer(1)]],
    const device uchar* zero_points [[buffer(2)]],
    const device ushort* pre_sign_bits [[buffer(3)]],
    const device ushort* post_sign_bits [[buffer(4)]],
    device char* staged [[buffer(5)]],
    device half* row_scales [[buffer(6)]],
    constant StageDownArguments& arguments [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
  threadgroup float partials[32];
  const uint zero_shift = (row & 1) * 4;

  float values[CHUNKS][VALUES];
  float peak = 0.0f;
  UNROLL
  for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
    const uint base = chunk * CHUNK + thread_id * VALUES;
    thread float* chunk_values = values[chunk];
    const uint group = base / 32;
    const float scale = float(weight_scales[ulong(group) * arguments.scale_stride + row]);
    const float zero = float((zero_points[ulong(group) * (arguments.scale_stride / 2) + row / 2] >> zero_shift) & 0xF);
    const uint2 packed = load_code_words(weight_codes, row, base / 8, arguments.code_row_words, arguments.code_layout);
    const float group_scale = dequantized_h32(chunk_values, packed, scale, zero, lane);
    const uint pre_bits = pre_sign_bits[base / VALUES];
    UNROLL
    for (ushort index = 0; index < VALUES; ++index) {
      chunk_values[index] = flip_sign(chunk_values[index], pre_bits, index) * group_scale;
    }
    butterfly<VALUES>(chunk_values);
    hadamard_across_lanes(chunk_values, lane, DOWN_BLOCK / VALUES);
    const uint post_bits = post_sign_bits[base / VALUES];
    UNROLL
    for (ushort index = 0; index < VALUES; ++index) {
      chunk_values[index] = flip_sign(chunk_values[index], post_bits, index);
    }
    peak = max(peak, values_peak(chunk_values));
  }
  const float normalization = rsqrt(float(DOWN_BLOCK));
  const float row_peak = max(threadgroup_peak(peak, partials, lane, thread_id / 32) * normalization, 1e-20f);
  const float inverse = 127.0f * normalization / row_peak;
  device char* destination = staged + ulong(row) * arguments.staged_row_stride;
  UNROLL
  for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
    store_codes(destination + chunk * CHUNK + thread_id * VALUES, values[chunk], inverse);
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * arguments.row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}

struct BlockStageDownArguments {
  uint scale_stride;
  uint staged_row_stride;
  uint row_scale_stride;
  uint code_row_words;
  uint code_layout;
  uint block_count;
};

template <ushort BLOCKS_PER_SIMDGROUP>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_down_rows_blocks(
    const device uchar* weight_codes [[buffer(0)]],
    const device bfloat* weight_scales [[buffer(1)]],
    const device uchar* zero_points [[buffer(2)]],
    const device ushort* pre_sign_bits [[buffer(3)]],
    const device ushort* post_sign_bits [[buffer(4)]],
    device char* staged [[buffer(5)]],
    device half* row_scales [[buffer(6)]],
    constant BlockStageDownArguments& arguments [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partials[32];
  const uint zero_shift = (row & 1) * 4;
  float values[BLOCKS_PER_SIMDGROUP][VALUES];
  float peak = 0.0f;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < arguments.block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      const uint group = base / 32;
      const float scale = float(weight_scales[ulong(group) * arguments.scale_stride + row]);
      const float zero = float(
          (zero_points[ulong(group) * (arguments.scale_stride / 2) + row / 2] >> zero_shift) & 0xF
      );
      const uint2 packed = load_code_words(weight_codes, row, base / 8, arguments.code_row_words, arguments.code_layout);
      thread float* block_values = values[slot];
      const float group_scale = dequantized_h32(block_values, packed, scale, zero, lane);
      const uint pre_bits = pre_sign_bits[base / VALUES];
      UNROLL
      for (ushort index = 0; index < VALUES; ++index) {
        block_values[index] = flip_sign(block_values[index], pre_bits, index) * group_scale;
      }
      block_hadamard_512(block_values, lane);
      const uint post_bits = post_sign_bits[base / VALUES];
      UNROLL
      for (ushort index = 0; index < VALUES; ++index) {
        block_values[index] = flip_sign(block_values[index], post_bits, index);
      }
      peak = max(peak, values_peak(block_values));
    }
  }

  const float normalization = rsqrt(float(BLOCK));
  const float row_peak = max(threadgroup_peak(peak, partials, lane, simdgroup) * normalization, 1e-20f);
  const float inverse = 127.0f * normalization / row_peak;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < arguments.block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      store_codes(staged + ulong(row) * arguments.staged_row_stride + base, values[slot], inverse);
    }
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * arguments.row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}

template [[host_name("stage_down_rows_1")]] kernel decltype(stage_down_rows<1>) stage_down_rows<1>;
template [[host_name("stage_down_rows_2")]] kernel decltype(stage_down_rows<2>) stage_down_rows<2>;
template [[host_name("stage_down_rows_3")]] kernel decltype(stage_down_rows<3>) stage_down_rows<3>;
template [[host_name("stage_down_rows_blocks_1")]] kernel decltype(stage_down_rows_blocks<1>) stage_down_rows_blocks<1>;
template [[host_name("stage_down_rows_blocks_2")]] kernel decltype(stage_down_rows_blocks<2>) stage_down_rows_blocks<2>;
template [[host_name("stage_down_rows_blocks_3")]] kernel decltype(stage_down_rows_blocks<3>) stage_down_rows_blocks<3>;
template [[host_name("stage_down_rows_blocks_4")]] kernel decltype(stage_down_rows_blocks<4>) stage_down_rows_blocks<4>;
template [[host_name("stage_down_rows_blocks_5")]] kernel decltype(stage_down_rows_blocks<5>) stage_down_rows_blocks<5>;
