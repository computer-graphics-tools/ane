// Compact U8/U2 model weights directly into ANE's row-major int8 surfaces.
// This file is concatenated after w8a8.metal and reuses its Hadamard, reduction, and store helpers.

constant uint STAGE_SOURCE_FORMAT [[function_constant(0)]];
constant uint STAGE_SOURCE_LAYOUT [[function_constant(1)]];
constant bool STAGE_SIGNED_CODES [[function_constant(2)]];
constant uint STAGE_SOURCE_BLOCK [[function_constant(3)]];
constant bool STAGE_HAS_POST_SIGNS [[function_constant(4)]];

struct StageCompactArguments {
  uint source_row_offset;
  uint rows;
  uint columns;
  uint scale_stride;
  uint group_size;
  uint staged_row_stride;
  uint row_scale_stride;
};

static uint stage_compact_mxu_lane(uint x_group, uint y) {
  const uint quad = (y & 4u) | (x_group & 2u);
  return quad * 4u + (y & 3u) * 2u + (x_group & 1u);
}

static uint stage_compact_mxu_code(const device uchar* codes, uint source_row, uint columns, uint column) {
  const device uint* words = reinterpret_cast<const device uint*>(codes);
  if (STAGE_SOURCE_FORMAT == 0) {
    constexpr uint BLOCK_ROWS = 32;
    constexpr uint LANE_WORDS = 8;
    const uint row = source_row % BLOCK_ROWS;
    const uint column_word = column / 4;
    const uint chunk = column_word / 8;
    const uint within_chunk = column_word % 8;
    const uint fragment_row = row / 16;
    const uint row_slot = row % 16 / 8;
    const uint fragment_column = within_chunk / 4;
    const uint x_group = within_chunk % 4;
    const uint word = fragment_row * 4 + row_slot * 2 + fragment_column;
    const uint block_words = BLOCK_ROWS * (columns / 4);
    const ulong physical = ulong(source_row / BLOCK_ROWS) * block_words + chunk * 32 * LANE_WORDS +
                           stage_compact_mxu_lane(x_group, row % 8) * LANE_WORDS + word;
    return (words[physical] >> (8 * (column % 4))) & 0xffu;
  }

  constexpr uint BLOCK_ROWS = 16;
  constexpr uint LANE_WORDS = 4;
  const uint row = source_row % BLOCK_ROWS;
  const uint chunk = column / 128;
  const uint within_chunk = column % 128;
  const uint fragment_column = within_chunk / 64;
  const uint within_fragment = within_chunk % 64;
  const uint source_word = within_fragment / 16;
  const uint x_group = within_fragment % 16 / 4;
  const uint source_code = within_fragment % 4;
  const uint packed_code = source_code * 4 + source_word;
  const uint row_slot = row / 8;
  const uint word = row_slot + fragment_column * 2;
  const uint block_words = BLOCK_ROWS * (columns / 16);
  const ulong physical = ulong(source_row / BLOCK_ROWS) * block_words + chunk * 32 * LANE_WORDS +
                         stage_compact_mxu_lane(x_group, row % 8) * LANE_WORDS + word;
  return (words[physical] >> (2 * packed_code)) & 0x3u;
}

static void stage_compact_decode_chunk(
    thread float* values,
    const device uchar* codes,
    const device bfloat* scales,
    const device bfloat* biases,
    constant StageCompactArguments& arguments,
    uint source_row,
    uint base
) {
  const uint group = base / arguments.group_size;
  const ulong metadata_offset = ulong(group) * arguments.scale_stride + source_row;
  const float scale = float(scales[metadata_offset]);
  if (STAGE_SOURCE_FORMAT == 0 && STAGE_SOURCE_LAYOUT == 0) {
    const device uint* packed =
        reinterpret_cast<const device uint*>(codes + ulong(source_row) * arguments.columns + base);
    UNROLL
    for (ushort word = 0; word < VALUES / 4; ++word) {
      const uint raw = packed[word];
      UNROLL
      for (ushort byte = 0; byte < 4; ++byte) {
        const uint code = ((raw >> (8 * byte)) & 0xffu) ^ (STAGE_SIGNED_CODES ? 0x80u : 0u);
        values[word * 4 + byte] = (float(code) - 128.0f) * scale;
      }
    }
  } else if (STAGE_SOURCE_FORMAT == 1 && STAGE_SOURCE_LAYOUT == 0) {
    const ulong row_offset = ulong(source_row) * arguments.columns / 4;
    const uint packed = *reinterpret_cast<const device uint*>(codes + row_offset + base / 4);
    const float bias = float(biases[metadata_offset]);
    UNROLL
    for (ushort index = 0; index < VALUES; ++index) {
      const uint code = (packed >> (2 * index)) & 0x3u;
      values[index] = fma(float(code), scale, bias);
    }
  } else {
    const float bias = STAGE_SOURCE_FORMAT == 1 ? float(biases[metadata_offset]) : 0.0f;
    UNROLL
    for (ushort index = 0; index < VALUES; ++index) {
      const uint column = base + index;
      const uint code = stage_compact_mxu_code(codes, source_row, arguments.columns, column);
      if (STAGE_SOURCE_FORMAT == 0) {
        const uint decoded = code ^ (STAGE_SIGNED_CODES ? 0x80u : 0u);
        values[index] = (float(decoded) - 128.0f) * scale;
      } else {
        values[index] = fma(float(code), scale, bias);
      }
    }
  }
}

template <ushort SOURCE_BLOCK>
static void stage_compact_source_hadamard(thread float* values, ushort lane) {
  butterfly<VALUES>(values);
  hadamard_across_lanes(values, lane, SOURCE_BLOCK / VALUES);
  const float normalization = rsqrt(float(SOURCE_BLOCK));
  UNROLL
  for (ushort index = 0; index < VALUES; ++index) {
    values[index] *= normalization;
  }
}

static void stage_compact_apply_pre_signs(thread float* values, const device float* pre_signs, uint base) {
  UNROLL
  for (ushort index = 0; index < VALUES; index += 4) {
    const float4 signs = *reinterpret_cast<const device float4*>(pre_signs + base + index);
    values[index] *= signs.x;
    values[index + 1] *= signs.y;
    values[index + 2] *= signs.z;
    values[index + 3] *= signs.w;
  }
}

static void stage_compact_store_codes(
    device char* destination,
    thread float* values,
    const device float* post_signs,
    uint base,
    float inverse
) {
  if (STAGE_HAS_POST_SIGNS) {
    UNROLL
    for (ushort index = 0; index < VALUES; ++index) {
      values[index] *= post_signs[base + index];
    }
  }
  store_codes(destination + base, values, inverse);
}

[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_compact_up_rows(
    const device uchar* weight_codes [[buffer(0)]],
    const device bfloat* weight_scales [[buffer(1)]],
    const device bfloat* weight_biases [[buffer(2)]],
    const device float* pre_signs [[buffer(3)]],
    const device float* post_signs [[buffer(4)]],
    device char* staged [[buffer(5)]],
    device half* row_scales [[buffer(6)]],
    constant StageCompactArguments& arguments [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float shared[CHUNK];
  threadgroup float partials[32];
  const uint source_row = arguments.source_row_offset + row;
  const uint base = thread_id * VALUES;
  float values[VALUES];
  stage_compact_decode_chunk(values, weight_codes, weight_scales, weight_biases, arguments, source_row, base);
  if (STAGE_SOURCE_BLOCK == 32) {
    stage_compact_source_hadamard<32>(values, lane);
  } else {
    stage_compact_source_hadamard<128>(values, lane);
  }
  stage_compact_apply_pre_signs(values, pre_signs, base);
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

template <ushort BLOCKS_PER_SIMDGROUP>
[[kernel, max_total_threads_per_threadgroup(THREADS)]] void stage_compact_down_rows_blocks(
    const device uchar* weight_codes [[buffer(0)]],
    const device bfloat* weight_scales [[buffer(1)]],
    const device bfloat* weight_biases [[buffer(2)]],
    const device float* pre_signs [[buffer(3)]],
    const device float* post_signs [[buffer(4)]],
    device char* staged [[buffer(5)]],
    device half* row_scales [[buffer(6)]],
    constant StageCompactArguments& arguments [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_position_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
  threadgroup float partials[32];
  const uint source_row = arguments.source_row_offset + row;
  const uint block_count = arguments.columns / BLOCK;
  float values[BLOCKS_PER_SIMDGROUP][VALUES];
  float peak = 0.0f;
  UNROLL
  for (ushort slot = 0; slot < BLOCKS_PER_SIMDGROUP; ++slot) {
    const uint block = simdgroup + slot * (THREADS / 32);
    if (block < block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      thread float* block_values = values[slot];
      stage_compact_decode_chunk(block_values, weight_codes, weight_scales, weight_biases, arguments, source_row, base);
      if (STAGE_SOURCE_BLOCK == 32) {
        stage_compact_source_hadamard<32>(block_values, lane);
      } else {
        stage_compact_source_hadamard<128>(block_values, lane);
      }
      stage_compact_apply_pre_signs(block_values, pre_signs, base);
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
    if (block < block_count) {
      const uint base = block * BLOCK + lane * VALUES;
      stage_compact_store_codes(
          staged + ulong(row) * arguments.staged_row_stride,
          values[slot],
          post_signs,
          base,
          inverse
      );
    }
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * arguments.row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}

template [[host_name(
    "stage_compact_down_rows_blocks_1"
)]] kernel decltype(stage_compact_down_rows_blocks<1>) stage_compact_down_rows_blocks<1>;
template [[host_name(
    "stage_compact_down_rows_blocks_2"
)]] kernel decltype(stage_compact_down_rows_blocks<2>) stage_compact_down_rows_blocks<2>;
template [[host_name(
    "stage_compact_down_rows_blocks_3"
)]] kernel decltype(stage_compact_down_rows_blocks<3>) stage_compact_down_rows_blocks<3>;
template [[host_name(
    "stage_compact_down_rows_blocks_4"
)]] kernel decltype(stage_compact_down_rows_blocks<4>) stage_compact_down_rows_blocks<4>;
template [[host_name(
    "stage_compact_down_rows_blocks_5"
)]] kernel decltype(stage_compact_down_rows_blocks<5>) stage_compact_down_rows_blocks<5>;
