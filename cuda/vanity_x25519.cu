// One candidate per CUDA thread.
//
// The field arithmetic uses five 51-bit limbs. CUDA supports 128-bit integer
// arithmetic, which keeps the intermediate products in registers and avoids
// the local-memory spills of a wider 16-limb representation.
// The private key stream is ChaCha20 keyed by a fresh host-provided seed. This
// keeps device-side candidates cryptographically unpredictable while avoiding
// a host-to-device transfer for every candidate.

#include <stdint.h>

using u32 = unsigned int;
using u64 = unsigned long long;
using u128 = unsigned __int128;

using fe = u64[5];
constexpr u64 kMask51 = (1ULL << 51) - 1;
constexpr u64 kSubBias0 = (2 * kMask51) - 36; // 2 * (2^51 - 19)
constexpr u64 kSubBias = 2 * kMask51;

__device__ __forceinline__ void fe_copy(fe out, const fe in) {
  for (int i = 0; i < 5; ++i) out[i] = in[i];
}

__device__ __forceinline__ void fe_carry(fe out) {
  for (int i = 0; i < 4; ++i) {
    out[i + 1] += out[i] >> 51;
    out[i] &= kMask51;
  }
  out[0] += (out[4] >> 51) * 19;
  out[4] &= kMask51;
  out[1] += out[0] >> 51;
  out[0] &= kMask51;
}

__device__ __forceinline__ void fe_add(fe out, const fe a, const fe b) {
  for (int i = 0; i < 5; ++i) out[i] = a[i] + b[i];
}

__device__ __forceinline__ void fe_sub(fe out, const fe a, const fe b) {
  out[0] = a[0] + kSubBias0 - b[0];
  for (int i = 1; i < 5; ++i) out[i] = a[i] + kSubBias - b[i];
}

__device__ __forceinline__ void fe_reduce(u128 h0, u128 h1, u128 h2, u128 h3,
                                          u128 h4, fe out) {
  h3 += h2 >> 51;
  u64 g2 = (u64)h2 & kMask51;
  h1 += h0 >> 51;
  u64 g0 = (u64)h0 & kMask51;
  h4 += h3 >> 51;
  u64 g3 = (u64)h3 & kMask51;
  g2 += (u64)(h1 >> 51);
  u64 g1 = (u64)h1 & kMask51;
  g0 += (u64)(h4 >> 51) * 19;
  u64 g4 = (u64)h4 & kMask51;
  g3 += g2 >> 51;
  g2 &= kMask51;
  g1 += g0 >> 51;
  g0 &= kMask51;
  out[0] = g0;
  out[1] = g1;
  out[2] = g2;
  out[3] = g3;
  out[4] = g4;
}

__device__ __forceinline__ void fe_mul(fe out, const fe a, const fe b) {
  u64 f0 = a[0], f1 = a[1], f2 = a[2], f3 = a[3], f4 = a[4];
  u64 g0 = b[0], g1 = b[1], g2 = b[2], g3 = b[3], g4 = b[4];
  u128 h0 = (u128)f0 * g0;
  u128 h1 = (u128)f0 * g1;
  u128 h2 = (u128)f0 * g2;
  u128 h3 = (u128)f0 * g3;
  u128 h4 = (u128)f0 * g4;
  f0 = f1;
  h0 += (u128)f0 * (g4 *= 19);
  h1 += (u128)f0 * g0;
  h2 += (u128)f0 * g1;
  h3 += (u128)f0 * g2;
  h4 += (u128)f0 * g3;
  f0 = f2;
  h0 += (u128)f0 * (g3 *= 19);
  h1 += (u128)f0 * g4;
  h2 += (u128)f0 * g0;
  h3 += (u128)f0 * g1;
  h4 += (u128)f0 * g2;
  f0 = f3;
  h0 += (u128)f0 * (g2 *= 19);
  h1 += (u128)f0 * g3;
  h2 += (u128)f0 * g4;
  h3 += (u128)f0 * g0;
  h4 += (u128)f0 * g1;
  f0 = f4;
  h0 += (u128)f0 * (g1 *= 19);
  h1 += (u128)f0 * g2;
  h2 += (u128)f0 * g3;
  h3 += (u128)f0 * g4;
  h4 += (u128)f0 * g0;
  fe_reduce(h0, h1, h2, h3, h4, out);
}

__device__ __forceinline__ void fe_square(fe out, const fe a) {
  u64 g0 = a[0], g1 = a[1], g2 = a[2], g3 = a[3], g4 = a[4];
  u128 h0 = (u128)g0 * g0;
  g0 *= 2;
  u128 h1 = (u128)g0 * g1;
  u128 h2 = (u128)g0 * g2;
  u128 h3 = (u128)g0 * g3;
  u128 h4 = (u128)g0 * g4;
  g0 = g4;
  h3 += (u128)g0 * (g4 *= 19);
  h2 += (u128)g1 * g1;
  g1 *= 2;
  h3 += (u128)g1 * g2;
  h4 += (u128)g1 * g3;
  h0 += (u128)g1 * g4;
  g0 = g3;
  h1 += (u128)g0 * (g3 *= 19);
  h2 += (u128)(g0 * 2) * g4;
  h4 += (u128)g2 * g2;
  g2 *= 2;
  h0 += (u128)g2 * g3;
  h1 += (u128)g2 * g4;
  fe_reduce(h0, h1, h2, h3, h4, out);
}

__device__ __forceinline__ void fe_add_mul_const(fe out, const fe a, const fe b,
                                                 u64 constant) {
  u128 h0 = (u128)a[0] + (u128)constant * b[0];
  u128 h1 = (u128)a[1] + (u128)constant * b[1];
  u128 h2 = (u128)a[2] + (u128)constant * b[2];
  u128 h3 = (u128)a[3] + (u128)constant * b[3];
  u128 h4 = (u128)a[4] + (u128)constant * b[4];
  fe_reduce(h0, h1, h2, h3, h4, out);
}

__device__ __forceinline__ void fe_mul_const(fe out, const fe a, u64 constant) {
  fe zero = {0, 0, 0, 0, 0};
  fe_add_mul_const(out, zero, a, constant);
}

__device__ __forceinline__ void fe_select(fe p, fe q, int bit) {
  const u64 mask = -(u64)bit;
  for (int i = 0; i < 5; ++i) {
    const u64 t = mask & (p[i] ^ q[i]);
    p[i] ^= t;
    q[i] ^= t;
  }
}

__device__ __forceinline__ void fe_unpack(fe out, const uint8_t in[32]) {
  u64 h0 = (u64)in[0] | (u64)in[1] << 8 | (u64)in[2] << 16 |
           (u64)in[3] << 24 | (u64)in[4] << 32 | (u64)in[5] << 40 |
           (u64)in[6] << 48;
  u64 h1 = ((u64)in[7] | (u64)in[8] << 8 | (u64)in[9] << 16 |
            (u64)in[10] << 24 | (u64)in[11] << 32 | (u64)in[12] << 40)
           << 5;
  u64 h2 = ((u64)in[13] | (u64)in[14] << 8 | (u64)in[15] << 16 |
            (u64)in[16] << 24 | (u64)in[17] << 32 | (u64)in[18] << 40 |
            (u64)in[19] << 48)
           << 2;
  u64 h3 = ((u64)in[20] | (u64)in[21] << 8 | (u64)in[22] << 16 |
            (u64)in[23] << 24 | (u64)in[24] << 32 | (u64)in[25] << 40)
           << 7;
  u64 h4 = (((u64)in[26] | (u64)in[27] << 8 | (u64)in[28] << 16 |
             (u64)in[29] << 24 | (u64)in[30] << 32 | (u64)in[31] << 40) &
            0x7fffffffffffULL)
           << 4;
  out[0] = h0 & kMask51;
  out[1] = (h1 | h0 >> 51) & kMask51;
  out[2] = (h2 | h1 >> 51) & kMask51;
  out[3] = (h3 | h2 >> 51) & kMask51;
  out[4] = (h4 | h3 >> 51) & kMask51;
}

__device__ __forceinline__ void fe_pack(uint8_t out[32], const fe in) {
  fe t;
  fe_copy(t, in);
  u64 q = (t[0] + 19) >> 51;
  q = (t[1] + q) >> 51;
  q = (t[2] + q) >> 51;
  q = (t[3] + q) >> 51;
  q = (t[4] + q) >> 51;
  t[0] += 19 * q;
  t[1] += t[0] >> 51;
  t[0] &= kMask51;
  t[2] += t[1] >> 51;
  t[1] &= kMask51;
  t[3] += t[2] >> 51;
  t[2] &= kMask51;
  t[4] += t[3] >> 51;
  t[3] &= kMask51;
  t[4] &= kMask51;
  out[0] = t[0] >> 0;
  out[1] = t[0] >> 8;
  out[2] = t[0] >> 16;
  out[3] = t[0] >> 24;
  out[4] = t[0] >> 32;
  out[5] = t[0] >> 40;
  out[6] = (t[0] >> 48) | (t[1] << 3);
  out[7] = t[1] >> 5;
  out[8] = t[1] >> 13;
  out[9] = t[1] >> 21;
  out[10] = t[1] >> 29;
  out[11] = t[1] >> 37;
  out[12] = (t[1] >> 45) | (t[2] << 6);
  out[13] = t[2] >> 2;
  out[14] = t[2] >> 10;
  out[15] = t[2] >> 18;
  out[16] = t[2] >> 26;
  out[17] = t[2] >> 34;
  out[18] = t[2] >> 42;
  out[19] = (t[2] >> 50) | (t[3] << 1);
  out[20] = t[3] >> 7;
  out[21] = t[3] >> 15;
  out[22] = t[3] >> 23;
  out[23] = t[3] >> 31;
  out[24] = t[3] >> 39;
  out[25] = (t[3] >> 47) | (t[4] << 4);
  out[26] = t[4] >> 4;
  out[27] = t[4] >> 12;
  out[28] = t[4] >> 20;
  out[29] = t[4] >> 28;
  out[30] = t[4] >> 36;
  out[31] = t[4] >> 44;
}

__device__ void fe_inverse(fe out, const fe z) {
  fe t0, t1, t2, t3;
  fe_square(t0, z);
  fe_square(t1, t0);
  fe_square(t1, t1);
  fe_mul(t1, z, t1);
  fe_mul(t0, t0, t1);
  fe_square(t2, t0);
  fe_mul(t1, t1, t2);
  fe_square(t2, t1);
  for (int i = 1; i < 5; ++i) fe_square(t2, t2);
  fe_mul(t1, t2, t1);
  fe_square(t2, t1);
  for (int i = 1; i < 10; ++i) fe_square(t2, t2);
  fe_mul(t2, t2, t1);
  fe_square(t3, t2);
  for (int i = 1; i < 20; ++i) fe_square(t3, t3);
  fe_mul(t2, t3, t2);
  for (int i = 0; i < 10; ++i) fe_square(t2, t2);
  fe_mul(t1, t2, t1);
  fe_square(t2, t1);
  for (int i = 1; i < 50; ++i) fe_square(t2, t2);
  fe_mul(t2, t2, t1);
  fe_square(t3, t2);
  for (int i = 1; i < 100; ++i) fe_square(t3, t3);
  fe_mul(t2, t3, t2);
  for (int i = 0; i < 50; ++i) fe_square(t2, t2);
  fe_mul(t1, t2, t1);
  for (int i = 0; i < 5; ++i) fe_square(t1, t1);
  fe_mul(out, t1, t0);
}

__device__ void x25519(uint8_t out[32], const uint8_t scalar_in[32],
                       const uint8_t public_u[32]) {
  uint8_t scalar[32];
  for (int i = 0; i < 32; ++i) scalar[i] = scalar_in[i];
  scalar[0] &= 248;
  scalar[31] &= 127;
  scalar[31] |= 64;

  fe x1, x2, z2, x3, z3, tmp0, tmp1;
  fe_unpack(x1, public_u);
  for (int i = 0; i < 5; ++i) {
    x2[i] = 0;
    z2[i] = 0;
    x3[i] = x1[i];
    z3[i] = 0;
  }
  x2[0] = 1;
  z3[0] = 1;

  int swap = 0;
  for (int pos = 254; pos >= 0; --pos) {
    const int bit = (scalar[pos >> 3] >> (pos & 7)) & 1;
    swap ^= bit;
    fe_select(x2, x3, swap);
    fe_select(z2, z3, swap);
    swap = bit;

    fe_sub(tmp0, x3, z3);
    fe_sub(tmp1, x2, z2);
    fe_add(x2, x2, z2);
    fe_add(z2, x3, z3);
    fe_mul(z3, tmp0, x2);
    fe_mul(z2, z2, tmp1);
    fe_square(tmp0, tmp1);
    fe_square(tmp1, x2);
    fe_add(x3, z3, z2);
    fe_sub(z2, z3, z2);
    fe_mul(x2, tmp1, tmp0);
    fe_sub(tmp1, tmp1, tmp0);
    fe_square(z2, z2);
    fe_mul_const(z3, tmp1, 121666);
    fe_square(x3, x3);
    fe_add(tmp0, tmp0, z3);
    fe_mul(z3, x1, z2);
    fe_mul(z2, tmp1, tmp0);
  }
  fe_select(x2, x3, swap);
  fe_select(z2, z3, swap);
  fe_inverse(tmp0, z2);
  fe_mul(tmp1, x2, tmp0);
  fe_pack(out, tmp1);
}

__device__ __forceinline__ u32 load32(const uint8_t *p) {
  return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

__device__ __forceinline__ void store32(uint8_t *p, u32 v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

__device__ __forceinline__ u32 rotl(u32 x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ __forceinline__ void qr(u32 &a, u32 &b, u32 &c, u32 &d) {
  a += b; d ^= a; d = rotl(d, 16);
  c += d; b ^= c; b = rotl(b, 12);
  a += b; d ^= a; d = rotl(d, 8);
  c += d; b ^= c; b = rotl(b, 7);
}

__device__ void chacha20_block(uint8_t out[64], const uint8_t key[32], u64 counter) {
  u32 x[16] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,
               load32(key + 0), load32(key + 4), load32(key + 8), load32(key + 12),
               load32(key + 16), load32(key + 20), load32(key + 24), load32(key + 28),
               (u32)counter, (u32)(counter >> 32), 0x243f6a88, 0x85a308d3};
  u32 initial[16];
  for (int i = 0; i < 16; ++i) initial[i] = x[i];
  for (int round = 0; round < 10; ++round) {
    qr(x[0], x[4], x[8], x[12]); qr(x[1], x[5], x[9], x[13]);
    qr(x[2], x[6], x[10], x[14]); qr(x[3], x[7], x[11], x[15]);
    qr(x[0], x[5], x[10], x[15]); qr(x[1], x[6], x[11], x[12]);
    qr(x[2], x[7], x[8], x[13]); qr(x[3], x[4], x[9], x[14]);
  }
  for (int i = 0; i < 16; ++i) store32(out + 4 * i, x[i] + initial[i]);
}

__device__ __forceinline__ uint8_t b64(uint8_t v) {
  return v < 26 ? 'A' + v : v < 52 ? 'a' + v - 26 : v < 62 ? '0' + v - 52 : v == 62 ? '+' : '/';
}

__device__ void base64_32(uint8_t out[44], const uint8_t in[32]) {
  int o = 0;
  for (int i = 0; i < 30; i += 3) {
    out[o++] = b64(in[i] >> 2);
    out[o++] = b64(((in[i] & 3) << 4) | (in[i + 1] >> 4));
    out[o++] = b64(((in[i + 1] & 15) << 2) | (in[i + 2] >> 6));
    out[o++] = b64(in[i + 2] & 63);
  }
  out[o++] = b64(in[30] >> 2);
  out[o++] = b64(((in[30] & 3) << 4) | (in[31] >> 4));
  out[o++] = b64((in[31] & 15) << 2);
  out[o] = '=';
}

__device__ bool glob_match_at(const uint8_t encoded[44], const uint8_t *pattern,
                              uint32_t pattern_len, uint32_t text_start, uint32_t end,
                              uint32_t case_sensitive) {
  uint32_t text = text_start;
  uint32_t pattern_index = 0;
  uint32_t star_index = 0xffffffffu;
  uint32_t star_text = text_start;
  while (text < end) {
    if (pattern_index == pattern_len) return true;
    uint8_t got = encoded[text];
    if (!case_sensitive && got >= 'A' && got <= 'Z') {
      got = (uint8_t)(got + ('a' - 'A'));
    }
    if (pattern_index < pattern_len && pattern[pattern_index] != '*' &&
        (pattern[pattern_index] == '?' || pattern[pattern_index] == got)) {
      ++pattern_index;
      ++text;
    } else if (pattern_index < pattern_len && pattern[pattern_index] == '*') {
      star_index = pattern_index++;
      star_text = text;
    } else if (star_index != 0xffffffffu) {
      pattern_index = star_index + 1;
      text = ++star_text;
    } else {
      return false;
    }
  }
  while (pattern_index < pattern_len && pattern[pattern_index] == '*') ++pattern_index;
  return pattern_index == pattern_len;
}

__device__ bool prefix_match(const uint8_t encoded[44], const uint8_t *prefix,
                             uint32_t prefix_len, uint32_t mode,
                             uint32_t case_sensitive, uint32_t start, uint32_t end) {
  if (prefix_len == 0 || end > 44 || start > end ||
      (mode == 0 && prefix_len > end - start)) return false;
  if (mode == 1) {
    for (uint32_t at = start; at <= end; ++at) {
      if (glob_match_at(encoded, prefix, prefix_len, at, end, case_sensitive)) return true;
    }
    return false;
  }
  for (uint32_t at = start; at + prefix_len <= end; ++at) {
    bool ok = true;
    for (uint32_t i = 0; i < prefix_len; ++i) {
      uint8_t got = encoded[at + i];
      if (!case_sensitive && got >= 'A' && got <= 'Z') {
        got = (uint8_t)(got + ('a' - 'A'));
      }
      if (got != prefix[i]) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

extern "C" __global__ void vanity_kernel(const uint8_t *seed, u64 base_counter, u64 count,
                                          const uint8_t *prefix, uint32_t prefix_len,
                                          uint32_t pattern_mode, uint32_t case_sensitive,
                                          uint32_t start, uint32_t end,
                                          int *found,
                                          uint8_t *private_out, uint8_t *public_out) {
  const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = tid < count && *found < 0;

  uint8_t private_key[32], public_key[32], encoded[44], stream[64];
  if (active) {
    chacha20_block(stream, seed, base_counter + tid);
    for (int i = 0; i < 32; ++i) private_key[i] = stream[i];
    const uint8_t basepoint[32] = {9};
    x25519(public_key, private_key, basepoint);
    base64_32(encoded, public_key);
    if (prefix_match(encoded, prefix, prefix_len, pattern_mode, case_sensitive, start, end) &&
        atomicCAS(found, -1, (int)tid) == -1) {
      for (int i = 0; i < 32; ++i) {
        private_out[i] = private_key[i];
        public_out[i] = public_key[i];
      }
    }
  }
}

__device__ __forceinline__ uint32_t base64_sextet_at(const uint8_t in[32], uint32_t position) {
  const uint32_t group = position >> 2;
  const uint32_t lane = position & 3;
  const uint32_t index = group * 3;
  switch (lane) {
    case 0:
      return in[index] >> 2;
    case 1:
      return ((in[index] & 3u) << 4) | (in[index + 1] >> 4);
    case 2:
      return ((in[index + 1] & 15u) << 2) |
             (index + 2 < 32 ? (in[index + 2] >> 6) : 0u);
    default:
      return in[index + 2] & 63u;
  }
}

__device__ __forceinline__ bool regex_match_base64(
    const uint8_t public_key[32], uint32_t start, uint32_t end,
    const uint32_t *__restrict__ transitions, const uint32_t *__restrict__ equals,
    const uint8_t *__restrict__ eoi_match) {
  constexpr uint32_t kDfaMatch = 1u << 31;
  constexpr uint32_t kDfaDead = 1u << 30;
  constexpr uint32_t kDfaStateMask = (1u << 30) - 1;
  uint32_t state = 0;
  for (uint32_t position = start; position < end; ++position) {
    const uint32_t edge = position == 43
        ? equals[state]
        : transitions[(state << 6) | base64_sextet_at(public_key, position)];
    if (edge & kDfaMatch) return true;
    if (edge & kDfaDead) return false;
    state = edge & kDfaStateMask;
  }
  return eoi_match[state] != 0;
}

extern "C" __global__ void vanity_regex_kernel(
    const uint8_t *seed, u64 base_counter, u64 count,
    const uint32_t *__restrict__ transitions, const uint32_t *__restrict__ equals,
    const uint8_t *__restrict__ eoi_match, uint32_t start, uint32_t end, int *found,
    uint8_t *private_out, uint8_t *public_out) {
  const u64 tid = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = tid < count && *found < 0;
  if (!active) return;

  uint8_t private_key[32], public_key[32], stream[64];
  chacha20_block(stream, seed, base_counter + tid);
  for (int i = 0; i < 32; ++i) private_key[i] = stream[i];
  const uint8_t basepoint[32] = {9};
  x25519(public_key, private_key, basepoint);

  if (regex_match_base64(public_key, start, end, transitions, equals, eoi_match) &&
      atomicCAS(found, -1, (int)tid) == -1) {
    for (int i = 0; i < 32; ++i) {
      private_out[i] = private_key[i];
      public_out[i] = public_key[i];
    }
  }
}

extern "C" __global__ void regex_match_test_kernel(
    const uint8_t *inputs, uint32_t stride, uint32_t count,
    const uint32_t *__restrict__ transitions, const uint32_t *__restrict__ equals,
    const uint8_t *__restrict__ eoi_match, uint32_t start, uint32_t end,
    uint8_t *results) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    results[index] = regex_match_base64(
        inputs + (size_t)index * stride, start, end, transitions, equals, eoi_match);
  }
}
