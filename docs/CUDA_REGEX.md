# CUDA Regex Implementation

CUDA regex matching keeps key generation and matching on the device. The host
compiles the Rust regex once, then sends a compact DFA table to each selected
GPU during searcher initialization.

## Compilation

`regex-automata` builds a forward dense DFA using the same regex syntax family
as the CPU `regex::bytes::Regex` path. The compiler walks only states reachable
from the slice start through the 64 standard Base64 characters and the final
`=` padding byte. Reachable states are re-numbered so state zero is the start
state.

The CUDA backend rejects empty patterns, oversized patterns, DFA state counts
above 4096, and compact tables above 1 MiB. These limits prevent unbounded
determinization or device allocation. Captures and match offsets are not
returned.

## Device Table

The compact representation contains:

```text
transitions[state * 64 + sextet] : u32
equals[state]                    : u32
eoi_match[state]                  : u8
```

Transition values use bit 31 for an accepting edge, bit 30 for a dead edge,
and bits 0..29 for the next compact state. The table is read from global
read-only memory. It is uploaded once per GPU and reused for every batch.

## Kernel Path

Each CUDA thread performs ChaCha20, X25519, and Base64 matching. It computes a
Base64 sextet directly from the public key rather than materializing a 44-byte
string. Position 43 uses the `equals` table; positions 0..42 use the 64-column
transition table. After `start..end` has been consumed, the matcher executes
the DFA EOI transition. A successful thread uses `atomicCAS` to publish the
first matching keypair.

For each later batch, host-to-device traffic is limited to the fresh seed and
the found flag. The regex table, regex text, and ordinary candidate strings are
not transferred per batch.

## Semantics and Testing

Matching is equivalent to applying the CPU regex to `public_key[start..end]`.
Therefore `^`, `$`, `\A`, `\z`, and word boundaries are relative to that slice.
The Rust compact-table simulator is tested against `regex::bytes::Regex` on
more than one million deterministic random Base64 inputs. A standalone CUDA
matcher kernel provides the next differential-test layer. End-to-end matches
return both keys so the CPU can verify the X25519 relationship and the regex.

The literal and glob kernels remain separate from the regex kernel. This keeps
their register allocation and performance independent of DFA state handling.
