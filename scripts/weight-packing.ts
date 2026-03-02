/**
 * Weight packing for ZKML constructor_pretrained.
 * Mirrors zkml/packing.nr: pack quantized values [-256, 255] into Fields.
 * 9 bits per value, 28 values per Field.
 */

import { Fr } from '@aztec/aztec.js/fields';

const OFFSET = 256;
const MASK_9_BITS = 511n;
const POW_2_9 = 512n;
const VALUES_PER_FIELD = 28;

function clampQuantized(v: number): number {
  if (v > 255) return 255;
  if (v < -256) return -256;
  return v;
}

/**
 * Pack N quantized values (each in [-256, 255]) into a single Field.
 */
export function pack(values: number[]): Fr {
  let packed = 0n;
  let shift = 1n;

  for (let i = 0; i < values.length && i < VALUES_PER_FIELD; i++) {
    const v = clampQuantized(values[i]);
    const vOffset = BigInt(v + OFFSET) & MASK_9_BITS;
    packed += vOffset * shift;
    shift *= POW_2_9;
  }

  return new Fr(packed);
}

/**
 * Pack weights or biases into Fields for constructor_pretrained.
 * Fills batches of 28, pads with zeros.
 */
export function packToFields(
  values: number[],
  packedCount: number,
  valuesPerField = VALUES_PER_FIELD
): Fr[] {
  const result: Fr[] = [];

  for (let packedIdx = 0; packedIdx < packedCount; packedIdx++) {
    const batch: number[] = [];
    const startIdx = packedIdx * valuesPerField;

    for (let i = 0; i < valuesPerField; i++) {
      const globalIdx = startIdx + i;
      batch.push(globalIdx < values.length ? values[globalIdx] : 0);
    }

    result.push(pack(batch));
  }

  return result;
}
