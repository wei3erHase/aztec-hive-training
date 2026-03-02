export const SCALING_FACTOR = 1_000_000;
export const FIELD_MODULUS =
  21888242871839275222246405745257275088548364400416034343698204186575808495617n;

export function fieldToSigned(value: bigint): number {
  if (value <= FIELD_MODULUS / 2n) return Number(value);
  return Number(value - FIELD_MODULUS);
}

export function downscaleTo8x8(pixels28x28: number[]): number[] {
  const pixels8x8: number[] = [];
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      let sum = 0;
      let count = 0;
      const startY = Math.floor(y * 3.5);
      const endY = Math.floor((y + 1) * 3.5);
      const startX = Math.floor(x * 3.5);
      const endX = Math.floor((x + 1) * 3.5);
      for (let py = startY; py < endY && py < 28; py++) {
        for (let px = startX; px < endX && px < 28; px++) {
          sum += pixels28x28[py * 28 + px];
          count++;
        }
      }
      pixels8x8.push(Math.round(sum / count));
    }
  }
  return pixels8x8;
}

export function pixelsToScaledBigInts(pixels: number[]): bigint[] {
  return pixels.map((p) => BigInt(Math.round((p * SCALING_FACTOR) / 255)));
}
