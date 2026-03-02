/**
 * Canvas Image Processor - Enforces exact 28x28 pixel format for proofs
 *
 * This module ensures all canvas images are processed to the exact format
 * required by the neural network proofs: 28x28 pixels with integer values 0-255
 */

export interface ProofImageData {
  pixels: number[];
  width: 28;
  height: 28;
}

/**
 * Process canvas to exact proof format: 28x28 pixels with integer values 0-255
 */
export function processCanvasForProof(
  canvas: HTMLCanvasElement
): ProofImageData {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Canvas context not available');
  }

  const sourceImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const pixels: number[] = new Array(784);
  const scaleX = canvas.width / 28;
  const scaleY = canvas.height / 28;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sourceX = Math.round(x * scaleX);
      const sourceY = Math.round(y * scaleY);
      const clampedX = Math.min(sourceX, canvas.width - 1);
      const clampedY = Math.min(sourceY, canvas.height - 1);
      const sourceIndex = (clampedY * canvas.width + clampedX) * 4;

      const r = sourceImageData.data[sourceIndex];
      const g = sourceImageData.data[sourceIndex + 1];
      const b = sourceImageData.data[sourceIndex + 2];
      const a = sourceImageData.data[sourceIndex + 3];

      let gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      if (a < 128) {
        gray = 255;
      }
      const inverted = 255 - gray;
      const pixel = Math.max(0, Math.min(255, Math.round(inverted)));
      pixels[y * 28 + x] = pixel;
    }
  }

  return {
    pixels,
    width: 28,
    height: 28,
  };
}

/**
 * Validate that image data meets proof requirements
 */
export function validateProofImageData(imageData: ProofImageData): boolean {
  if (!imageData || typeof imageData !== 'object') return false;
  if (imageData.width !== 28 || imageData.height !== 28) return false;
  if (!Array.isArray(imageData.pixels)) return false;
  if (imageData.pixels.length !== 784) return false;

  return imageData.pixels.every(
    (pixel) => Number.isInteger(pixel) && pixel >= 0 && pixel <= 255
  );
}

/**
 * Convert proof image data to the format expected by neural network circuits
 */
export function imageDataToCircuitFormat(imageData: ProofImageData): number[] {
  if (!validateProofImageData(imageData)) {
    throw new Error('Invalid image data format for proof');
  }
  return [...imageData.pixels];
}

/**
 * Create a visual representation of the processed image for debugging
 */
export function createDebugCanvas(
  imageData: ProofImageData
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 28;
  canvas.height = 28;
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not create debug canvas context');
  }

  const debugImageData = ctx.createImageData(28, 28);

  for (let i = 0; i < 784; i++) {
    const pixel = imageData.pixels[i];
    const dataIndex = i * 4;
    const value = 255 - pixel;
    debugImageData.data[dataIndex] = value;
    debugImageData.data[dataIndex + 1] = value;
    debugImageData.data[dataIndex + 2] = value;
    debugImageData.data[dataIndex + 3] = 255;
  }

  ctx.putImageData(debugImageData, 0, 0);
  return canvas;
}

/**
 * Get statistics about the processed image
 */
export function getImageStats(imageData: ProofImageData): {
  min: number;
  max: number;
  mean: number;
  nonZeroPixels: number;
  coverage: number;
} {
  const pixels = imageData.pixels;
  const min = Math.min(...pixels);
  const max = Math.max(...pixels);
  const sum = pixels.reduce((a, b) => a + b, 0);
  const mean = sum / pixels.length;
  const nonZeroPixels = pixels.filter((p) => p > 0).length;
  const coverage = (nonZeroPixels / pixels.length) * 100;

  return { min, max, mean, nonZeroPixels, coverage };
}
