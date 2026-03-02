import { describe, it, expect, beforeEach } from 'vitest';
import {
  ImageProcessor,
  createMNISTProcessor,
  MNIST_PROCESSING_OPTIONS,
  type ImageData,
} from '@/services/core/image_processor';

function make28x28Gradient(): ImageData {
  return {
    pixels: Array.from({ length: 784 }, (_, i) => Math.floor((i / 784) * 255)),
    width: 28,
    height: 28,
    channels: 1,
  };
}

function makePassthroughProcessor(): ImageProcessor {
  return new ImageProcessor({
    normalize: false,
    centerCrop: false,
    resize: false,
    invertColors: false,
    addNoise: false,
  });
}

describe('ImageProcessor', () => {
  describe('validateImageData — error paths', () => {
    let processor: ImageProcessor;
    beforeEach(() => {
      processor = makePassthroughProcessor();
    });

    it('throws when pixel count does not match width×height×channels', () => {
      const bad: ImageData = {
        pixels: [1, 2, 3],
        width: 28,
        height: 28,
        channels: 1,
      };
      expect(() => processor.processImage(bad)).toThrow(/Expected \d+ pixels/);
    });

    it('throws when width is zero', () => {
      const bad: ImageData = { pixels: [], width: 0, height: 28, channels: 1 };
      expect(() => processor.processImage(bad)).toThrow(
        'Image dimensions must be positive'
      );
    });

    it('throws when height is zero', () => {
      const bad: ImageData = { pixels: [], width: 28, height: 0, channels: 1 };
      expect(() => processor.processImage(bad)).toThrow(
        'Image dimensions must be positive'
      );
    });

    it('throws when channel count exceeds 4', () => {
      const bad: ImageData = {
        pixels: Array(28 * 28 * 5).fill(0),
        width: 28,
        height: 28,
        channels: 5,
      };
      expect(() => processor.processImage(bad)).toThrow(
        'Image channels must be 1-4'
      );
    });

    it('throws when channel count is zero', () => {
      const bad: ImageData = { pixels: [], width: 28, height: 28, channels: 0 };
      expect(() => processor.processImage(bad)).toThrow(
        'Image channels must be 1-4'
      );
    });

    it('throws when a pixel value exceeds 255', () => {
      const pixels = Array(784).fill(128);
      pixels[0] = 300;
      const bad: ImageData = { pixels, width: 28, height: 28, channels: 1 };
      expect(() => processor.processImage(bad)).toThrow(/invalid pixel/);
    });

    it('throws when a pixel value is negative', () => {
      const pixels = Array(784).fill(128);
      pixels[10] = -1;
      const bad: ImageData = { pixels, width: 28, height: 28, channels: 1 };
      expect(() => processor.processImage(bad)).toThrow(/invalid pixel/);
    });
  });

  describe('processImage output guarantees', () => {
    it('always returns exactly 784 values', () => {
      const processor = makePassthroughProcessor();
      const result = processor.processImage(make28x28Gradient());
      expect(result).toHaveLength(784);
    });

    it('all output values are in [0, 255]', () => {
      const processor = makePassthroughProcessor();
      const result = processor.processImage(make28x28Gradient());
      for (const px of result) {
        expect(px).toBeGreaterThanOrEqual(0);
        expect(px).toBeLessThanOrEqual(255);
      }
    });

    it('all output values are integers', () => {
      const processor = makePassthroughProcessor();
      const result = processor.processImage(make28x28Gradient());
      for (const px of result) {
        expect(Number.isInteger(px)).toBe(true);
      }
    });
  });

  describe('resize', () => {
    it('resizes a 56×56 input to 784 output pixels', () => {
      const processor = new ImageProcessor({
        normalize: false,
        centerCrop: false,
        resize: true,
        invertColors: false,
        addNoise: false,
      });
      const bigImage: ImageData = {
        pixels: Array.from({ length: 56 * 56 }, (_, i) => i % 256),
        width: 56,
        height: 56,
        channels: 1,
      };
      expect(processor.processImage(bigImage)).toHaveLength(784);
    });
  });

  describe('invertColors', () => {
    it('produces a different output than non-inverted on a gradient image', () => {
      const image = make28x28Gradient();
      const normal = makePassthroughProcessor().processImage(image);
      const inverted = new ImageProcessor({
        normalize: false,
        centerCrop: false,
        resize: false,
        invertColors: true,
        addNoise: false,
      }).processImage(image);
      // Inversion reverses the gradient ordering before range-stretch, so outputs differ
      expect(inverted).not.toEqual(normal);
    });
  });

  describe('grayscale conversion', () => {
    it('converts 3-channel RGB to single-channel output (784 pixels)', () => {
      const processor = makePassthroughProcessor();
      // uniform R=200, G=100, B=50 for all 784 pixels → constant grayscale → normalizeToRange → all 0
      const rgbImage: ImageData = {
        pixels: Array(784 * 3)
          .fill(0)
          .map((_, i) => {
            const ch = i % 3;
            return ch === 0 ? 200 : ch === 1 ? 100 : 50;
          }),
        width: 28,
        height: 28,
        channels: 3,
      };
      const result = processor.processImage(rgbImage);
      expect(result).toHaveLength(784);
      // constant grayscale → normalizeToRange maps to all 0
      for (const px of result) {
        expect(px).toBe(0);
      }
    });
  });

  describe('centerCrop', () => {
    it('crops a 32×32 image to exactly 784 output pixels', () => {
      const processor = new ImageProcessor({
        normalize: false,
        centerCrop: true,
        resize: false,
        invertColors: false,
        addNoise: false,
      });
      const bigImage: ImageData = {
        pixels: Array.from({ length: 32 * 32 }, (_, i) => i % 256),
        width: 32,
        height: 32,
        channels: 1,
      };
      const result = processor.processImage(bigImage);
      expect(result).toHaveLength(784);
    });

    it('crops from the center — border pixels are excluded from the output', () => {
      // 32×32 → 28×28: center crop starts at (startX=2, startY=2).
      // Place a sentinel pixel at (col=1, row=1) — inside the 2px border, excluded by center crop.
      // All other pixels are 0.
      // Correct center crop: all output pixels are 0 → normalizeToRange → all 0.
      // Wrong crop from (0,0): sentinel (col=1, row=1) would be included → output has a 200
      //   → normalizeToRange produces a 255 at that position → NOT all zero.
      const srcW = 32;
      const pixels = Array(srcW * srcW).fill(0);
      pixels[1 * srcW + 1] = 200; // at (col=1, row=1), inside the 2px border

      const processor = new ImageProcessor({
        normalize: false,
        centerCrop: true,
        resize: false,
        invertColors: false,
        addNoise: false,
      });
      const result = processor.processImage({
        pixels,
        width: 32,
        height: 32,
        channels: 1,
      });
      // The sentinel at (1,1) must NOT appear in the center 28×28 crop.
      // normalizeToRange on an all-zero image returns all 0.
      expect(result.every((v) => v === 0)).toBe(true);
    });

    it('does NOT crop when image is exactly 28×28', () => {
      const image = make28x28Gradient();
      const withCrop = new ImageProcessor({
        normalize: false,
        centerCrop: true,
        resize: false,
        invertColors: false,
        addNoise: false,
      }).processImage(image);
      const withoutCrop = makePassthroughProcessor().processImage(image);
      expect(withCrop).toEqual(withoutCrop);
    });
  });

  describe('addNoise', () => {
    it('produces output different from input (noise is applied)', () => {
      const processor = new ImageProcessor({
        normalize: false,
        centerCrop: false,
        resize: false,
        invertColors: false,
        addNoise: true,
        noiseLevel: 0.5,
      });
      const image = make28x28Gradient();
      const original = makePassthroughProcessor().processImage(image);
      const noisy = processor.processImage(image);
      // With noiseLevel=0.5 (±127 noise) at least one pixel will differ
      const differ = noisy.some((v, i) => v !== original[i]);
      expect(differ).toBe(true);
    });

    it('all noisy output values remain in [0, 255]', () => {
      const processor = new ImageProcessor({
        normalize: false,
        centerCrop: false,
        resize: false,
        invertColors: false,
        addNoise: true,
        noiseLevel: 1.0,
      });
      const result = processor.processImage(make28x28Gradient());
      for (const px of result) {
        expect(px).toBeGreaterThanOrEqual(0);
        expect(px).toBeLessThanOrEqual(255);
      }
    });

    it('zero noiseLevel produces identical output to passthrough', () => {
      const processor = new ImageProcessor({
        normalize: false,
        centerCrop: false,
        resize: false,
        invertColors: false,
        addNoise: true,
        noiseLevel: 0,
      });
      const image = make28x28Gradient();
      const result = processor.processImage(image);
      const expected = makePassthroughProcessor().processImage(image);
      expect(result).toEqual(expected);
    });
  });

  describe('createMNISTProcessor / MNIST_PROCESSING_OPTIONS', () => {
    it('createMNISTProcessor returns an ImageProcessor', () => {
      expect(createMNISTProcessor()).toBeInstanceOf(ImageProcessor);
    });

    it('MNIST options disable normalize (keeps raw 0-255 integers)', () => {
      expect(MNIST_PROCESSING_OPTIONS.normalize).toBe(false);
    });

    it('MNIST options enable resize', () => {
      expect(MNIST_PROCESSING_OPTIONS.resize).toBe(true);
    });

    it('MNIST options enable centerCrop', () => {
      expect(MNIST_PROCESSING_OPTIONS.centerCrop).toBe(true);
    });

    it('MNIST processor produces 784 values', () => {
      const p = createMNISTProcessor();
      expect(p.processImage(make28x28Gradient())).toHaveLength(784);
    });
  });
});
