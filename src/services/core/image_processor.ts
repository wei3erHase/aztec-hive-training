/**
 * Image Processing Module
 * Handles 28x28 image normalization, validation, and preprocessing for neural network input
 */

export interface ImageData {
  pixels: number[];
  width: number;
  height: number;
  channels: number;
}

export interface ProcessingOptions {
  normalize: boolean;
  centerCrop: boolean;
  resize: boolean;
  invertColors: boolean;
  addNoise: boolean;
  noiseLevel: number;
}

export class ImageProcessor {
  private options: ProcessingOptions;

  constructor(options: Partial<ProcessingOptions> = {}) {
    this.options = {
      normalize: true,
      centerCrop: true,
      resize: true,
      invertColors: false,
      addNoise: false,
      noiseLevel: 0.05,
      ...options,
    };
  }

  public processImage(imageData: ImageData): number[] {
    let pixels = [...imageData.pixels];
    this.validateImageData(imageData);

    if (
      this.options.resize &&
      (imageData.width !== 28 || imageData.height !== 28)
    ) {
      pixels = this.resizeImage(
        pixels,
        imageData.width,
        imageData.height,
        28,
        28
      );
    }

    if (
      this.options.centerCrop &&
      imageData.width > 28 &&
      imageData.height > 28
    ) {
      pixels = this.centerCrop(
        pixels,
        imageData.width,
        imageData.height,
        28,
        28
      );
    }

    if (imageData.channels > 1) {
      pixels = this.convertToGrayscale(pixels, imageData.channels);
    }

    if (this.options.invertColors) {
      pixels = pixels.map((pixel) => 255 - pixel);
    }

    pixels = this.normalizeToRange(pixels, 0, 255);

    if (this.options.addNoise) {
      pixels = this.addNoise(pixels, this.options.noiseLevel);
    }

    if (this.options.normalize) {
      pixels = pixels.map((pixel) => pixel / 255);
    }

    pixels = pixels.map((pixel) => {
      let value = pixel;
      if (this.options.normalize && pixel <= 1.0) {
        value = pixel * 255;
      }
      return Math.max(0, Math.min(255, Math.round(value)));
    });

    if (pixels.length !== 784) {
      throw new Error(`Expected 784 pixels, got ${pixels.length}`);
    }

    return pixels;
  }

  private validateImageData(imageData: ImageData): void {
    if (!imageData.pixels || !Array.isArray(imageData.pixels)) {
      throw new Error('Image pixels must be an array');
    }
    if (imageData.width <= 0 || imageData.height <= 0) {
      throw new Error('Image dimensions must be positive');
    }
    if (imageData.channels <= 0 || imageData.channels > 4) {
      throw new Error('Image channels must be 1-4');
    }
    const expectedPixels =
      imageData.width * imageData.height * imageData.channels;
    if (imageData.pixels.length !== expectedPixels) {
      throw new Error(
        `Expected ${expectedPixels} pixels, got ${imageData.pixels.length}`
      );
    }
    const invalidPixels = imageData.pixels.filter(
      (pixel) => !Number.isFinite(pixel) || pixel < 0 || pixel > 255
    );
    if (invalidPixels.length > 0) {
      throw new Error(
        `Found ${invalidPixels.length} invalid pixel values (must be 0-255)`
      );
    }
  }

  private resizeImage(
    pixels: number[],
    srcWidth: number,
    srcHeight: number,
    dstWidth: number,
    dstHeight: number
  ): number[] {
    const resized: number[] = [];
    const xRatio = srcWidth / dstWidth;
    const yRatio = srcHeight / dstHeight;

    for (let y = 0; y < dstHeight; y++) {
      for (let x = 0; x < dstWidth; x++) {
        const srcX = x * xRatio;
        const srcY = y * yRatio;
        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        const x2 = Math.min(x1 + 1, srcWidth - 1);
        const y2 = Math.min(y1 + 1, srcHeight - 1);
        const dx = srcX - x1;
        const dy = srcY - y1;

        const topLeft = pixels[y1 * srcWidth + x1];
        const topRight = pixels[y1 * srcWidth + x2];
        const bottomLeft = pixels[y2 * srcWidth + x1];
        const bottomRight = pixels[y2 * srcWidth + x2];

        const top = topLeft * (1 - dx) + topRight * dx;
        const bottom = bottomLeft * (1 - dx) + bottomRight * dx;
        const interpolated = top * (1 - dy) + bottom * dy;
        resized.push(Math.round(interpolated));
      }
    }
    return resized;
  }

  private centerCrop(
    pixels: number[],
    srcWidth: number,
    srcHeight: number,
    cropWidth: number,
    cropHeight: number
  ): number[] {
    const cropped: number[] = [];
    const startX = Math.floor((srcWidth - cropWidth) / 2);
    const startY = Math.floor((srcHeight - cropHeight) / 2);

    for (let y = 0; y < cropHeight; y++) {
      for (let x = 0; x < cropWidth; x++) {
        const srcIdx = (startY + y) * srcWidth + (startX + x);
        cropped.push(pixels[srcIdx]);
      }
    }
    return cropped;
  }

  private convertToGrayscale(pixels: number[], channels: number): number[] {
    if (channels === 1) return pixels;
    const grayscale: number[] = [];
    for (let i = 0; i < pixels.length; i += channels) {
      if (channels === 3 || channels === 4) {
        const r = pixels[i];
        const g = pixels[i + 1];
        const b = pixels[i + 2];
        grayscale.push(Math.round(0.299 * r + 0.587 * g + 0.114 * b));
      } else {
        grayscale.push(pixels[i]);
      }
    }
    return grayscale;
  }

  private normalizeToRange(
    pixels: number[],
    min: number,
    max: number
  ): number[] {
    const currentMin = Math.min(...pixels);
    const currentMax = Math.max(...pixels);
    if (currentMin === currentMax) return pixels.map(() => min);
    const scale = (max - min) / (currentMax - currentMin);
    return pixels.map((pixel) =>
      Math.round(min + (pixel - currentMin) * scale)
    );
  }

  private addNoise(pixels: number[], noiseLevel: number): number[] {
    return pixels.map((pixel) => {
      const noise = (Math.random() - 0.5) * 2 * noiseLevel * 255;
      return Math.max(0, Math.min(255, Math.round(pixel + noise)));
    });
  }
}

export const MNIST_PROCESSING_OPTIONS: ProcessingOptions = {
  normalize: false,
  centerCrop: true,
  resize: true,
  invertColors: false,
  addNoise: false,
  noiseLevel: 0.02,
};

export function createMNISTProcessor(): ImageProcessor {
  return new ImageProcessor(MNIST_PROCESSING_OPTIONS);
}
