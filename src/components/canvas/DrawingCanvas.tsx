import React, { useRef, useState, useEffect, useCallback } from 'react';
import { motion } from 'motion/react';
import {
  processCanvasForProof,
  type ProofImageData,
} from '../../utils/canvas_processor';
import { downscaleTo8x8 } from '../../utils/zkml';
import { CanvasPreview } from './CanvasPreview';

const CANVAS_SIZE = 280;

export interface DrawingCanvasProps {
  onDraw?: (imageData: ProofImageData) => void;
  onClear?: () => void;
  disabled?: boolean;
  showPreview?: boolean;
  brushSize?: number;
  className?: string;
  previewHeatmap?: number[];
  beforeHeatmap?: number[];
  afterHeatmap?: number[];
  neuronHeatmap?: number[];
}

export function DrawingCanvas({
  onDraw,
  onClear,
  disabled = false,
  showPreview = true,
  brushSize = 20,
  className = '',
  previewHeatmap,
  beforeHeatmap,
  afterHeatmap,
  neuronHeatmap,
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasContent, setHasContent] = useState(false);
  const lastPos = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_SIZE * dpr;
    canvas.height = CANVAS_SIZE * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = brushSize;
  }, [brushSize]);

  const getCoordinates = useCallback(
    (e: React.MouseEvent | React.TouchEvent): { x: number; y: number } => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const scaleX = CANVAS_SIZE / rect.width;
      const scaleY = CANVAS_SIZE / rect.height;
      if ('touches' in e) {
        const touch = e.touches[0];
        return {
          x: (touch.clientX - rect.left) * scaleX,
          y: (touch.clientY - rect.top) * scaleY,
        };
      }
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    },
    []
  );

  const startDrawing = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (disabled) return;
      if (!('touches' in e)) e.preventDefault();
      setIsDrawing(true);
      const coords = getCoordinates(e);
      lastPos.current = coords;
      const ctx = canvasRef.current?.getContext('2d');
      if (ctx) {
        ctx.beginPath();
        ctx.arc(coords.x, coords.y, brushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        setHasContent(true);
      }
    },
    [disabled, getCoordinates, brushSize]
  );

  const draw = useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      if (!isDrawing || disabled) return;
      if (!('touches' in e)) e.preventDefault();
      const ctx = canvasRef.current?.getContext('2d');
      if (!ctx || !lastPos.current) return;
      const coords = getCoordinates(e);
      ctx.beginPath();
      ctx.moveTo(lastPos.current.x, lastPos.current.y);
      ctx.lineTo(coords.x, coords.y);
      ctx.stroke();
      lastPos.current = coords;
      setHasContent(true);
    },
    [isDrawing, disabled, getCoordinates]
  );

  const updatePreview = useCallback((imageData: ProofImageData) => {
    const ctx = previewRef.current?.getContext('2d');
    if (!ctx) return;
    const pixels8x8 = downscaleTo8x8(imageData.pixels);
    const previewImageData = ctx.createImageData(8, 8);
    for (let i = 0; i < 64; i++) {
      const pixel = 255 - pixels8x8[i];
      const idx = i * 4;
      previewImageData.data[idx] = pixel;
      previewImageData.data[idx + 1] = pixel;
      previewImageData.data[idx + 2] = pixel;
      previewImageData.data[idx + 3] = 255;
    }
    ctx.putImageData(previewImageData, 0, 0);
  }, []);

  const stopDrawing = useCallback(() => {
    if (!isDrawing) return;
    setIsDrawing(false);
    lastPos.current = null;
    const canvas = canvasRef.current;
    if (canvas && hasContent && onDraw) {
      try {
        const imageData = processCanvasForProof(canvas);
        onDraw(imageData);
        if (showPreview) updatePreview(imageData);
      } catch (error) {
        console.error('Failed to process canvas:', error);
      }
    }
  }, [isDrawing, hasContent, onDraw, showPreview, updatePreview]);

  const clearCanvas = useCallback(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setHasContent(false);
    const previewCtx = previewRef.current?.getContext('2d');
    if (previewCtx) {
      previewCtx.fillStyle = '#FFFFFF';
      previewCtx.fillRect(0, 0, 8, 8);
    }
    onClear?.();
  }, [onClear]);

  return (
    <div
      className={`flex w-full max-w-full flex-col items-center gap-4 ${className}`}
    >
      <div className="relative w-full max-w-[280px]">
        <motion.div
          className="relative w-full overflow-hidden rounded-2xl border-2 bg-white"
          style={{ aspectRatio: '1', borderColor: '#374151' }}
          whileHover={
            !disabled ? { borderColor: 'rgba(168, 85, 247, 0.5)' } : undefined
          }
        >
          <canvas
            ref={canvasRef}
            className="block w-full"
            style={{
              aspectRatio: '1',
              cursor: disabled ? 'not-allowed' : 'crosshair',
              touchAction: 'none',
            }}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
          />
          <div
            className="absolute inset-0 pointer-events-none opacity-10"
            style={{
              backgroundImage: `linear-gradient(to right, #8b5cf6 1px, transparent 1px), linear-gradient(to bottom, #8b5cf6 1px, transparent 1px)`,
              backgroundSize: '12.5% 12.5%',
            }}
          />
          {disabled && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <span className="text-center text-sm text-gray-400 sm:text-base">
                Connect wallet to draw
              </span>
            </div>
          )}
        </motion.div>
        {hasContent && !disabled && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute -top-2 -right-2 w-8 h-8 bg-red-500 hover:bg-red-400 rounded-full flex items-center justify-center shadow-lg"
            onClick={clearCanvas}
            aria-label="Clear canvas"
            data-testid="canvas-clear-button"
          >
            <span className="text-white text-lg">×</span>
          </motion.button>
        )}
      </div>

      {showPreview && (
        <CanvasPreview
          previewRef={previewRef}
          previewHeatmap={previewHeatmap}
          neuronHeatmap={neuronHeatmap}
          beforeHeatmap={beforeHeatmap}
          afterHeatmap={afterHeatmap}
        />
      )}
    </div>
  );
}

export default DrawingCanvas;
