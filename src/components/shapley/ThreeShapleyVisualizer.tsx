import React, { useEffect, useMemo, useRef } from 'react';
import { OrbitControls } from '@react-three/drei';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useIsLightTheme } from '../../store/theme';
import {
  clamp01,
  nodeColor,
  buildHiddenValues,
  buildNodes,
  buildEdges,
  type LayerNode,
  type Edge,
  type NodeMap,
} from './sceneUtils';
import type { ArchitectureId } from '../../config/contracts';

interface ThreeShapleyVisualizerProps {
  architecture: ArchitectureId;
  probabilities?: number[] | null;
  predictedDigit?: number | null;
  focusedDigit?: number | null;
  targetDigit?: number | null;
  inputShapley?: number[] | null;
  neuronShapley?: number[] | null;
  beforeInputShapley?: number[] | null;
  afterInputShapley?: number[] | null;
  mode?: 'predict' | 'train-target' | 'train-delta';
  className?: string;
}

function NodeCloud({
  nodes,
  predictedDigit,
}: {
  nodes: LayerNode[];
  predictedDigit: number | null | undefined;
}) {
  return (
    <>
      {nodes.map((node) => {
        const absValue = clamp01(Math.abs(node.value));
        const isOutputWinner =
          node.kind === 'output' &&
          predictedDigit !== null &&
          predictedDigit === node.index;
        const baseScale =
          node.kind === 'output' ? 0.58 : node.kind === 'gap' ? 0.52 : 0.48;
        const scale = baseScale + absValue * 0.88 + (isOutputWinner ? 0.16 : 0);
        const color = nodeColor(node, predictedDigit);
        return (
          <mesh
            key={node.id}
            position={node.position}
            scale={[scale, scale, scale]}
          >
            <sphereGeometry args={[0.22, 12, 12]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={
                node.kind === 'output'
                  ? 0.14 + absValue * 0.75 + (isOutputWinner ? 0.28 : 0)
                  : 0.1 + absValue * 0.55
              }
              roughness={0.36}
              metalness={0.62}
            />
          </mesh>
        );
      })}
    </>
  );
}

function EdgeCloud({
  edges,
  deltaPalette,
  isLightTheme,
}: {
  edges: Edge[];
  deltaPalette: boolean;
  isLightTheme: boolean;
}) {
  const materialRef = useRef<THREE.LineBasicMaterial>(null);
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const colors: number[] = [];
    for (const edge of edges) {
      const strength = clamp01(Math.abs(edge.signal));
      const tint = deltaPalette
        ? edge.signal >= 0
          ? isLightTheme
            ? [0.08, 0.52, 0.23]
            : [0.13, 0.77, 0.37]
          : isLightTheme
            ? [0.74, 0.18, 0.18]
            : [0.94, 0.27, 0.27]
        : edge.signal >= 0
          ? isLightTheme
            ? [0.29, 0.44, 0.05]
            : [0.83, 1, 0.16]
          : isLightTheme
            ? [0.35, 0.12, 0.35]
            : [0.66, 0.33, 0.55];
      const weighted = tint.map(
        (ch) =>
          ch * (isLightTheme ? 0.38 + strength * 0.62 : 0.16 + strength * 0.84)
      );
      positions.push(
        edge.from[0],
        edge.from[1],
        edge.from[2],
        edge.to[0],
        edge.to[1],
        edge.to[2]
      );
      colors.push(
        weighted[0],
        weighted[1],
        weighted[2],
        weighted[0],
        weighted[1],
        weighted[2]
      );
    }
    const buf = new THREE.BufferGeometry();
    buf.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(positions, 3)
    );
    buf.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return buf;
  }, [edges, deltaPalette, isLightTheme]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  useFrame((state) => {
    if (!materialRef.current) return;
    materialRef.current.opacity =
      (isLightTheme ? 0.56 : 0.34) +
      ((Math.sin(state.clock.elapsedTime * 1.7) + 1) *
        (isLightTheme ? 0.09 : 0.13)) /
        2;
  });

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial
        ref={materialRef}
        transparent
        opacity={isLightTheme ? 0.6 : 0.4}
        vertexColors
        depthWrite={false}
        blending={isLightTheme ? THREE.NormalBlending : THREE.AdditiveBlending}
      />
    </lineSegments>
  );
}

function NeuralScene({
  architecture,
  nodes,
  edges,
  predictedDigit,
  deltaPalette,
  isLightTheme,
}: {
  architecture: ArchitectureId;
  nodes: NodeMap;
  edges: Edge[];
  predictedDigit: number | null | undefined;
  deltaPalette: boolean;
  isLightTheme: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const { invalidate } = useThree();

  useEffect(() => {
    const interval = window.setInterval(() => invalidate(), 56);
    return () => window.clearInterval(interval);
  }, [invalidate]);

  useFrame((state) => {
    if (!groupRef.current) return;
    const baseY = architecture === 'cnnGap' ? 0.26 : 0.35;
    groupRef.current.rotation.y =
      baseY + Math.sin(state.clock.elapsedTime * 0.35) * 0.08;
  });

  return (
    <group
      ref={groupRef}
      rotation={[0.16, architecture === 'cnnGap' ? 0.26 : 0.35, 0]}
    >
      <NodeCloud nodes={nodes.inputNodes} predictedDigit={predictedDigit} />
      <NodeCloud nodes={nodes.hiddenNodes} predictedDigit={predictedDigit} />
      <NodeCloud nodes={nodes.gapNodes} predictedDigit={predictedDigit} />
      <NodeCloud nodes={nodes.outputNodes} predictedDigit={predictedDigit} />
      <EdgeCloud
        edges={edges}
        deltaPalette={deltaPalette}
        isLightTheme={isLightTheme}
      />
    </group>
  );
}

export function ThreeShapleyVisualizer({
  architecture,
  probabilities,
  predictedDigit,
  focusedDigit = null,
  targetDigit = null,
  inputShapley,
  neuronShapley,
  beforeInputShapley,
  afterInputShapley,
  mode = 'predict',
  className = '',
}: ThreeShapleyVisualizerProps) {
  const isLightTheme = useIsLightTheme();
  const hiddenValues = useMemo(
    () => buildHiddenValues(architecture, neuronShapley),
    [architecture, neuronShapley]
  );

  const outputValues = useMemo(
    () =>
      probabilities?.length === 10 ? probabilities : new Array(10).fill(0),
    [probabilities]
  );

  const hasBothDeltas =
    beforeInputShapley?.length === 64 && afterInputShapley?.length === 64;

  const beforeValues = hasBothDeltas
    ? beforeInputShapley
    : inputShapley?.length === 64
      ? new Array(64).fill(0)
      : null;

  const afterValues = hasBothDeltas
    ? afterInputShapley
    : inputShapley?.length === 64
      ? inputShapley
      : null;

  const deltaValues = useMemo(
    () =>
      beforeValues && afterValues
        ? beforeValues.map((b, i) => (afterValues[i] ?? 0) - b)
        : null,
    [beforeValues, afterValues]
  );

  const resolvedMode: 'predict' | 'train-target' | 'train-delta' =
    useMemo(() => {
      if (mode === 'train-delta' && deltaValues) return 'train-delta';
      if (mode === 'train-target') return 'train-target';
      return 'predict';
    }, [mode, deltaValues]);

  const sceneData = useMemo(() => {
    if (!afterValues) return null;
    const nodes = buildNodes(
      architecture,
      afterValues,
      hiddenValues,
      outputValues
    );
    const edgeInputValues =
      resolvedMode === 'train-delta' && deltaValues ? deltaValues : afterValues;
    const edgeNodes = buildNodes(
      architecture,
      edgeInputValues,
      hiddenValues,
      outputValues
    );
    const edges = buildEdges(architecture, edgeNodes, {
      fullPaths: resolvedMode === 'train-delta',
      focusedDigit,
      targetDigit,
      deltaInputValues:
        resolvedMode === 'train-delta' && deltaValues ? deltaValues : null,
    });
    return { nodes, edges };
  }, [
    architecture,
    afterValues,
    hiddenValues,
    outputValues,
    resolvedMode,
    deltaValues,
    focusedDigit,
    targetDigit,
  ]);

  const isDeltaMode = resolvedMode === 'train-delta';
  const isTargetMode = resolvedMode === 'train-target';
  const isSynthesizedBaseline =
    !isDeltaMode && !isTargetMode && inputShapley?.length === 64;

  return (
    <div
      className={`relative min-w-0 overflow-hidden rounded-2xl border border-default bg-surface dark:border-signal-soft dark:bg-[#070a12] ${className}`}
      data-testid="three-shapley-visualizer"
    >
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-default px-3 py-3 sm:px-4 dark:border-signal-soft">
        <div>
          <p className="mono-label text-muted">Neural Attribution Space</p>
          <p className="text-base text-default sm:text-sm">
            {isDeltaMode
              ? 'Training Delta View'
              : isTargetMode
                ? 'Training Target View'
                : 'Inference Contribution View'}
          </p>
        </div>
        <div className="shrink-0 text-right text-xs text-muted sm:text-[11px]">
          <div>
            Input: <span className="text-[#d4ff28]">64</span>
          </div>
          <div>
            Hidden:{' '}
            <span className="text-[#d4ff28]">{hiddenValues.length}</span>
          </div>
          {architecture === 'cnnGap' && (
            <div>
              GAP: <span className="text-[#80e3ff]">auto</span>
            </div>
          )}
          <div>
            Output: <span className="text-[#d4ff28]">10</span>
          </div>
          {isDeltaMode && deltaValues && (
            <div>
              Delta:{' '}
              <span className="text-[#d4ff28]">
                {deltaValues
                  .reduce((acc, v) => acc + Math.abs(v), 0)
                  .toFixed(2)}
              </span>
            </div>
          )}
        </div>
      </div>

      <div
        className={`relative h-[280px] sm:h-[340px] ${
          isLightTheme
            ? 'bg-gradient-to-b from-[var(--bg-surface)] to-[var(--bg-surface-secondary)]'
            : 'bg-gradient-to-b from-[#070c17] to-[#06070b]'
        }`}
      >
        {sceneData ? (
          <Canvas
            frameloop="demand"
            dpr={[1, 1.25]}
            performance={{ min: 0.62 }}
            camera={{ position: [0, 4, 19], fov: 42 }}
            gl={{
              antialias: false,
              alpha: true,
              powerPreference: 'low-power',
              preserveDrawingBuffer: false,
            }}
          >
            <color
              attach="background"
              args={[isLightTheme ? '#f2f5fa' : '#06080c']}
            />
            <ambientLight intensity={isLightTheme ? 0.62 : 0.44} />
            <pointLight
              position={[16, 14, 10]}
              intensity={isLightTheme ? 1.3 : 2.0}
              color={isLightTheme ? '#7f9f12' : '#d4ff28'}
            />
            <pointLight
              position={[-14, -10, -8]}
              intensity={isLightTheme ? 0.9 : 1.45}
              color={isLightTheme ? '#475569' : '#80336a'}
            />
            <NeuralScene
              architecture={architecture}
              nodes={sceneData.nodes}
              edges={sceneData.edges}
              predictedDigit={predictedDigit}
              deltaPalette={isDeltaMode}
              isLightTheme={isLightTheme}
            />
            <OrbitControls
              makeDefault
              enablePan={false}
              enableDamping={false}
              minDistance={12}
              maxDistance={36}
              rotateSpeed={0.48}
              zoomSpeed={0.6}
              minPolarAngle={Math.PI * 0.25}
              maxPolarAngle={Math.PI * 0.75}
            />
          </Canvas>
        ) : (
          <div className="flex h-full items-center justify-center px-4 text-center text-base text-muted sm:px-6 sm:text-sm">
            Draw and predict to populate live contribution and network output
            data.
          </div>
        )}

        <div className="pointer-events-none absolute left-2 top-2 rounded-md border border-default bg-white/75 px-2 py-1 text-xs text-default dark:border-signal-soft dark:bg-black/45 dark:text-gray-300 sm:left-3 sm:top-3 sm:text-[10px]">
          {isDeltaMode ? (
            <>
              <span className="mr-2 text-[#22c55e]">reinforced paths</span>
              <span className="text-[#ef4444]">weakened paths</span>
            </>
          ) : (
            <>
              <span className="mr-2 text-[#d4ff28]">+ positive</span>
              <span className="text-[#b8679f]">- negative</span>
            </>
          )}
        </div>

        <div className="pointer-events-none absolute bottom-2 right-2 hidden rounded-md border border-default bg-white/75 px-2 py-1 text-xs text-default dark:border-signal-soft dark:bg-black/45 dark:text-gray-300 sm:bottom-3 sm:right-3 sm:block sm:text-[10px]">
          Input nodes {'->'} feature nodes {'->'} summary nodes {'->'} output
          digits
        </div>

        {isSynthesizedBaseline && (
          <div className="pointer-events-none absolute bottom-2 left-2 rounded-md border border-default bg-white/75 px-2 py-1 text-xs text-default dark:border-signal-soft dark:bg-black/45 dark:text-gray-300 sm:bottom-3 sm:left-3 sm:text-[10px]">
            Delta baseline synthesized from current snapshot.
          </div>
        )}
      </div>
    </div>
  );
}

export default ThreeShapleyVisualizer;
