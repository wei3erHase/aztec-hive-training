import type { ArchitectureId } from '../../config/contracts';

export type NodeKind = 'input' | 'hidden' | 'gap' | 'output';

export interface LayerNode {
  id: string;
  index: number;
  kind: NodeKind;
  value: number;
  position: [number, number, number];
}

export interface Edge {
  from: [number, number, number];
  to: [number, number, number];
  signal: number;
}

export interface NodeMap {
  inputNodes: LayerNode[];
  hiddenNodes: LayerNode[];
  gapNodes: LayerNode[];
  outputNodes: LayerNode[];
}

export function clamp01(value: number): number {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

export function nodeColor(
  node: LayerNode,
  predictedDigit: number | null | undefined
): string {
  if (node.kind === 'output') {
    return predictedDigit === node.index ? '#d4ff28' : '#7a8ca8';
  }
  if (node.kind === 'gap') {
    return node.value >= 0 ? '#80e3ff' : '#6f69f9';
  }
  return node.value >= 0 ? '#d4ff28' : '#80336a';
}

function topIndices(values: number[], count: number): number[] {
  return values
    .map((value, index) => ({ index, score: Math.abs(value) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, count)
    .map((entry) => entry.index);
}

function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function expandValues(values: number[], targetLength: number): number[] {
  if (targetLength <= 0) return [];
  if (values.length === targetLength) return values;
  if (values.length === 0) return new Array(targetLength).fill(0);
  const expanded: number[] = [];
  for (let i = 0; i < targetLength; i++) {
    const sourceIndex = Math.floor((i / targetLength) * values.length);
    expanded.push(values[sourceIndex] ?? 0);
  }
  return expanded;
}

export function buildHiddenValues(
  architecture: ArchitectureId,
  neuronShapley: number[] | null | undefined
): number[] {
  if (architecture === 'singleLayer') return [];
  const rawValues =
    neuronShapley && neuronShapley.length > 0 ? neuronShapley : [];
  return expandValues(rawValues, architecture === 'cnnGap' ? 48 : 16);
}

export function buildNodes(
  architecture: ArchitectureId,
  inputValues: number[],
  hiddenValues: number[],
  outputValues: number[]
): NodeMap {
  const inputNodes: LayerNode[] = [];
  const hiddenNodes: LayerNode[] = [];
  const gapNodes: LayerNode[] = [];
  const outputNodes: LayerNode[] = [];

  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const idx = y * 8 + x;
      inputNodes.push({
        id: `in-${idx}`,
        index: idx,
        kind: 'input',
        value: inputValues[idx] ?? 0,
        position: [-8.6, (3.5 - y) * 0.95, (x - 3.5) * 0.95],
      });
    }
  }

  if (architecture === 'cnnGap') {
    const convCount = hiddenValues.length;
    const groups = 8;
    const perGroup = Math.max(1, Math.ceil(convCount / groups));
    for (let i = 0; i < convCount; i++) {
      const group = Math.floor(i / perGroup);
      const local = i % perGroup;
      const yCenter = (perGroup - 1) / 2;
      hiddenNodes.push({
        id: `conv-${i}`,
        index: i,
        kind: 'hidden',
        value: hiddenValues[i] ?? 0,
        position: [
          -1.6,
          (yCenter - local) * 0.88,
          (group - (groups - 1) / 2) * 1.18,
        ],
      });
    }
    const gapCount = groups;
    const bucketSize = Math.ceil(convCount / gapCount);
    for (let i = 0; i < gapCount; i++) {
      const start = i * bucketSize;
      const bucket = hiddenValues.slice(
        start,
        Math.min(convCount, start + bucketSize)
      );
      gapNodes.push({
        id: `gap-${i}`,
        index: i,
        kind: 'gap',
        value: average(bucket),
        position: [3.1, 0, (i - (gapCount - 1) / 2) * 1.18],
      });
    }
  } else {
    const hiddenCount = hiddenValues.length;
    for (let i = 0; i < hiddenCount; i++) {
      const angle = (i / Math.max(hiddenCount, 1)) * Math.PI * 2;
      const radius = hiddenCount > 20 ? 5.1 : 4.1;
      hiddenNodes.push({
        id: `hidden-${i}`,
        index: i,
        kind: 'hidden',
        value: hiddenValues[i] ?? 0,
        position: [0, Math.cos(angle) * radius, Math.sin(angle) * radius],
      });
    }
  }

  for (let i = 0; i < 10; i++) {
    outputNodes.push({
      id: `out-${i}`,
      index: i,
      kind: 'output',
      value: outputValues[i] ?? 0,
      position: [8.6, (4.5 - i) * 0.95, 0],
    });
  }

  return { inputNodes, hiddenNodes, gapNodes, outputNodes };
}

export function buildEdges(
  architecture: ArchitectureId,
  nodes: NodeMap,
  options: {
    fullPaths: boolean;
    focusedDigit: number | null;
    targetDigit: number | null;
    deltaInputValues: number[] | null;
  }
): Edge[] {
  const { fullPaths, focusedDigit, targetDigit, deltaInputValues } = options;
  const threshold = fullPaths
    ? architecture === 'cnnGap'
      ? 0.000001
      : 0.0000025
    : architecture === 'cnnGap'
      ? 0.006
      : 0.018;

  const edgeMap = new Map<string, Edge>();
  const pathCandidates: Array<{
    chain: LayerNode[];
    signal: number;
    score: number;
  }> = [];
  const outputMean = average(nodes.outputNodes.map((n) => n.value));
  const hasTargetDirection =
    fullPaths &&
    targetDigit !== null &&
    targetDigit >= 0 &&
    targetDigit < nodes.outputNodes.length;

  const outputTargets =
    !fullPaths &&
    focusedDigit !== null &&
    focusedDigit >= 0 &&
    focusedDigit < nodes.outputNodes.length
      ? [focusedDigit]
      : fullPaths
        ? nodes.outputNodes.map((_, i) => i)
        : topIndices(
            nodes.outputNodes.map((n) => n.value),
            3
          );

  const applyOutputDirection = (
    rawSignal: number,
    outputNode: LayerNode,
    outputFactor: number
  ): number => {
    if (!hasTargetDirection || targetDigit === null)
      return rawSignal * outputFactor;
    const direction = outputNode.index === targetDigit ? 1 : -1;
    const outputWeight =
      Math.abs(outputNode.value - outputMean) +
      (outputNode.index === targetDigit ? 0.45 : 0.15);
    return direction * Math.abs(rawSignal) * outputWeight;
  };

  const topInput = topIndices(
    nodes.inputNodes.map((n) => n.value),
    fullPaths ? 18 : 12
  );

  const addEdge = (
    from: LayerNode | undefined,
    to: LayerNode | undefined,
    signal: number
  ) => {
    if (!from || !to) return;
    const key = `${from.id}->${to.id}`;
    const existing = edgeMap.get(key);
    if (existing) {
      existing.signal += signal;
      return;
    }
    edgeMap.set(key, { from: from.position, to: to.position, signal });
  };

  const addPathCandidate = (chain: LayerNode[], signal: number) => {
    if (chain.length < 2) return;
    const score = Math.abs(signal);
    if (score < threshold) return;
    pathCandidates.push({ chain, signal, score });
  };

  const flushPaths = (maxPaths: number) => {
    pathCandidates
      .sort((a, b) => b.score - a.score)
      .slice(0, maxPaths)
      .forEach((candidate) => {
        for (let i = 0; i < candidate.chain.length - 1; i++) {
          addEdge(candidate.chain[i], candidate.chain[i + 1], candidate.signal);
        }
      });
  };

  if (architecture === 'cnnGap' && nodes.hiddenNodes.length > 0) {
    const gapCount = Math.max(nodes.gapNodes.length, 1);
    const convPerGap = Math.max(
      1,
      Math.floor(nodes.hiddenNodes.length / gapCount)
    );
    for (const outIdx of outputTargets) {
      const outputNode = nodes.outputNodes[outIdx];
      if (!outputNode) continue;
      const outputFactor = fullPaths
        ? outputNode.value - outputMean
        : outputNode.value;
      const topGap = topIndices(
        nodes.gapNodes.map((g) => Math.abs(g.value * outputFactor)),
        Math.min(fullPaths ? 6 : 3, nodes.gapNodes.length)
      );
      for (const gapIdx of topGap) {
        const gapNode = nodes.gapNodes[gapIdx];
        if (!gapNode) continue;
        const convStart = gapIdx * convPerGap;
        const localConv = nodes.hiddenNodes.slice(
          convStart,
          Math.min(nodes.hiddenNodes.length, convStart + convPerGap)
        );
        const topConvLocal = topIndices(
          localConv.map((c) => Math.abs(c.value * gapNode.value)),
          Math.min(fullPaths ? 6 : 3, localConv.length)
        );
        for (const localIdx of topConvLocal) {
          const convNode = localConv[localIdx];
          if (!convNode) continue;
          const inputCandidates = topIndices(
            nodes.inputNodes.map((inp) => Math.abs(inp.value * convNode.value)),
            fullPaths ? 16 : 8
          );
          for (const inputIdx of inputCandidates) {
            if (!fullPaths && !topInput.includes(inputIdx)) continue;
            const inputNode = nodes.inputNodes[inputIdx];
            const baseSignal = deltaInputValues
              ? (deltaInputValues[inputIdx] ?? 0)
              : inputNode.value;
            const rawSignal = baseSignal * convNode.value * gapNode.value;
            addPathCandidate(
              [inputNode, convNode, gapNode, outputNode],
              applyOutputDirection(rawSignal, outputNode, outputFactor)
            );
          }
        }
      }
    }
    flushPaths(
      fullPaths
        ? focusedDigit !== null
          ? 260
          : 320
        : focusedDigit !== null
          ? 90
          : 130
    );
    return Array.from(edgeMap.values());
  }

  if (nodes.hiddenNodes.length > 0) {
    for (const outIdx of outputTargets) {
      const outputNode = nodes.outputNodes[outIdx];
      if (!outputNode) continue;
      const outputFactor = fullPaths
        ? outputNode.value - outputMean
        : outputNode.value;
      const topHidden = topIndices(
        nodes.hiddenNodes.map((h) => Math.abs(h.value * outputFactor)),
        Math.min(fullPaths ? 10 : 6, nodes.hiddenNodes.length)
      );
      for (const hiddenIdx of topHidden) {
        const hiddenNode = nodes.hiddenNodes[hiddenIdx];
        if (!hiddenNode) continue;
        const inputCandidates = topIndices(
          nodes.inputNodes.map((inp) => Math.abs(inp.value * hiddenNode.value)),
          fullPaths ? 16 : 8
        );
        for (const inputIdx of inputCandidates) {
          if (!fullPaths && !topInput.includes(inputIdx)) continue;
          const inputNode = nodes.inputNodes[inputIdx];
          const baseSignal = deltaInputValues
            ? (deltaInputValues[inputIdx] ?? 0)
            : inputNode.value;
          addPathCandidate(
            [inputNode, hiddenNode, outputNode],
            applyOutputDirection(
              baseSignal * hiddenNode.value,
              outputNode,
              outputFactor
            )
          );
        }
      }
    }
    flushPaths(
      fullPaths
        ? focusedDigit !== null
          ? 190
          : 260
        : focusedDigit !== null
          ? 80
          : 120
    );
    return Array.from(edgeMap.values());
  }

  for (const outIdx of outputTargets) {
    const outputNode = nodes.outputNodes[outIdx];
    if (!outputNode) continue;
    const outputFactor = fullPaths
      ? outputNode.value - outputMean
      : outputNode.value;
    for (const inputIdx of topInput) {
      const inputNode = nodes.inputNodes[inputIdx];
      const baseSignal = deltaInputValues
        ? (deltaInputValues[inputIdx] ?? 0)
        : inputNode.value;
      addPathCandidate(
        [inputNode, outputNode],
        applyOutputDirection(baseSignal, outputNode, outputFactor)
      );
    }
  }
  flushPaths(fullPaths ? 120 : 64);
  return Array.from(edgeMap.values());
}
