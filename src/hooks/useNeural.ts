import { useState, useCallback, useEffect } from 'react';
import { AztecAddress } from '@aztec/aztec.js/addresses';
import { Fr } from '@aztec/aztec.js/fields';
import { useAztecWallet } from '../aztec-wallet/hooks/useAztecWallet';
import { SharedPXEService } from '../aztec-wallet/services/aztec/pxe';
import {
  getDeploymentConfig,
  getContractKeyForArchitecture,
  type ArchitectureId,
} from '../config/contracts';
import {
  SCALING_FACTOR,
  fieldToSigned,
  downscaleTo8x8,
  pixelsToScaledBigInts,
} from '../utils/zkml';
import type { ProofImageData } from '../utils/canvas_processor';

export interface PredictionResult {
  digit: number;
  confidence: number;
  probabilities: number[];
  processingTime: number;
}

export interface ExplanationResult {
  classDigit: number;
  inputContribs: number[];
  neuronContribs: number[];
  baseBias: number;
  logit: number;
  normalizedInputs: number[];
  normalizedNeurons: number[];
  processingTime: number;
}

export interface TrainingState {
  isTraining: boolean;
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  currentLoss: number;
  accuracy: number;
}

export interface NeuralState {
  isInitialized: boolean;
  prediction: PredictionResult | null;
  explanation: ExplanationResult | null;
  explanationError: string | null;
  isExplaining: boolean;
  trainingState: TrainingState;
  architecture: ArchitectureId;
}

interface CachedSetup {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  contract: any;
  callerAddress: AztecAddress;
}

// Keyed by `${networkId}:${arch}:${contractAddress}`.
// `contractSetupCache` stores completed setups; `contractSetupInFlight`
// stores in-progress promises so concurrent callers wait on the same work
// instead of racing to write to IndexedDB simultaneously (which causes
// "IDBTransaction not active" errors and silent prediction hangs).
const contractSetupCache = new Map<string, CachedSetup>();
const contractSetupInFlight = new Map<string, Promise<CachedSetup>>();

async function getContractClass(arch: ArchitectureId) {
  switch (arch) {
    case 'singleLayer':
      return (await import('../artifacts/SingleLayer')).SingleLayerContract;
    case 'cnnGap':
      return (await import('../artifacts/CNNGAP')).CNNGAPContract;
    default:
      return (await import('../artifacts/MultiLayerPerceptron'))
        .MultiLayerPerceptronContract;
  }
}

async function getContractArtifact(arch: ArchitectureId) {
  switch (arch) {
    case 'singleLayer':
      return (await import('../artifacts/SingleLayer'))
        .SingleLayerContractArtifact;
    case 'cnnGap':
      return (await import('../artifacts/CNNGAP')).CNNGAPContractArtifact;
    default:
      return (await import('../artifacts/MultiLayerPerceptron'))
        .MultiLayerPerceptronContractArtifact;
  }
}

async function setupContract(
  arch: ArchitectureId,
  contractAddress: string,
  nodeUrl: string,
  networkId: string
) {
  const cacheKey = `${networkId}:${arch}:${contractAddress}`;
  const cached = contractSetupCache.get(cacheKey);
  if (cached) return cached;

  // If another call is already setting up the same contract, wait for it
  // rather than racing to perform concurrent IndexedDB writes.
  const inFlight = contractSetupInFlight.get(cacheKey);
  if (inFlight) return inFlight;

  const setupPromise = (async () => {
    try {
      return await doSetupContract(
        arch,
        contractAddress,
        nodeUrl,
        networkId,
        cacheKey
      );
    } finally {
      contractSetupInFlight.delete(cacheKey);
    }
  })();
  contractSetupInFlight.set(cacheKey, setupPromise);
  return setupPromise;
}

async function doSetupContract(
  arch: ArchitectureId,
  contractAddress: string,
  nodeUrl: string,
  networkId: string,
  cacheKey: string
) {
  const pxeInstance = await SharedPXEService.getCurrentInstance(
    nodeUrl,
    networkId as 'local-network' | 'testnet'
  );
  const { pxe, wallet } = pxeInstance;

  // predict_all and shapley_for_class are unconstrained (public) functions.
  // They do NOT access private state, so the caller identity is irrelevant.
  // Using AztecAddress.ZERO avoids AccountManager.create (which lazily
  // initialises the ECDSA account contract artifact and can hang on a cold
  // browser context, blocking all predictions indefinitely).
  const callerAddress = AztecAddress.ZERO;

  const artifact = await getContractArtifact(arch);
  const address = AztecAddress.fromString(contractAddress);
  let contractInstance = await pxe.getContractInstance(address);
  if (!contractInstance) {
    // Contract not yet registered — fetch from node and register once.
    // Skipping when already registered avoids a redundant IndexedDB write
    // that can race with the block-stream and cause TransactionInactiveError.
    contractInstance = await pxeInstance.aztecNode.getContract(address);
    if (contractInstance && artifact) {
      await wallet.registerContract(contractInstance, artifact);
    }
  }
  const ContractClass = await getContractClass(arch);
  const contract = ContractClass.at(
    address,
    wallet as Parameters<typeof ContractClass.at>[1]
  );

  const result: CachedSetup = { contract, callerAddress };
  contractSetupCache.set(cacheKey, result);
  return result;
}

function pixelsToFields(pixels: number[]): Fr[] {
  return pixelsToScaledBigInts(pixels).map((scaled) => new Fr(scaled));
}

function parseProbabilities(result: unknown): number[] {
  const toBigInt = (r: unknown): bigint =>
    typeof r === 'bigint'
      ? r
      : ((r as { toBigInt?: () => bigint })?.toBigInt?.() ??
        BigInt(r as bigint));
  return (Array.isArray(result) ? result : [result]).map(
    (r) => Number(fieldToSigned(toBigInt(r))) / SCALING_FACTOR
  );
}

function parseExplanationResult(result: unknown): {
  inputContribsRaw: unknown[];
  neuronContribsRaw: unknown[];
  baseBiasRaw: unknown;
  logitRaw: unknown;
} {
  if (Array.isArray(result)) {
    if (result.length === 3) {
      return {
        inputContribsRaw: result[0] as unknown[],
        neuronContribsRaw: [],
        baseBiasRaw: result[1],
        logitRaw: result[2],
      };
    }
    return {
      inputContribsRaw: result[0] as unknown[],
      neuronContribsRaw: result[1] as unknown[],
      baseBiasRaw: result[2],
      logitRaw: result[3],
    };
  }
  const r = result as Record<string, unknown>;
  const idx1 = r[1] ?? r.neuron_contribs ?? r.channel_contribs;
  const isFourTuple = Array.isArray(idx1) && idx1.length > 0;
  if (!isFourTuple) {
    return {
      inputContribsRaw: (r[0] ?? r.input_contribs) as unknown[],
      neuronContribsRaw: [],
      baseBiasRaw: r[1] ?? r.base_bias,
      logitRaw: r[2] ?? r.logit,
    };
  }
  return {
    inputContribsRaw: (r[0] ?? r.input_contribs) as unknown[],
    neuronContribsRaw: idx1 as unknown[],
    baseBiasRaw: r[2] ?? r.base_bias,
    logitRaw: r[3] ?? r.logit,
  };
}

export function useNeural(initialArchitecture: ArchitectureId = 'mlp') {
  const { network } = useAztecWallet();
  const networkId = network?.name ?? 'local-network';

  const [state, setState] = useState<NeuralState>({
    isInitialized: true,
    prediction: null,
    explanation: null,
    explanationError: null,
    isExplaining: false,
    trainingState: {
      isTraining: false,
      progress: 0,
      currentEpoch: 0,
      totalEpochs: 0,
      currentLoss: 0,
      accuracy: 0,
    },
    architecture: initialArchitecture,
  });

  const setArchitecture = useCallback((arch: ArchitectureId) => {
    setState((prev) => ({
      ...prev,
      architecture: arch,
      prediction: null,
      explanation: null,
      explanationError: null,
    }));
  }, []);

  useEffect(() => {
    const config = getDeploymentConfig(networkId).contracts;
    const hasAny =
      config.singleLayer.address ||
      config.multiLayerPerceptron.address ||
      config.cnnGap.address;
    if (!hasAny) {
      console.warn(`No contract addresses configured for ${networkId}`);
    }
  }, [networkId]);

  const predict = useCallback(
    async (imageData: ProofImageData): Promise<PredictionResult | null> => {
      const arch = state.architecture;
      const contractKey = getContractKeyForArchitecture(arch);
      const contractAddress =
        getDeploymentConfig(networkId).contracts[contractKey].address;
      if (!contractAddress) {
        const label =
          arch === 'singleLayer'
            ? 'Single Layer'
            : arch === 'cnnGap'
              ? 'CNN'
              : 'Multi-Layer Network';
        throw new Error(
          `${label} is not deployed for ${networkId}. Deploy with yarn deploy-contracts for the local network.`
        );
      }
      const nodeUrl = network?.nodeUrl ?? 'http://localhost:8080';
      const { contract, callerAddress } = await setupContract(
        arch,
        contractAddress,
        nodeUrl,
        networkId
      );

      const startTime = performance.now();
      const inputPixels = pixelsToFields(downscaleTo8x8(imageData.pixels));
      const { result } = await contract.methods
        .predict_all(inputPixels)
        .simulate({ from: callerAddress });

      const probs = parseProbabilities(result);
      const digit = probs.length > 0 ? probs.indexOf(Math.max(...probs)) : 0;
      const predictionResult: PredictionResult = {
        digit,
        confidence: probs[digit] ?? 0,
        probabilities: probs.length === 10 ? probs : Array(10).fill(0),
        processingTime: performance.now() - startTime,
      };
      setState((prev) => ({
        ...prev,
        prediction: predictionResult,
        explanationError: null,
      }));
      return predictionResult;
    },
    [state.architecture, networkId, network?.nodeUrl]
  );

  const explain = useCallback(
    async (
      imageData: ProofImageData,
      targetClass: number
    ): Promise<ExplanationResult | null> => {
      const arch = state.architecture;
      const contractKey = getContractKeyForArchitecture(arch);
      const contractAddress =
        getDeploymentConfig(networkId).contracts[contractKey].address;
      if (!contractAddress) return null;

      const nodeUrl = network?.nodeUrl ?? 'http://localhost:8080';
      const startTime = performance.now();
      const { contract, callerAddress } = await setupContract(
        arch,
        contractAddress,
        nodeUrl,
        networkId
      );

      const inputPixels = pixelsToFields(downscaleTo8x8(imageData.pixels));

      let rawResult: unknown;
      try {
        ({ result: rawResult } = await contract.methods
          .shapley_for_class(inputPixels, new Fr(BigInt(targetClass)))
          .simulate({ from: callerAddress }));
      } catch (simError) {
        const errMsg =
          simError instanceof Error ? simError.message : String(simError);
        if (
          arch === 'cnnGap' &&
          (errMsg.includes('gas') || errMsg.includes('SimulationError'))
        ) {
          setState((prev) => ({
            ...prev,
            explanationError:
              'CNN Explain exceeds gas limit. Switch to Multi-Layer Network.',
            isExplaining: false,
          }));
          return null;
        }
        throw simError;
      }

      const { inputContribsRaw, neuronContribsRaw, baseBiasRaw, logitRaw } =
        parseExplanationResult(rawResult);

      const toBigInt = (c: unknown): bigint =>
        typeof c === 'bigint'
          ? c
          : ((c as { toBigInt?: () => bigint })?.toBigInt?.() ??
            BigInt(c as bigint));

      const toScaled = (raw: unknown[]) =>
        raw.map((c) => Number(fieldToSigned(toBigInt(c))) / SCALING_FACTOR);

      const inputContribs = toScaled(inputContribsRaw ?? []);
      const neuronContribs = toScaled(neuronContribsRaw);
      const baseBias =
        Number(fieldToSigned(toBigInt(baseBiasRaw))) / SCALING_FACTOR;
      const logit = Number(fieldToSigned(toBigInt(logitRaw))) / SCALING_FACTOR;

      const normalize = (vals: number[]) => {
        const maxAbs = Math.max(...vals.map((c) => Math.abs(c)), 1e-9);
        return vals.map((c) => c / maxAbs);
      };

      const explanationResult: ExplanationResult = {
        classDigit: targetClass,
        inputContribs,
        neuronContribs,
        baseBias,
        logit,
        normalizedInputs: normalize(inputContribs),
        normalizedNeurons: normalize(neuronContribs),
        processingTime: performance.now() - startTime,
      };
      setState((prev) => ({
        ...prev,
        explanation: explanationResult,
        explanationError: null,
        isExplaining: false,
      }));
      return explanationResult;
    },
    [state.architecture, networkId, network?.nodeUrl]
  );

  const prepareTrainingTx = useCallback(
    (imageData: ProofImageData, label: number) => ({
      inputPixels: pixelsToScaledBigInts(downscaleTo8x8(imageData.pixels)),
      label: BigInt(label),
      architecture: state.architecture,
    }),
    [state.architecture]
  );

  const clearPrediction = useCallback(
    () => setState((prev) => ({ ...prev, prediction: null })),
    []
  );

  const clearExplanation = useCallback(
    () =>
      setState((prev) => ({
        ...prev,
        explanation: null,
        explanationError: null,
      })),
    []
  );

  // Predictions are unconstrained simulations — no wallet required.
  const canPredict = true;

  const isArchitectureAvailable =
    !!getDeploymentConfig(networkId).contracts[
      getContractKeyForArchitecture(state.architecture)
    ].address;

  return {
    ...state,
    setArchitecture,
    isArchitectureAvailable,
    predict,
    explain,
    prepareTrainingTx,
    clearPrediction,
    clearExplanation,
    canPredict,
  };
}
