import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { useAztecWallet } from '../aztec-wallet/hooks/useAztecWallet';
import { useConnectModal } from '../aztec-wallet/hooks/useConnectModal';
import { DrawingCanvas } from '../components/canvas';
import { ArchitectureSelector } from '../components/home/ArchitectureSelector';
import { HeroSection } from '../components/home/HeroSection';
import { LocalNetworkWarning } from '../components/home/LocalNetworkWarning';
import { PredictionMatrix } from '../components/home/PredictionMatrix';
import { PredictionStatus } from '../components/home/PredictionStatus';
import { TrainingControls } from '../components/home/TrainingControls';
import ThreeShapleyVisualizer from '../components/shapley/ThreeShapleyVisualizer';
import { getDeploymentConfig } from '../config/contracts';
import {
  useAppNavigation,
  useNeural,
  useNetworkStatus,
  useToast,
  useTrainOnChain,
} from '../hooks';
import { getTrainingService } from '../services';
import type { ProofImageData } from '../utils/canvas_processor';

type TrainStatus =
  | 'idle'
  | 'labeling'
  | 'training'
  | 'submitting'
  | 'success'
  | 'error';

export const Home: React.FC = () => {
  const { isConnected, network } = useAztecWallet();
  const { open: openConnectModal } = useConnectModal();
  const { navigate } = useAppNavigation();
  const {
    isInitialized,
    architecture,
    setArchitecture,
    isArchitectureAvailable,
    prediction,
    explanation,
    explanationError,
    isExplaining,
    trainingState,
    predict,
    explain,
    prepareTrainingTx,
    clearPrediction,
    clearExplanation,
    canPredict,
  } = useNeural();
  const { submitTraining, isPending: isSubmittingOnChain } = useTrainOnChain();
  const toast = useToast();

  const networkId = network?.name ?? 'local-network';
  const nodeUrl = network?.nodeUrl ?? 'http://localhost:8080';
  const networkStatus = useNetworkStatus(networkId, nodeUrl);

  const localNetworkDeployed = (() => {
    const c = getDeploymentConfig('local-network').contracts;
    return !!(
      c.singleLayer.address ||
      c.multiLayerPerceptron.address ||
      c.cnnGap.address
    );
  })();
  const isLocalMissingContracts =
    networkId === 'local-network' && !localNetworkDeployed;

  const [currentImage, setCurrentImage] = useState<ProofImageData | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<number | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus>('idle');
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [predictError, setPredictError] = useState<string | null>(null);

  const [beforeExplanation, setBeforeExplanation] = useState<Awaited<
    ReturnType<typeof explain>
  > | null>(null);
  const [afterExplanation, setAfterExplanation] = useState<Awaited<
    ReturnType<typeof explain>
  > | null>(null);
  const [predictBeforeExplanation, setPredictBeforeExplanation] =
    useState<Awaited<ReturnType<typeof explain>> | null>(null);
  const lastPredictExplanationRef = useRef<Awaited<
    ReturnType<typeof explain>
  > | null>(null);

  useEffect(() => {
    if (selectedLabel !== null || !explanation) return;
    setPredictBeforeExplanation(lastPredictExplanationRef.current);
    lastPredictExplanationRef.current = explanation;
  }, [explanation, selectedLabel]);

  useEffect(() => {
    setPredictBeforeExplanation(null);
    lastPredictExplanationRef.current = null;
  }, [architecture]);

  useEffect(() => {
    if (!currentImage || !isArchitectureAvailable || !canPredict) return;
    setPredictError(null);
    predict(currentImage)
      .then((result) => {
        if (result) {
          setSelectedLabel(null);
          setBeforeExplanation(null);
          setAfterExplanation(null);
          explain(currentImage!, result.digit).catch((err) =>
            console.error('[Home] Explanation failed:', err)
          );
        }
      })
      .catch((err) => {
        setPredictError(err instanceof Error ? err.message : String(err));
      });
  }, [
    architecture,
    currentImage,
    isArchitectureAvailable,
    canPredict,
    predict,
    explain,
  ]);

  const handleDraw = useCallback(
    async (imageData: ProofImageData) => {
      setCurrentImage(imageData);
      setPredictError(null);
      setSelectedLabel(null);
      setBeforeExplanation(null);
      setAfterExplanation(null);
      try {
        const result = await predict(imageData);
        if (result) {
          await explain(imageData, result.digit).catch((err) =>
            console.error('[Home] Explanation failed:', err)
          );
        }
      } catch (err) {
        setPredictError(err instanceof Error ? err.message : String(err));
      }
    },
    [predict, explain]
  );

  const handleClear = useCallback(() => {
    setCurrentImage(null);
    setPredictError(null);
    clearPrediction();
    clearExplanation();
    setSelectedLabel(null);
    setTrainStatus('idle');
    setStatusMessage('');
    setBeforeExplanation(null);
    setAfterExplanation(null);
    setPredictBeforeExplanation(null);
    lastPredictExplanationRef.current = null;
  }, [clearPrediction, clearExplanation]);

  const handleLabelSelect = useCallback(
    async (label: number) => {
      if (selectedLabel === label) {
        setSelectedLabel(null);
        setBeforeExplanation(null);
        setAfterExplanation(null);
        if (currentImage && prediction)
          await explain(currentImage, prediction.digit);
        return;
      }
      setSelectedLabel(label);
      setAfterExplanation(null);
      if (currentImage)
        setBeforeExplanation(await explain(currentImage, label));
    },
    [selectedLabel, currentImage, prediction, explain]
  );

  const handleSubmitTraining = useCallback(async () => {
    if (!currentImage || selectedLabel === null) {
      toast.warning('Draw a digit and select a label first');
      return;
    }
    setTrainStatus('submitting');
    setStatusMessage('Preparing training data...');

    const loadingId = toast.loading('Submitting training transaction…');

    try {
      if (!beforeExplanation) {
        setStatusMessage('Capturing baseline contribution snapshot...');
        setBeforeExplanation(await explain(currentImage, selectedLabel));
      }

      const txData = prepareTrainingTx(currentImage, selectedLabel);
      if (!txData) throw new Error('Failed to prepare training data');

      const service = getTrainingService(networkId, nodeUrl);
      const connected = await service.checkConnection();
      if (!connected)
        throw new Error(
          service.getConnectionError() ?? 'Aztec network not available.'
        );

      const blockNumber = await service.getBlockNumber();
      setStatusMessage(
        `Connected to ${networkStatus.networkName} (Block #${blockNumber}). Submitting...`
      );

      setStatusMessage('Submitting training to contract...');
      const result = await submitTraining(txData);
      if (!result.success)
        throw new Error(result.error || 'Failed to submit training');

      const shortHash =
        result.txHash && result.txHash.length > 16
          ? `${result.txHash.slice(0, 10)}...${result.txHash.slice(-6)}`
          : result.txHash || 'confirmed';
      toast.dismiss(loadingId);
      setStatusMessage(`Training submitted! TX: ${shortHash}`);
      setTrainStatus('success');
      toast.success(
        'Training submitted!',
        `TX: ${shortHash} (Block #${result.blockNumber})`
      );

      if (currentImage && selectedLabel !== null) {
        setStatusMessage(
          `TX: ${shortHash} — Re-predicting with updated weights...`
        );
        await predict(currentImage);
        setStatusMessage(
          `TX: ${shortHash} — Fetching contribution comparison...`
        );
        setAfterExplanation(await explain(currentImage, selectedLabel));
        setStatusMessage(`Training complete! TX: ${shortHash}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      toast.dismiss(loadingId);
      toast.error('Training failed', message);
      setStatusMessage(`Error: ${message}`);
      setTrainStatus('error');
    }
  }, [
    currentImage,
    selectedLabel,
    beforeExplanation,
    prepareTrainingTx,
    networkStatus.networkName,
    submitTraining,
    explain,
    predict,
    networkId,
    nodeUrl,
    toast,
  ]);

  const showTrainDelta =
    selectedLabel !== null && !!beforeExplanation && !!afterExplanation;
  const showTrainTarget = selectedLabel !== null;

  const visualInput =
    showTrainTarget || showTrainDelta
      ? (beforeExplanation?.normalizedInputs ??
        explanation?.normalizedInputs ??
        null)
      : (explanation?.normalizedInputs ?? null);
  const visualNeurons =
    showTrainTarget || showTrainDelta
      ? (beforeExplanation?.normalizedNeurons ??
        explanation?.normalizedNeurons ??
        null)
      : (explanation?.normalizedNeurons ?? null);
  const visualBefore = showTrainDelta
    ? (beforeExplanation?.normalizedInputs ?? null)
    : (predictBeforeExplanation?.normalizedInputs ?? null);
  const visualAfter = showTrainDelta
    ? (afterExplanation?.normalizedInputs ?? null)
    : (explanation?.normalizedInputs ?? null);
  const visualMode: 'predict' | 'train-target' | 'train-delta' = showTrainDelta
    ? 'train-delta'
    : showTrainTarget
      ? 'train-target'
      : 'predict';

  const displayedExplanation = showTrainDelta
    ? (afterExplanation ?? beforeExplanation)
    : selectedLabel !== null
      ? (beforeExplanation ?? explanation)
      : explanation;

  return (
    <>
      <div
        className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 sm:py-12"
        data-testid="home-page"
      >
        <HeroSection
          isConnected={isConnected}
          networkStatus={networkStatus}
          onConnect={openConnectModal}
          onNavigateDocs={() => navigate('docs')}
        />

        {isLocalMissingContracts && <LocalNetworkWarning />}

        <motion.section
          id="demo"
          className="w-full rounded-2xl sm:rounded-3xl border border-signal-soft bg-surface/95 p-4 sm:p-6 md:p-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.15 }}
        >
          <div className="grid min-w-0 items-start gap-4 sm:gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="min-w-0 space-y-4 sm:space-y-6">
              <ArchitectureSelector
                architecture={architecture}
                isArchitectureAvailable={isArchitectureAvailable}
                onSelect={setArchitecture}
              />
              <div
                className="panel min-w-0 rounded-2xl p-4 sm:p-5"
                data-testid="canvas-panel"
              >
                <DrawingCanvas
                  onDraw={handleDraw}
                  onClear={handleClear}
                  disabled={
                    !isInitialized || !canPredict || !isArchitectureAvailable
                  }
                  showPreview={true}
                  brushSize={18}
                  previewHeatmap={
                    selectedLabel === null && prediction && explanation
                      ? explanation.normalizedInputs
                      : undefined
                  }
                  neuronHeatmap={
                    selectedLabel === null && prediction && explanation
                      ? explanation.normalizedNeurons
                      : undefined
                  }
                  beforeHeatmap={
                    selectedLabel !== null && beforeExplanation
                      ? beforeExplanation.normalizedInputs
                      : undefined
                  }
                  afterHeatmap={
                    selectedLabel !== null && afterExplanation
                      ? afterExplanation.normalizedInputs
                      : undefined
                  }
                />
              </div>
            </div>

            <div className="min-w-0 space-y-4 sm:space-y-6">
              <div className="relative min-w-0">
                <ThreeShapleyVisualizer
                  architecture={architecture}
                  probabilities={prediction?.probabilities}
                  predictedDigit={selectedLabel ?? prediction?.digit ?? null}
                  focusedDigit={
                    visualMode === 'train-delta' ? null : selectedLabel
                  }
                  targetDigit={selectedLabel}
                  inputShapley={visualInput}
                  neuronShapley={visualNeurons}
                  beforeInputShapley={visualBefore}
                  afterInputShapley={visualAfter}
                  mode={visualMode}
                />
              </div>
              <PredictionMatrix
                prediction={prediction}
                selectedLabel={selectedLabel}
                isExplaining={isExplaining}
                currentImage={currentImage}
                onLabelSelect={handleLabelSelect}
              />
              <PredictionStatus
                prediction={prediction}
                selectedLabel={selectedLabel}
                explanationProcessingTime={
                  displayedExplanation?.processingTime ?? null
                }
                predictError={predictError}
                explanationError={explanationError}
              />
              <TrainingControls
                currentImage={currentImage}
                selectedLabel={selectedLabel}
                trainStatus={trainStatus}
                statusMessage={statusMessage}
                isSubmittingOnChain={isSubmittingOnChain}
                trainingState={trainingState}
                isConnected={isConnected}
                onSubmit={handleSubmitTraining}
                onConnect={openConnectModal}
              />
            </div>
          </div>
        </motion.section>
      </div>
    </>
  );
};
