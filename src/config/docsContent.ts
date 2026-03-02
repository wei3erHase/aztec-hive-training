export const features = [
  {
    icon: '🔒',
    title: 'Inputs Stay Private',
    description:
      'Your drawings and labels never leave your device. Only a ZK proof of correct training is sent on-chain.',
  },
  {
    icon: '🛡️',
    title: 'Zero-Knowledge Proofs',
    description:
      'Proofs are generated on your machine. Aztec verifies without seeing your data—client-side proofs keep sensitive data where it belongs.',
  },
  {
    icon: '⛓️',
    title: 'Public Weights',
    description:
      'Model weights are public on-chain. Train together, verify together. Anyone can audit and use the collectively-trained model.',
  },
  {
    icon: '✏️',
    title: 'Draw, Predict, Train',
    description:
      'Draw digits on a canvas, get predictions, and contribute training updates. See Shapley explanations of which pixels mattered most.',
  },
] as const;

export const architectureLayers = [
  {
    label: 'Blockchain (Aztec-first)',
    items: ['Noir Circuits', 'Aztec Network', 'ZK Proofs'],
    color: 'from-aztec-jade to-aztec-turquoise',
  },
  {
    label: 'Core Logic',
    items: ['Neural Trainer', 'Image Processor', 'Proof Generator'],
    color: 'from-neural-500 to-neural-600',
  },
  {
    label: 'Application',
    items: ['Drawing Canvas', 'Wallet Integration'],
    color: 'from-hive-500 to-hive-600',
  },
] as const;

export const workflow = [
  {
    step: 1,
    title: 'Draw',
    description: 'Sketch a digit on the canvas',
    icon: '✏️',
  },
  {
    step: 2,
    title: 'Process',
    description: 'Image is normalized to 8×8 pixels',
    icon: '🖼️',
  },
  {
    step: 3,
    title: 'Predict',
    description: 'Neural network classifies the digit',
    icon: '🐝',
  },
  {
    step: 4,
    title: 'Verify',
    description: 'ZK proof validates the computation',
    icon: '🛡️',
  },
] as const;

export const eli5Architectures = [
  {
    name: 'Simple Linear (64→10)',
    eli5: 'Each pixel votes for each digit. Add up the votes, pick the winner. Simple and fast, but basic.',
  },
  {
    name: 'Multi-Layer Network (64→16→10)',
    eli5: '16 helper neurons learn shapes like curves and strokes, then decide the digit. Good balance of speed and accuracy.',
  },
  {
    name: 'CNN + GAP (Conv 4×4×3 → GAP → 3→10)',
    eli5: 'Slides small windows over your drawing to spot patterns (loops, lines, corners). Best at handwriting; can show which parts mattered most.',
  },
] as const;

export const techSpecs = [
  {
    icon: '🐝',
    iconBg: 'bg-neural-600/20',
    title: 'Neural Network',
    dotColor: 'bg-neural-500',
    items: [
      '64 input pixels (8×8 grayscale)',
      '10 output classes (digits 0-9)',
      'SingleLayer, MLP, CNN+GAP',
      'Softmax activation',
    ],
  },
  {
    icon: '⚡',
    iconBg: 'bg-hive-600/20',
    title: 'Performance',
    dotColor: 'bg-hive-500',
    items: [
      'Training proof: ~10-15 s per contribution',
      'Inference: <500 ms',
      'SingleLayer ~12 s, MLP ~14.5 s, CNNGAP ~10 s (M1)',
    ],
  },
  {
    icon: '🛠️',
    iconBg: 'bg-aztec-jade/20',
    title: 'Tech Stack',
    dotColor: 'bg-aztec-jade',
    items: ['Noir (ZK circuits)', 'Aztec Network', 'Three.js', 'TypeScript'],
  },
] as const;
