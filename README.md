# HIVE Neural Network

A privacy-preserving on-chain neural network training application built on Aztec Network. Draw digits, run inference privately, and submit training updates that improve the shared model — all with zero-knowledge proofs.

## Features

- 🧠 **Three neural network architectures** — Single Layer (64→10), MLP (64→16→10), CNN+GAP
- 🔐 **Private training** — gradient updates computed client-side, committed on-chain via hash
- ✏️ **Draw-to-predict** — canvas input processed through MNIST pipeline, classified by the contract
- ⛽ **Sponsored fee payments** — gasless transactions through SponsoredFPC
- 🔄 **Network switching** between Devnet and local network

---

## Quick Start

### Prerequisites

- Node.js >= 22.0.0
- Yarn package manager
- [Foundry](https://book.getfoundry.sh/getting-started/installation) (for Anvil, the local L1 chain)

### Installation

```bash
git clone <repo-url>
cd aztec-hyve-training
yarn install
```

### Start Local Network & Deploy

```bash
# Confirm pinned Aztec version for this repo
cat .aztecrc

# Terminal 1: Start Anvil (local L1 chain)
anvil --host 0.0.0.0 -p 8545 --block-time 12

# Terminal 2: Start the Aztec local network
gaztec start --local-network --l1-rpc-urls http://localhost:8545

# Terminal 3: Build, deploy, and run
yarn build && yarn deploy-contracts:local-network && yarn dev
```

The application will be available at **http://localhost:3000**

---

## Available Commands

### Contracts

```bash
yarn build-contracts          # Compile all Noir contracts + generate TS bindings
yarn test:nr                  # Run Noir contract tests (gaztec test)
```

### Deployment

```bash
yarn deploy-contracts                    # Deploy to local network (default)
yarn deploy-contracts:local-network      # Deploy to local network
yarn deploy-contracts:devnet             # Deploy to devnet
yarn deploy-contracts:all                # Deploy to both
```

Deployed addresses are written to:
- `config/deployed.local.json` — local network (gitignored)
- `config/deployed.json` — devnet (committed)

### App

```bash
yarn dev                      # Start Vite dev server
yarn build                    # Build contracts + app
yarn build-app                # Build app only
yarn build-app:ci             # CI build (reduced memory)
yarn serve                    # Preview production build
```

### Testing & Quality

```bash
yarn test:js                  # Run JS unit tests (vitest)
yarn test:nr                  # Run Noir contract tests
yarn test:e2e                 # Run Playwright E2E tests (deploys + builds first)
yarn test:e2e:local-network   # Run E2E tests against a running local network
yarn lint                     # Check formatting (ESLint + Prettier)
yarn lint:fix                 # Auto-fix formatting
```

### Environment Variables

Copy `.env.example` to `.env`:

```bash
VITE_AZTEC_NODE_URL=http://localhost:8080   # Aztec node URL
VITE_PROVER_ENABLED=false                   # Disable prover for faster development
```

---

## Project Structure

```
aztec-hyve-training/
├── contracts/                     # Noir smart contracts
│   ├── zkml/                      # Shared zkml library (forward pass, backprop, packing)
│   │   └── src/
│   │       ├── architectures/     # arch_64_10, arch_64_16_10, arch_cnn_gap
│   │       ├── test/              # Pure unit tests for all architectures
│   │       └── *.nr               # conv, pooling, relu, softmax, packing, hash
│   ├── single_layer_contract/     # On-chain single-layer network (64→10)
│   ├── multi_layer_perceptron/    # On-chain MLP (64→16→10)
│   └── cnn_gap_contract/          # On-chain CNN+GAP network
├── scripts/
│   ├── deploy.ts                  # Deployment script (local network + devnet)
│   ├── build-contracts.ts         # Contract compile + codegen helper
│   ├── pretrained-weights.ts      # Pre-trained weight loaders
│   └── weight-packing.ts          # 9-bit weight packing utilities
├── config/
│   ├── deployed.json              # Devnet deployment addresses (committed)
│   └── deployed.local.json        # Local deployment addresses (gitignored)
├── src/
│   ├── artifacts/                 # Generated contract TypeScript bindings
│   ├── aztec-wallet/              # Modular Aztec wallet library
│   │   ├── adapters/              # Browser wallet adapters (Azguard)
│   │   ├── components/            # ConnectButton and modals
│   │   ├── connectors/            # Wallet connector implementations
│   │   ├── hooks/                 # useAztecWallet, useConnectModal, etc.
│   │   ├── providers/             # AztecWalletProvider
│   │   ├── services/              # PXE client service
│   │   ├── signers/               # Account signing implementations
│   │   ├── store/                 # Zustand stores
│   │   └── types/                 # Configuration types
│   ├── components/                # React UI components
│   │   ├── canvas/                # Drawing canvas and controls
│   │   ├── home/                  # Prediction panel, architecture selector
│   │   ├── settings/              # Config panel
│   │   ├── shapley/               # Shapley value visualisation
│   │   └── ui/                    # Primitive components (Button, Card, Dialog, etc.)
│   ├── config/
│   │   ├── contracts.ts           # Contract address registry
│   │   ├── walletConfig.ts        # Aztec wallet configuration
│   │   └── networks/              # Network constants
│   ├── hooks/
│   │   ├── mutations/             # useTrainOnChain (on-chain training mutation)
│   │   ├── network/               # useAztecNode, useNetworkStore
│   │   ├── useNeural.ts           # Prediction + training orchestration
│   │   ├── useNetworkStatus.ts    # Node connection status
│   │   └── useToast.ts            # Toast notifications (Zustand)
│   ├── pages/
│   │   ├── Home.tsx               # Main draw-predict-train page
│   │   └── DocsPage.tsx           # Documentation page
│   ├── services/
│   │   ├── TrainingService.ts     # Aztec node + contract interaction service
│   │   ├── aztec/                 # Low-level Aztec client helpers
│   │   └── core/                  # Image processor, neural trainer (JS-side)
│   ├── store/                     # Global Zustand stores
│   ├── styles/                    # Tailwind CSS + CVA theme
│   └── utils/                     # Utility functions
└── tests/
    ├── unit/
    │   ├── config/                # contracts.ts tests
    │   └── services/              # TrainingService, ImageProcessor, NeuralTrainer tests
    └── e2e/
        ├── app.test.ts            # Unauthenticated smoke tests
        └── local-network/         # Connected-wallet E2E tests
```

---

## Architecture Overview

### On-Chain Training Flow

Training on Aztec uses a **hash-commitment pattern** to prevent race conditions:

1. **Client** computes the forward pass + backward pass (gradient update) entirely off-chain inside the PXE prover
2. **`submit_training_input`** (private) — packs current weights, trains, repacks new weights, enqueues a public call with the hash of new weights
3. **`apply_training_update`** (public, `only_self`) — verifies the committed hash and writes the new packed weights to public storage

This means the gradient computation is proven correct by the ZK circuit, and the public state update is atomic and replay-safe.

### Neural Network Architectures

| Contract | Architecture | Weights | Biases |
| -------- | ------------ | ------- | ------ |
| `SingleLayerContract` | 64→10 linear | 640 | 10 |
| `MultiLayerPerceptronContract` | 64→16→10 (ReLU) | 1,184 | 26 |
| `CNNGAPContract` | Conv(4×4×3)→ReLU→GAP→FC(3→10) | 78 | 13 |

Weights are stored packed (9-bit, 28 values per Field) to minimise on-chain storage.

### Wallet Options

| Wallet | Description | Use Case |
| ------ | ----------- | -------- |
| **Embedded** | Keys generated and stored in browser | Quick testing |
| **Azguard** | Browser extension wallet | Production, user-controlled keys |

---

## Using Azguard Wallet

1. Install the [Azguard](https://azguardwallet.io/) browser extension
2. Navigate to **Settings → Fee Configuration**
3. Select **FPC** in the "Pay fee with" dropdown
4. Click **"Create New FPC"** and enter the Sponsored FPC address:
   ```
   0x1586f476995be97f07ebd415340a14be48dc28c6c661cc6bdddb80ae790caa4e
   ```

---

## Adding a New Browser Wallet

To add support for a new browser wallet (e.g., Obsidian):

1. Create `src/aztec-wallet/adapters/obsidian/` with `ObsidianAdapter.ts`, `ObsidianWalletService.ts`, and `index.ts`
2. Implement `IBrowserWalletAdapter` in the adapter class
3. Register it in `src/aztec-wallet/config/aztecWallets.ts`
4. Add `'obsidian'` to the `aztecWallets` array in `src/config/walletConfig.ts`

No changes needed to hooks, providers, or `BrowserWalletConnector` — the adapter pattern handles all wallet-specific logic.

---

## Network Information

| Network | Node URL | Usage |
| ------- | -------- | ----- |
| Local Network | `http://localhost:8080` | Development |
| Devnet | `https://v4-devnet-2.aztec-labs.com` | Public testnet |

---

## UI Development

This project uses **Tailwind CSS v4** and **Radix UI Primitives**.

All Tailwind classes **must** be defined in a `styles` object at the top of the component — never inline:

```tsx
// ✅ Correct
const styles = {
  container: 'flex flex-col gap-4',
  title: 'text-lg font-semibold text-default',
} as const;

// ❌ Wrong
<div className="flex flex-col gap-4">
```

---

## Resources

- [Aztec Documentation](https://docs.aztec.network)
- [Noir Language Guide](https://noir-lang.org/docs)
- [Aztec.js API Reference](https://docs.aztec.network/reference/aztec-js)
- [Azguard Wallet](https://azguardwallet.io/)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

Built with [Aztec Network](https://aztec.network)
