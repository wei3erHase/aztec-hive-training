#!/usr/bin/env tsx
/**
 * Deploy ZKML pretrained contracts to local-network and/or testnet.
 *
 * Usage:
 *   yarn deploy-contracts                           # deploys to local-network (default)
 *   yarn deploy-contracts --network local-network   # deploy to local network
 *   yarn deploy-contracts --network testnet         # deploy to testnet
 *   yarn deploy-contracts --network all             # deploy to both local-network and testnet
 *
 * Prerequisites:
 *   - Local network: run `aztec start --local-network` on port 8080
 *   - Testnet: uses SPONSOR_FPC_ADDRESS from `.env` (or canonical genesis FPC as fallback)
 *   - Contracts must be built: yarn build-contracts
 */

import { NO_FROM } from '@aztec/aztec.js/account';
import { loadContractArtifact } from '@aztec/aztec.js/abi';
import { AztecAddress } from '@aztec/aztec.js/addresses';
import { getContractInstanceFromInstantiationParams } from '@aztec/aztec.js/contracts';
import { SponsoredFeePaymentMethod } from '@aztec/aztec.js/fee';
import { Fr } from '@aztec/aztec.js/fields';
import { getFeeJuiceBalance } from '@aztec/aztec.js/utils';
import { SPONSORED_FPC_SALT } from '@aztec/constants';
import { createAztecNodeClient, waitForNode } from '@aztec/aztec.js/node';
import { createStore } from '@aztec/kv-store/lmdb';
import { SponsoredFPCContractArtifact } from '@aztec/noir-contracts.js/SponsoredFPC';
import { createPXE } from '@aztec/pxe/client/bundle';
import { getPXEConfig } from '@aztec/pxe/config';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  getSingleLayerWeights,
  getSingleLayerBiases,
  getMLPWeights,
  getMLPBiases,
  getCNNWeights,
  getCNNBiases,
} from './pretrained-weights.js';
import { packToFields, MAX_TRAINER_PACKED_BIAS_FIELDS, MAX_TRAINER_PACKED_WEIGHT_FIELDS } from './weight-packing.js';
import { SPONSORED_FPC_CANONICAL_ADDRESS } from '../src/config/networks/sponsoredFpc.js';

const NETWORK_URLS: Record<string, string> = {
  'local-network': 'http://localhost:8080',
  testnet: 'https://rpc.testnet.aztec-labs.com/',
};

/**
 * Load project `.env` when present. Values from `.env` override any existing
 * process env (including Node's `--env-file`), so local `.env` is authoritative.
 */
function loadOptionalDotEnv(): void {
  const envPath = path.join(process.cwd(), '.env');
  if (!fs.existsSync(envPath)) return;
  const raw = fs.readFileSync(envPath, 'utf-8');
  for (let line of raw.split('\n')) {
    line = line.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    let value = line.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

loadOptionalDotEnv();

/**
 * Resolves the SponsoredFPC instance per network.
 *
 * - local-network: derived from the canonical genesis salt (@aztec/constants).
 *   `aztecNode.getContract` does not expose genesis contracts, so we always
 *   compute the instance locally from the known salt.
 * - testnet: fetch by SPONSOR_FPC_ADDRESS from `.env`, or fall back to the
 *   canonical genesis FPC (SPONSORED_FPC_CANONICAL_ADDRESS).
 * - testnet (SPONSOR_FPC_SALT set, no address): derived from the declared salt.
 *
 * Returning the full instance (not just the address) lets the caller register
 * it with the PXE without a separate `getContract` call that may return undefined.
 */
const getSponsoredFpcInstance = async (
  networkId: string,
  aztecNode: Awaited<ReturnType<typeof createAztecNodeClient>>
) => {
  if (networkId === 'local-network') {
    return getContractInstanceFromInstantiationParams(
      SponsoredFPCContractArtifact,
      { salt: new Fr(SPONSORED_FPC_SALT) }
    );
  }

  const envAddress =
    process.env.SPONSOR_FPC_ADDRESS ?? SPONSORED_FPC_CANONICAL_ADDRESS;
  const instance = await aztecNode.getContract(
    AztecAddress.fromString(envAddress)
  );
  if (instance) {
    return instance;
  }

  if (process.env.SPONSOR_FPC_ADDRESS) {
    throw new Error(
      `SponsoredFPC at SPONSOR_FPC_ADDRESS=${process.env.SPONSOR_FPC_ADDRESS} not found on node. Verify the address is correct.`
    );
  }

  const envSalt = process.env.SPONSOR_FPC_SALT;
  if (envSalt) {
    return getContractInstanceFromInstantiationParams(
      SponsoredFPCContractArtifact,
      { salt: Fr.fromHexString(envSalt) }
    );
  }

  throw new Error(
    `SponsoredFPC at canonical address ${SPONSORED_FPC_CANONICAL_ADDRESS} not found on node`
  );
};

const CONFIG_DIR = path.join(process.cwd(), 'config');

async function getOrCreateStore(networkName: string) {
  const storeName = `deploy-pxe-${networkName}`;
  fs.mkdirSync(path.join(process.cwd(), '.deploy-store', networkName), {
    recursive: true,
  });
  return createStore(storeName, {
    dataDirectory: path.join(process.cwd(), '.deploy-store'),
    dataStoreMapSizeKb: 50 * 1024,
  });
}

async function deployToNetwork(networkId: string): Promise<void> {
  const nodeUrl = NETWORK_URLS[networkId];
  if (!nodeUrl) {
    throw new Error(`Unknown network: ${networkId}`);
  }

  console.log(`\n========================================`);
  console.log(`  Deploying to ${networkId} (${nodeUrl})`);
  console.log(`========================================\n`);

  // Stand-alone PXE for the deploy script (not the browser SharedPXEService)
  const aztecNode = createAztecNodeClient(nodeUrl);
  await waitForNode(aztecNode);

  const l1Contracts = await aztecNode.getL1ContractAddresses();
  const pxeStore = await getOrCreateStore(networkId);
  const pxeConfig = getPXEConfig();
  pxeConfig.l1Contracts = l1Contracts;
  // Local network accepts dummy proofs; testnet requires real proofs from the client prover.
  pxeConfig.proverEnabled = networkId !== 'local-network';
  const pxe = await createPXE(aztecNode, pxeConfig, { store: pxeStore });

  // Signerless wallet: handles NO_FROM via DefaultEntrypoint (no account contract)
  const { MinimalWallet } = await import('../src/utils/MinimalWallet.js');
  const deployWallet = new MinimalWallet(pxe, aztecNode);

  // SponsoredFPC is pre-deployed at genesis on both local-network and testnet.
  // Register the artifact so the PXE can resolve its function selectors during simulation.
  // NOTE: `aztecNode.getContract` returns undefined for genesis-deployed contracts on fresh
  // local networks, so we always derive the instance from the declared salt instead.
  const sponsoredFPCInstance = await getSponsoredFpcInstance(
    networkId,
    aztecNode
  );
  const sponsoredFPCAddress = sponsoredFPCInstance.address;
  const fpcFJBalance = await getFeeJuiceBalance(sponsoredFPCAddress, aztecNode);
  console.log(`  SponsoredFPC: ${sponsoredFPCAddress.toString()}`);
  console.log(`  FPC Fee Juice balance: ${fpcFJBalance}`);
  if (fpcFJBalance === 0n) {
    throw new Error(
      `SponsoredFPC at ${sponsoredFPCAddress.toString()} has zero Fee Juice balance. Fund it before deploying.`
    );
  }

  {
    // Try to register with the current artifact.  If the class ID doesn't match
    // (the on-chain FPC was compiled with a different aztec-nr version), fall back
    // to `registerContractClass` with the legacy artifact so PXE can prove the
    // private `sponsor_unconditionally` call without a class ID check.
    let registered = false;
    try {
      await pxe.registerContract({
        instance: sponsoredFPCInstance,
        artifact: SponsoredFPCContractArtifact,
      });
      registered = true;
    } catch (regErr) {
      console.warn(
        `  [warn] New FPC artifact class mismatch, falling back to legacy artifact: ${(regErr as Error).message}`
      );
    }

    if (!registered) {
      // Use the legacy SponsoredFPC artifact (compiled with aztec-nr 4.3.0)
      // to register the on-chain FPC class so PXE can prove `sponsor_unconditionally()`.
      try {
        const _require = createRequire(import.meta.url);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const legacySponsoredFPCJson = _require(
          './utils/SponsoredFPC_legacy.json'
        ) as any;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const legacyArtifact = loadContractArtifact(
          legacySponsoredFPCJson as any
        );
        await (
          pxe as unknown as {
            registerContractClass: (a: unknown) => Promise<void>;
          }
        ).registerContractClass(legacyArtifact);
        await pxe.registerContract({
          instance: sponsoredFPCInstance,
          artifact: legacyArtifact,
        });
        console.log('  Registered FPC with legacy artifact (4.3.0).');
      } catch (legacyErr) {
        console.warn(
          `  [warn] Legacy FPC artifact registration also failed: ${(legacyErr as Error).message}`
        );
      }
    }
  }
  const paymentMethod = new SponsoredFeePaymentMethod(sponsoredFPCAddress);

  // Contract artifacts
  const { SingleLayerContract } = await import(
    '../src/artifacts/SingleLayer.js'
  );
  const { MultiLayerPerceptronContract } = await import(
    '../src/artifacts/MultiLayerPerceptron.js'
  );
  const { CNNGAPContract } = await import('../src/artifacts/CNNGAP.js');

  const sendOpts = {
    from: NO_FROM,
    fee: { paymentMethod },
    // Class publication is idempotent (no-op if already registered), so always publish.
    // Skipping it would fail if the class hasn't been published to this network yet.
    skipClassPublication: false,
  };

  const results: Record<string, { address: string; constructorName: string }> =
    {};

  // 1. SingleLayer
  try {
    console.log('Deploying SingleLayer...');
    const slWeights = getSingleLayerWeights();
    const slBiases = getSingleLayerBiases();
    const slPackedWeights = packToFields(
      slWeights,
      MAX_TRAINER_PACKED_WEIGHT_FIELDS
    );
    const slPackedBiases = packToFields(slBiases, MAX_TRAINER_PACKED_BIAS_FIELDS, 10);
    const { contract: singleLayer } = await SingleLayerContract.deployWithOpts(
      { method: 'constructor_pretrained', wallet: deployWallet },
      slPackedWeights,
      slPackedBiases
    ).send(sendOpts);
    results.singleLayer = {
      address: singleLayer.address.toString(),
      constructorName: 'constructor_pretrained',
    };
    console.log(`  SingleLayer: ${results.singleLayer.address}`);
  } catch (err) {
    console.warn('SingleLayer deployment failed:', (err as Error).message);
  }

  // 2. MultiLayerPerceptron
  try {
    console.log('Deploying MultiLayerPerceptron...');
    const mlpWeights = getMLPWeights();
    const mlpBiases = getMLPBiases();
    const mlpPackedWeights = packToFields(
      mlpWeights,
      MAX_TRAINER_PACKED_WEIGHT_FIELDS
    );
    const mlpPackedBiases = packToFields(mlpBiases, MAX_TRAINER_PACKED_BIAS_FIELDS, 26);
    const { contract: mlp } = await MultiLayerPerceptronContract.deployWithOpts(
      { method: 'constructor_pretrained', wallet: deployWallet },
      mlpPackedWeights,
      mlpPackedBiases
    ).send(sendOpts);
    results.multiLayerPerceptron = {
      address: mlp.address.toString(),
      constructorName: 'constructor_pretrained',
    };
    console.log(
      `  MultiLayerPerceptron: ${results.multiLayerPerceptron.address}`
    );
  } catch (err) {
    console.warn(
      'MultiLayerPerceptron deployment failed:',
      (err as Error).message
    );
  }

  // 3. CNNGAP
  try {
    console.log('Deploying CNNGAP...');
    const cnnWeights = getCNNWeights();
    const cnnBiases = getCNNBiases();
    const cnnPackedWeights = packToFields(
      cnnWeights,
      MAX_TRAINER_PACKED_WEIGHT_FIELDS
    );
    const cnnPackedBiases = packToFields(cnnBiases, MAX_TRAINER_PACKED_BIAS_FIELDS, 28);
    const { contract: cnn } = await CNNGAPContract.deployWithOpts(
      { method: 'constructor_pretrained', wallet: deployWallet },
      cnnPackedWeights,
      cnnPackedBiases
    ).send(sendOpts);
    results.cnnGap = {
      address: cnn.address.toString(),
      constructorName: 'constructor_pretrained',
    };
    console.log(`  CNNGAP: ${results.cnnGap.address}`);
  } catch (err) {
    console.warn('CNNGAP deployment failed:', (err as Error).message);
  }

  // Write deployment config
  // - local-network → config/deployed.local.json (gitignored, injected by Vite plugin)
  // - testnet       → config/deployed.json        (committed, read directly by the app)
  const outputPath =
    networkId === 'local-network'
      ? path.join(CONFIG_DIR, 'deployed.local.json')
      : path.join(CONFIG_DIR, 'deployed.json');

  let existing: Record<string, unknown> = {};
  if (fs.existsSync(outputPath)) {
    existing = JSON.parse(fs.readFileSync(outputPath, 'utf-8'));
  }

  const updated = {
    ...existing,
    [networkId]: results,
  };

  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(outputPath, JSON.stringify(updated, null, 2), 'utf-8');
  console.log(`\nWrote deployment to ${outputPath}`);
}

async function main() {
  const args = process.argv.slice(2);
  const networkFromEq = args.find((a) => a.startsWith('--network='));
  const networkFromSplit = (() => {
    const idx = args.indexOf('--network');
    return idx >= 0 ? args[idx + 1] : undefined;
  })();
  const network =
    (networkFromEq ? networkFromEq.split('=')[1] : networkFromSplit) ??
    process.env.DEPLOY_NETWORK ??
    'local-network';

  const networks = network === 'all' ? ['local-network', 'testnet'] : [network];

  // Drop any leftover LMDB state from a prior crash (native malloc errors on reopen).
  fs.rmSync(path.join(process.cwd(), '.deploy-store'), {
    recursive: true,
    force: true,
  });

  try {
    for (const n of networks) {
      await deployToNetwork(n);
    }
    console.log('\nDone.');
  } catch (err) {
    console.error(`Deployment failed:`, err);
    process.exit(1);
  } finally {
    // Always clean up the PXE store, even on failure
    fs.rmSync(path.join(process.cwd(), '.deploy-store'), {
      recursive: true,
      force: true,
    });
  }
}

main().then(() => process.exit(0));
