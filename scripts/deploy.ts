#!/usr/bin/env tsx
/**
 * Deploy ZKML pretrained contracts to local-network and/or devnet.
 *
 * Usage:
 *   yarn deploy-contracts                          # deploys to local-network (default)
 *   yarn deploy-contracts --network local-network  # deploy to local network
 *   yarn deploy-contracts --network devnet         # deploy to devnet
 *   yarn deploy-contracts --network all            # deploy to both local-network and devnet
 *
 * Prerequisites:
 *   - Local network: run `gaztec start --local-network` on port 8080
 *   - Devnet: ensure network is reachable
 *   - Contracts must be built: yarn build-contracts
 */

import { AztecAddress } from '@aztec/aztec.js/addresses';
import { getContractInstanceFromInstantiationParams } from '@aztec/aztec.js/contracts';
import { SponsoredFeePaymentMethod } from '@aztec/aztec.js/fee';
import { Fr } from '@aztec/aztec.js/fields';
import { createAztecNodeClient, waitForNode } from '@aztec/aztec.js/node';
import { SPONSORED_FPC_SALT } from '@aztec/constants';
import { createStore } from '@aztec/kv-store/lmdb';
import { SponsoredFPCContractArtifact } from '@aztec/noir-contracts.js/SponsoredFPC';
import { createPXE } from '@aztec/pxe/client/bundle';
import { getPXEConfig } from '@aztec/pxe/config';
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
import { packToFields } from './weight-packing.js';

const NETWORK_URLS: Record<string, string> = {
  'local-network': 'http://localhost:8080',
  devnet: 'https://v4-devnet-2.aztec-labs.com',
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
  // Local network accepts dummy proofs; devnet requires real proofs from the client prover.
  pxeConfig.proverEnabled = networkId !== 'local-network';
  const pxe = await createPXE(aztecNode, pxeConfig, { store: pxeStore });

  // Signerless wallet: handles AztecAddress.ZERO via SignerlessAccount
  const { MinimalWallet } = await import('../src/utils/MinimalWallet.js');
  const deployWallet = new MinimalWallet(pxe as any, aztecNode);

  // SponsoredFPC is pre-deployed at genesis on both local-network and devnet.
  // Register the artifact so the PXE can resolve its function selectors during simulation.
  const sponsoredFPCInstance = await getContractInstanceFromInstantiationParams(
    SponsoredFPCContractArtifact,
    { salt: new Fr(SPONSORED_FPC_SALT) }
  );

  await (pxe as any).registerContract({
    instance: sponsoredFPCInstance,
    artifact: SponsoredFPCContractArtifact,
  });
  const paymentMethod = new SponsoredFeePaymentMethod(
    sponsoredFPCInstance.address
  );
  console.log(`  SponsoredFPC: ${sponsoredFPCInstance.address.toString()}`);

  // Contract artifacts
  const { SingleLayerContract } = await import(
    '../src/artifacts/SingleLayer.js'
  );
  const { MultiLayerPerceptronContract } = await import(
    '../src/artifacts/MultiLayerPerceptron.js'
  );
  const { CNNGAPContract } = await import('../src/artifacts/CNNGAP.js');

  const sendOpts = {
    from: AztecAddress.ZERO,
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
    const slPackedWeights = packToFields(slWeights, 23, 28);
    const slPackedBiases = packToFields(slBiases, 1, 10);
    const singleLayer = await SingleLayerContract.deployWithOpts(
      { method: 'constructor_pretrained', wallet: deployWallet as any },
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
    const mlpPackedWeights = packToFields(mlpWeights, 43, 28);
    const mlpPackedBiases = packToFields(mlpBiases, 1, 26);
    const mlp = await MultiLayerPerceptronContract.deployWithOpts(
      { method: 'constructor_pretrained', wallet: deployWallet as any },
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
    const cnnPackedWeights = packToFields(cnnWeights, 3, 28);
    const cnnPackedBiases = packToFields(cnnBiases, 1, 28);
    const cnn = await CNNGAPContract.deployWithOpts(
      { method: 'constructor_pretrained', wallet: deployWallet as any },
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
  const outputPath =
    networkId === 'local-network'
      ? path.join(CONFIG_DIR, 'deployed.local.json')
      : path.join(CONFIG_DIR, 'deployed.json');

  let existing: Record<string, unknown> = {};
  if (networkId !== 'local-network' && fs.existsSync(outputPath)) {
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
    'all';

  const networks = network === 'all' ? ['local-network', 'devnet'] : [network];

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
