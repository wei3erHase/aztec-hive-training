import {
  Benchmark,
  type BenchmarkContext,
} from '@defi-wonderland/aztec-benchmark';
import { AztecAddress } from '@aztec/aztec.js/addresses';
import type { ContractFunctionInteractionCallIntent } from '@aztec/aztec.js/authorization';
import { getContractInstanceFromInstantiationParams } from '@aztec/aztec.js/contracts';
import { SponsoredFeePaymentMethod } from '@aztec/aztec.js/fee';
import { Fr } from '@aztec/aztec.js/fields';
import { createAztecNodeClient, waitForNode } from '@aztec/aztec.js/node';
import { SPONSORED_FPC_SALT } from '@aztec/constants';
import { SponsoredFPCContract } from '@aztec/noir-contracts.js/SponsoredFPC';
import { EmbeddedWallet } from '@aztec/wallets/embedded';
import { registerInitialLocalNetworkAccountsInWallet } from '@aztec/wallets/testing';
import { randomBytes } from 'crypto';
import {
  getSingleLayerWeights,
  getSingleLayerBiases,
  getMLPWeights,
  getMLPBiases,
  getCNNWeights,
  getCNNBiases,
} from '../scripts/pretrained-weights.js';
import { packToFields } from '../scripts/weight-packing.js';
import { CNNGAPContract } from '../src/artifacts/CNNGAP.js';
import { MultiLayerPerceptronContract } from '../src/artifacts/MultiLayerPerceptron.js';
import { SingleLayerContract } from '../src/artifacts/SingleLayer.js';
import { FIELD_MODULUS } from '../src/utils/zkml.js';

// Contract constants
const SCALING_FACTOR = 1_000_000;
const INPUT_SIZE = 64;

interface TrainingArchitecture {
  contract: SingleLayerContract | MultiLayerPerceptronContract | CNNGAPContract;
  packedWeights: Fr[];
  packedBiases: Fr[];
}

interface TrainingBenchmarkContext extends BenchmarkContext {
  wallet: EmbeddedWallet;
  deployer: AztecAddress;
  accounts: AztecAddress[];
  singleLayer: TrainingArchitecture;
  mlp: TrainingArchitecture;
  cnnGap: TrainingArchitecture;
  feePaymentMethod: SponsoredFeePaymentMethod;
}

/**
 * Generate mock training data for benchmarking
 */
function generateMockTrainingData(): { inputPixels: Fr[]; label: Fr } {
  const inputPixels: Fr[] = [];
  for (let i = 0; i < INPUT_SIZE; i++) {
    const pixelValue = BigInt(Math.floor(((i % 16) * SCALING_FACTOR) / 16));
    inputPixels.push(new Fr(pixelValue));
  }
  return { inputPixels, label: new Fr(5n) };
}

/**
 * Benchmark class comparing training step across SingleLayer, MLP, and CNNGAP.
 * Returns named benchmark cases so reports show results side by side.
 */
export default class TrainingBenchmark extends Benchmark {
  async setup(): Promise<TrainingBenchmarkContext> {
    const aztecNode = createAztecNodeClient('http://localhost:8080');
    await waitForNode(aztecNode);

    const skipProving = process.argv.includes('--skip-proving');
    const wallet = await EmbeddedWallet.create('http://localhost:8080', {
      ephemeral: true,
      pxeConfig: { proverEnabled: !skipProving },
    });
    const accounts: AztecAddress[] =
      await registerInitialLocalNetworkAccountsInWallet(wallet);
    const [deployer] = accounts;

    const sponsoredFPCInstance =
      await getContractInstanceFromInstantiationParams(
        SponsoredFPCContract.artifact,
        { salt: new Fr(SPONSORED_FPC_SALT) }
      );
    await wallet.registerContract(
      sponsoredFPCInstance,
      SponsoredFPCContract.artifact
    );
    const feePaymentMethod = new SponsoredFeePaymentMethod(
      sponsoredFPCInstance.address
    );

    const mkSalt = () =>
      new Fr(BigInt('0x' + randomBytes(32).toString('hex')) % FIELD_MODULUS);

    // Deploy SingleLayer
    const singlePackedWeights = packToFields(getSingleLayerWeights(), 23, 28);
    const singlePackedBiases = packToFields(getSingleLayerBiases(), 1, 10);
    const singleDeployed = await SingleLayerContract.deployWithOpts(
      { wallet, method: 'constructor_pretrained' },
      singlePackedWeights,
      singlePackedBiases
    ).send({
      from: AztecAddress.ZERO,
      fee: { paymentMethod: feePaymentMethod },
      universalDeploy: true,
      skipInitialization: false,
      contractAddressSalt: mkSalt(),
    });
    const singleContract = SingleLayerContract.at(
      singleDeployed.address,
      wallet
    );
    const singlePackedWeightsLive = (await singleContract.methods
      .get_all_packed_weights()
      .simulate({ from: deployer })) as Fr[];
    const singlePackedBiasesLive = await singleContract.methods
      .get_packed_biases()
      .simulate({ from: deployer });
    const singleBiasesArr = Array.isArray(singlePackedBiasesLive)
      ? (singlePackedBiasesLive as Fr[])
      : [singlePackedBiasesLive as Fr];

    console.log(
      `SingleLayer deployed at: ${singleDeployed.address.toString()}`
    );

    // Deploy MLP
    const mlpPackedWeights = packToFields(getMLPWeights(), 43, 28);
    const mlpPackedBiases = packToFields(getMLPBiases(), 1, 26);
    const mlpDeployed = await MultiLayerPerceptronContract.deployWithOpts(
      { wallet, method: 'constructor_pretrained' },
      mlpPackedWeights,
      mlpPackedBiases
    ).send({
      from: AztecAddress.ZERO,
      fee: { paymentMethod: feePaymentMethod },
      universalDeploy: true,
      skipInitialization: false,
      contractAddressSalt: mkSalt(),
    });
    const mlpContract = MultiLayerPerceptronContract.at(
      mlpDeployed.address,
      wallet
    );
    const mlpPackedWeightsLive = (await mlpContract.methods
      .get_all_packed_weights()
      .simulate({ from: deployer })) as Fr[];
    const mlpPackedBiasesLive = await mlpContract.methods
      .get_packed_biases()
      .simulate({ from: deployer });
    const mlpBiasesArr = Array.isArray(mlpPackedBiasesLive)
      ? (mlpPackedBiasesLive as Fr[])
      : [mlpPackedBiasesLive as Fr];

    console.log(`MLP deployed at: ${mlpDeployed.address.toString()}`);

    // Deploy CNNGAP
    const cnnPackedWeights = packToFields(getCNNWeights(), 3, 28);
    const cnnPackedBiases = packToFields(getCNNBiases(), 1, 28);
    const cnnDeployed = await CNNGAPContract.deployWithOpts(
      { wallet, method: 'constructor_pretrained' },
      cnnPackedWeights,
      cnnPackedBiases
    ).send({
      from: AztecAddress.ZERO,
      fee: { paymentMethod: feePaymentMethod },
      universalDeploy: true,
      skipInitialization: false,
      contractAddressSalt: mkSalt(),
    });
    const cnnContract = CNNGAPContract.at(cnnDeployed.address, wallet);
    const cnnPackedWeightsLive = (await cnnContract.methods
      .get_all_packed_weights()
      .simulate({ from: deployer })) as Fr[];
    const cnnPackedBiasesLive = (await cnnContract.methods
      .get_packed_biases()
      .simulate({ from: deployer })) as Fr[];

    console.log(`CNNGAP deployed at: ${cnnDeployed.address.toString()}`);

    return {
      wallet,
      deployer,
      accounts,
      singleLayer: {
        contract: singleContract,
        packedWeights: singlePackedWeightsLive,
        packedBiases: singleBiasesArr,
      },
      mlp: {
        contract: mlpContract,
        packedWeights: mlpPackedWeightsLive,
        packedBiases: mlpBiasesArr,
      },
      cnnGap: {
        contract: cnnContract,
        packedWeights: cnnPackedWeightsLive,
        packedBiases: cnnPackedBiasesLive,
      },
      feePaymentMethod,
    };
  }

  getMethods(
    context: TrainingBenchmarkContext
  ): Array<
    | ContractFunctionInteractionCallIntent
    | { name: string; interaction: ContractFunctionInteractionCallIntent }
  > {
    const { wallet, deployer, singleLayer, mlp, cnnGap } = context;
    const { inputPixels, label } = generateMockTrainingData();

    const singleIntent: ContractFunctionInteractionCallIntent = {
      caller: deployer,
      action: singleLayer.contract
        .withWallet(wallet)
        .methods.submit_training_input(
          inputPixels,
          label,
          singleLayer.packedWeights,
          singleLayer.packedBiases
        ),
    };
    const mlpIntent: ContractFunctionInteractionCallIntent = {
      caller: deployer,
      action: mlp.contract
        .withWallet(wallet)
        .methods.submit_training_input(
          inputPixels,
          label,
          mlp.packedWeights,
          mlp.packedBiases
        ),
    };
    const cnnIntent: ContractFunctionInteractionCallIntent = {
      caller: deployer,
      action: cnnGap.contract
        .withWallet(wallet)
        .methods.submit_training_input(
          inputPixels,
          label,
          cnnGap.packedWeights,
          cnnGap.packedBiases
        ),
    };

    return [
      { name: 'SingleLayer: submit_training_input', interaction: singleIntent },
      { name: 'MLP: submit_training_input', interaction: mlpIntent },
      { name: 'CNNGAP: submit_training_input', interaction: cnnIntent },
    ];
  }
}
