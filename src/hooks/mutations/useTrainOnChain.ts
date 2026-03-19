/**
 * useTrainOnChain Hook
 * Handles on-chain training transaction submission
 * Supports both embedded wallet (SharedPXEService) and Wallet SDK paths.
 */

import { useState, useCallback } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useAztecWallet } from '../../aztec-wallet/hooks/useAztecWallet';
import { deployAccountIfNotExists } from '../../aztec-wallet/services/wallet/deployAccount';
import { getSavedAccount } from '../../aztec-wallet/services/wallet/embeddedAccount';
import { useWalletStore } from '../../aztec-wallet/store/wallet';
import { WalletType } from '../../aztec-wallet/types/aztec';
import {
  getDeploymentConfig,
  getContractKeyForArchitecture,
  type ArchitectureId,
} from '../../config/contracts';
import { FIELD_MODULUS } from '../../utils/zkml';
import type {
  TrainingTxData,
  TrainingTxResult,
} from '../../services/TrainingService';

interface UseTrainOnChainReturn {
  submitTraining: (data: TrainingTxData) => Promise<TrainingTxResult>;
  isPending: boolean;
  error: string | null;
}

if (typeof window !== 'undefined') {
  const originalError = console.error;
  console.error = (...args: unknown[]) => {
    const message = args
      .map((arg) => {
        if (typeof arg === 'string') return arg;
        if (arg && typeof arg === 'object') {
          if ('message' in (arg as Error))
            return String((arg as Error).message);
          if (
            typeof (arg as { toString?: () => string }).toString === 'function'
          )
            return (arg as { toString: () => string }).toString();
          return JSON.stringify(arg);
        }
        return String(arg);
      })
      .join(' ');
    if (
      message.includes('Error processing block stream') &&
      (message.includes('TransactionInactiveError') ||
        message.includes('IDBCursor') ||
        message.includes('transaction has finished') ||
        message.includes('transaction is not active'))
    ) {
      return;
    }
    originalError.apply(console, args);
  };
}

type ContractType = {
  at: (
    a: unknown,
    w: unknown
  ) => {
    methods: {
      submit_training_input: (
        a: unknown[],
        b: unknown,
        c: unknown[],
        d: unknown[]
      ) => { send: (opts: unknown) => Promise<unknown> };
      get_all_packed_weights: () => {
        simulate: (o: unknown) => Promise<unknown>;
      };
      get_packed_biases: () => { simulate: (o: unknown) => Promise<unknown> };
    };
  };
};

async function loadContractModules(arch: ArchitectureId) {
  let ContractClass: ContractType;
  let ContractArtifact: unknown;
  if (arch === 'singleLayer') {
    const m = await import('../../artifacts/SingleLayer');
    ContractClass = m.SingleLayerContract as ContractType;
    ContractArtifact = m.SingleLayerContractArtifact;
  } else if (arch === 'cnnGap') {
    const m = await import('../../artifacts/CNNGAP');
    ContractClass = m.CNNGAPContract as ContractType;
    ContractArtifact = m.CNNGAPContractArtifact;
  } else {
    const m = await import('../../artifacts/MultiLayerPerceptron');
    ContractClass = m.MultiLayerPerceptronContract as ContractType;
    ContractArtifact = m.MultiLayerPerceptronContractArtifact;
  }
  return { ContractClass, ContractArtifact };
}

export const useTrainOnChain = (): UseTrainOnChainReturn => {
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { network } = useAztecWallet();
  const networkId = network?.name ?? 'local-network';
  const nodeUrl = network?.nodeUrl ?? 'http://localhost:8080';

  const { walletType, sdkWallet, sdkAccount } = useWalletStore(
    useShallow((state) => ({
      walletType: state.walletType,
      sdkWallet: state.sdkWallet,
      sdkAccount: state.account,
    }))
  );

  const submitTraining = useCallback(
    async (data: TrainingTxData): Promise<TrainingTxResult> => {
      setIsPending(true);
      setError(null);

      const arch: ArchitectureId = data.architecture ?? 'mlp';
      const contractKey = getContractKeyForArchitecture(arch);
      const deploymentConfig = getDeploymentConfig(networkId);
      const contractAddress =
        deploymentConfig.contracts[contractKey].address ?? '';

      if (!contractAddress) {
        const msg = `Contract not configured for ${networkId}`;
        setError(msg);
        setIsPending(false);
        return { success: false, error: msg };
      }

      try {
        if (walletType === WalletType.WALLET_SDK) {
          return await submitWithSDKWallet({
            arch,
            contractAddress,
            networkId,
            nodeUrl,
            data,
            sdkWallet,
            sdkAccount,
          });
        }

        return await submitWithEmbeddedWallet({
          arch,
          contractAddress,
          networkId,
          nodeUrl,
          data,
        });
      } catch (err) {
        const errorMsg =
          err instanceof Error
            ? err.message
            : 'Unknown error during training submission';
        setError(errorMsg);
        return { success: false, error: errorMsg };
      } finally {
        setIsPending(false);
      }
    },
    [networkId, nodeUrl, walletType, sdkWallet, sdkAccount]
  );

  return { submitTraining, isPending, error };
};

async function submitWithSDKWallet({
  arch,
  contractAddress,
  networkId,
  nodeUrl,
  data,
  sdkWallet,
  sdkAccount,
}: {
  arch: ArchitectureId;
  contractAddress: string;
  networkId: string;
  nodeUrl: string;
  data: TrainingTxData;
  sdkWallet: import('@aztec/aztec.js/wallet').Wallet | null;
  sdkAccount: import('@aztec/aztec.js/account').AccountWithSecretKey | null;
}): Promise<TrainingTxResult> {
  if (!sdkWallet || !sdkAccount) {
    return {
      success: false,
      error: 'No SDK wallet connected. Please connect your wallet first.',
    };
  }

  const [
    { AztecAddress },
    { Fr },
    { TxStatus },
    { SponsoredFeePaymentMethod },
    { AVAILABLE_NETWORKS },
    { ContractClass, ContractArtifact },
  ] = await Promise.all([
    import('@aztec/aztec.js/addresses'),
    import('@aztec/aztec.js/fields'),
    import('@aztec/stdlib/tx'),
    import('@aztec/aztec.js/fee'),
    import('../../config/networks'),
    loadContractModules(arch),
  ]);

  const accountAddress = sdkAccount.getAddress();

  // Resolve the sponsored FPC address for fee payment.
  // For named networks the address comes from the network config;
  // for local-network it is computed from the canonical salt.
  const networkConfig = AVAILABLE_NETWORKS.find((n) => n.name === networkId);
  let fpcAddress: import('@aztec/aztec.js/addresses').AztecAddress;
  if (networkConfig?.sponsoredFpcAddress) {
    fpcAddress = AztecAddress.fromString(networkConfig.sponsoredFpcAddress);
  } else {
    const [
      { getContractInstanceFromInstantiationParams },
      { SponsoredFPCContractArtifact },
      { SPONSORED_FPC_SALT },
    ] = await Promise.all([
      import('@aztec/aztec.js/contracts'),
      import('@aztec/noir-contracts.js/SponsoredFPC'),
      import('@aztec/constants'),
    ]);
    const fpcInstance = await getContractInstanceFromInstantiationParams(
      SponsoredFPCContractArtifact,
      { salt: new Fr(SPONSORED_FPC_SALT) }
    );
    fpcAddress = fpcInstance.address;
  }
  const paymentMethod = new SponsoredFeePaymentMethod(fpcAddress);

  const trainingAddress = AztecAddress.fromString(contractAddress);

  // Register the contract in the SDK wallet's PXE (inside the extension) if
  // it is not already known.  The SDK wallet exposes the same PXE interface.
  let contractInstance = await (
    sdkWallet as { getContractInstance: (a: unknown) => Promise<unknown> }
  ).getContractInstance(trainingAddress);
  if (!contractInstance) {
    const { createAztecNodeClient } = await import('@aztec/aztec.js/node');
    const aztecNode = createAztecNodeClient(nodeUrl);
    contractInstance = await aztecNode.getContract(trainingAddress);
    if (contractInstance) {
      await (
        sdkWallet as {
          registerContract: (i: unknown, a: unknown) => Promise<void>;
        }
      ).registerContract(contractInstance, ContractArtifact);
    }
  }

  const contract = ContractClass.at(trainingAddress, sdkWallet);

  const toFr = (value: bigint) => {
    if (value < 0n) return new Fr(FIELD_MODULUS + value);
    return new Fr(value);
  };
  const ensureFr = (value: unknown) => {
    if (value instanceof Fr) return value;
    if (
      value &&
      typeof (value as { toBigInt?: () => bigint }).toBigInt === 'function'
    )
      return new Fr((value as { toBigInt: () => bigint }).toBigInt());
    return new Fr(BigInt(value as bigint));
  };

  const inputPixelsFr = data.inputPixels.map((p) => toFr(p));
  const labelFr = new Fr(data.label);

  const { result: packedWeightsResult } = await contract.methods
    .get_all_packed_weights()
    .simulate({ from: accountAddress });
  const { result: packedBiasesResult } = await contract.methods
    .get_packed_biases()
    .simulate({ from: accountAddress });

  const currentPackedWeightsFr = (
    Array.isArray(packedWeightsResult)
      ? packedWeightsResult
      : [packedWeightsResult]
  ).map((w) => ensureFr(w));
  const currentPackedBiasesFr = (
    Array.isArray(packedBiasesResult)
      ? packedBiasesResult
      : [packedBiasesResult]
  ).map((b) => ensureFr(b));

  const tx = contract.methods.submit_training_input(
    inputPixelsFr,
    labelFr,
    currentPackedWeightsFr,
    currentPackedBiasesFr
  );

  const sendResult = await tx.send({
    from: accountAddress,
    fee: { paymentMethod },
    wait: { waitForStatus: TxStatus.PROPOSED, timeout: 120 },
  });

  const rec =
    (
      sendResult as {
        receipt?: {
          txHash?: { toString?: () => string };
          blockNumber?: number;
        };
      }
    ).receipt ??
    (sendResult as {
      txHash?: { toString?: () => string };
      blockNumber?: number;
    });
  return {
    success: true,
    txHash: rec.txHash?.toString?.(),
    blockNumber: rec.blockNumber,
  };
}

async function submitWithEmbeddedWallet({
  arch,
  contractAddress,
  networkId,
  nodeUrl,
  data,
}: {
  arch: ArchitectureId;
  contractAddress: string;
  networkId: string;
  nodeUrl: string;
  data: TrainingTxData;
}): Promise<TrainingTxResult> {
  const saved = getSavedAccount();
  if (!saved) {
    return {
      success: false,
      error: 'No embedded wallet found. Please connect your wallet first.',
    };
  }

  const [
    { SharedPXEService },
    { AztecAddress },
    { Fr },
    { AccountManager },
    { EcdsaRAccountContract },
    { TxStatus },
    { ContractClass, ContractArtifact },
  ] = await Promise.all([
    import('../../aztec-wallet/services/aztec/pxe'),
    import('@aztec/aztec.js/addresses'),
    import('@aztec/aztec.js/fields'),
    import('@aztec/aztec.js/wallet'),
    import('@aztec/accounts/ecdsa/lazy'),
    import('@aztec/stdlib/tx'),
    loadContractModules(arch),
  ]);

  const pxeInstance = await SharedPXEService.getCurrentInstance(
    nodeUrl,
    networkId as 'local-network' | 'testnet'
  );
  const { pxe } = pxeInstance;
  const wallet = pxeInstance.wallet;

  let accountAddress: import('@aztec/aztec.js/addresses').AztecAddress;

  const accounts = await (
    wallet as { getAccounts: () => Promise<{ item: unknown }[]> }
  ).getAccounts();
  const savedAddr = saved.address;
  const existing = accounts.find(
    (a) => (a.item as { toString: () => string }).toString() === savedAddr
  );

  if (existing) {
    accountAddress = existing.item as typeof accountAddress;
  } else {
    const secretKey = Fr.fromString(saved.secretKey);
    const salt = Fr.fromString(saved.salt);
    const signingKey = Buffer.from(saved.signingKey, 'hex');
    const accountContract = new EcdsaRAccountContract(signingKey);
    const accountManager = await AccountManager.create(
      wallet as Parameters<typeof AccountManager.create>[0],
      secretKey,
      accountContract,
      salt
    );
    const account = await accountManager.getAccount();
    const instance = accountManager.getInstance();
    const artifact = await accountManager
      .getAccountContract()
      .getContractArtifact();
    await (
      wallet as {
        registerContract: (
          i: unknown,
          a: unknown,
          sk?: unknown
        ) => Promise<void>;
      }
    ).registerContract(instance, artifact, accountManager.getSecretKey());
    (wallet as { addAccount: (a: unknown) => void }).addAccount(account);
    accountAddress = accountManager.address;

    try {
      await deployAccountIfNotExists(accountManager, pxeInstance);
    } catch (deployErr) {
      console.warn(
        '[useTrainOnChain] Account deployment failed:',
        deployErr instanceof Error ? deployErr.message : deployErr
      );
      throw deployErr;
    }
  }

  const trainingAddress = AztecAddress.fromString(contractAddress);
  let contractInstance = await pxe.getContractInstance(trainingAddress);
  if (!contractInstance) {
    contractInstance = await pxeInstance.aztecNode.getContract(trainingAddress);
    if (contractInstance) {
      await (
        wallet as {
          registerContract: (i: unknown, a: unknown) => Promise<void>;
        }
      ).registerContract(contractInstance, ContractArtifact);
    }
  }

  const paymentMethod = await pxeInstance.getSponsoredFeePaymentMethod();
  const contract = ContractClass.at(trainingAddress, wallet);

  const toFr = (value: bigint) => {
    if (value < 0n) return new Fr(FIELD_MODULUS + value);
    return new Fr(value);
  };
  const ensureFr = (value: unknown) => {
    if (value instanceof Fr) return value;
    if (
      value &&
      typeof (value as { toBigInt?: () => bigint }).toBigInt === 'function'
    )
      return new Fr((value as { toBigInt: () => bigint }).toBigInt());
    return new Fr(BigInt(value as bigint));
  };

  const inputPixelsFr = data.inputPixels.map((p) => toFr(p));
  const labelFr = new Fr(data.label);

  const { result: packedWeightsResult } = await contract.methods
    .get_all_packed_weights()
    .simulate({ from: accountAddress });
  const { result: packedBiasesResult } = await contract.methods
    .get_packed_biases()
    .simulate({ from: accountAddress });

  const currentPackedWeightsFr = (
    Array.isArray(packedWeightsResult)
      ? packedWeightsResult
      : [packedWeightsResult]
  ).map((w) => ensureFr(w));
  const currentPackedBiasesFr = (
    Array.isArray(packedBiasesResult)
      ? packedBiasesResult
      : [packedBiasesResult]
  ).map((b) => ensureFr(b));

  const tx = contract.methods.submit_training_input(
    inputPixelsFr,
    labelFr,
    currentPackedWeightsFr,
    currentPackedBiasesFr
  );

  // PROPOSED status confirms sequencer accepted the tx without waiting for proof
  const sendResult = await tx.send({
    from: accountAddress,
    fee: { paymentMethod },
    wait: { waitForStatus: TxStatus.PROPOSED, timeout: 120 },
  });

  const rec =
    (
      sendResult as {
        receipt?: {
          txHash?: { toString?: () => string };
          blockNumber?: number;
        };
      }
    ).receipt ??
    (sendResult as {
      txHash?: { toString?: () => string };
      blockNumber?: number;
    });
  return {
    success: true,
    txHash: rec.txHash?.toString?.(),
    blockNumber: rec.blockNumber,
  };
}
