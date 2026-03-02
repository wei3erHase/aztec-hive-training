/**
 * useTrainOnChain Hook
 * Handles on-chain training transaction submission
 * Uses getSavedAccount and SharedPXEService from aztec-wallet
 */

import { useState, useCallback } from 'react';
import { useAztecWallet } from '../../aztec-wallet/hooks/useAztecWallet';
import { deployAccountIfNotExists } from '../../aztec-wallet/services/wallet/deployAccount';
import { getSavedAccount } from '../../aztec-wallet/services/wallet/embeddedAccount';
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

export const useTrainOnChain = (): UseTrainOnChainReturn => {
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { network } = useAztecWallet();
  const networkId = network?.name ?? 'local-network';
  const nodeUrl = network?.nodeUrl ?? 'http://localhost:8080';

  const submitTraining = useCallback(
    async (data: TrainingTxData): Promise<TrainingTxResult> => {
      setIsPending(true);
      setError(null);

      const saved = getSavedAccount();
      if (!saved) {
        const msg =
          'No embedded wallet found. Please connect your wallet first.';
        setError(msg);
        return { success: false, error: msg };
      }

      const arch: ArchitectureId = data.architecture ?? 'mlp';
      const contractKey = getContractKeyForArchitecture(arch);
      const deploymentConfig = getDeploymentConfig(networkId);
      const contractAddress =
        deploymentConfig.contracts[contractKey].address ?? '';

      if (!contractAddress) {
        const msg = `Contract not configured for ${networkId}`;
        setError(msg);
        return { success: false, error: msg };
      }

      try {
        const [
          { SharedPXEService },
          { AztecAddress },
          { Fr },
          { AccountManager },
          { EcdsaRAccountContract },
          { TxStatus },
        ] = await Promise.all([
          import('../../aztec-wallet/services/aztec/pxe'),
          import('@aztec/aztec.js/addresses'),
          import('@aztec/aztec.js/fields'),
          import('@aztec/aztec.js/wallet'),
          import('@aztec/accounts/ecdsa/lazy'),
          import('@aztec/stdlib/tx'),
        ]);

        let ContractClass: {
          at: (address: unknown, wallet: unknown) => unknown;
        };
        let ContractArtifact: unknown;
        if (arch === 'singleLayer') {
          const m = await import('../../artifacts/SingleLayer');
          ContractClass = m.SingleLayerContract as typeof ContractClass;
          ContractArtifact = m.SingleLayerContractArtifact;
        } else if (arch === 'cnnGap') {
          const m = await import('../../artifacts/CNNGAP');
          ContractClass = m.CNNGAPContract as typeof ContractClass;
          ContractArtifact = m.CNNGAPContractArtifact;
        } else {
          const m = await import('../../artifacts/MultiLayerPerceptron');
          ContractClass =
            m.MultiLayerPerceptronContract as typeof ContractClass;
          ContractArtifact = m.MultiLayerPerceptronContractArtifact;
        }

        const pxeInstance = await SharedPXEService.getCurrentInstance(
          nodeUrl,
          networkId as 'local-network' | 'devnet'
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

          // Deploy account on this network if not yet initialized (e.g. after
          // switching from devnet to local-network — account exists on devnet
          // but must be deployed on local-network before sending transactions).
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
          contractInstance =
            await pxeInstance.aztecNode.getContract(trainingAddress);
        }
        if (contractInstance) {
          await (
            wallet as {
              registerContract: (i: unknown, a: unknown) => Promise<void>;
            }
          ).registerContract(contractInstance, ContractArtifact);
        }

        const paymentMethod = await pxeInstance.getSponsoredFeePaymentMethod();
        const contract = (
          ContractClass as {
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
                get_packed_biases: () => {
                  simulate: (o: unknown) => Promise<unknown>;
                };
              };
            };
          }
        ).at(trainingAddress, wallet);

        const toFr = (value: bigint): InstanceType<typeof Fr> => {
          if (value < 0n) return new Fr(FIELD_MODULUS + value);
          return new Fr(value);
        };

        const inputPixelsFr = data.inputPixels.map((p) => toFr(p));
        const labelFr = new Fr(data.label);

        const packedWeightsResult = await contract.methods
          .get_all_packed_weights()
          .simulate({ from: accountAddress });
        const packedBiasesResult = await contract.methods
          .get_packed_biases()
          .simulate({ from: accountAddress });

        const ensureFr = (value: unknown): InstanceType<typeof Fr> => {
          if (value instanceof Fr) return value;
          if (
            value &&
            typeof (value as { toBigInt?: () => bigint }).toBigInt ===
              'function'
          )
            return new Fr((value as { toBigInt: () => bigint }).toBigInt());
          return new Fr(BigInt(value as bigint));
        };

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

        // In Aztec v4, send() handles waiting internally.
        // Default waitForStatus is CHECKPOINTED which requires a proof on devnet (can take 10-30+ min).
        // PROPOSED is sufficient to confirm the sequencer accepted the tx, and is available immediately.
        const receipt = await tx.send({
          from: accountAddress,
          fee: { paymentMethod },
          wait: { waitForStatus: TxStatus.PROPOSED, timeout: 120 },
        });

        const rec = receipt as {
          txHash?: { toString?: () => string };
          blockNumber?: number;
        };
        const txHash = rec.txHash?.toString?.() ?? 'unknown';
        const blockNumber = rec.blockNumber;

        return { success: true, txHash, blockNumber };
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
    [networkId, nodeUrl]
  );

  return { submitTraining, isPending, error };
};
