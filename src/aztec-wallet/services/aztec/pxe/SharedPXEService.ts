import { AztecAddress } from '@aztec/aztec.js/addresses';
import { getContractInstanceFromInstantiationParams } from '@aztec/aztec.js/contracts';
import { SponsoredFeePaymentMethod } from '@aztec/aztec.js/fee';
import { Fr } from '@aztec/aztec.js/fields';
import { createLogger } from '@aztec/aztec.js/log';
import type { AztecNode } from '@aztec/aztec.js/node';
import { SPONSORED_FPC_SALT } from '@aztec/constants';
import { createStore } from '@aztec/kv-store/indexeddb';
import { SponsoredFPCContractArtifact } from '@aztec/noir-contracts.js/SponsoredFPC';
import { createPXE } from '@aztec/pxe/client/bundle';
import { getPXEConfig } from '@aztec/pxe/config';
import type { PXE } from '@aztec/pxe/server';
import { getDeploymentConfig } from '../../../../config/contracts';
import { AVAILABLE_NETWORKS } from '../../../../config/networks';
import { FeePaymentRegister } from '../../../../services/aztec/feePayment/FeePaymentRegister';
import { ZKMLContractRegister } from '../../../../services/aztec/zkml/ZKMLContractRegister';
import { MinimalWallet } from '../../../../utils/MinimalWallet';
import { NetworkService } from '../network';
import { AztecStorageService } from '../storage';
import type { AztecNetwork } from '../../../../config/networks/constants';

const logger = createLogger('shared-pxe-service');
const pxeLogger = createLogger('pxe');
const getProverEnabled = (networkName: AztecNetwork): boolean => {
  const networkConfig = AVAILABLE_NETWORKS.find((n) => n.name === networkName);
  if (!networkConfig) {
    throw new Error(`Network configuration not found for: ${networkName}`);
  }
  return networkConfig.proverEnabled;
};

export interface SharedPXEInstance {
  pxe: PXE;
  aztecNode: AztecNode;
  wallet: MinimalWallet;
  storageService: AztecStorageService;
  getSponsoredFeePaymentMethod: () => Promise<SponsoredFeePaymentMethod>;
}

interface PXEInstanceEntry {
  instance: SharedPXEInstance;
  nodeUrl: string;
  networkName: AztecNetwork;
}

/**
 * SharedPXEService manages PXE instances across the application.
 * Uses singleton pattern with lazy initialization.
 */
class SharedPXEServiceClass {
  private instances: Map<string, PXEInstanceEntry> = new Map();
  private initPromises: Map<string, Promise<SharedPXEInstance>> = new Map();
  private cachedPaymentMethods: Map<string, SponsoredFeePaymentMethod> =
    new Map();
  private storePromises: Map<
    string,
    Promise<Awaited<ReturnType<typeof createStore>>>
  > = new Map();
  private static readonly PERSISTED_STORE_KB = 5e5; // 500MB
  private static readonly FALLBACK_STORE_KB = 1e5; // 100MB

  /**
   * Get the current network's PXE instance.
   * Convenience method for callers that have network config.
   */
  async getCurrentInstance(
    nodeUrl: string,
    networkName: AztecNetwork
  ): Promise<SharedPXEInstance> {
    return this.getInstance(nodeUrl, networkName);
  }

  /**
   * Get or create a PXE instance for a specific network.
   * If already initializing, returns the same promise to avoid duplicate initialization.
   */
  async getInstance(
    nodeUrl: string,
    networkName: AztecNetwork
  ): Promise<SharedPXEInstance> {
    const normalizedNodeUrl = this.normalizeNodeUrl(nodeUrl);
    const key = this.getInstanceKey(networkName);

    // Return existing instance if available
    const existing = this.instances.get(key);
    if (existing) {
      // Reuse if URL matches, else replace stale one for this network
      if (existing.nodeUrl === normalizedNodeUrl) {
        return existing.instance;
      }
      this.clearInstance(existing.networkName);
    }

    // Return in-progress initialization if exists
    const initPromise = this.initPromises.get(key);
    if (initPromise) {
      return initPromise;
    }

    // Start new initialization
    const promise = this.initializeInstance(
      normalizedNodeUrl,
      networkName,
      key
    );
    this.initPromises.set(key, promise);

    try {
      const instance = await promise;
      return instance;
    } finally {
      this.initPromises.delete(key);
    }
  }

  /**
   * Check if a PXE instance is initialized for a network
   */
  isInitialized(networkName: AztecNetwork): boolean {
    const key = this.getInstanceKey(networkName);
    return this.instances.has(key);
  }

  /**
   * Check if initialization is in progress for a network
   */
  isInitializing(networkName: AztecNetwork): boolean {
    const key = this.getInstanceKey(networkName);
    return this.initPromises.has(key);
  }

  /**
   * Get existing instance without initialization (returns null if not initialized)
   */
  getExistingInstance(networkName: AztecNetwork): SharedPXEInstance | null {
    const key = this.getInstanceKey(networkName);
    return this.instances.get(key)?.instance ?? null;
  }

  /**
   * Clear PXE store for a network (used for reorg recovery).
   * Clears cached instance; next getInstance will reinitialize.
   */
  async clearPXEStore(networkName: string): Promise<void> {
    this.clearInstance(networkName as AztecNetwork);
  }

  /**
   * Clear a specific PXE instance (useful for network switching)
   */
  clearInstance(networkName: AztecNetwork): void {
    const key = this.getInstanceKey(networkName);
    this.instances.delete(key);
    this.cachedPaymentMethods.delete(key);
    logger.info(`Cleared PXE instance for ${networkName}`);
  }

  /**
   * Clear all PXE instances
   */
  clearAll(): void {
    this.instances.clear();
    this.cachedPaymentMethods.clear();
    logger.info('Cleared all PXE instances');
  }

  private getInstanceKey(networkName: AztecNetwork | string): string {
    return `${networkName}`;
  }

  private getFeePaymentConfig(networkName: string) {
    const networkConfig = AVAILABLE_NETWORKS.find(
      (n) => n.name === networkName
    );
    return networkConfig?.feePaymentContracts;
  }

  private getZKMLConfig(networkName: string) {
    const networkConfig = AVAILABLE_NETWORKS.find(
      (n) => n.name === networkName
    );
    return networkConfig?.zkmlContracts;
  }

  private normalizeNodeUrl(nodeUrl: string): string {
    if (!nodeUrl) {
      return nodeUrl;
    }
    return nodeUrl.endsWith('/') ? nodeUrl.slice(0, -1) : nodeUrl;
  }

  private async initializeInstance(
    nodeUrl: string,
    networkName: AztecNetwork,
    key: string
  ): Promise<SharedPXEInstance> {
    logger.info(`Initializing PXE for network: ${networkName}`);

    const aztecNode = NetworkService.getNodeClient(nodeUrl);

    // Get L1 contracts for network-specific database
    const l1Contracts = await aztecNode.getL1ContractAddresses();
    const storeName = `aztec-pxe-${networkName}`;

    // Reuse a single store per network
    const pxeStore = await this.getOrCreateStore(networkName, storeName);

    const config = getPXEConfig();
    config.l1Contracts = l1Contracts;
    config.proverEnabled = getProverEnabled(networkName);

    const pxe = await createPXE(aztecNode, config, {
      store: pxeStore,
    });

    const wallet = new MinimalWallet(pxe, aztecNode);

    // Register the SponsoredFPC artifact so the PXE can resolve its function
    // selectors during fee payment simulation (e.g. when deploying an account).
    await this.registerSponsoredFPC(pxe, aztecNode, networkName);

    // Register fee payment contracts (look up config by network name)
    const feePaymentConfig = this.getFeePaymentConfig(networkName);
    const feePaymentRegister = new FeePaymentRegister();
    await feePaymentRegister.registerAll(pxe, feePaymentConfig);

    // Register ZKML contracts when configured (populated after deployment)
    const zkmlConfig = this.getZKMLConfig(networkName);
    const zkmlRegister = new ZKMLContractRegister();
    await zkmlRegister.registerAll(pxe, zkmlConfig);

    // Pre-register all deployed ZKML contracts with their artifacts so that
    // the first predict() call does not need to do IndexedDB writes while the
    // block-stream processor is also writing (causing "transaction not active").
    await this.preRegisterDeployedZKML(pxe, aztecNode, networkName);

    // Initialize storage service
    const storageService = new AztecStorageService();

    // Register saved senders
    await this.registerSavedSenders(pxe, storageService);

    const nodeInfo = await aztecNode.getNodeInfo();
    logger.info(`PXE connected to ${networkName}`, nodeInfo);

    const instance: SharedPXEInstance = {
      pxe,
      aztecNode,
      wallet,
      storageService,
      getSponsoredFeePaymentMethod: () =>
        this.getSponsoredFeePaymentMethod(key, aztecNode, networkName),
    };

    this.instances.set(key, {
      instance,
      nodeUrl,
      networkName,
    });

    return instance;
  }

  private async getOrCreateStore(
    networkName: AztecNetwork,
    storeName: string
  ): Promise<Awaited<ReturnType<typeof createStore>>> {
    const existingPromise = this.storePromises.get(networkName);
    if (existingPromise) {
      return existingPromise;
    }

    const createPromise = this.createPXEStoreWithFallback(storeName);
    this.storePromises.set(networkName, createPromise);
    return createPromise;
  }

  private async createPXEStoreWithFallback(
    storeName: string
  ): Promise<Awaited<ReturnType<typeof createStore>>> {
    try {
      return await createStore(
        storeName,
        {
          dataDirectory: 'pxe',
          dataStoreMapSizeKb: SharedPXEServiceClass.PERSISTED_STORE_KB,
        },
        undefined,
        pxeLogger
      );
    } catch (error) {
      logger.warn(
        `Failed to create persistent PXE store (limit ${
          SharedPXEServiceClass.PERSISTED_STORE_KB
        } KB). Retrying with smaller ephemeral store.`,
        { error }
      );

      return await createStore(
        `${storeName}-tmp`,
        {
          dataDirectory: 'pxe-tmp',
          dataStoreMapSizeKb: SharedPXEServiceClass.FALLBACK_STORE_KB,
        },
        undefined,
        pxeLogger
      );
    }
  }

  private async registerSponsoredFPC(
    pxe: PXE,
    aztecNode: AztecNode,
    networkName: AztecNetwork
  ): Promise<void> {
    const instance = await this.getSponsoredPFCInstance(aztecNode, networkName);
    await pxe.registerContract({
      instance,
      artifact: SponsoredFPCContractArtifact,
    });
    logger.info(`Registered SponsoredFPC at ${instance.address}`);
  }

  /**
   * Returns the SponsoredFPC contract instance.
   *
   * For networks that declare a `sponsoredFpcAddress`, the instance is fetched
   * directly from the node — the salt never appears in client-side code.
   * For local-network (no address configured), falls back to computing it from
   * the canonical salt (0).
   */
  private async getSponsoredPFCInstance(
    aztecNode: AztecNode,
    networkName: AztecNetwork
  ) {
    const networkConfig = AVAILABLE_NETWORKS.find(
      (n) => n.name === networkName
    );

    if (networkConfig?.sponsoredFpcAddress) {
      const address = AztecAddress.fromString(
        networkConfig.sponsoredFpcAddress
      );
      const instance = await aztecNode.getContract(address);
      if (!instance) {
        throw new Error(
          `SponsoredFPC instance not found on node at ${networkConfig.sponsoredFpcAddress}`
        );
      }
      return instance;
    }

    return await getContractInstanceFromInstantiationParams(
      SponsoredFPCContractArtifact,
      { salt: new Fr(SPONSORED_FPC_SALT) }
    );
  }

  private async getSponsoredFeePaymentMethod(
    key: string,
    aztecNode: AztecNode,
    networkName: AztecNetwork
  ): Promise<SponsoredFeePaymentMethod> {
    const cached = this.cachedPaymentMethods.get(key);
    if (cached) {
      return cached;
    }

    const instance = await this.getSponsoredPFCInstance(aztecNode, networkName);
    const paymentMethod = new SponsoredFeePaymentMethod(instance.address);

    this.cachedPaymentMethods.set(key, paymentMethod);

    return paymentMethod;
  }

  /**
   * Pre-registers all deployed ZKML contracts with the PXE so that the first
   * predict() call does not trigger IndexedDB writes that compete with the
   * running block-stream processor (which would cause "transaction not active"
   * errors and silent prediction hangs).
   *
   * Failures are non-fatal — a contract that is not deployed yet simply has no
   * address and is skipped.
   */
  private async preRegisterDeployedZKML(
    pxe: PXE,
    aztecNode: AztecNode,
    networkName: AztecNetwork
  ): Promise<void> {
    try {
      const config = getDeploymentConfig(networkName);

      const candidates: {
        key: keyof typeof config.contracts;
        getArtifact: () => Promise<unknown>;
      }[] = [
        {
          key: 'singleLayer',
          getArtifact: async () =>
            (await import('../../../../artifacts/SingleLayer'))
              .SingleLayerContractArtifact,
        },
        {
          key: 'multiLayerPerceptron',
          getArtifact: async () =>
            (await import('../../../../artifacts/MultiLayerPerceptron'))
              .MultiLayerPerceptronContractArtifact,
        },
        {
          key: 'cnnGap',
          getArtifact: async () =>
            (await import('../../../../artifacts/CNNGAP'))
              .CNNGAPContractArtifact,
        },
      ];

      for (const { key, getArtifact } of candidates) {
        const contractAddress = config.contracts[key]?.address;
        if (!contractAddress) continue;

        try {
          const address = AztecAddress.fromString(contractAddress);
          let instance = await pxe.getContractInstance(address);
          if (!instance) {
            instance = await aztecNode.getContract(address);
          }
          if (!instance) continue;

          const artifact = await getArtifact();
          await pxe.registerContract({
            instance,
            artifact: artifact as Parameters<
              typeof pxe.registerContract
            >[0]['artifact'],
          });
          logger.info(
            `Pre-registered ZKML contract ${key} at ${contractAddress}`
          );
        } catch (err) {
          logger.warn(`Failed to pre-register ZKML contract ${key}:`, err);
        }
      }
    } catch (err) {
      logger.warn('Failed to pre-register ZKML contracts:', err);
    }
  }

  private async registerSavedSenders(
    pxe: PXE,
    storageService: AztecStorageService
  ): Promise<void> {
    if (typeof window === 'undefined') return; // localStorage not available in Node.js
    try {
      const savedSenders = storageService.getSenders();

      if (savedSenders.length === 0) {
        return;
      }

      logger.info(`Registering ${savedSenders.length} saved senders with PXE`);

      for (const senderAddressString of savedSenders) {
        try {
          const senderAddress = AztecAddress.fromString(senderAddressString);
          await pxe.registerSender(senderAddress);
        } catch {
          // Sender might already be registered
          logger.warn(`Failed to register sender ${senderAddressString}`);
        }
      }
    } catch (error) {
      logger.error('Error registering saved senders:', error);
    }
  }
}

// Export singleton instance
export const SharedPXEService = new SharedPXEServiceClass();
