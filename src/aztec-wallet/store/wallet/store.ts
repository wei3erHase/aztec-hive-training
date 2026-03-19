import { create } from 'zustand';
import type { AccountWithSecretKey } from '@aztec/aztec.js/account';
import { AztecAddress } from '@aztec/aztec.js/addresses';
import type { AppCapabilities, GrantedAccountsCapability, Wallet } from '@aztec/aztec.js/wallet';
import type { WalletProvider } from '@aztec/wallet-sdk/manager';
import { WALLET_SDK_CONNECTOR_ID } from '../../connectors/WalletSDKConnector';
import { NetworkService } from '../../services/aztec/network';
import { WalletType } from '../../types/aztec';
import { getNetworkStore } from '../network';
import { getDeploymentConfig } from '../../../config/contracts';
import { createEmbeddedActions } from './actions/embedded';
import { isValidPXETransition } from './types';
import type { WalletStore, WalletState } from './types';
import type {
  WalletConnector,
  WalletConnectorId,
} from '../../../types/walletConnector';

export type { WalletStore, PXEStatus, NetworkStatus } from './types';

const WALLET_CONNECTION_STORAGE_KEY = 'aztec-wallet-connection';

/**
 * Builds a capability manifest scoped to the contracts deployed on the given
 * network. Simulation and transaction scopes are declared upfront so the wallet
 * only shows one permission dialog instead of per-operation popups.
 */
function buildCapabilityManifest(networkId: string): AppCapabilities {
  const deploymentConfig = getDeploymentConfig(networkId);
  const { singleLayer, multiLayerPerceptron, cnnGap } =
    deploymentConfig.contracts;

  const contractAddresses = [
    singleLayer.address,
    multiLayerPerceptron.address,
    cnnGap.address,
  ]
    .filter((addr): addr is string => !!addr)
    .map((addr) => AztecAddress.fromString(addr));

  const capabilities: AppCapabilities['capabilities'] = [
    { type: 'accounts', canGet: true },
  ];

  if (contractAddresses.length > 0) {
    capabilities.push({
      type: 'contracts',
      contracts: contractAddresses,
      canRegister: true,
    });

    // Unconstrained (utility) reads — auto-approved after initial grant
    capabilities.push({
      type: 'simulation',
      utilities: {
        scope: contractAddresses.flatMap((addr) => [
          { contract: addr, function: 'get_all_packed_weights' },
          { contract: addr, function: 'get_packed_biases' },
        ]),
      },
    });

    // Transaction submissions — wallet will still prompt per-tx, but
    // declaring scope lets it show a meaningful name instead of raw calldata
    capabilities.push({
      type: 'transaction',
      scope: contractAddresses.flatMap((addr) => [
        { contract: addr, function: 'submit_training_input' },
      ]),
    });
  }

  return {
    version: '1.0' as const,
    metadata: {
      name: 'Hive Neural Network',
      version: '1.0.0',
      description: 'Train neural networks on Aztec',
      url: typeof window !== 'undefined' ? window.location.origin : '',
    },
    capabilities,
  };
}

interface StoredWalletConnection {
  connectorId: WalletConnectorId;
  walletType: WalletType;
}

const saveWalletConnection = (data: StoredWalletConnection): void => {
  try {
    localStorage.setItem(WALLET_CONNECTION_STORAGE_KEY, JSON.stringify(data));
  } catch {
    // Ignore storage errors
  }
};

export const clearWalletConnection = (): void => {
  try {
    localStorage.removeItem(WALLET_CONNECTION_STORAGE_KEY);
  } catch {
    // Ignore
  }
};

export const getStoredWalletConnection = (): StoredWalletConnection | null => {
  try {
    const data = localStorage.getItem(WALLET_CONNECTION_STORAGE_KEY);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
};

const INITIAL_STATE: WalletState = {
  account: null,
  walletType: null,
  status: 'disconnected',
  error: null,
  networkStatus: 'idle',
  networkError: null,
  pxeStatus: 'idle',
  pxeError: null,
  signerType: null,
  connectedRdns: null,
  sdkWallet: null,
  connectors: [],
  activeConnectorId: null,
  connectingConnectorId: null,
};

export const useWalletStore = create<WalletStore>((set, get) => ({
  ...INITIAL_STATE,

  // Shared connect orchestration
  _connectWith: async <T>(
    connectorId: WalletConnectorId,
    run: (connector: WalletConnector) => Promise<T>
  ) => {
    const connector = get().connectors.find((item) => item.id === connectorId);
    if (!connector) {
      set({
        error: `Connector "${connectorId}" not found`,
        status: 'disconnected',
        connectingConnectorId: null,
      });
      throw new Error(`Connector "${connectorId}" not found`);
    }

    if (get().connectingConnectorId === connectorId) {
      throw new Error(`Connector "${connectorId}" is already connecting`);
    }

    set({
      connectingConnectorId: connectorId,
      error: null,
    });

    try {
      const result = await run(connector);
      set({
        activeConnectorId: connector.id,
        walletType: connector.type,
        status: 'connected',
        connectingConnectorId: null,
      });
      saveWalletConnection({
        connectorId: connector.id,
        walletType: connector.type,
      });
      return result;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to connect to wallet';
      set({
        status: 'disconnected',
        error: message,
        activeConnectorId: null,
        walletType: null,
        connectingConnectorId: null,
      });
      throw err;
    }
  },

  _disconnectWith: async (cleanup?: () => Promise<void> | void) => {
    const { activeConnectorId, connectors } = get();
    if (!activeConnectorId) return;

    try {
      if (cleanup) {
        await cleanup();
      }
    } finally {
      clearWalletConnection();
      set({
        account: null,
        walletType: null,
        status: 'disconnected',
        error: null,
        signerType: null,
        connectedRdns: null,
        sdkWallet: null,
        activeConnectorId: null,
        connectingConnectorId: null,
        pxeStatus: 'idle',
        pxeError: null,
        connectors,
      });
    }
  },

  setConnectors: (connectors) => set({ connectors }),

  /**
   * Generic connect method that delegates to the connector.
   *
   * NOTE: This does NOT wrap in `_connectWith` because each connector's
   * `connect()` method is responsible for calling the appropriate store action
   * (e.g., `connectEmbedded`, `connectExternalSigner`) which already uses
   * `_connectWith` internally. This avoids double-wrapping.
   */
  connect: async (connectorId) => {
    const connector = get().connectors.find((c) => c.id === connectorId);
    if (!connector) {
      set({
        error: `Connector "${connectorId}" not found`,
        status: 'disconnected',
      });
      throw new Error(`Connector "${connectorId}" not found`);
    }
    await connector.connect();
  },

  // Embedded actions
  ...createEmbeddedActions(set, get),

  // Wallet SDK actions
  connectWalletSDK: async (
    wallet: Wallet,
    provider: WalletProvider,
    connectorId: WalletConnectorId = WALLET_SDK_CONNECTOR_ID
  ): Promise<AccountWithSecretKey> => {
    const connectWith = get()._connectWith;
    return connectWith(connectorId, async () => {
      set({ status: 'connecting', error: null });

      const { currentConfig } = getNetworkStore();
      const networkId = currentConfig.name;
      const manifest = buildCapabilityManifest(networkId);

      // Primary flow: requestCapabilities() shows the unified permission dialog
      // and returns granted accounts — use that instead of calling getAccounts() directly
      // (which triggers a separate broken dialog on multi-account wallets).
      let accounts: { item: AztecAddress; alias?: string }[] = [];
      try {
        const capabilities = await wallet.requestCapabilities(manifest);
        const accountsCap = capabilities.granted.find(
          (c): c is GrantedAccountsCapability => c.type === 'accounts'
        );
        accounts = accountsCap?.accounts ?? [];
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes('Network no longer exists')) {
          throw new Error(
            'The Aztec testnet is not configured in your wallet. ' +
              'Please open Azguard, go to Networks, and add the testnet ' +
              'using https://rpc.testnet.aztec-labs.com/ as the RPC URL.'
          );
        }
        // Other errors: fall through to getAccounts() fallback below
        console.warn('[connectWalletSDK] requestCapabilities failed:', msg);
      }

      // Fallback for wallets that don't support capabilities
      if (accounts.length === 0) {
        const raw = await wallet.getAccounts();
        accounts = raw.map((a) => {
          // Aliased<AztecAddress> can have .item or .address
          const inner =
            (a as unknown as { item?: AztecAddress; address?: AztecAddress })
              .item ??
            (a as unknown as { address?: AztecAddress }).address ??
            (a as unknown as AztecAddress);
          return { item: inner };
        });
      }

      const primary = accounts[0];
      if (!primary) {
        throw new Error('Wallet returned no accounts');
      }

      const address = primary.item;
      const fakeAccount = {
        getAddress: () => address,
      } as unknown as AccountWithSecretKey;

      // Store the SDK wallet for later use (e.g., sending transactions)
      const connector = get().connectors.find((c) => c.id === connectorId);
      if (connector && 'setupWallet' in connector) {
        (
          connector as import('../../connectors/WalletSDKConnector').WalletSDKConnector
        ).setupWallet(provider, wallet);
      }

      set({
        account: fakeAccount,
        sdkWallet: wallet,
        walletType: WalletType.WALLET_SDK,
      });

      return fakeAccount;
    });
  },

  // Shared actions
  disconnect: async (cleanup?: () => Promise<void> | void) => {
    await get()._disconnectWith(cleanup);
  },

  setError: (error) => set({ error }),

  setNetworkStatus: (networkStatus, networkError = null) => {
    set({ networkStatus, networkError });
  },

  setPXEStatus: (pxeStatus, pxeError = null) => {
    const currentStatus = get().pxeStatus;
    if (!isValidPXETransition(currentStatus, pxeStatus)) {
      console.warn(
        `Invalid PXE state transition: ${currentStatus} → ${pxeStatus}`
      );
    }
    set({ pxeStatus, pxeError });
  },

  checkNetwork: async () => {
    const { networkStatus } = get();
    if (networkStatus === 'checking' || networkStatus === 'available') {
      return;
    }

    const { currentConfig } = getNetworkStore();
    set({ networkStatus: 'checking', networkError: null });

    try {
      await NetworkService.checkAvailability(currentConfig.nodeUrl);
      set({ networkStatus: 'available' });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Failed to connect to network';
      set({ networkStatus: 'error', networkError: message });
    }
  },

  reset: () =>
    set((state) => ({
      ...INITIAL_STATE,
      connectors: state.connectors,
    })),

  syncFromStorage: async () => {
    const { activeConnectorId, connectExistingEmbedded } = get();
    const stored = getStoredWalletConnection();

    if (stored && !activeConnectorId) {
      if (stored.walletType === 'embedded') {
        try {
          await connectExistingEmbedded(stored.connectorId);
        } catch {
          // Silent fail - user can manually reconnect
        }
      } else {
        // SDK wallets (and any future types) cannot be auto-reconnected.
        // Clear the stale entry so the user starts from a clean disconnected state.
        clearWalletConnection();
      }
    } else if (!stored && activeConnectorId) {
      // Another tab disconnected - clear our state
      clearWalletConnection();
      set((state) => ({
        ...INITIAL_STATE,
        connectors: state.connectors,
      }));
    }
  },
}));

export const getWalletStore = () => useWalletStore.getState();

let isWalletListenerSetup = false;

export const setupWalletCrossTabSync = (): void => {
  if (isWalletListenerSetup) return;
  isWalletListenerSetup = true;

  window.addEventListener('storage', (e: StorageEvent) => {
    if (e.key === WALLET_CONNECTION_STORAGE_KEY) {
      getWalletStore().syncFromStorage();
    }
  });
};
