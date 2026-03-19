import type { AccountWithSecretKey } from '@aztec/aztec.js/account';
import type { Wallet } from '@aztec/aztec.js/wallet';
import type { WalletProvider } from '@aztec/wallet-sdk/manager';
import type {
  ConnectionStatus,
  WalletConnector,
  WalletConnectorId,
} from '../../../types/walletConnector';
import type { ExternalSignerType, WalletType } from '../../types/aztec';

export type PXEStatus = 'idle' | 'initializing' | 'ready' | 'error';
export type NetworkStatus = 'idle' | 'checking' | 'available' | 'error';

/**
 * Valid PXE state transitions:
 * - idle → initializing (starting PXE init)
 * - initializing → ready (PXE init success)
 * - initializing → error (PXE init failed)
 * - ready → idle (disconnect/reset)
 * - error → idle (disconnect/reset)
 * - error → initializing (retry)
 */
export const VALID_PXE_TRANSITIONS: Record<PXEStatus, PXEStatus[]> = {
  idle: ['initializing'],
  initializing: ['ready', 'error'],
  ready: ['idle'],
  error: ['idle', 'initializing'],
};

export const isValidPXETransition = (
  from: PXEStatus,
  to: PXEStatus
): boolean => {
  if (from === to) return true;
  return VALID_PXE_TRANSITIONS[from]?.includes(to) ?? false;
};

export type WalletState = {
  // Core state (always relevant)
  account: AccountWithSecretKey | null;
  walletType: WalletType | null;
  status: ConnectionStatus;
  error: string | null;

  // Network availability check
  networkStatus: NetworkStatus;
  networkError: string | null;

  // PXE initialization flags (services stay in singleton)
  pxeStatus: PXEStatus;
  pxeError: string | null;

  // ExternalSigner-specific (only read when walletType === EXTERNAL_SIGNER)
  signerType: ExternalSignerType | null;
  connectedRdns: string | null;

  // WalletSDK-specific (only set when walletType === WALLET_SDK)
  sdkWallet: Wallet | null;

  // Connector management
  connectors: WalletConnector[];
  activeConnectorId: WalletConnectorId | null;
  connectingConnectorId: WalletConnectorId | null;
};

export type WalletActions = {
  _connectWith: <T>(
    connectorId: WalletConnectorId,
    run: (connector: WalletConnector) => Promise<T>
  ) => Promise<T>;
  _disconnectWith: (cleanup?: () => Promise<void> | void) => Promise<void>;
  // Connector management
  setConnectors: (connectors: WalletConnector[]) => void;
  connect: (connectorId: WalletConnectorId) => Promise<void>;

  // Embedded
  connectEmbedded: (
    connectorId?: WalletConnectorId
  ) => Promise<AccountWithSecretKey>;
  connectExistingEmbedded: (
    connectorId?: WalletConnectorId
  ) => Promise<AccountWithSecretKey | null>;
  hasSavedEmbeddedAccount: () => boolean;

  // Wallet SDK
  connectWalletSDK: (
    wallet: Wallet,
    provider: WalletProvider,
    connectorId?: WalletConnectorId
  ) => Promise<AccountWithSecretKey>;

  // Shared
  disconnect: (cleanup?: () => Promise<void> | void) => Promise<void>;
  setError: (error: string | null) => void;
  setNetworkStatus: (status: NetworkStatus, error?: string | null) => void;
  setPXEStatus: (status: PXEStatus, error?: string | null) => void;
  checkNetwork: () => Promise<void>;
  reset: () => void;
  syncFromStorage: () => Promise<void>;
};

export type WalletStore = WalletState & WalletActions;

// Helper types for action creators
export type SetState = (
  partial: Partial<WalletStore> | ((state: WalletStore) => Partial<WalletStore>)
) => void;

export type GetState = () => WalletStore;
