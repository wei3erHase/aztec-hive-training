import type { EmbeddedGroupConfig, WalletSDKGroupConfig } from '../types';

/**
 * Default labels for wallet groups
 */
export const DEFAULT_LABELS = {
  embedded: 'Embedded Wallet',
  walletSdk: 'Aztec Wallet',
  aztecWallets: 'Aztec Wallet',
  evmWallets: 'EVM Wallet',
} as const;

/**
 * Default modal configuration
 */
export const DEFAULT_MODAL_CONFIG = {
  title: 'Connect Wallet',
  subtitle:
    'Choose how you want to connect. Each option offers a different balance of convenience and security.',
} as const;

/**
 * Default embedded wallet configuration
 */
export const DEFAULT_EMBEDDED_CONFIG: EmbeddedGroupConfig = {
  label: DEFAULT_LABELS.embedded,
  enabled: true,
};

/**
 * Default wallet-sdk configuration
 */
export const DEFAULT_WALLET_SDK_CONFIG: WalletSDKGroupConfig = {
  label: DEFAULT_LABELS.walletSdk,
};

/**
 * App identifier sent to Aztec Wallet SDK during discovery and handshake.
 * Must be consistent across getAvailableWallets() and establishSecureChannel().
 */
export const WALLET_SDK_APP_ID = 'hive-neural-network' as const;
