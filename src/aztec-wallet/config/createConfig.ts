import { isStringArray } from '../../utils';
import { createEmbeddedConnector } from '../connectors/EmbeddedConnector';
import { createWalletSDKConnector } from '../connectors/WalletSDKConnector';
import {
  DEFAULT_LABELS,
  DEFAULT_MODAL_CONFIG,
  DEFAULT_EMBEDDED_CONFIG,
  DEFAULT_WALLET_SDK_CONFIG,
} from './defaults';
import { getAztecWalletPreset, AZTEC_WALLET_PRESETS } from './walletPresets';
import type { ConnectorFactory } from '../connectors/registry';
import type {
  AztecWalletConfig,
  AztecBrowserWalletConfig,
  AztecWalletsGroupConfig,
  EmbeddedGroupConfig,
  WalletSDKGroupConfig,
  EVMWalletsGroupConfig,
  ResolvedAztecWalletConfig,
} from '../types';

/**
 * Check if value is a full group config (has 'wallets' property)
 */
function isFullGroupConfig(
  value: unknown
): value is { wallets: unknown[]; label?: string } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'wallets' in value &&
    Array.isArray((value as { wallets: unknown }).wallets)
  );
}

/**
 * Resolve Aztec wallet IDs to full wallet configs
 */
function resolveAztecWallets(ids: string[]): AztecBrowserWalletConfig[] {
  const wallets: AztecBrowserWalletConfig[] = [];

  for (const id of ids) {
    const preset = getAztecWalletPreset(id);
    if (preset) {
      wallets.push({
        id: preset.id,
        name: preset.name,
        icon: preset.icon,
        adapter: preset.getAdapter,
        checkInstalled: preset.checkInstalled,
      });
    } else {
      const available = Object.keys(AZTEC_WALLET_PRESETS).join(', ');
      console.warn(
        `AztecWallet: Unknown Aztec wallet "${id}". Available: ${available}`
      );
    }
  }

  return wallets;
}

/**
 * Create connector factories from resolved wallet groups.
 * This is called internally by createAztecWalletConfig.
 */
function createConnectorsFromWalletGroups(walletGroups: {
  embedded: EmbeddedGroupConfig | false;
  walletSdk: WalletSDKGroupConfig | false;
  aztecWallets: AztecWalletsGroupConfig | false;
  evmWallets: EVMWalletsGroupConfig | false;
}): ConnectorFactory[] {
  const connectors: ConnectorFactory[] = [];

  // Add embedded connector
  if (walletGroups.embedded !== false) {
    connectors.push(createEmbeddedConnector);
  }

  // Add wallet-sdk connector
  if (walletGroups.walletSdk !== false) {
    connectors.push(createWalletSDKConnector);
  }

  // Legacy aztecWallets group is kept for backward compatibility but unused

  return connectors;
}

/**
 * Create an AztecWallet configuration with defaults applied
 *
 * @param config - User configuration
 * @returns Resolved configuration with all defaults applied
 *
 * @example Simple config (recommended)
 * ```ts
 * const config = createAztecWalletConfig({
 *   networks: [{ name: 'testnet', nodeUrl: '...' }],
 *   walletGroups: {
 *     embedded: true,
 *     walletSdk: true,
 *   },
 * });
 * ```
 */
export function createAztecWalletConfig(
  config: AztecWalletConfig
): ResolvedAztecWalletConfig {
  const { walletGroups } = config;

  // Resolve embedded config
  let resolvedEmbedded: ResolvedAztecWalletConfig['walletGroups']['embedded'];
  if (walletGroups.embedded === false) {
    resolvedEmbedded = false;
  } else if (walletGroups.embedded === true || !walletGroups.embedded) {
    resolvedEmbedded = { ...DEFAULT_EMBEDDED_CONFIG };
  } else {
    resolvedEmbedded = {
      ...DEFAULT_EMBEDDED_CONFIG,
      ...walletGroups.embedded,
    };
  }

  // Resolve wallet-sdk config
  let resolvedWalletSdk: WalletSDKGroupConfig | false;
  if (walletGroups.walletSdk === false) {
    resolvedWalletSdk = false;
  } else if (walletGroups.walletSdk === true || !walletGroups.walletSdk) {
    resolvedWalletSdk = { ...DEFAULT_WALLET_SDK_CONFIG };
  } else {
    resolvedWalletSdk = {
      ...DEFAULT_WALLET_SDK_CONFIG,
      ...(walletGroups.walletSdk as WalletSDKGroupConfig),
    };
  }

  // Resolve legacy Aztec wallets config
  let resolvedAztecWallets: AztecWalletsGroupConfig | false;
  if (walletGroups.aztecWallets === false || !walletGroups.aztecWallets) {
    resolvedAztecWallets = false;
  } else if (isStringArray(walletGroups.aztecWallets)) {
    const wallets = resolveAztecWallets(walletGroups.aztecWallets);
    resolvedAztecWallets =
      wallets.length > 0
        ? { label: DEFAULT_LABELS.aztecWallets, wallets }
        : false;
  } else if (isFullGroupConfig(walletGroups.aztecWallets)) {
    resolvedAztecWallets = {
      label: walletGroups.aztecWallets.label ?? DEFAULT_LABELS.aztecWallets,
      wallets: walletGroups.aztecWallets.wallets,
    };
  } else {
    resolvedAztecWallets = false;
  }

  // EVM wallets stripped for HIVE
  const resolvedEvmWallets: EVMWalletsGroupConfig | false = false;

  const resolvedWalletGroups: ResolvedAztecWalletConfig['walletGroups'] = {
    embedded: resolvedEmbedded,
    walletSdk: resolvedWalletSdk,
    aztecWallets: resolvedAztecWallets,
    evmWallets: resolvedEvmWallets,
  };

  const connectors = createConnectorsFromWalletGroups(resolvedWalletGroups);

  return {
    ...config,
    defaultNetwork: config.defaultNetwork ?? config.networks[0]?.name,
    modal: {
      ...DEFAULT_MODAL_CONFIG,
      ...config.modal,
    },
    walletGroups: resolvedWalletGroups,
    connectors,
  };
}
