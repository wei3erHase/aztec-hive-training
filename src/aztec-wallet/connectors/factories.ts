import { createEmbeddedConnector } from './EmbeddedConnector';
import { createWalletSDKConnector } from './WalletSDKConnector';
import type { ConnectorFactory } from './registry';

/**
 * Embedded wallet connector preset.
 * Uses app-managed PXE with internal signing.
 * Usage: connectors: [embedded()]
 */
export const embedded = (): ConnectorFactory => createEmbeddedConnector;

/**
 * Wallet SDK connector preset.
 * Discovers any Aztec extension wallet that implements the SDK protocol.
 * Usage: connectors: [walletSdk()]
 */
export const walletSdk = (): ConnectorFactory => createWalletSDKConnector;
