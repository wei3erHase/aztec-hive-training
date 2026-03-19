/**
 * Wallet Presets Registry
 *
 * Pre-configured wallet definitions for legacy Aztec browser wallets.
 * For new integrations, prefer the walletSdk group which uses the
 * Aztec Wallet SDK protocol for automatic wallet discovery.
 */

import type { IBrowserWalletAdapter } from '../../types/browserWallet';
import type { IconType } from '../types';

// =============================================================================
// Types
// =============================================================================

export interface EVMWalletPreset {
  id: string;
  name: string;
  icon: IconType;
  rdns: string;
}

export interface AztecWalletPreset {
  id: string;
  name: string;
  icon: IconType;
  /** Lazy adapter factory - only imported when needed (async for dynamic imports) */
  getAdapter: () => Promise<IBrowserWalletAdapter>;
  /** Check if wallet extension is installed (optional, async) */
  checkInstalled?: () => Promise<boolean>;
}

// =============================================================================
// EVM Wallet Presets (stripped for HIVE)
// =============================================================================

export const EVM_WALLET_PRESETS: Record<string, EVMWalletPreset> = {};

// =============================================================================
// Aztec Wallet Presets (legacy - kept for backward compat)
// =============================================================================

export const AZTEC_WALLET_PRESETS: Record<string, AztecWalletPreset> = {
  // Add legacy browser wallet presets here if needed
  // New integrations should use walletSdk group instead
};

// =============================================================================
// Helper functions
// =============================================================================

export function getEVMWalletPreset(id: string): EVMWalletPreset | undefined {
  return EVM_WALLET_PRESETS[id];
}

export function getAztecWalletPreset(
  id: string
): AztecWalletPreset | undefined {
  return AZTEC_WALLET_PRESETS[id];
}

export function getAvailableEVMWalletIds(): string[] {
  return Object.keys(EVM_WALLET_PRESETS);
}

export function getAvailableAztecWalletIds(): string[] {
  return Object.keys(AZTEC_WALLET_PRESETS);
}

export type EVMWalletId = keyof typeof EVM_WALLET_PRESETS;
export type AztecWalletId = keyof typeof AZTEC_WALLET_PRESETS;
