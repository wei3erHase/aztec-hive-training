/**
 * WalletSDKConnector - Connector for Aztec Wallet SDK extension wallets.
 *
 * Discovers any wallet that implements the Aztec Wallet SDK protocol.
 * The connection flow is driven by the UI (WalletSDKDiscoveryView →
 * WalletSDKVerificationView) which calls `finalize()` once the user
 * has confirmed the emoji verification hash.
 */

import type { AccountWithSecretKey } from '@aztec/aztec.js/account';
import type { Wallet } from '@aztec/aztec.js/wallet';
import type { WalletProvider } from '@aztec/wallet-sdk/manager';
import { getWalletStore } from '../store/wallet';
import { WalletType } from '../types/aztec';
import type {
  WalletConnector,
  ConnectorStatus,
  WalletConnectorId,
} from '../../types/walletConnector';

export const WALLET_SDK_CONNECTOR_ID = 'wallet-sdk' as const;

/**
 * Connector that integrates Aztec extension wallets via the Wallet SDK protocol.
 *
 * The actual discovery and secure-channel handshake happen in
 * WalletSDKDiscoveryView / WalletSDKVerificationView; this class holds
 * the resulting wallet reference and wires up disconnect events.
 */
export class WalletSDKConnector implements WalletConnector {
  readonly id: WalletConnectorId;
  readonly label: string;
  readonly type = WalletType.WALLET_SDK;

  private _wallet: Wallet | null = null;
  private _provider: WalletProvider | null = null;
  private _unsubDisconnect: (() => void) | null = null;

  constructor(
    id: string = WALLET_SDK_CONNECTOR_ID,
    label: string = 'Aztec Wallet'
  ) {
    this.id = id;
    this.label = label;
  }

  /**
   * Called by the UI after the full SDK flow completes (discovery → secure
   * channel → emoji verification → confirm).  Stores the provider + wallet
   * and registers a disconnect listener.
   */
  setupWallet(provider: WalletProvider, wallet: Wallet): void {
    this._provider = provider;
    this._wallet = wallet;

    this._unsubDisconnect = provider.onDisconnect(async () => {
      this._wallet = null;
      this._provider = null;
      this._unsubDisconnect = null;
      await getWalletStore().disconnect();
    });
  }

  /** The raw SDK wallet, used for transaction signing. */
  getSDKWallet(): Wallet | null {
    return this._wallet;
  }

  getStatus(): ConnectorStatus {
    const state = getWalletStore();
    const isActive = state.activeConnectorId === this.id;

    return {
      isInstalled: true,
      status: isActive ? state.status : 'disconnected',
      error: isActive ? state.error : null,
    };
  }

  getAccount(): AccountWithSecretKey | null {
    const state = getWalletStore();
    return state.activeConnectorId === this.id ? state.account : null;
  }

  /** Connection is UI-driven; this method is not used. */
  async connect(): Promise<void> {
    throw new Error(
      'WalletSDKConnector: use WalletSDKDiscoveryView to initiate connection.'
    );
  }

  async disconnect(): Promise<void> {
    this._unsubDisconnect?.();
    this._unsubDisconnect = null;

    if (this._provider) {
      await this._provider.disconnect().catch(() => {});
      this._provider = null;
      this._wallet = null;
    }

    await getWalletStore().disconnect();
  }
}

export const createWalletSDKConnector = (): WalletSDKConnector =>
  new WalletSDKConnector();
