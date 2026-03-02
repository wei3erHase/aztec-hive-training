import type { PXE } from '@aztec/pxe/server';
import type { FeePaymentContractsConfig } from '../../../config/networks/types';

/**
 * Registers fee payment contracts with the PXE.
 * No-op when config is empty (HIVE uses sponsored fees or no fee contracts).
 */
export class FeePaymentRegister {
  async registerAll(
    _pxe: PXE,
    config?: FeePaymentContractsConfig | null
  ): Promise<void> {
    if (!config || Object.keys(config).length === 0) {
      return;
    }
    // Fee payment contract registration would go here when configured
  }
}
