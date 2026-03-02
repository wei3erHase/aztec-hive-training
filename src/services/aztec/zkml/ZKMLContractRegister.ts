import type { PXE } from '@aztec/pxe/server';
import type { DeployedContractConfig } from '../../../config/networks/types';

/**
 * Registers ZKML contract addresses with the PXE.
 * Populated when ZKML contracts are deployed (see commits 5-6).
 */
export class ZKMLContractRegister {
  async registerAll(
    _pxe: PXE,
    config?: Record<string, DeployedContractConfig> | null
  ): Promise<void> {
    if (!config || Object.keys(config).length === 0) {
      return;
    }
    // ZKML contract registration will be implemented when contracts exist
  }
}
