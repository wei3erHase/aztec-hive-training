import { AztecAddress } from '@aztec/stdlib/aztec-address';
import { SPONSORED_FPC_CANONICAL_ADDRESS } from './sponsoredFpc';
import { NetworkConfig } from './types';

/**
 * Testnet configuration for public Aztec testnet.
 */
export const TESTNET_CONFIG: NetworkConfig = {
  name: 'testnet',
  displayName: 'Testnet',
  description: 'Public Aztec testnet with real contracts',
  nodeUrl:
    typeof location !== 'undefined'
      ? `${location.origin}/api/testnet`
      : 'https://rpc.testnet.aztec-labs.com/',
  publicNodeUrl: 'https://rpc.testnet.aztec-labs.com/',
  deployerAddress: AztecAddress.ZERO.toString(),
  proverEnabled: true,
  isTestnet: true,
  sponsoredFpcAddress: SPONSORED_FPC_CANONICAL_ADDRESS,
};
