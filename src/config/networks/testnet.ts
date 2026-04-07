import { AztecAddress } from '@aztec/stdlib/aztec-address';
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
  sponsoredFpcAddress:
    '0x1e2e59ede08460bf957ab1a349d64d84ff8ffecc03528bd6e111c608a2ba2c6c',
};
