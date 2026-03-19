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
    '0x1aa49f0ed1b28cf27b5747deceb89addc85d53ab5bc66406c4f429ccf9e1b8ee',
};
