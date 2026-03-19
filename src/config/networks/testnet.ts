import { AztecAddress } from '@aztec/stdlib/aztec-address';
import { NetworkConfig } from './types';

/**
 * Testnet configuration for public Aztec testnet.
 *
 * Contract addresses are hardcoded for the public testnet.
 * These contracts are already deployed and available for testing.
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
  dripperContractAddress:
    '0x0000000000000000000000000000000000000000000000000000000000000000',
  tokenContractAddress:
    '0x0000000000000000000000000000000000000000000000000000000000000000',
  deployerAddress: AztecAddress.ZERO.toString(),
  dripperDeploymentSalt: '1337',
  tokenDeploymentSalt: '1337',
  proverEnabled: true,
  isTestnet: true,
  sponsoredFpcAddress:
    '0x1aa49f0ed1b28cf27b5747deceb89addc85d53ab5bc66406c4f429ccf9e1b8ee',
};
