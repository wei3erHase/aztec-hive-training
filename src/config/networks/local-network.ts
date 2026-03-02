import { NetworkConfig } from './types';

const LOCAL_NETWORK_NODE_URL = 'http://localhost:8080';

/**
 * Local network configuration for local development.
 * Run `yarn deploy-contracts` to deploy and update addresses.
 */
export const LOCAL_NETWORK_CONFIG: NetworkConfig = {
  name: 'local-network',
  displayName: 'Local Network',
  description: 'Local network - run "yarn deploy-contracts" to deploy',
  nodeUrl: LOCAL_NETWORK_NODE_URL,
  dripperContractAddress:
    '0x0000000000000000000000000000000000000000000000000000000000000000',
  tokenContractAddress:
    '0x0000000000000000000000000000000000000000000000000000000000000000',
  deployerAddress:
    '0x0000000000000000000000000000000000000000000000000000000000000000',
  dripperDeploymentSalt: '1337',
  tokenDeploymentSalt: '1337',
  proverEnabled: false,
  isTestnet: false,
};
