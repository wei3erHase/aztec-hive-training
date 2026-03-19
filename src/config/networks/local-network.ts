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
  deployerAddress:
    '0x0000000000000000000000000000000000000000000000000000000000000000',
  proverEnabled: false,
  isTestnet: false,
};
