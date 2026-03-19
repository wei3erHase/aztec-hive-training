export * from './types';
export * from './constants';
export * from './local-network';
export * from './testnet';

import { LOCAL_NETWORK_CONFIG } from './local-network';
import { TESTNET_CONFIG } from './testnet';
import { NetworkConfig } from './types';

export const AVAILABLE_NETWORKS: NetworkConfig[] = [
  TESTNET_CONFIG,
  LOCAL_NETWORK_CONFIG,
];
