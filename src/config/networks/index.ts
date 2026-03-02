export * from './types';
export * from './constants';
export * from './local-network';
export * from './devnet';

import { DEVNET_CONFIG } from './devnet';
import { LOCAL_NETWORK_CONFIG } from './local-network';
import { NetworkConfig } from './types';

export const AVAILABLE_NETWORKS: NetworkConfig[] = [
  DEVNET_CONFIG,
  LOCAL_NETWORK_CONFIG,
];
