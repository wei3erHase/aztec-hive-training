import { useQuery } from '@tanstack/react-query';
import { useAztecWallet, hasAppManagedPXE } from '../../aztec-wallet';
import { SharedPXEService } from '../../aztec-wallet/services/aztec/pxe';
import { getNetworkStore } from '../../aztec-wallet/store/network';

const DEFAULT_REFRESH_INTERVAL = 30_000;

export interface NetworkHealth {
  blockHeight: number | null;
  latency: number | null;
  lastSynced: Date | null;
  isHealthy: boolean;
  isLoading: boolean;
  error: string | null;
}

interface HealthData {
  blockHeight: number;
  latency: number;
  lastSynced: Date;
}

function getAztecNode() {
  const config = getNetworkStore().currentConfig;
  const instance = SharedPXEService.getExistingInstance(config?.name ?? '');
  return instance?.aztecNode ?? null;
}

export function useNetworkHealth(
  refreshInterval: number = DEFAULT_REFRESH_INTERVAL
): NetworkHealth {
  const { isConnected, isPXEInitialized, connector } = useAztecWallet();

  const canFetch = Boolean(
    isConnected && isPXEInitialized && connector && hasAppManagedPXE(connector)
  );

  const { data, isLoading, error } = useQuery<HealthData, Error>({
    queryKey: ['networkHealth', connector?.id],
    queryFn: async (): Promise<HealthData> => {
      const aztecNode = getAztecNode();
      if (!aztecNode) {
        throw new Error('AztecNode not available');
      }

      const startTime = performance.now();
      const blockNumber = await aztecNode.getBlockNumber();
      const endTime = performance.now();

      return {
        blockHeight: blockNumber,
        latency: Math.round(endTime - startTime),
        lastSynced: new Date(),
      };
    },
    enabled: canFetch,
    refetchInterval: canFetch ? refreshInterval : false,
    staleTime: refreshInterval / 2,
    retry: false,
  });

  if (!canFetch) {
    return {
      blockHeight: null,
      latency: null,
      lastSynced: null,
      isHealthy: false,
      isLoading: false,
      error: null,
    };
  }

  return {
    blockHeight: data?.blockHeight ?? null,
    latency: data?.latency ?? null,
    lastSynced: data?.lastSynced ?? null,
    isHealthy: data ? data.latency < 5000 : false,
    isLoading,
    error: error?.message ?? null,
  };
}
