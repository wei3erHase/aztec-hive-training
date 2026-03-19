import { useQuery } from '@tanstack/react-query';
import { LOCAL_NETWORK_CONFIG } from '../../config/networks/local-network';
import { TESTNET_CONFIG } from '../../config/networks/testnet';
import type { AztecNetwork } from '../../config/networks/constants';

const CHECK_INTERVAL = 30000;
const TIMEOUT = 5000;

export type AvailabilityStatus = 'checking' | 'available' | 'unavailable';

export interface NetworkAvailability {
  networks: Record<AztecNetwork, AvailabilityStatus>;
  isChecking: boolean;
}

const NETWORK_CONFIGS: Record<AztecNetwork, { nodeUrl: string }> = {
  'local-network': LOCAL_NETWORK_CONFIG,
  testnet: TESTNET_CONFIG,
};

function isLocalhost(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1';
  } catch {
    return false;
  }
}

async function checkNetworkAvailable(
  nodeUrl: string,
  proxyPath?: string
): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT);

  const baseUrl = nodeUrl.endsWith('/') ? nodeUrl.slice(0, -1) : nodeUrl;
  const url = isLocalhost(nodeUrl) ? `${baseUrl}/status` : proxyPath;

  if (!url) {
    clearTimeout(timeoutId);
    return true;
  }

  try {
    const response = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    clearTimeout(timeoutId);
    return false;
  }
}

async function fetchNetworkAvailability(): Promise<
  Record<AztecNetwork, boolean>
> {
  const entries = Object.entries(NETWORK_CONFIGS) as [
    AztecNetwork,
    { nodeUrl: string },
  ][];

  const results = await Promise.all(
    entries.map(async ([network, config]) => {
      const proxyPath = `/api/${network}-status`;
      const available = await checkNetworkAvailable(config.nodeUrl, proxyPath);
      return [network, available] as const;
    })
  );

  return Object.fromEntries(results) as Record<AztecNetwork, boolean>;
}

export function useNetworkAvailability(): NetworkAvailability {
  const { data, isLoading } = useQuery({
    queryKey: ['networkAvailability'],
    queryFn: fetchNetworkAvailability,
    refetchInterval: CHECK_INTERVAL,
    staleTime: CHECK_INTERVAL - 1000,
  });

  const networks = (['local-network', 'testnet'] as const).reduce(
    (acc, network) => {
      const available = data?.[network];
      acc[network] =
        available === undefined
          ? 'checking'
          : available
            ? 'available'
            : 'unavailable';
      return acc;
    },
    {} as Record<AztecNetwork, AvailabilityStatus>
  );

  return {
    networks,
    isChecking: isLoading,
  };
}
