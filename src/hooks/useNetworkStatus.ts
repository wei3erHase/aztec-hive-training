import { useState, useEffect } from 'react';
import { getTrainingService } from '../services';

export interface NetworkStatus {
  connected: boolean;
  blockNumber: number | null;
  networkName: string;
  connectionError: string | null;
}

export function useNetworkStatus(
  networkId: string,
  nodeUrl: string
): NetworkStatus {
  const [status, setStatus] = useState<NetworkStatus>({
    connected: false,
    blockNumber: null,
    networkName: 'Unknown',
    connectionError: null,
  });

  useEffect(() => {
    const check = async () => {
      const service = getTrainingService(networkId, nodeUrl);
      try {
        const connected = await service.checkConnection();
        const blockNumber = connected ? await service.getBlockNumber() : null;
        const networkInfo = service.getNetworkInfo();
        setStatus({
          connected,
          blockNumber,
          networkName: networkInfo.name,
          connectionError: connected ? null : service.getConnectionError(),
        });
      } catch {
        setStatus({
          connected: false,
          blockNumber: null,
          networkName: 'Unknown',
          connectionError: service.getConnectionError(),
        });
      }
    };

    check();
    const interval = setInterval(check, 10000);
    return () => clearInterval(interval);
  }, [networkId, nodeUrl]);

  return status;
}
