/**
 * Training Service
 * Connects to Aztec network and manages contract interactions
 * Supports per-network configuration via getDeploymentConfig
 */

import { getDeploymentConfig } from '../config/contracts';
import type { ArchitectureId } from '../config/contracts';

export interface ContractAddresses {
  multiLayerPerceptron?: string;
}

export interface TrainingTxData {
  inputPixels: bigint[];
  label: bigint;
  architecture: ArchitectureId;
}

export interface TrainingTxResult {
  success: boolean;
  txHash?: string;
  commitmentHash?: bigint;
  error?: string;
  blockNumber?: number;
}

function formatConnectionError(
  error: unknown,
  nodeUrl: string,
  networkId: string
): string {
  const msg = error instanceof Error ? error.message : String(error);
  const isLocal =
    networkId === 'local-network' ||
    nodeUrl.includes('localhost') ||
    nodeUrl === '/rpc';

  const connectionFailedPatterns = [
    'failed to fetch',
    'fetch failed',
    'network error',
    'connection refused',
    'econnrefused',
    'econnreset',
    'net::err_connection_refused',
    'net::err_connection_reset',
  ];
  const isConnectionFailure = connectionFailedPatterns.some((p) =>
    msg.toLowerCase().includes(p)
  );

  if (isConnectionFailure) {
    if (isLocal) {
      return 'Local network is not running. Start it with `yarn deploy-contracts` or `aztec start` before deploying.';
    }
    return `Cannot connect to ${networkId}. Check your internet connection or try again later.`;
  }

  if (msg.includes('RPC failed') || msg.includes('RPC error')) {
    if (isLocal) {
      return 'Local network returned an error. Ensure it is running.';
    }
    return `Network error (${msg}). Please try again.`;
  }

  return msg;
}

async function rpcCall<T>(
  url: string,
  method: string,
  params: unknown[] = []
): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      method,
      params,
      id: Date.now(),
    }),
  });
  if (!response.ok) throw new Error(`RPC failed: ${response.status}`);
  const data = await response.json();
  if (data.error) throw new Error(data.error.message || 'RPC error');
  return data.result as T;
}

function parseBlockNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'bigint') return Number(value);
  if (typeof value !== 'string') return null;
  const parsed = value.startsWith('0x')
    ? Number.parseInt(value, 16)
    : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Training Service for Aztec contract interactions
 */
export class TrainingService {
  private addresses: ContractAddresses;
  private isConnected = false;
  private pxeUrl: string;
  private networkId: string;
  private nodeVersion: string | null = null;
  private lastConnectionError: string | null = null;

  constructor(
    networkId: string,
    nodeUrl: string,
    addresses: ContractAddresses = {}
  ) {
    this.networkId = networkId;
    this.pxeUrl =
      typeof window !== 'undefined' && networkId === 'local-network'
        ? '/rpc'
        : nodeUrl;

    const deploymentConfig = getDeploymentConfig(networkId);
    this.addresses = {
      multiLayerPerceptron:
        deploymentConfig.contracts.multiLayerPerceptron.address ||
        addresses.multiLayerPerceptron,
    };
  }

  async initialize(): Promise<void> {
    try {
      const nodeInfo = await rpcCall<{ nodeVersion: string }>(
        this.pxeUrl,
        'node_getNodeInfo'
      );
      this.nodeVersion = nodeInfo.nodeVersion;
      this.isConnected = true;
      this.lastConnectionError = null;
    } catch (error) {
      this.lastConnectionError = formatConnectionError(
        error,
        this.pxeUrl,
        this.networkId
      );
      this.isConnected = false;
    }
  }

  async checkConnection(): Promise<boolean> {
    try {
      const nodeInfo = await rpcCall<{ nodeVersion: string }>(
        this.pxeUrl,
        'node_getNodeInfo'
      );
      this.nodeVersion = nodeInfo.nodeVersion;
      this.isConnected = true;
      this.lastConnectionError = null;
      return true;
    } catch (error) {
      this.lastConnectionError = formatConnectionError(
        error,
        this.pxeUrl,
        this.networkId
      );
      this.isConnected = false;
      return false;
    }
  }

  getConnectionError(): string | null {
    return this.lastConnectionError;
  }

  async getBlockNumber(): Promise<number | null> {
    try {
      const direct = await rpcCall<unknown>(this.pxeUrl, 'node_getBlockNumber');
      const directNumber = parseBlockNumber(direct);
      if (directNumber !== null) return directNumber;
    } catch {
      // fallthrough
    }
    try {
      const tips = await rpcCall<{ latest?: { number?: unknown } }>(
        this.pxeUrl,
        'node_getL2Tips'
      );
      return parseBlockNumber(tips.latest?.number);
    } catch {
      return null;
    }
  }

  getNetworkInfo(): { id: string; name: string; nodeUrl: string } {
    const name =
      this.networkId === 'local-network' ? 'Local Network' : 'Aztec Testnet';
    return {
      id: this.networkId,
      name,
      nodeUrl: this.pxeUrl,
    };
  }

  getAddresses(): ContractAddresses {
    return { ...this.addresses };
  }

  isTrainingContractDeployed(): boolean {
    return !!this.addresses.multiLayerPerceptron;
  }
}

let instance: TrainingService | null = null;
let instanceNetworkId: string | null = null;
let initialized = false;

export function getTrainingService(
  networkId?: string,
  nodeUrl?: string
): TrainingService {
  const targetNetworkId = networkId || 'local-network';
  const targetNodeUrl =
    nodeUrl ||
    (targetNetworkId === 'local-network'
      ? 'http://localhost:8080'
      : 'https://rpc.testnet.aztec-labs.com/');

  if (!instance || instanceNetworkId !== targetNetworkId) {
    instance = new TrainingService(targetNetworkId, targetNodeUrl);
    instanceNetworkId = targetNetworkId;
    initialized = false;
  }

  if (!initialized) {
    instance
      .initialize()
      .then(() => {
        initialized = true;
      })
      .catch(console.error);
  }
  return instance;
}

export async function initializeTrainingService(
  networkId?: string,
  nodeUrl?: string
): Promise<TrainingService> {
  const targetNetworkId = networkId || 'local-network';
  const targetNodeUrl =
    nodeUrl ||
    (targetNetworkId === 'local-network'
      ? 'http://localhost:8080'
      : 'https://rpc.testnet.aztec-labs.com/');

  if (!instance || instanceNetworkId !== targetNetworkId) {
    instance = new TrainingService(targetNetworkId, targetNodeUrl);
    instanceNetworkId = targetNetworkId;
    initialized = false;
  }

  if (!initialized) {
    await instance.initialize();
    initialized = true;
  }
  return instance;
}

export function resetTrainingService(): void {
  instance = null;
  instanceNetworkId = null;
  initialized = false;
}
