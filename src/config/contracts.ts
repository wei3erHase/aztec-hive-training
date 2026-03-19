/**
 * Contract Configuration
 * Reads deployment addresses from:
 *   config/deployed.json       – committed, public networks (testnet, etc.)
 *   config/deployed.local.json – gitignored, local-network per-developer (injected by Vite plugin)
 */

import _localConfigModule from 'virtual:deployed-local-config';
import deployedConfig from '../../config/deployed.json' with { type: 'json' };

// Virtual module injected by Vite plugin; reads config/deployed.local.json at build/dev time
const _localConfig: Record<string, unknown> = (
  typeof _localConfigModule === 'object' && _localConfigModule !== null
    ? _localConfigModule
    : {}
) as Record<string, unknown>;

const _mergedConfig = {
  ...deployedConfig,
  ..._localConfig,
} as typeof deployedConfig;

export interface ContractConfig {
  address?: string;
  salt?: string;
  constructorName?: string;
  artifact?: string;
  lazyRegister?: boolean;
}

export type ArchitectureId = 'singleLayer' | 'mlp' | 'cnnGap';

export interface ContractsConfig {
  singleLayer: ContractConfig;
  multiLayerPerceptron: ContractConfig;
  cnnGap: ContractConfig;
}

export interface DeploymentConfig {
  networkId: string;
  contracts: ContractsConfig;
  deployedAt?: string;
}

interface DeployedJson {
  [networkId: string]: {
    singleLayer?: { address?: string; salt?: string; constructorName?: string };
    multiLayerPerceptron?: {
      address?: string;
      salt?: string;
      constructorName?: string;
    };
    cnnGap?: { address?: string; salt?: string; constructorName?: string };
    trainingVerification?: {
      address?: string;
      salt?: string;
      constructorName?: string;
    };
  };
}

let testOverride: Partial<Record<string, DeploymentConfig>> | null = null;

export function setDeploymentConfigOverride(
  networkId: string,
  config: {
    contracts?: {
      singleLayer?: Partial<ContractConfig>;
      multiLayerPerceptron?: Partial<ContractConfig>;
      cnnGap?: Partial<ContractConfig>;
      trainingVerification?: Partial<ContractConfig>;
    };
  }
): void {
  testOverride = testOverride || {};
  const base = ((_mergedConfig as DeployedJson)[networkId] ??
    {}) as NonNullable<DeployedJson[string]>;
  const existing = (testOverride[networkId]?.contracts ??
    {}) as Partial<ContractsConfig>;
  const override = (config.contracts ?? {}) as Partial<ContractsConfig> & {
    trainingVerification?: Partial<ContractConfig>;
  };
  testOverride[networkId] = {
    networkId,
    contracts: {
      singleLayer: {
        address: '',
        salt: '',
        constructorName: 'constructor',
        artifact: 'single_layer_contract-SingleLayer',
        lazyRegister: true,
        ...base.singleLayer,
        ...existing.singleLayer,
        ...override.singleLayer,
      },
      multiLayerPerceptron: {
        address: '',
        salt: '',
        constructorName: 'constructor',
        artifact: 'multi_layer_perceptron-MultiLayerPerceptron',
        lazyRegister: true,
        ...base.multiLayerPerceptron,
        ...base.trainingVerification,
        ...existing.multiLayerPerceptron,
        ...override.multiLayerPerceptron,
        ...override.trainingVerification,
      },
      cnnGap: {
        address: '',
        salt: '',
        constructorName: 'constructor',
        artifact: 'cnn_gap_contract-CNNGAP',
        lazyRegister: true,
        ...base.cnnGap,
        ...existing.cnnGap,
        ...override.cnnGap,
      },
    },
  } as DeploymentConfig;
}

export function clearDeploymentConfigOverride(): void {
  testOverride = null;
}

/**
 * Load deployment config for a specific network.
 * Maps legacy 'sandbox' id to 'local-network' for compatibility.
 */
export function getDeploymentConfig(
  networkId: string = 'local-network'
): DeploymentConfig {
  const mappedId = networkId === 'sandbox' ? 'local-network' : networkId;

  if (testOverride?.[mappedId]) {
    return testOverride[mappedId] as DeploymentConfig;
  }
  if (testOverride?.[networkId]) {
    return testOverride[networkId] as DeploymentConfig;
  }

  const config = _mergedConfig as DeployedJson;
  const net = config[mappedId] ?? config[networkId] ?? {};

  return {
    networkId: mappedId,
    contracts: {
      singleLayer: {
        address: net.singleLayer?.address || '',
        salt: net.singleLayer?.salt || '',
        constructorName: net.singleLayer?.constructorName || 'constructor',
        artifact: 'single_layer_contract-SingleLayer',
        lazyRegister: true,
      },
      multiLayerPerceptron: {
        address:
          net.multiLayerPerceptron?.address ??
          net.trainingVerification?.address ??
          '',
        salt:
          net.multiLayerPerceptron?.salt ??
          net.trainingVerification?.salt ??
          '',
        constructorName:
          net.multiLayerPerceptron?.constructorName ??
          net.trainingVerification?.constructorName ??
          'constructor',
        artifact: 'multi_layer_perceptron-MultiLayerPerceptron',
        lazyRegister: true,
      },
      cnnGap: {
        address: net.cnnGap?.address || '',
        salt: net.cnnGap?.salt || '',
        constructorName: net.cnnGap?.constructorName || 'constructor',
        artifact: 'cnn_gap_contract-CNNGAP',
        lazyRegister: true,
      },
    },
  };
}

export const contractsConfig = getDeploymentConfig();

export function getContractAddress(
  contractName: keyof ContractsConfig,
  networkId?: string
): string | undefined {
  const config = networkId ? getDeploymentConfig(networkId) : contractsConfig;
  return config.contracts[contractName]?.address;
}

export function getContractKeyForArchitecture(
  arch: ArchitectureId
): keyof ContractsConfig {
  switch (arch) {
    case 'singleLayer':
      return 'singleLayer';
    case 'mlp':
      return 'multiLayerPerceptron';
    case 'cnnGap':
      return 'cnnGap';
    default:
      return 'multiLayerPerceptron';
  }
}
