import type { AztecNetwork } from './constants';

/**
 * Configuration for a deployed contract on a network.
 * Contains address, deployment salt, and deployer address.
 */
export interface DeployedContractConfig {
  /** Contract address */
  address?: string;
  /** Deployment salt (for deterministic addresses) */
  salt?: string;
  /** Deployer address */
  deployer?: string;
}

/**
 * Fee Payment Contracts configuration - a map of contract names to their deployment configs.
 */
export type FeePaymentContractsConfig = Record<string, DeployedContractConfig>;

export interface NetworkConfig {
  name: AztecNetwork;
  displayName: string;
  description: string;
  nodeUrl: string;
  /** Human-readable node URL shown in the UI (may differ from nodeUrl when a proxy is used). */
  publicNodeUrl?: string;
  deployerAddress: string;
  proverEnabled: boolean;
  isTestnet: boolean;
  /** Address of the SponsoredFPC contract on this network. When set, the instance is fetched from the node instead of computing from salt. Defaults to canonical salt (0) if omitted. */
  sponsoredFpcAddress?: string;
  /** Fee payment contracts configuration (keyed by contract name, e.g., 'metered') */
  feePaymentContracts?: Record<string, DeployedContractConfig>;
  /** ZKML contract addresses for PXE registration (populated when deployed) */
  zkmlContracts?: Record<string, DeployedContractConfig>;
}
