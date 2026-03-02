import { describe, it, expect, afterEach } from 'vitest';
import {
  getDeploymentConfig,
  getContractAddress,
  getContractKeyForArchitecture,
  setDeploymentConfigOverride,
  clearDeploymentConfigOverride,
  type ArchitectureId,
} from '@/config/contracts';

// Read addresses straight from the source so tests never go stale after a redeployment
import deployedJson from '../../../config/deployed.json';
const DEVNET_MLP_ADDRESS: string =
  deployedJson.devnet.multiLayerPerceptron.address;
const DEVNET_CNN_ADDRESS: string = deployedJson.devnet.cnnGap.address;

afterEach(() => {
  clearDeploymentConfigOverride();
});

describe('getDeploymentConfig', () => {
  describe('devnet', () => {
    it('returns the correct networkId', () => {
      expect(getDeploymentConfig('devnet').networkId).toBe('devnet');
    });

    it('returns the MLP address from deployed.json', () => {
      const config = getDeploymentConfig('devnet');
      expect(config.contracts.multiLayerPerceptron.address).toBe(
        DEVNET_MLP_ADDRESS
      );
    });

    it('returns the CNN-GAP address from deployed.json', () => {
      const config = getDeploymentConfig('devnet');
      expect(config.contracts.cnnGap.address).toBe(DEVNET_CNN_ADDRESS);
    });

    it('includes the correct artifact identifier for each contract', () => {
      const config = getDeploymentConfig('devnet');
      expect(config.contracts.singleLayer.artifact).toBe(
        'single_layer_contract-SingleLayer'
      );
      expect(config.contracts.multiLayerPerceptron.artifact).toBe(
        'multi_layer_perceptron-MultiLayerPerceptron'
      );
      expect(config.contracts.cnnGap.artifact).toBe('cnn_gap_contract-CNNGAP');
    });

    it('sets lazyRegister to true for all contracts', () => {
      const config = getDeploymentConfig('devnet');
      expect(config.contracts.singleLayer.lazyRegister).toBe(true);
      expect(config.contracts.multiLayerPerceptron.lazyRegister).toBe(true);
      expect(config.contracts.cnnGap.lazyRegister).toBe(true);
    });
  });

  describe('local-network (no deployment)', () => {
    it('returns networkId "local-network"', () => {
      expect(getDeploymentConfig('local-network').networkId).toBe(
        'local-network'
      );
    });

    it('returns empty string addresses for all contracts', () => {
      const config = getDeploymentConfig('local-network');
      expect(config.contracts.singleLayer.address).toBe('');
      expect(config.contracts.multiLayerPerceptron.address).toBe('');
      expect(config.contracts.cnnGap.address).toBe('');
    });

    it('defaults the constructorName to "constructor"', () => {
      const config = getDeploymentConfig('local-network');
      expect(config.contracts.singleLayer.constructorName).toBe('constructor');
    });
  });

  describe('sandbox → local-network alias (backward compat)', () => {
    it('maps sandbox to local-network (same networkId)', () => {
      expect(getDeploymentConfig('sandbox').networkId).toBe('local-network');
    });

    it('returns the same addresses as local-network', () => {
      const localNetwork = getDeploymentConfig('local-network');
      const legacy = getDeploymentConfig('sandbox');
      expect(legacy.contracts.multiLayerPerceptron.address).toBe(
        localNetwork.contracts.multiLayerPerceptron.address
      );
    });
  });
});

describe('setDeploymentConfigOverride / clearDeploymentConfigOverride', () => {
  it('overrides a contract address for a specific network', () => {
    setDeploymentConfigOverride('local-network', {
      contracts: { singleLayer: { address: '0xdeadbeef' } },
    });
    const config = getDeploymentConfig('local-network');
    expect(config.contracts.singleLayer.address).toBe('0xdeadbeef');
  });

  it('does not affect other networks', () => {
    setDeploymentConfigOverride('local-network', {
      contracts: { singleLayer: { address: '0xdeadbeef' } },
    });
    // devnet should still return the real address
    expect(
      getDeploymentConfig('devnet').contracts.multiLayerPerceptron.address
    ).toBe(DEVNET_MLP_ADDRESS);
  });

  it('clearDeploymentConfigOverride restores the original config', () => {
    setDeploymentConfigOverride('local-network', {
      contracts: { singleLayer: { address: '0xdeadbeef' } },
    });
    clearDeploymentConfigOverride();
    expect(
      getDeploymentConfig('local-network').contracts.singleLayer.address
    ).toBe('');
  });
});

describe('getContractAddress', () => {
  it('returns the MLP address for devnet', () => {
    expect(getContractAddress('multiLayerPerceptron', 'devnet')).toBe(
      DEVNET_MLP_ADDRESS
    );
  });

  it('returns the CNN-GAP address for devnet', () => {
    expect(getContractAddress('cnnGap', 'devnet')).toBe(DEVNET_CNN_ADDRESS);
  });

  it('returns empty string for local-network (no deployment)', () => {
    expect(getContractAddress('singleLayer', 'local-network')).toBe('');
  });

  it('reflects overrides set via setDeploymentConfigOverride', () => {
    setDeploymentConfigOverride('local-network', {
      contracts: { cnnGap: { address: '0xoverride' } },
    });
    expect(getContractAddress('cnnGap', 'local-network')).toBe('0xoverride');
  });
});

describe('getContractKeyForArchitecture', () => {
  const cases: Array<[ArchitectureId, string]> = [
    ['singleLayer', 'singleLayer'],
    ['mlp', 'multiLayerPerceptron'],
    ['cnnGap', 'cnnGap'],
  ];

  for (const [arch, expectedKey] of cases) {
    it(`maps "${arch}" → "${expectedKey}"`, () => {
      expect(getContractKeyForArchitecture(arch)).toBe(expectedKey);
    });
  }
});
