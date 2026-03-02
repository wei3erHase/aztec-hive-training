import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  TrainingService,
  getTrainingService,
  resetTrainingService,
} from '@/services/TrainingService';

afterEach(() => {
  resetTrainingService();
  vi.unstubAllGlobals();
});

describe('TrainingService — network info', () => {
  it('reports "Local Network" name for the local network', () => {
    const svc = new TrainingService('local-network', 'http://localhost:8080');
    expect(svc.getNetworkInfo().name).toBe('Local Network');
  });

  it('reports "Aztec Devnet" name for a devnet network', () => {
    const svc = new TrainingService('devnet', 'https://node.devnet.com');
    expect(svc.getNetworkInfo().name).toBe('Aztec Devnet');
  });

  it('exposes the network id', () => {
    const svc = new TrainingService('local-network', 'http://localhost:8080');
    expect(svc.getNetworkInfo().id).toBe('local-network');
  });

  it('exposes the nodeUrl in network info', () => {
    const url = 'http://localhost:8080';
    const svc = new TrainingService('local-network', url);
    expect(svc.getNetworkInfo().nodeUrl).toBe(url);
  });
});

describe('TrainingService — contract deployment detection', () => {
  it('reports deployed when the devnet MLP address is present in deployed.json', () => {
    const svc = new TrainingService('devnet', 'https://node.devnet.com');
    expect(svc.isTrainingContractDeployed()).toBe(true);
  });

  it('reports not deployed for local-network (no address in deployed.json)', () => {
    const svc = new TrainingService('local-network', 'http://localhost:8080');
    expect(svc.isTrainingContractDeployed()).toBe(false);
  });

  it('getAddresses reflects the deployment config', () => {
    const svc = new TrainingService('devnet', 'https://node.devnet.com');
    const addresses = svc.getAddresses();
    expect(typeof addresses.multiLayerPerceptron).toBe('string');
    expect((addresses.multiLayerPerceptron ?? '').length).toBeGreaterThan(0);
  });
});

describe('TrainingService — connection state', () => {
  it('getConnectionError is null before any connection attempt', () => {
    const svc = new TrainingService('local-network', 'http://localhost:8080');
    expect(svc.getConnectionError()).toBeNull();
  });

  it('checkConnection returns true and clears error on successful RPC response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: vi.fn().mockResolvedValue({ result: { nodeVersion: '1.0.0' } }),
      })
    );

    const svc = new TrainingService('local-network', 'http://localhost:8080');
    const result = await svc.checkConnection();

    expect(result).toBe(true);
    expect(svc.getConnectionError()).toBeNull();
  });

  it('checkConnection returns false and sets a user-friendly error for local network failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('connection refused'))
    );

    const svc = new TrainingService('local-network', 'http://localhost:8080');
    const result = await svc.checkConnection();

    expect(result).toBe(false);
    expect(svc.getConnectionError()).toContain('Local network is not running');
  });

  it('checkConnection formats a remote network error differently', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('connection refused'))
    );

    const svc = new TrainingService('devnet', 'https://node.devnet.com');
    const result = await svc.checkConnection();

    expect(result).toBe(false);
    expect(svc.getConnectionError()).toContain('Cannot connect to devnet');
  });

  it('checkConnection returns false when RPC response is not ok', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: vi.fn(),
      })
    );

    const svc = new TrainingService('local-network', 'http://localhost:8080');
    const result = await svc.checkConnection();

    expect(result).toBe(false);
    expect(svc.getConnectionError()).not.toBeNull();
  });
});

describe('getTrainingService singleton', () => {
  it('returns the same instance for the same network', () => {
    const a = getTrainingService('local-network');
    const b = getTrainingService('local-network');
    expect(a).toBe(b);
  });

  it('returns a new instance for a different network', () => {
    const localNetwork = getTrainingService('local-network');
    const devnet = getTrainingService('devnet');
    expect(localNetwork).not.toBe(devnet);
  });

  it('returns a new instance after resetTrainingService', () => {
    const first = getTrainingService('local-network');
    resetTrainingService();
    const second = getTrainingService('local-network');
    expect(first).not.toBe(second);
  });
});
