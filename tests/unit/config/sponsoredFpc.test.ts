import { describe, expect, it } from 'vitest';
import {
  SPONSORED_FPC_CANONICAL_ADDRESS,
  SPONSORED_FPC_CLASS_ID,
  computeSponsoredFpcConstants,
} from '../../../src/config/networks/sponsoredFpc';
import { TESTNET_CONFIG } from '../../../src/config/networks/testnet';

describe('SponsoredFPC constants', () => {
  it('matches the bundled @aztec/noir-contracts.js artifact', async () => {
    const computed = await computeSponsoredFpcConstants();
    expect(computed.classId).toBe(SPONSORED_FPC_CLASS_ID);
    expect(computed.address).toBe(SPONSORED_FPC_CANONICAL_ADDRESS);
  });

  it('points testnet at the canonical FPC address', () => {
    expect(TESTNET_CONFIG.sponsoredFpcAddress).toBe(
      SPONSORED_FPC_CANONICAL_ADDRESS
    );
  });
});
