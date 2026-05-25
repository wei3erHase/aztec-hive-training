import { getContractInstanceFromInstantiationParams } from '@aztec/aztec.js/contracts';
import { Fr } from '@aztec/aztec.js/fields';
import { SPONSORED_FPC_SALT } from '@aztec/constants';
import { SponsoredFPCContractArtifact } from '@aztec/noir-contracts.js/SponsoredFPC';
import { getContractClassFromArtifact } from '@aztec/stdlib/contract';

/**
 * SponsoredFPC class ID for @aztec/noir-contracts.js matching package.json config.aztecVersion (4.3.0).
 * Verified in tests/unit/config/sponsoredFpc.test.ts.
 */
export const SPONSORED_FPC_CLASS_ID =
  '0x216c0b6e1ada02642e83be3ab790eb2c10979c325de34e71df5c34b9159b7eb7';

/**
 * Canonical genesis SponsoredFPC address (SPONSORED_FPC_SALT from @aztec/constants).
 * Same on local-network and testnet for the current Aztec release.
 */
export const SPONSORED_FPC_CANONICAL_ADDRESS =
  '0x08b888c4be63ed67f61a622fdd013ea028326bac22a8982a3b5a7e9ec62f765b';

/** Recompute class ID and address from the bundled artifact (for tests and drift checks). */
export async function computeSponsoredFpcConstants(): Promise<{
  classId: string;
  address: string;
}> {
  const contractClass = await getContractClassFromArtifact(
    SponsoredFPCContractArtifact
  );
  const instance = await getContractInstanceFromInstantiationParams(
    SponsoredFPCContractArtifact,
    { salt: new Fr(SPONSORED_FPC_SALT) }
  );

  return {
    classId: contractClass.id.toString(),
    address: instance.address.toString(),
  };
}
