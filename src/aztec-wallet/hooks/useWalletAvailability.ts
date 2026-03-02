import { useMemo, useState, useEffect } from 'react';

/** EVM wallets disabled in HIVE - hooks return false/empty for API compatibility. */
export function useIsEvmWalletInstalled(_rdns: string | undefined): boolean {
  return false;
}

/** EVM wallets disabled in HIVE. */
export function useDiscoveredWallets(): Array<{ info: { rdns: string } }> {
  return [];
}

/** EVM wallets disabled in HIVE. */
export function useEvmWalletsAvailability(
  wallets: Array<{ rdns?: string }>
): Record<string, boolean> {
  return useMemo(() => {
    const availability: Record<string, boolean> = {};
    for (const wallet of wallets) {
      if (wallet.rdns) availability[wallet.rdns] = false;
    }
    return availability;
  }, [wallets]);
}

/**
 * Hook to get availability info for Aztec browser wallets
 *
 * Calls each wallet's `checkInstalled` function asynchronously.
 *
 * @param wallets - Array of wallet configs with optional checkInstalled function
 * @returns Map of wallet id to installation status (undefined while loading)
 *
 * @example
 * ```tsx
 * const availability = useAztecWalletsAvailability([
 *   { id: 'azguard', checkInstalled: async () => { ... } },
 * ]);
 * // availability = { 'azguard': true }
 * ```
 */
export function useAztecWalletsAvailability(
  wallets: Array<{ id: string; checkInstalled?: () => Promise<boolean> }>
): Record<string, boolean | undefined> {
  const [availability, setAvailability] = useState<
    Record<string, boolean | undefined>
  >({});

  // Create a stable key for the wallets array to avoid unnecessary re-runs
  const walletsKey = useMemo(
    () => wallets.map((w) => w.id).join(','),
    [wallets]
  );

  useEffect(() => {
    let cancelled = false;

    const checkAll = async () => {
      const results: Record<string, boolean | undefined> = {};

      // Initialize all as undefined (loading)
      for (const wallet of wallets) {
        results[wallet.id] = undefined;
      }
      setAvailability({ ...results });

      // Check each wallet in parallel
      await Promise.all(
        wallets.map(async (wallet) => {
          if (wallet.checkInstalled) {
            try {
              const isInstalled = await wallet.checkInstalled();
              if (!cancelled) {
                setAvailability((prev) => ({
                  ...prev,
                  [wallet.id]: isInstalled,
                }));
              }
            } catch (error) {
              console.warn(
                `Failed to check if ${wallet.id} is installed:`,
                error
              );
              if (!cancelled) {
                setAvailability((prev) => ({
                  ...prev,
                  [wallet.id]: false,
                }));
              }
            }
          } else {
            // No checkInstalled function, assume unknown (don't show status)
            if (!cancelled) {
              setAvailability((prev) => ({
                ...prev,
                [wallet.id]: undefined,
              }));
            }
          }
        })
      );
    };

    checkAll();

    return () => {
      cancelled = true;
    };
  }, [walletsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  return availability;
}
