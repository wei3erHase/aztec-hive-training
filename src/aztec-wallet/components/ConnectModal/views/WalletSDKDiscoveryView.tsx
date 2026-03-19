import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Fr } from '@aztec/aztec.js/fields';
import { createAztecNodeClient } from '@aztec/aztec.js/node';
import { WalletManager } from '@aztec/wallet-sdk/manager';
import type {
  DiscoverySession,
  WalletProvider,
} from '@aztec/wallet-sdk/manager';
import { WALLET_SDK_APP_ID } from '../../../config/defaults';
import { useNetworkStore } from '../../../store/network';
import { BackButton } from '../../shared';
import { useConnectModalContext } from '../context';

const styles = {
  container: 'flex flex-col',
  backButton: 'mb-3',
  description: 'text-sm text-muted mb-4',
  searching: 'flex flex-col items-center gap-3 py-6',
  spinner:
    'w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin',
  searchingText: 'text-sm text-muted text-center',
  walletList: 'flex flex-col gap-2',
  walletButton:
    'flex items-center gap-3 w-full px-4 py-3 rounded-xl bg-surface-secondary hover:bg-surface-tertiary border border-default transition-colors cursor-pointer',
  walletIcon:
    'w-9 h-9 rounded-lg bg-surface flex items-center justify-center overflow-hidden flex-shrink-0',
  walletIconImg: 'w-7 h-7 object-contain',
  walletIconPlaceholder: 'text-lg',
  walletName: 'text-sm font-medium text-default',
  walletType: 'text-xs text-muted capitalize',
  noWallets: 'text-center py-6',
  noWalletsTitle: 'text-sm font-medium text-default mb-1',
  noWalletsText: 'text-xs text-muted',
  errorText:
    'text-sm text-red-500 bg-red-500/10 px-4 py-3 rounded-xl border border-red-500/20',
} as const;

const DISCOVERY_TIMEOUT_MS = 30_000;

/**
 * View that drives the Wallet SDK discovery flow.
 *
 * On mount it queries the Aztec node for chain info, then starts a
 * WalletManager discovery session. Discovered wallets appear in a list.
 * Clicking one initiates the ECDH key exchange and transitions to the
 * verification view.
 */
export const WalletSDKDiscoveryView: React.FC = () => {
  const { goBack, setView, setWalletSDKPending, setError, error } =
    useConnectModalContext();
  const nodeUrl = useNetworkStore((state) => state.currentConfig.nodeUrl);

  const [providers, setProviders] = useState<WalletProvider[]>([]);
  const [isDiscovering, setIsDiscovering] = useState(true);
  const [connecting, setConnecting] = useState<string | null>(null);
  const discoveryRef = useRef<DiscoverySession | null>(null);

  useEffect(() => {
    let cancelled = false;

    const startDiscovery = async () => {
      try {
        const node = createAztecNodeClient(nodeUrl);
        const nodeInfo = await node.getNodeInfo();

        // StrictMode runs effects twice; bail out if cleanup already ran so we
        // don't create a second WalletManager session and trigger duplicate
        // Azguard popups.
        if (cancelled) return;

        // chainId must be the L1 chain ID (e.g. Sepolia = 11155111), not the
        // Aztec/L2 chain ID.  rollupVersion is the version of the rollup contract.
        const chainInfo = {
          chainId: new Fr(nodeInfo.l1ChainId),
          version: new Fr(nodeInfo.rollupVersion),
        };

        const session = WalletManager.configure({
          extensions: { enabled: true },
        }).getAvailableWallets({
          chainInfo,
          appId: WALLET_SDK_APP_ID,
          timeout: DISCOVERY_TIMEOUT_MS,
          onWalletDiscovered: (provider) => {
            if (!cancelled) {
              setProviders((prev) => {
                if (prev.some((p) => p.id === provider.id)) return prev;
                return [...prev, provider];
              });
            }
          },
        });

        discoveryRef.current = session;
        await session.done;

        if (!cancelled) {
          setIsDiscovering(false);
        }
      } catch (err) {
        if (!cancelled) {
          console.error('[WalletSDKDiscovery] error:', err);
          setIsDiscovering(false);
        }
      }
    };

    startDiscovery();

    return () => {
      cancelled = true;
      discoveryRef.current?.cancel();
      discoveryRef.current = null;
    };
  }, [nodeUrl]);

  const handleSelectProvider = useCallback(
    async (provider: WalletProvider) => {
      if (connecting) return;
      setConnecting(provider.id);
      setError(null);

      try {
        discoveryRef.current?.cancel();
        const pending =
          await provider.establishSecureChannel(WALLET_SDK_APP_ID);
        setWalletSDKPending({ provider, pending });
        setView('wallet-sdk-verification');
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : 'Failed to connect to wallet';
        setError(msg);
        setConnecting(null);
      }
    },
    [connecting, setError, setWalletSDKPending, setView]
  );

  return (
    <div className={styles.container}>
      <BackButton onClick={goBack} className={styles.backButton} />

      <p className={styles.description}>
        Open your Aztec wallet extension and approve the connection request.
        Discovered wallets will appear below.
      </p>

      {error && <div className={styles.errorText}>{error}</div>}

      {providers.length === 0 && isDiscovering && (
        <div className={styles.searching}>
          <div className={styles.spinner} />
          <p className={styles.searchingText}>
            Searching for Aztec wallets&hellip;
          </p>
        </div>
      )}

      {providers.length === 0 && !isDiscovering && (
        <div className={styles.noWallets}>
          <p className={styles.noWalletsTitle}>No wallets found</p>
          <p className={styles.noWalletsText}>
            Make sure an Aztec wallet extension is installed and unlocked.
          </p>
        </div>
      )}

      {providers.length > 0 && (
        <div className={styles.walletList}>
          {providers.map((provider) => (
            <button
              key={provider.id}
              className={styles.walletButton}
              onClick={() => handleSelectProvider(provider)}
              disabled={!!connecting}
            >
              <div className={styles.walletIcon}>
                {provider.icon &&
                !provider.icon.startsWith('chrome-extension://') ? (
                  <img
                    src={provider.icon}
                    alt={provider.name}
                    className={styles.walletIconImg}
                  />
                ) : (
                  <span className={styles.walletIconPlaceholder}>⬡</span>
                )}
              </div>
              <div>
                <div className={styles.walletName}>{provider.name}</div>
                <div className={styles.walletType}>{provider.type}</div>
              </div>
              {connecting === provider.id && (
                <div
                  className={styles.spinner}
                  style={{ marginLeft: 'auto' }}
                />
              )}
            </button>
          ))}

          {isDiscovering && (
            <div className={styles.searching}>
              <div className={styles.spinner} />
              <p className={styles.searchingText}>
                Looking for more wallets&hellip;
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
