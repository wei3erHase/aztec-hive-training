import React, { useCallback, useState } from 'react';
import { hashToEmoji } from '@aztec/wallet-sdk/crypto';
import { useWalletStore } from '../../../store/wallet';
import { BackButton } from '../../shared';
import { useConnectModalContext } from '../context';

const CONFIRM_TIMEOUT_MS = 60_000;

const styles = {
  container: 'flex flex-col',
  backButton: 'mb-3',
  description: 'text-sm text-muted mb-4',
  emojiGrid:
    'grid grid-cols-3 gap-2 mb-5 bg-surface-secondary p-4 rounded-xl border border-default',
  emojiCell:
    'flex items-center justify-center text-2xl h-12 rounded-lg bg-surface',
  instructions: 'text-sm text-muted mb-5 text-center',
  actions: 'flex gap-3',
  confirmButton:
    'flex-1 px-4 py-2.5 bg-accent text-on-accent text-sm font-medium rounded-xl hover:opacity-90 disabled:opacity-50 transition-opacity cursor-pointer',
  cancelButton:
    'flex-1 px-4 py-2.5 bg-surface-secondary text-default text-sm font-medium rounded-xl hover:bg-surface-tertiary disabled:opacity-50 transition-colors cursor-pointer border border-default',
  spinner:
    'w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin inline-block mr-2',
  errorText:
    'text-sm text-red-500 bg-red-500/10 px-4 py-3 rounded-xl border border-red-500/20 mb-4',
} as const;

/**
 * Displays a 3×3 emoji grid derived from the ECDH verification hash so the
 * user can confirm the connection is not being intercepted (MITM protection).
 *
 * The user should verify the emojis match what their wallet extension shows,
 * then click "Confirm" to finalize the secure channel.
 */
export const WalletSDKVerificationView: React.FC = () => {
  const {
    goBack,
    walletSDKPending,
    setSuccessState,
    setView,
    setError,
    error,
  } = useConnectModalContext();
  const connectWalletSDK = useWalletStore((state) => state.connectWalletSDK);

  const [isConfirming, setIsConfirming] = useState(false);

  const emojis = walletSDKPending
    ? Array.from(
        hashToEmoji(walletSDKPending.pending.verificationHash.toString())
      )
    : [];

  const handleConfirm = useCallback(async () => {
    if (!walletSDKPending || isConfirming) return;
    setIsConfirming(true);
    setError(null);

    try {
      const timeout = new Promise<never>((_, reject) =>
        setTimeout(
          () =>
            reject(
              new Error('Wallet did not respond in time. Please try again.')
            ),
          CONFIRM_TIMEOUT_MS
        )
      );
      const wallet = await Promise.race([
        walletSDKPending.pending.confirm(),
        timeout,
      ]);
      const account = await connectWalletSDK(wallet, walletSDKPending.provider);
      const address = account.getAddress().toString();
      setSuccessState({ address });
      setView('success');
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : 'Failed to confirm connection';
      setError(msg);
      setIsConfirming(false);
    }
  }, [
    walletSDKPending,
    isConfirming,
    connectWalletSDK,
    setSuccessState,
    setView,
    setError,
  ]);

  const handleCancel = useCallback(() => {
    walletSDKPending?.pending.cancel();
    goBack();
  }, [walletSDKPending, goBack]);

  if (!walletSDKPending) {
    return null;
  }

  return (
    <div className={styles.container}>
      <BackButton onClick={handleCancel} className={styles.backButton} />

      <p className={styles.description}>
        Verify these emojis match what your wallet extension is showing. This
        confirms your connection is secure.
      </p>

      {error && <div className={styles.errorText}>{error}</div>}

      <div className={styles.emojiGrid}>
        {emojis.map((emoji, i) => (
          <div key={i} className={styles.emojiCell}>
            {emoji}
          </div>
        ))}
      </div>

      <p className={styles.instructions}>
        If the emojis match, click <strong>Confirm</strong>. Otherwise click{' '}
        <strong>Cancel</strong> and try again.
      </p>

      <div className={styles.actions}>
        <button
          className={styles.cancelButton}
          onClick={handleCancel}
          disabled={isConfirming}
        >
          Cancel
        </button>
        <button
          className={styles.confirmButton}
          onClick={handleConfirm}
          disabled={isConfirming}
        >
          {isConfirming && <span className={styles.spinner} />}
          Confirm
        </button>
      </div>
    </div>
  );
};
