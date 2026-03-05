import React from 'react';
import { motion } from 'motion/react';

const styles = {
  wrapper:
    'mb-6 rounded-2xl border border-amber-500/30 bg-amber-600/10 p-5 sm:p-6',
  header: 'mb-2 flex items-center gap-2',
  title: 'font-semibold text-amber-300',
  body: 'mb-4 text-sm text-amber-200/80',
  steps: 'mb-4 list-decimal space-y-2 pl-5 text-sm text-amber-100/90',
  code: 'rounded bg-amber-900/40 px-1.5 py-0.5 font-mono text-amber-200',
  hint: 'text-xs text-amber-300/60',
  strong: 'text-amber-300',
} as const;

export const LocalNetworkWarning: React.FC = () => (
  <motion.div
    className={styles.wrapper}
    initial={{ opacity: 0, y: -8 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.35 }}
  >
    <div className={styles.header}>
      <span className="text-lg">⚠️</span>
      <span className={styles.title}>
        No contracts deployed on Local Network
      </span>
    </div>
    <p className={styles.body}>
      The app is pointed at your local Aztec network, but no contract addresses
      were found. Start the local network and run the deploy script to get
      going:
    </p>
    <ol className={styles.steps}>
      <li>
        Start the local network:{' '}
        <code className={styles.code}>gaztec start --local-network</code>
      </li>
      <li>
        Deploy with: <code className={styles.code}>yarn deploy-contracts</code>
      </li>
    </ol>
    <p className={styles.hint}>
      Alternatively, switch to <strong className={styles.strong}>Devnet</strong>{' '}
      in the network selector — contracts are already deployed there.
    </p>
  </motion.div>
);
