import React from 'react';
import { motion } from 'motion/react';
import { Badge, Button } from '../ui';
import type { NetworkStatus } from '../../hooks';

interface HeroSectionProps {
  isConnected: boolean;
  networkStatus: NetworkStatus;
  onConnect: () => void;
  onNavigateDocs: () => void;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  isConnected,
  networkStatus,
  onConnect,
  onNavigateDocs,
}) => (
  <motion.section
    className="relative z-10 mb-10 w-full rounded-2xl sm:rounded-3xl panel-strong p-5 sm:p-8 md:p-10"
    initial={{ opacity: 0, y: 24 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.45 }}
  >
    <div className="relative z-10 mb-5 flex flex-wrap items-center gap-3">
      <Badge variant="signal" className="relative z-10">
        <span className="h-2 w-2 rounded-full bg-[#d4ff28] shadow-[0_0_10px_rgba(212,255,40,0.9)]" />
        Hive Training
      </Badge>

      <Badge
        variant={networkStatus.connected ? 'success' : 'error'}
        className="font-mono"
        {...(!networkStatus.connected && networkStatus.connectionError
          ? { title: networkStatus.connectionError }
          : {})}
      >
        <span
          className={`h-2 w-2 rounded-full ${
            networkStatus.connected ? 'bg-emerald-400' : 'bg-red-400'
          }`}
        />
        {networkStatus.connected
          ? `${networkStatus.networkName} #${networkStatus.blockNumber ?? '...'}`
          : (networkStatus.connectionError ?? 'Offline')}
      </Badge>
    </div>

    <h1 className="mb-4 max-w-3xl font-display text-2xl font-bold leading-tight text-default sm:text-4xl md:text-6xl">
      Draw, Predict, Train. Your data stays on your device.
    </h1>
    <p className="mb-8 max-w-3xl text-base text-muted sm:text-lg">
      Contribute to a shared model via ZK proofs. Weights are public on Aztec.
    </p>

    <div className="flex flex-wrap items-center gap-3">
      {isConnected ? (
        <motion.a
          href="#demo"
          data-testid="hero-start-button"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Button variant="signal" size="lg">
            Open Control Deck
          </Button>
        </motion.a>
      ) : (
        <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
          <Button
            variant="signal"
            size="lg"
            onClick={onConnect}
            data-testid="hero-connect-button"
          >
            Connect Wallet
          </Button>
        </motion.div>
      )}
      <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
        <Button
          variant="ghost"
          size="lg"
          onClick={onNavigateDocs}
          data-testid="hero-docs-button"
        >
          Open Docs
        </Button>
      </motion.div>
    </div>
  </motion.section>
);
