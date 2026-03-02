import React from 'react';
import { motion } from 'motion/react';
import { Button } from '../ui';

interface ConnectPromptProps {
  onConnect: () => void;
}

export const ConnectPrompt: React.FC<ConnectPromptProps> = ({ onConnect }) => (
  <div className="mx-auto w-full max-w-xl rounded-2xl panel p-6 text-center sm:p-8">
    <div className="mb-4 text-5xl">🐝</div>
    <h3 className="mb-2 text-lg font-semibold text-default sm:text-xl">
      Connect Wallet to Start
    </h3>
    <p className="mb-6 text-base text-muted sm:text-sm">
      Connect your wallet to unlock prediction, explanation, and on-chain
      training flows.
    </p>
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className="inline-block"
    >
      <Button
        variant="signal"
        size="lg"
        onClick={onConnect}
        data-testid="demo-connect-wallet-button"
      >
        Connect Wallet
      </Button>
    </motion.div>
  </div>
);
