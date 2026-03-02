import React from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { useAztecWallet, AztecWalletProvider } from './aztec-wallet';
import { Header, NetworkError } from './components';
import { TooltipProvider, Toaster } from './components/ui';
import { walletConfig } from './config/walletConfig';
import { Layout } from './containers';
import { useAppNavigation } from './hooks';
import { queryClient } from './lib/queryClient';
import { AppNavigationProvider } from './providers/AppNavigationProvider';
import { cn } from './utils';

const styles = {
  container: 'min-h-screen',
  bgSettings: 'bg-page',
  bgDefault: 'bg-surface flex flex-col',
  main: 'flex flex-col',
  errorContainer: 'w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-6',
} as const;

const AppContent: React.FC = () => {
  const { networkStatus, networkError, networkName, checkNetwork } =
    useAztecWallet();
  const { activeTab } = useAppNavigation();

  const showNetworkError = networkStatus === 'error';

  return (
    <div
      className={cn(
        styles.container,
        activeTab === 'settings' ? styles.bgSettings : styles.bgDefault
      )}
    >
      <Header />

      {showNetworkError && (
        <div className={styles.errorContainer}>
          <NetworkError
            error={networkError}
            networkName={networkName}
            onRetry={checkNetwork}
          />
        </div>
      )}

      <main className={styles.main}>
        <Layout />
      </main>
    </div>
  );
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AztecWalletProvider config={walletConfig}>
          <AppNavigationProvider defaultTab="home">
            <AppContent />
            <Toaster />
          </AppNavigationProvider>
        </AztecWalletProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
