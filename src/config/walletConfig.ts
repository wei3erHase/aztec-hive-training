import { createAztecWalletConfig } from '../aztec-wallet';
import { AVAILABLE_NETWORKS } from './networks';

export const walletConfig = createAztecWalletConfig({
  networks: AVAILABLE_NETWORKS.map((n) => ({
    name: n.name,
    nodeUrl: n.nodeUrl,
    displayName: n.displayName,
  })),
  defaultNetwork: 'devnet',
  showNetworkPicker: 'full',
  walletGroups: {
    embedded: true,
    aztecWallets: ['azguard'],
  },
});
