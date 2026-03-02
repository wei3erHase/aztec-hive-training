/**
 * Playwright fixture for local-network e2e tests.
 *
 * Strategy — "as the frontend does":
 *   1. Worker-scoped `contractsDeployed`: ensures contracts are deployed to local network.
 *      The Vite virtual module (virtual:deployed-local-config) reads deployed.local.json at
 *      server startup — no browser-side override is needed or possible.
 *   2. Worker-scoped `walletCredentials`: drives the full UI flow exactly once
 *      (Connect Wallet → Embedded Wallet). This creates + deploys a fresh ECDSA
 *      account on the local network via sponsored fees, then reads the resulting
 *      credentials from localStorage.
 *   3. Test-scoped `connectedPage`: seeds those credentials into a fresh browser
 *      page before navigation. The app auto-reconnects (account already on-chain)
 *      so each test starts in the connected state without the deployment wait.
 *
 * Prerequisites:
 *   - Local Aztec network running: `aztec start --local-network`
 *   - App dev/preview server running on port 3000 with deployed.local.json present
 */

import { test as base, expect } from '@playwright/test';
import type { Page } from '@playwright/test';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// ---- types mirrored from src/aztec-wallet/services/wallet/embeddedAccount.ts ----
interface StoredAccountData {
  address: string;
  signingKey: string;
  secretKey: string;
  salt: string;
}

// ---- fixture types ----
type WorkerFixtures = {
  /** Ensures contracts are deployed to local network before any test runs. */
  contractsDeployed: boolean;
  /** Embedded-account credentials, obtained once per worker via the full UI flow. */
  walletCredentials: StoredAccountData;
};

type TestFixtures = {
  /** A fresh page with the wallet already connected to local-network. */
  connectedPage: Page;
};

// ============================================================================
// Helpers
// ============================================================================

function ensureContractsDeployed(): boolean {
  const deploymentFile = path.join(
    process.cwd(),
    'config',
    'deployed.local.json'
  );

  if (fs.existsSync(deploymentFile)) {
    try {
      const deployed = JSON.parse(fs.readFileSync(deploymentFile, 'utf-8'));
      if (deployed['local-network']) {
        console.log('✅ Contracts already deployed to local network');
        return true;
      }
    } catch {
      console.log('⚠️  Could not parse deployment file, will redeploy');
    }
  }

  console.log('🔄 Deploying contracts to local network...');
  execSync('yarn deploy-contracts --network=local-network', {
    stdio: 'inherit',
    cwd: process.cwd(),
  });
  console.log('✅ Contracts deployed successfully');
  return true;
}

// ============================================================================
// Fixture implementation
// ============================================================================

export const test = base.extend<TestFixtures, WorkerFixtures>({
  // ------------------------------------------------------------------
  // Worker-scoped: ensure contracts are deployed before wallet/tests
  // NOTE: The Vite dev server must be restarted after first deployment
  // so the virtual:deployed-local-config module picks up deployed.local.json.
  // ------------------------------------------------------------------
  contractsDeployed: [
    async ({}, use) => {
      const deployed = ensureContractsDeployed();
      await use(deployed);
    },
    { scope: 'worker' },
  ],

  // ------------------------------------------------------------------
  // Worker-scoped: connect via UI once, share credentials across tests.
  // Depends on contractsDeployed so contracts exist before wallet creation.
  // ------------------------------------------------------------------
  walletCredentials: [
    async ({ browser, contractsDeployed: _ }, use) => {
      const context = await browser.newContext();
      const page = await context.newPage();

      // Capture browser console for diagnostics
      page.on('console', (msg) => {
        if (msg.type() === 'error') {
          console.error(`[browser:error] ${msg.text()}`);
        }
      });
      page.on('pageerror', (err) => {
        console.error(`[browser:pageerror] ${err.message}`);
      });

      // Select local-network before the app initialises its network store
      await page.addInitScript(() => {
        localStorage.setItem('aztec-wallet-network', 'local-network');
      });

      await page.goto('/');

      // Open the Connect Wallet modal
      const connectBtn = page
        .getByRole('button', { name: /connect wallet/i })
        .first();
      await expect(connectBtn).toBeVisible({ timeout: 15_000 });
      await connectBtn.click();

      // Select "Embedded Wallet" — creates + deploys a fresh ECDSA account
      const embeddedBtn = page.locator('[data-testid="wallet-group-embedded"]');
      await expect(embeddedBtn).toBeVisible({ timeout: 10_000 });
      await embeddedBtn.click();

      // Wait until the canvas panel appears — signals a successful connection
      // (account deployed on-chain via sponsored fees, PXE ready)
      await expect(page.locator('[data-testid="canvas-panel"]')).toBeVisible({
        timeout: 180_000,
      });

      // Read credentials that the app stored in localStorage
      const credentials = await page.evaluate<StoredAccountData>(() =>
        JSON.parse(localStorage.getItem('aztec-embedded-account') ?? '{}')
      );

      await context.close();
      await use(credentials);
    },
    { scope: 'worker' },
  ],

  // ------------------------------------------------------------------
  // Test-scoped: fresh page, credentials pre-seeded, auto-reconnects.
  // The account is already on-chain so reconnection is fast.
  // ------------------------------------------------------------------
  connectedPage: async ({ page, walletCredentials }, use) => {
    // Capture browser console for diagnostics — surfaces real errors during tests
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        console.error(`[browser:error] ${msg.text()}`);
      } else if (msg.type() === 'warning') {
        console.warn(`[browser:warn] ${msg.text()}`);
      }
    });
    page.on('pageerror', (err) => {
      console.error(`[browser:pageerror] ${err.message}`);
    });

    // Inject credentials + network selection before the first script runs.
    // The Vite virtual module already has contract addresses baked in from
    // deployed.local.json — no runtime override mechanism is needed.
    await page.addInitScript(
      ({ creds }) => {
        localStorage.setItem('aztec-embedded-account', JSON.stringify(creds));
        localStorage.setItem(
          'aztec-wallet-connection',
          JSON.stringify({ connectorId: 'embedded', walletType: 'embedded' })
        );
        localStorage.setItem('aztec-wallet-network', 'local-network');
      },
      { creds: walletCredentials }
    );

    await page.goto('/');

    // The account is already on-chain, so the browser PXE only needs to
    // initialise and register (no deployment wait). Still allow generous time
    // for the WASM PXE bundle to load and sync.
    await expect(page.locator('[data-testid="canvas-panel"]')).toBeVisible({
      timeout: 180_000,
    });

    await use(page);
  },
});

export { expect };
