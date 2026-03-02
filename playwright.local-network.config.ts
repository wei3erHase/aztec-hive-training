/**
 * Playwright config for local-network e2e tests.
 *
 * These tests require a full local Aztec network (sequencer + node + PXE).
 * Playwright manages both the app server and the Aztec node as webServer
 * processes, so the suite is fully self-contained:
 *
 *   yarn test:e2e:local-network          # dev machine (reuses existing servers)
 *   CI=true yarn test:e2e:local-network  # CI (starts everything from scratch)
 *
 * Contracts are automatically deployed via the wallet fixture before tests run.
 */
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e/local-network',
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  reporter: 'list',

  use: {
    baseURL: 'http://localhost:3000',
    // Default to headless for stability; opt-in to headed mode with PW_HEADED=true.
    headless: process.env.PW_HEADED !== 'true',
    launchOptions: { devtools: process.env.PW_DEVTOOLS === 'true' },
  },

  expect: { timeout: 30_000 },

  timeout: 300_000,

  projects: [
    {
      name: 'local-network',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: [
    {
      command: 'PORT=3000 yarn serve',
      url: 'http://localhost:3000',
      reuseExistingServer: true,
      timeout: 30_000,
    },
    {
      command:
        'aztec start --local-network --l1-rpc-urls http://localhost:8545',
      url: 'http://localhost:8080/status',
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});
