import { defineConfig, devices } from '@playwright/test';

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  workers: process.env.PLAYWRIGHT_NUM_WORKERS
    ? Number(process.env.PLAYWRIGHT_NUM_WORKERS)
    : 1,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:3000',
    // Default to headless for stability; opt-in to headed mode with PW_HEADED=true.
    headless: process.env.PW_HEADED !== 'true',
    launchOptions: {
      devtools: process.env.PW_DEVTOOLS === 'true',
    },
  },
  expect: {
    timeout: process.env.CI ? 45_000 : 20_000,
  },
  timeout: 400_000,
  projects: [
    {
      name: 'webkit',
      // local-network tests live in their own config (playwright.local-network.config.ts)
      testIgnore: ['**/local-network/**'],
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'chromium',
      testIgnore: ['**/local-network/**'],
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      testIgnore: ['**/local-network/**'],
      use: { ...devices['Desktop Firefox'] },
    },
  ],

  webServer: {
    command: process.env.CI
      ? 'PORT=3000 yarn dev'
      : 'PORT=3000 yarn serve',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: process.env.CI ? 120_000 : 30_000,
  },
});
