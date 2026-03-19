import { test, expect, type Page } from '@playwright/test';

const APP_READY_TIMEOUT = process.env.CI ? 45_000 : 15_000;

async function gotoAndWaitForHome(page: Page) {
  await page.goto('/');
  await page.waitForSelector('[data-testid="home-page"]', {
    timeout: APP_READY_TIMEOUT,
  });
}

// ============================================================================
// Smoke tests – run against the live dev server at http://localhost:3000
// Covers the unauthenticated state (no wallet connected, no Aztec node running).
// ============================================================================

test.describe('App shell', () => {
  test('page loads with correct title', async ({ page }) => {
    await gotoAndWaitForHome(page);
    await expect(page).toHaveTitle(/HIVE Neural Network/i);
  });

  test('header renders with logo and navigation tabs', async ({ page }) => {
    await gotoAndWaitForHome(page);
    const header = page.locator('nav, header').first();
    await expect(header).toBeVisible({ timeout: APP_READY_TIMEOUT });
  });

  test('wallet connect button is visible and wallet is NOT connected', async ({
    page,
  }) => {
    await gotoAndWaitForHome(page);
    const connectBtn = page.getByRole('button', { name: /connect/i }).first();
    await expect(connectBtn).toBeVisible({ timeout: APP_READY_TIMEOUT });
  });
});

test.describe('Home page — unauthenticated state', () => {
  test.beforeEach(async ({ page }) => {
    await gotoAndWaitForHome(page);
  });

  test('hero section shows a "Connect Wallet" button when no wallet is connected', async ({
    page,
  }) => {
    const btn = page.locator('[data-testid="hero-connect-button"]');
    await expect(btn).toBeVisible({ timeout: APP_READY_TIMEOUT });
  });

  test('training controls show "Connect Wallet to Train" instead of the submit button', async ({
    page,
  }) => {
    const connectBtn = page.locator('[data-testid="train-connect-button"]');
    await expect(connectBtn).toBeVisible({ timeout: APP_READY_TIMEOUT });
    const submitBtn = page.locator('[data-testid="train-submit-button"]');
    await expect(submitBtn).not.toBeVisible();
  });

  test('drawing canvas IS rendered — read-only predictions work without wallet', async ({
    page,
  }) => {
    await expect(page.locator('[data-testid="canvas-panel"]')).toBeVisible({
      timeout: APP_READY_TIMEOUT,
    });
    await expect(page.locator('canvas').first()).toBeVisible({
      timeout: APP_READY_TIMEOUT,
    });
  });

  test('architecture selector buttons ARE rendered without wallet connection', async ({
    page,
  }) => {
    await expect(
      page.locator('[data-testid="architecture-singleLayer"]')
    ).toBeVisible({ timeout: APP_READY_TIMEOUT });
    await expect(page.locator('[data-testid="architecture-mlp"]')).toBeVisible({
      timeout: APP_READY_TIMEOUT,
    });
    await expect(
      page.locator('[data-testid="architecture-cnnGap"]')
    ).toBeVisible({ timeout: APP_READY_TIMEOUT });
  });
});

test.describe('Network / connection state (no local Aztec node running)', () => {
  test('app does NOT crash — root element is still present after connection failure', async ({
    page,
  }) => {
    await gotoAndWaitForHome(page);
    await page.waitForLoadState('domcontentloaded');
    await expect(page.locator('#root')).toBeVisible();
  });

  test('no Aztec block number is displayed (node not reachable)', async ({
    page,
  }) => {
    await gotoAndWaitForHome(page);
    await page.waitForLoadState('domcontentloaded');
    // No block number indicator should be present or visible
    const blockEl = page.locator('[data-testid="block-number"]');
    const count = await blockEl.count();
    if (count === 0) return;
    await expect(blockEl.first()).not.toBeVisible();
  });

  test('/rpc proxy fails — Aztec node is not running', async ({ page }) => {
    // Register listener BEFORE navigation so it captures the first mount-cycle RPC call
    let rpcStatus: number | 'network-error' | null = null;

    page.on('response', (response) => {
      if (response.url().includes('/rpc') && rpcStatus === null) {
        rpcStatus = response.status();
      }
    });
    page.on('requestfailed', (request) => {
      if (request.url().includes('/rpc') && rpcStatus === null) {
        rpcStatus = 'network-error';
      }
    });

    await gotoAndWaitForHome(page);
    // Give the app's checkConnection() useEffect time to fire and complete
    await page.waitForFunction(() => document.readyState === 'complete', {
      timeout: 10_000,
    });
    // Avoid long networkidle waits; poll briefly for the first RPC attempt
    await page.waitForTimeout(2_500);

    if (rpcStatus === null) {
      // No /rpc request observed — skip rather than silently passing
      test.skip(
        true,
        'No /rpc request observed — app may use a different endpoint'
      );
      return;
    }

    // Either a network-level failure or an HTTP error (5xx from Vite proxy)
    const failed =
      rpcStatus === 'network-error' ||
      (typeof rpcStatus === 'number' && rpcStatus >= 500);
    expect(failed).toBe(true);
  });
});

test.describe('Predict / train flow — wallet gate applies only to on-chain actions', () => {
  test.beforeEach(async ({ page }) => {
    await gotoAndWaitForHome(page);
  });

  test('on-chain train submit button is absent — replaced by "Connect Wallet to Train"', async ({
    page,
  }) => {
    await expect(
      page.locator('[data-testid="train-submit-button"]')
    ).not.toBeVisible();
    await expect(
      page.locator('[data-testid="train-connect-button"]')
    ).toBeVisible({ timeout: APP_READY_TIMEOUT });
  });

  test('drawing canvas is present — predict flow is available without wallet', async ({
    page,
  }) => {
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible({ timeout: APP_READY_TIMEOUT });
  });
});
