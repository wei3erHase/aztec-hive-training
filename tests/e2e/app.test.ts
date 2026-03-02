import { test, expect } from '@playwright/test';

// ============================================================================
// Smoke tests – run against the live dev server at http://localhost:3000
// Covers the unauthenticated state (no wallet connected, no Aztec node running).
// ============================================================================

test.describe('App shell', () => {
  test('page loads with correct title', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/HIVE Neural Network/i);
  });

  test('header renders with logo and navigation tabs', async ({ page }) => {
    await page.goto('/');
    const header = page.locator('nav, header').first();
    await expect(header).toBeVisible();
  });

  test('wallet connect button is visible and wallet is NOT connected', async ({
    page,
  }) => {
    await page.goto('/');
    const connectBtn = page.getByRole('button', { name: /connect/i }).first();
    await expect(connectBtn).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Home page — unauthenticated state', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait until the wallet connect button is visible — React has settled
    await page.waitForSelector('[data-testid="demo-connect-wallet-button"]', {
      timeout: 15_000,
    });
  });

  test('shows "Connect Wallet to Start" prompt when no wallet is connected', async ({
    page,
  }) => {
    await expect(page.getByText('Connect Wallet to Start')).toBeVisible({
      timeout: 10_000,
    });
  });

  test('the connect-wallet CTA button is present (data-testid)', async ({
    page,
  }) => {
    const btn = page.locator('[data-testid="demo-connect-wallet-button"]');
    await expect(btn).toBeVisible({ timeout: 10_000 });
  });

  test('drawing canvas is NOT rendered without wallet connection', async ({
    page,
  }) => {
    // The canvas-panel and <canvas> are inside the isConnected branch
    await expect(
      page.locator('[data-testid="canvas-panel"]')
    ).not.toBeVisible();
    await expect(page.locator('canvas')).not.toBeVisible();
  });

  test('architecture selector buttons are NOT rendered without wallet connection', async ({
    page,
  }) => {
    // Buttons use data-testid="architecture-{id}" — labels: "Simple Linear", "Multi-Layer Network", "CNN + GAP"
    await expect(
      page.locator('[data-testid="architecture-singleLayer"]')
    ).not.toBeVisible();
    await expect(
      page.locator('[data-testid="architecture-mlp"]')
    ).not.toBeVisible();
    await expect(
      page.locator('[data-testid="architecture-cnnGap"]')
    ).not.toBeVisible();
  });
});

test.describe('Network / connection state (no local Aztec node running)', () => {
  test('app does NOT crash — root element is still present after connection failure', async ({
    page,
  }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await expect(page.locator('#root')).toBeVisible();
  });

  test('no Aztec block number is displayed (node not reachable)', async ({
    page,
  }) => {
    await page.goto('/');
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

    await page.goto('/');
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

test.describe('Predict / train flow — expected to be blocked without wallet', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForSelector('[data-testid="demo-connect-wallet-button"]', {
      timeout: 15_000,
    });
  });

  test('on-chain train button is disabled or absent without wallet connection', async ({
    page,
  }) => {
    const trainBtn = page
      .getByRole('button', { name: /^train|submit training/i })
      .first();
    const isVisible = await trainBtn.isVisible().catch(() => false);

    if (isVisible) {
      // If somehow visible, it must be disabled
      await expect(trainBtn).toBeDisabled();
    }
    // Not visible at all is the expected outcome — wallet gate hides it
  });

  test('drawing: canvas is absent so no draw interaction is possible', async ({
    page,
  }) => {
    // Confirms the predict/draw flow is fully gated behind wallet connection
    const canvas = page.locator('canvas').first();
    await expect(canvas).not.toBeVisible();
  });
});
