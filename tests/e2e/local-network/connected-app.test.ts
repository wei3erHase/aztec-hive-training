/**
 * E2E tests for the connected application state (local network).
 *
 * These tests require:
 *   - A running local Aztec network: `aztec start --local-network`
 *   - Contracts deployed: `yarn deploy-contracts --network local-network`
 *   - App server running on port 3000: `yarn dev` or `yarn serve`
 *
 * Run with: yarn test:e2e:local-network
 */

import { test, expect } from './wallet.fixture';
import type { Page } from '@playwright/test';

// ============================================================================
// Connected UI structure
// ============================================================================

test.describe('Connected state — UI structure', () => {
  test('canvas panel, architecture buttons, and predictions panel are visible', async ({
    connectedPage: page,
  }) => {
    await expect(page.locator('[data-testid="canvas-panel"]')).toBeVisible();
    await expect(
      page.locator('[data-testid="architecture-singleLayer"]')
    ).toBeVisible();
    await expect(
      page.locator('[data-testid="architecture-mlp"]')
    ).toBeVisible();
    await expect(
      page.locator('[data-testid="architecture-cnnGap"]')
    ).toBeVisible();
    await expect(
      page.locator('[data-testid="predictions-panel"]')
    ).toBeVisible();
    await expect(
      page.locator('[data-testid="prediction-status-card"]')
    ).toContainText(/draw a digit/i);
  });
});

// ============================================================================
// Draw → Predict flow
// ============================================================================

async function drawAndPredict(page: Page, architectureTestId: string) {
  await page.locator(`[data-testid="${architectureTestId}"]`).click();

  const canvas = page.locator('[data-testid="canvas-panel"] canvas').first();
  await expect(canvas).toBeVisible();

  const drawStroke = async () => {
    // Scroll the canvas into the viewport and get a fresh bounding box each time.
    // The canvas may not be in the visible area on the first attempt (page renders
    // below the fold), so we can't rely on a box computed before the first draw.
    await canvas.scrollIntoViewIfNeeded();
    const box = await canvas.boundingBox();
    if (!box) throw new Error('Canvas bounding box is null');

    // Use canvas.hover() to move the cursor onto the canvas via Playwright's
    // locator-aware action (handles any remaining scroll offset automatically).
    // Then use page.mouse for the full drag gesture. Each `await` between
    // mousedown and mouseup lets React commit isDrawing=true before stopDrawing
    // runs, which is required for onDraw to fire.
    const hw = box.width * 0.5;
    const hh = box.height * 0.5;
    const offsets = [
      { x: -40, y: -30 },
      { x: -10, y: -50 },
      { x: +30, y: -10 },
      { x: 0, y: +25 },
      { x: +40, y: +40 },
    ];

    await canvas.hover({ position: { x: hw - 40, y: hh - 30 } });
    await page.mouse.down();
    for (const { x, y } of offsets) {
      await canvas.hover({ position: { x: hw + x, y: hh + y } });
    }
    await page.mouse.up();
  };

  const clearCanvas = async () => {
    const clearBtn = page.locator('[data-testid="canvas-clear-button"]');
    if ((await clearBtn.count()) === 0) return;
    const visible = await clearBtn
      .first()
      .isVisible()
      .catch(() => false);
    if (visible) {
      await clearBtn.first().click();
    }
  };

  const waitForPredictedDigit = async (
    timeoutMs: number
  ): Promise<number | null> => {
    const deadline = Date.now() + timeoutMs;
    const status = page.locator('[data-testid="prediction-status-card"]');

    while (Date.now() < deadline) {
      const text = (await status.textContent()) ?? '';
      const match = text.match(/predicted:\s*(\d)/i);
      if (match) return Number(match[1]);
      if (
        /error:/i.test(text) ||
        /not deployed/i.test(text) ||
        /failed/i.test(text)
      ) {
        throw new Error(`Prediction failed: ${text}`);
      }
      await page.waitForTimeout(1_000);
    }

    return null;
  };

  for (let attempt = 1; attempt <= 3; attempt++) {
    await drawStroke();
    // 180s per attempt — the MLP Noir circuit takes ~140s on a typical machine.
    // Keeping 3 attempts for robustness; total max wait ≤ 9 min (< 10 min timeout).
    const digit = await waitForPredictedDigit(180_000);
    if (digit !== null) {
      await expect(
        page.locator(`[data-testid="prediction-cell-${digit}"]`)
      ).toBeVisible({ timeout: 5_000 });
      return digit;
    }
    await clearCanvas();
  }

  throw new Error('Prediction was not produced after 3 draw attempts');
}

test.describe('Draw → Predict flow', () => {
  test('drawing on the canvas produces a digit prediction (MLP)', async ({
    connectedPage: page,
  }) => {
    await drawAndPredict(page, 'architecture-mlp');
  });

  test('drawing on the canvas produces a digit prediction (Single Layer)', async ({
    connectedPage: page,
  }) => {
    await drawAndPredict(page, 'architecture-singleLayer');
  });

  test('drawing on the canvas produces a digit prediction (CNN+GAP)', async ({
    connectedPage: page,
  }) => {
    await drawAndPredict(page, 'architecture-cnnGap');
  });
});

// ============================================================================
// Training submission — full on-chain flow
// ============================================================================

test.describe('Training submission', () => {
  test('train button is visible and disabled before drawing', async ({
    connectedPage: page,
  }) => {
    const trainBtn = page.locator('[data-testid="train-submit-button"]');
    await expect(trainBtn).toBeVisible();
    await expect(trainBtn).toBeDisabled();
  });

  test('full on-chain train flow: draw → predict → label → submit → success', async ({
    connectedPage: page,
  }) => {
    const digit = await drawAndPredict(page, 'architecture-mlp');
    await page.locator(`[data-testid="prediction-cell-${digit}"]`).click();

    // Submit training on-chain
    const trainBtn = page.locator('[data-testid="train-submit-button"]');
    await expect(trainBtn).toBeEnabled({ timeout: 5_000 });
    await trainBtn.click();

    // Wait for the status message to appear and track its progress
    const statusMsg = page.locator('[data-testid="training-status-message"]');
    await expect(statusMsg).toBeVisible({ timeout: 10_000 });

    // The tx must reach PROPOSED status (waitForStatus: PROPOSED in useTrainOnChain).
    // On local-network this typically completes within 30-60s.
    // If this assertion fails, check the browser:error logs above for the real cause.
    await expect(statusMsg).toContainText(
      /Training submitted|Training complete/i,
      {
        timeout: 180_000,
      }
    );

    // Status badge should reflect success (no error)
    await expect(statusMsg).not.toContainText(/^Error:/i);
  });
});
