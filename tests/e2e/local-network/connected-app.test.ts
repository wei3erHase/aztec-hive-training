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

  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  if (!box) {
    throw new Error('Canvas bounding box is null');
  }

  const drawStroke = async () => {
    const cx = box.x + box.width * 0.5;
    const cy = box.y + box.height * 0.5;
    const points = [
      { x: cx - 55, y: cy - 40 },
      { x: cx - 25, y: cy - 65 },
      { x: cx + 20, y: cy - 20 },
      { x: cx - 10, y: cy + 25 },
      { x: cx + 35, y: cy + 45 },
      { x: cx + 55, y: cy + 5 },
    ];

    // Prefer synthetic events (more deterministic in CI/headless); fallback to native mouse.
    await canvas.evaluate((el, localPoints) => {
      const fire = (type: string, x: number, y: number, buttons: number) => {
        const event = new MouseEvent(type, {
          bubbles: true,
          cancelable: true,
          clientX: x,
          clientY: y,
          buttons,
        });
        el.dispatchEvent(event);
      };

      const first = localPoints[0];
      fire('mousedown', first.x, first.y, 1);
      for (let i = 1; i < localPoints.length; i++) {
        const p = localPoints[i];
        fire('mousemove', p.x, p.y, 1);
      }
      const last = localPoints[localPoints.length - 1];
      fire('mouseup', last.x, last.y, 0);
    }, points);

    const clearBtn = page.locator('[data-testid="canvas-clear-button"]');
    const hasClearBtn = await clearBtn
      .first()
      .isVisible()
      .catch(() => false);
    if (hasClearBtn) return;

    await page.mouse.move(points[0].x, points[0].y);
    await page.mouse.down();
    for (let i = 1; i < points.length; i++) {
      await page.mouse.move(points[i].x, points[i].y, { steps: 12 });
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
    const digit = await waitForPredictedDigit(20_000);
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
