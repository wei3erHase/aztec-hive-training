/**
 * Fallback when virtual:deployed-local-config is not available (e.g. Vitest).
 * Vite's injectDeployedLocal plugin provides the real config in dev/build.
 */
export default {} as Record<string, unknown>;
