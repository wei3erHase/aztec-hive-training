export { MinimalWallet } from './MinimalWallet';
export { queuePxeCall } from './pxeQueue';
export { toTitleCase } from './string';
export { parseAddressFromCaip, getChainFromCaipAccount } from './azguard';
export { cn } from './cn';
export { iconSize, type IconSize } from './iconSize';
export { truncateAddress } from './truncateAddress';
export { formatRelativeTime } from './format';

export function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((x) => typeof x === 'string');
}

export const isValidConfig = (
  config: { nodeUrl?: string; name?: string } | null
): boolean => !!(config?.nodeUrl && config?.name);
