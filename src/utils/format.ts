/**
 * Formatting utility functions for displaying values.
 */

/**
 * Format a relative time string (e.g., "2s ago", "1m ago")
 */
export const formatRelativeTime = (date: Date | number | null): string => {
  if (date == null) return 'N/A';

  const ms = typeof date === 'number' ? date : date.getTime();
  const seconds = Math.floor((Date.now() - ms) / 1000);

  if (seconds < 5) return 'Now';
  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
};
