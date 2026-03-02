export type IconSize = 'sm' | 'md' | 'lg' | 'xl' | '2xl';

const SIZES: Record<IconSize, number> = {
  sm: 14,
  md: 18,
  lg: 20,
  xl: 24,
  '2xl': 32,
};

export function iconSize(size: IconSize = 'md'): number {
  return SIZES[size] ?? SIZES.md;
}
