import React from 'react';
import { cn } from '../../utils';

export interface SkeletonProps {
  className?: string;
  rounded?: 'sm' | 'md' | 'lg' | 'full';
}

const roundedStyles = {
  sm: 'rounded',
  md: 'rounded-lg',
  lg: 'rounded-xl',
  full: 'rounded-full',
} as const;

export const Skeleton: React.FC<SkeletonProps> = ({
  className,
  rounded = 'md',
}) => (
  <div
    className={cn(
      'animate-pulse bg-gray-800/80',
      roundedStyles[rounded],
      className
    )}
  />
);

export const SkeletonText: React.FC<{ lines?: number; className?: string }> = ({
  lines = 3,
  className,
}) => (
  <div className={cn('space-y-2', className)}>
    {Array.from({ length: lines }).map((_, i) => (
      <Skeleton
        key={i}
        className={cn('h-4', i === lines - 1 && lines > 1 ? 'w-3/4' : 'w-full')}
      />
    ))}
  </div>
);
