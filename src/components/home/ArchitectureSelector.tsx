import React from 'react';
import { Button, Badge } from '../ui';
import type { ArchitectureId } from '../../config/contracts';

interface ArchitectureSelectorProps {
  architecture: ArchitectureId;
  isArchitectureAvailable: boolean;
  onSelect: (id: ArchitectureId) => void;
}

const ARCHITECTURES: { id: ArchitectureId; label: string }[] = [
  { id: 'singleLayer', label: 'Simple Linear' },
  { id: 'mlp', label: 'Multi-Layer Network' },
  { id: 'cnnGap', label: 'CNN + GAP' },
];

const ARCH_LABELS: Record<ArchitectureId, string> = {
  singleLayer: 'Single Layer',
  mlp: 'Multi-Layer Network',
  cnnGap: 'CNN',
};

export const ArchitectureSelector: React.FC<ArchitectureSelectorProps> = ({
  architecture,
  isArchitectureAvailable,
  onSelect,
}) => (
  <div className="panel rounded-2xl p-4 sm:p-5 md:p-6">
    <div className="mb-3 flex items-center justify-between gap-2">
      <span className="mono-label text-muted">Architectures</span>
      <span className="text-right text-sm text-muted sm:text-xs">
        Switch live inference target
      </span>
    </div>

    <div className="mb-4 grid gap-2 sm:grid-cols-3">
      {ARCHITECTURES.map(({ id, label }) => (
        <Button
          key={id}
          data-testid={`architecture-${id}`}
          onClick={() => onSelect(id)}
          variant={architecture === id ? 'primary' : 'ghost'}
          size="sm"
          className="w-full justify-center"
        >
          {label}
        </Button>
      ))}
    </div>

    {!isArchitectureAvailable && (
      <Badge variant="warning" className="mb-4 w-full justify-start">
        {ARCH_LABELS[architecture]} is not deployed on this network. Use the
        local network for all architectures, or choose Multi-Layer Network on
        devnet.
      </Badge>
    )}

    <p className="rounded-lg border border-default bg-interactive px-3 py-2 text-base text-default sm:text-sm">
      Draw, review the predicted digit, pick a training target in the matrix,
      then submit one on-chain update.
    </p>
  </div>
);
