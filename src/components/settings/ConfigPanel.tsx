import React from 'react';
import { Link, Rocket } from 'lucide-react';
import { getDeploymentConfig } from '../../config/contracts';
import { iconSize } from '../../utils';
import { WeightsViewer } from '../docs/WeightsViewer';
import { ConfigPanelHeader } from './ConfigPanelHeader';
import { ConfigSection } from './ConfigSection';
import { ConfigValueRow } from './ConfigValueRow';
import type { NetworkConfig } from '../../config/networks/types';

export interface ConfigPanelProps {
  config: NetworkConfig;
  action?: React.ReactNode;
}

const styles = {
  container:
    'flex-1 flex flex-col gap-4 md:gap-7 bg-page px-4 py-4 lg:px-10 lg:py-8',
  sectionsContainer: 'flex flex-col gap-4 md:gap-5',
  grid2Col: 'grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4',
  contractsGap: 'flex flex-col gap-3 md:gap-4',
} as const;

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ config, action }) => {
  const deployment = getDeploymentConfig(config.name);
  const contracts = deployment.contracts;
  const deployedCount = [
    contracts.singleLayer.address,
    contracts.multiLayerPerceptron.address,
    contracts.cnnGap.address,
  ].filter(Boolean).length;

  return (
    <div className={styles.container}>
      <ConfigPanelHeader
        displayName={config.displayName}
        description={config.description}
        action={action}
      />

      <div className={styles.sectionsContainer}>
        <ConfigSection
          icon={<Link size={iconSize()} />}
          iconVariant="green"
          title="Connection"
          badge={{ text: 'Online', variant: 'online' }}
        >
          <ConfigValueRow
            label="Node URL"
            value={config.publicNodeUrl ?? config.nodeUrl}
          />
        </ConfigSection>

        <ConfigSection
          icon={<Rocket size={iconSize()} />}
          iconVariant="amber"
          title="Deployment Info"
          badge={{
            text: `${deployedCount} deployed`,
            variant: 'count',
          }}
        >
          <div className={styles.contractsGap}>
            <ConfigValueRow
              label="Single Layer"
              value={contracts.singleLayer.address || '—'}
              badge={{ text: '64→10', variant: 'blue' }}
            />
            <ConfigValueRow
              label="Multi-Layer Perceptron"
              value={contracts.multiLayerPerceptron.address || '—'}
              badge={{ text: '64→16→10', variant: 'purple' }}
            />
            <ConfigValueRow
              label="CNN-GAP"
              value={contracts.cnnGap.address || '—'}
              badge={{ text: 'CNN', variant: 'red' }}
            />
            <WeightsViewer />
          </div>
        </ConfigSection>
      </div>
    </div>
  );
};
