import React from 'react';
import { useAppNavigation } from '../hooks';
import { DocsPage } from '../pages/DocsPage';
import { Home } from '../pages/Home';
import { SettingsCard } from './SettingsCard';

const styles = {
  main: 'flex flex-col flex-1',
} as const;

export const MainContent: React.FC = () => {
  const { activeTab } = useAppNavigation();

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <Home />;
      case 'docs':
        return <DocsPage />;
      case 'settings':
        return (
          <div className="w-full max-w-[1400px] mx-auto px-0 lg:px-6 xl:px-10 py-0 lg:py-6">
            <SettingsCard />
          </div>
        );
      default:
        return <Home />;
    }
  };

  return <main className={styles.main}>{renderContent()}</main>;
};
