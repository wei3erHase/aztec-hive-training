import { createContext, useContext } from 'react';
import type { TabType } from '../types';

interface AppNavigationContextValue {
  activeTab: TabType;
  setActiveTab: (tab: TabType) => void;
  navigate: (tab: TabType) => void;
}

export const AppNavigationContext =
  createContext<AppNavigationContextValue | null>(null);

export function useAppNavigation(): AppNavigationContextValue {
  const ctx = useContext(AppNavigationContext);
  if (!ctx) {
    throw new Error(
      'useAppNavigation must be used within AppNavigationProvider'
    );
  }
  return ctx;
}
