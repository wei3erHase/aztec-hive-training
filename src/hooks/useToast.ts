import { create } from 'zustand';

export type ToastVariant = 'success' | 'error' | 'warning' | 'info' | 'loading';

export interface ToastItem {
  id: string;
  title: string;
  description?: string;
  variant: ToastVariant;
  duration?: number;
}

interface ToastStore {
  toasts: ToastItem[];
  add: (toast: Omit<ToastItem, 'id'>) => string;
  dismiss: (id: string) => void;
  clear: () => void;
}

let counter = 0;

const useToastStore = create<ToastStore>((set) => ({
  toasts: [],
  add: (toast) => {
    const id = String(++counter);
    set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
    const duration =
      toast.duration ?? (toast.variant === 'loading' ? Infinity : 5000);
    if (duration !== Infinity) {
      setTimeout(
        () =>
          set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
        duration
      );
    }
    return id;
  },
  dismiss: (id) =>
    set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
  clear: () => set({ toasts: [] }),
}));

export function useToast() {
  const { add, dismiss, clear } = useToastStore();

  return {
    success: (title: string, description?: string) =>
      add({ variant: 'success', title, description }),
    error: (title: string, description?: string) =>
      add({ variant: 'error', title, description }),
    warning: (title: string, description?: string) =>
      add({ variant: 'warning', title, description }),
    info: (title: string, description?: string) =>
      add({ variant: 'info', title, description }),
    loading: (title: string, description?: string) =>
      add({ variant: 'loading', title, description, duration: Infinity }),
    dismiss,
    clear,
  };
}

export { useToastStore };
