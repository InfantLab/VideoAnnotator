/**
 * One way to tell the user something happened, with the next step as a
 * button (spec 029 FR-025): "Run started on 48 videos" + Show run. Over the
 * sonner toaster mounted in App.
 */
import { toast } from 'sonner';

export interface NotifyAction {
  label: string;
  /** A viewer address to go to, e.g. `/videos?run=b1`. */
  to?: string;
  onClick?: () => void;
}

export interface Notice {
  title: string;
  body?: string;
  action?: NotifyAction;
  tone?: 'info' | 'success' | 'error';
}

let navigate: ((to: string) => void) | null = null;

/** Lets actions with `to` move within the viewer; set by `NotifyNavigator`. */
export function setNotifyNavigator(fn: ((to: string) => void) | null): void {
  navigate = fn;
}

function go(to: string): void {
  if (navigate) navigate(to);
  else window.location.assign(`${import.meta.env.BASE_URL.replace(/\/$/, '')}${to}`);
}

export function notify({ title, body, action, tone = 'info' }: Notice): string | number {
  const show = tone === 'success' ? toast.success : tone === 'error' ? toast.error : toast;
  return show(title, {
    description: body,
    duration: tone === 'error' ? 10_000 : 6_000,
    action: action
      ? {
          label: action.label,
          onClick: () => {
            action.onClick?.();
            if (action.to) go(action.to);
          },
        }
      : undefined,
  });
}
