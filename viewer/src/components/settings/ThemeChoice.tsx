import { Monitor, Moon, Sun } from 'lucide-react';
import { useTheme } from 'next-themes';

import { Button } from '@/components/ui/button';

const CHOICES = [
  { value: 'system', label: 'Like this computer', icon: Monitor },
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
] as const;

/** Light or dark: the system's unless chosen here (viewer design P9, UAT #53). */
export function ThemeChoice() {
  const { theme = 'system', setTheme } = useTheme();
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm" role="radiogroup" aria-label="Appearance" data-testid="theme-choice">
      <span className="text-muted-foreground">Appearance</span>
      {CHOICES.map(({ value, label, icon: Icon }) => (
        <Button
          key={value}
          size="sm"
          variant={theme === value ? 'default' : 'outline'}
          role="radio"
          aria-checked={theme === value}
          onClick={() => setTheme(value)}
        >
          <Icon className="mr-1 h-4 w-4" />
          {label}
        </Button>
      ))}
    </div>
  );
}
