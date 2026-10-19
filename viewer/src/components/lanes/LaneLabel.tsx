interface LaneLabelProps {
  title: string;
  subtitle?: string;
}

export function LaneLabel({ title, subtitle }: LaneLabelProps) {
  return (
    <div className="min-w-0 truncate text-xs leading-tight" title={subtitle ? `${title} · ${subtitle}` : title}>
      <span className="font-medium">{title}</span>
      {subtitle && <span className="text-muted-foreground"> · {subtitle}</span>}
    </div>
  );
}
