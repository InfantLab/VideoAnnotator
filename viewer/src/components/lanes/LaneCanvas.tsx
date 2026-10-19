import { useEffect, useRef } from 'react';

import {
  densityBins,
  formatValue,
  rowHeight,
  showsDensity,
  visibleSlice,
  type Lane,
  type LaneEvent,
  type LaneRow,
  type ViewRange,
} from '@/lib/lanes';

interface LaneCanvasProps {
  lane: Lane;
  view: ViewRange;
  width: number;
  /** The event whose table row is selected, outlined (SC-005). */
  selectedRow?: number | null;
}

const x = (t: number, view: ViewRange, width: number) => ((t - view.start) / (view.end - view.start)) * width;

function drawDensity(ctx: CanvasRenderingContext2D, row: LaneRow, view: ViewRange, width: number, top: number, height: number) {
  const bins = densityBins(row.events, view, width, row.maxDuration);
  let most = 0;
  for (const n of bins) if (n > most) most = n;
  if (!most) return;
  ctx.fillStyle = row.colour;
  for (let column = 0; column < bins.length; column++) {
    if (!bins[column]) continue;
    const h = Math.max(1, (bins[column] / most) * (height - 2));
    ctx.fillRect(column, top + height - 1 - h, 1, h);
  }
}

function drawNumeric(ctx: CanvasRenderingContext2D, row: LaneRow, events: LaneEvent[], view: ViewRange, width: number, top: number, height: number) {
  const { min = 0, max = 0 } = row;
  const range = max - min || 1;
  const y = (v: number) => top + height - 3 - ((v - min) / range) * (height - 6);
  ctx.strokeStyle = row.colour;
  ctx.fillStyle = row.colour;
  ctx.lineWidth = 1.5;
  let previous: LaneEvent | null = null;
  ctx.beginPath();
  for (const e of events) {
    if (e.value === null) continue;
    const px = x(e.start, view, width);
    // Joined only within the sampling interval: no line across a gap (Principle VI).
    const joined =
      previous !== null && row.interval !== null && e.start - previous.start <= row.interval * 1.001 + 1e-6;
    if (joined) ctx.lineTo(px, y(e.value));
    else {
      ctx.moveTo(px, y(e.value));
      ctx.fillRect(px - 1, y(e.value) - 1, 2, 2);
    }
    previous = e;
  }
  ctx.stroke();
  ctx.fillStyle = 'rgba(128, 128, 128, 0.9)';
  ctx.font = '9px sans-serif';
  const unit = row.unit ? ` ${row.unit}` : '';
  ctx.fillText(`${formatValue(max)}${unit}`, width - 44, top + 9);
  ctx.fillText(`${formatValue(min)}${unit}`, width - 44, top + height - 2);
}

const NOTE_COLOUR = 'hsl(38, 92%, 50%)';

/** Researchers' notes: outlined, never filled like pipeline output (Principle VI). */
function drawNotes(ctx: CanvasRenderingContext2D, events: LaneEvent[], view: ViewRange, width: number, top: number, height: number) {
  ctx.strokeStyle = NOTE_COLOUR;
  ctx.fillStyle = 'hsla(38, 92%, 50%, 0.18)';
  ctx.lineWidth = 1.5;
  for (const e of events) {
    const start = x(e.start, view, width);
    if (e.end === null) {
      ctx.beginPath();
      ctx.moveTo(start - 4, top + 2);
      ctx.lineTo(start + 4, top + 2);
      ctx.lineTo(start, top + height - 2);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      continue;
    }
    const end = x(e.end, view, width);
    ctx.fillRect(start, top + 2, Math.max(2, end - start), height - 4);
    ctx.strokeRect(start, top + 2, Math.max(2, end - start), height - 4);
  }
}

function drawRow(
  ctx: CanvasRenderingContext2D,
  row: LaneRow,
  view: ViewRange,
  width: number,
  top: number,
  selectedRow?: number | null,
  notes = false,
) {
  const height = rowHeight(row);
  const [from, to] = visibleSlice(row.events, view, row.maxDuration);
  const events = row.events.slice(from, to);
  if (notes) {
    drawNotes(ctx, events, view, width, top, height);
    return;
  }
  if (row.form === 'numeric') {
    drawNumeric(ctx, row, events, view, width, top, height);
    return;
  }
  if (showsDensity(row, view, width)) {
    drawDensity(ctx, row, view, width, top, height);
    return;
  }
  ctx.fillStyle = row.colour;
  for (const e of events) {
    const start = x(e.start, view, width);
    if (row.form === 'moment' || (row.form === 'span' && e.end === null)) {
      ctx.fillRect(start - 1, top + 2, 2, height - 4);
      continue;
    }
    const end = x(e.end ?? e.start, view, width);
    const inset = row.form === 'presence' ? 5 : 2;
    ctx.fillRect(start, top + inset, Math.max(2, end - start), height - inset * 2);
    if (row.form === 'span' && e.label && end - start > 30) {
      ctx.save();
      ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
      ctx.font = '10px sans-serif';
      const text = e.label.length * 6 > end - start - 6 ? `${e.label.slice(0, Math.max(1, Math.floor((end - start - 12) / 6)))}…` : e.label;
      ctx.fillText(text, Math.max(start, 0) + 3, top + height - 5);
      ctx.restore();
    }
  }
  if (selectedRow !== null && selectedRow !== undefined) {
    const selected = events.find((e) => e.row === selectedRow);
    if (selected) {
      const start = x(selected.start, view, width);
      const end = x(selected.end ?? selected.start, view, width);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.lineWidth = 2;
      ctx.strokeRect(start - 2, top + 1, Math.max(4, end - start + 4), height - 2);
    }
  }
}

/** One lane's rows, drawn for the current view (spec 028 R6). */
export function LaneCanvas({ lane, view, width, selectedRow }: LaneCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const height = lane.rows.reduce((sum, row) => sum + rowHeight(row), 0);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || width <= 0 || height <= 0) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    let top = 0;
    for (const row of lane.rows) {
      drawRow(ctx, row, view, width, top, selectedRow, lane.kind === 'notes');
      top += rowHeight(row);
    }
  }, [lane, view, width, height, selectedRow]);

  return (
    <canvas
      ref={ref}
      className="block"
      style={{ width: width > 0 ? `${width}px` : '100%', height: `${height}px` }}
      data-testid={`lane-canvas-${lane.id}`}
      aria-hidden="true"
    />
  );
}
