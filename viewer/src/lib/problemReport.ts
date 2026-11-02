/**
 * A problem report as text, and where it can go (UAT #68): an email, a GitHub
 * issue, or the clipboard. The viewer sends nothing itself; the researcher
 * reads every detail first and sends it from their own mail or browser.
 */
import { formatWhen } from '@/lib/when';
import type { LoggedError } from '@/lib/errorLog';
import { GITHUB_URL, VERSION } from '@/utils/version';

export const REPORT_EMAIL = 'infantologist@gmail.com';
const ISSUE_TEMPLATE = 'bug_report.yml';
// Mail programs and GitHub both refuse very long addresses.
const MAX_BODY = 1800;

export interface ReportDetails {
  viewer: string;
  server: string | null;
  address: string | null;
  browser: string;
  screen: string;
  page: string;
  errors: LoggedError[];
}

export interface Report {
  what: string;
  expected?: string;
  /** Left out when the researcher unticks the details. */
  details: ReportDetails | null;
  /** Only when they leave it ticked. */
  video?: string | null;
}

const PAGES: [RegExp, string][] = [
  [/^\/(home)?$/, 'Home'],
  [/^\/videos/, 'Videos'],
  [/^\/runs/, 'Runs'],
  [/^\/run$/, 'New run'],
  [/^\/ask/, 'Ask AI'],
  [/^\/pipelines/, 'Pipelines'],
  [/^\/settings/, 'Settings'],
  [/^\/help/, 'Help'],
  [/^\/compare/, 'Compare'],
  [/^\/video$/, 'the Video page'],
  [/^\/viewer$/, 'the Video page, files from this computer'],
];

/** The page in the viewer's words, never its address (which can hold a path). */
export function pageName(pathname: string): string {
  const path = pathname.replace(/^\/viewer(?=\/)/, '') || '/';
  return PAGES.find(([pattern]) => pattern.test(path))?.[1] ?? 'another page';
}

/** "Firefox 132 on Linux", from the user agent; enough to reproduce, no more. */
export function browserName(userAgent: string): string {
  // Edge and Opera also say Chrome, and Chrome also says Safari: the most specific first.
  const found = [
    ['Edge', /Edg\/(\d+)/],
    ['Opera', /OPR\/(\d+)/],
    ['Firefox', /Firefox\/(\d+)/],
    ['Chrome', /Chrome\/(\d+)/],
    ['Safari', /Version\/(\d+).*Safari/],
  ]
    .map(([name, pattern]) => [name, (pattern as RegExp).exec(userAgent)?.[1]] as const)
    .find(([, version]) => version);
  const browser = found ? `${found[0]} ${found[1]}` : 'An unknown browser';
  const systems: [RegExp, string][] = [
    [/Windows/, 'Windows'],
    [/iPhone|iPad/, 'iOS'],
    [/Mac OS X|Macintosh/, 'macOS'],
    [/Android/, 'Android'],
    [/Linux/, 'Linux'],
  ];
  const os = systems.find(([pattern]) => pattern.test(userAgent))?.[1] ?? 'an unknown system';
  return `${browser} on ${os}`;
}

export function collectDetails({
  server,
  address,
  errors,
}: {
  server: string | null;
  address: string | null;
  errors: LoggedError[];
}): ReportDetails {
  return {
    viewer: VERSION,
    server,
    address,
    browser: browserName(navigator.userAgent),
    screen: `${window.innerWidth} × ${window.innerHeight}`,
    page: pageName(window.location.pathname),
    errors,
  };
}

/** The details as the lines the dialog shows and the report carries. */
export function detailLines(details: ReportDetails, video?: string | null): string[] {
  const server = details.server
    ? `Server ${details.server}${details.address ? ` at ${details.address}` : ''}`
    : 'Not connected to a server';
  return [
    `Viewer ${details.viewer} · ${server}`,
    `${details.browser} · ${details.screen}`,
    `Page: ${details.page}${video ? ` · video ${video}` : ''}`,
    ...(details.errors.length
      ? details.errors.map((e) => `Error at ${formatWhen(e.at)}: ${e.message}`)
      : ['No errors recorded']),
  ];
}

export function reportTitle(report: Report): string {
  const first = report.what.trim().split('\n')[0].trim();
  return first.length > 70 ? `${first.slice(0, 69)}…` : first || 'A problem in the viewer';
}

export function reportBody(report: Report): string {
  const parts = [`What happened:\n${report.what.trim()}`];
  if (report.expected?.trim()) parts.push(`What I expected:\n${report.expected.trim()}`);
  if (report.details) {
    parts.push(`Details:\n${detailLines(report.details, report.video).map((l) => `- ${l}`).join('\n')}`);
  }
  // Dated in full: the report is read on another day.
  parts.push(`Sent from the viewer, ${formatWhen(new Date(), new Date(0))}.`);
  return parts.join('\n\n');
}

function shorten(text: string): string {
  return text.length > MAX_BODY ? `${text.slice(0, MAX_BODY - 1)}…` : text;
}

export function mailtoUrl(report: Report): string {
  const subject = `VideoAnnotator: ${reportTitle(report)}`;
  return `mailto:${REPORT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(shorten(reportBody(report)))}`;
}

/** A new issue with the template's fields filled in (GitHub fills an issue form from fields named by id). */
export function githubIssueUrl(report: Report): string {
  const params = new URLSearchParams({
    template: ISSUE_TEMPLATE,
    title: reportTitle(report),
    what: shorten(report.what.trim()),
  });
  if (report.expected?.trim()) params.set('expected', report.expected.trim());
  if (report.details) params.set('details', detailLines(report.details, report.video).join('\n'));
  return `${GITHUB_URL}/issues/new?${params}`;
}
