/** A problem report's text and where it goes (UAT #68). */
import { afterEach, describe, expect, it } from 'vitest';

import { clearErrors, recentErrors, recordError, scrub } from '@/lib/errorLog';
import {
  REPORT_EMAIL,
  browserName,
  detailLines,
  githubIssueUrl,
  mailtoUrl,
  pageName,
  reportBody,
  reportTitle,
  type Report,
  type ReportDetails,
} from '@/lib/problemReport';

const details: ReportDetails = {
  viewer: '1.7.0',
  server: '1.7.0',
  address: '127.0.0.1:18011',
  browser: 'Firefox 132 on Linux',
  screen: '1440 × 900',
  page: 'the Video page',
  errors: [],
};
const report = (extra: Partial<Report> = {}): Report => ({
  what: 'The Speech lane is empty\nbut the Data tab shows words',
  expected: 'Words in the lane',
  details,
  video: null,
  ...extra,
});

afterEach(() => clearErrors());

describe('problem report', () => {
  it('says what happened, what was expected and the details', () => {
    const body = reportBody(report());
    expect(body).toContain('What happened:\nThe Speech lane is empty');
    expect(body).toContain('What I expected:\nWords in the lane');
    expect(body).toContain('- Viewer 1.7.0 · Server 1.7.0 at 127.0.0.1:18011');
    expect(body).toContain('- Firefox 132 on Linux · 1440 × 900');
    expect(body).toContain('- No errors recorded');
    expect(reportTitle(report())).toBe('The Speech lane is empty');
  });

  it('carries no details, and no video name, unless left ticked', () => {
    expect(reportBody(report({ details: null }))).not.toContain('Details');
    expect(reportBody(report())).not.toContain('video ');
    expect(detailLines(details, '2UWdXP.joke1.mp4')).toContain('Page: the Video page · video 2UWdXP.joke1.mp4');
  });

  it('goes to the project address, or a GitHub issue with the form filled in', () => {
    const mail = new URL(mailtoUrl(report()));
    expect(mail.protocol).toBe('mailto:');
    expect(mail.pathname).toBe(REPORT_EMAIL);
    expect(mail.searchParams.get('subject')).toBe('VideoAnnotator: The Speech lane is empty');
    const issue = new URL(githubIssueUrl(report()));
    expect(issue.pathname).toBe('/InfantLab/VideoAnnotator/issues/new');
    expect(issue.searchParams.get('template')).toBe('bug_report.yml');
    expect(issue.searchParams.get('what')).toContain('Data tab shows words');
    expect(issue.searchParams.get('expected')).toBe('Words in the lane');
    expect(issue.searchParams.get('details')).toContain('Firefox 132 on Linux');
  });

  it('names pages in the viewer’s words, never by their address', () => {
    expect(pageName('/viewer/video')).toBe('the Video page');
    expect(pageName('/viewer/videos')).toBe('Videos');
    expect(pageName('/viewer/run')).toBe('New run');
    expect(pageName('/viewer/home')).toBe('Home');
    expect(pageName('/somewhere')).toBe('another page');
  });

  it('names the browser and system', () => {
    const chrome =
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36';
    expect(browserName(chrome)).toBe('Chrome 130 on Windows');
    expect(browserName(`${chrome} Edg/130.0.0.0`)).toBe('Edge 130 on Windows');
    expect(browserName('Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0')).toBe('Firefox 132 on Linux');
    expect(
      browserName('Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15'),
    ).toBe('Safari 18 on macOS');
  });
});

describe('error log', () => {
  it('keeps the latest errors, once each, without paths or keys', () => {
    recordError(new Error("Can't read /home/ada/Studies/BabyJokes/child01.vtt"));
    recordError(new Error("Can't read /home/ada/Studies/BabyJokes/child01.vtt"));
    recordError(`Refused va_${'a'.repeat(43)} at /viewer-connect?token=abc123`);
    expect(recentErrors().map((e) => e.message)).toEqual([
      'Refused va_… at /viewer-connect?token=…',
      "Can't read …/child01.vtt",
    ]);
  });

  it('scrubs Windows paths too', () => {
    expect(scrub('Missing C:\\Users\\ada\\Videos\\p01.mp4')).toBe('Missing …/p01.mp4');
  });
});
