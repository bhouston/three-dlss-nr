import { readFileSync, appendFileSync, existsSync } from 'node:fs';
const path = 'coverage/coverage-summary.json';
if (existsSync(path)) {
  const { total } = JSON.parse(readFileSync(path, 'utf8'));
  const lines = [
    '| Coverage | Percent |',
    '| --- | ---: |',
    ...['statements', 'branches', 'functions', 'lines'].map((key) => `| ${key} | ${total[key].pct}% |`),
  ];
  const summary = `${lines.join('\n')}\n`;
  console.log(summary);
  if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, summary);
}
