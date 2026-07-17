#!/usr/bin/env node
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--coursemapper') args.coursemapper = argv[++index];
    else if (arg === '--manifest') args.manifest = argv[++index];
    else if (arg === '--dataset-manifest') args.datasetManifest = argv[++index];
    else if (arg === '--repo-root') args.repoRoot = argv[++index];
    else if (arg === '--tier') args.tier = argv[++index];
    else if (arg === '--verify-training-run') args.verifyTrainingRun = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.coursemapper || !args.manifest || !args.datasetManifest || !args.repoRoot || !args.tier) {
    throw new Error('--coursemapper, --manifest, --dataset-manifest, --repo-root, and --tier are required');
  }
  return args;
}

async function sha256File(filePath) {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const coursemapper = path.resolve(args.coursemapper);
  const manifestPath = path.resolve(args.manifest);
  const packageRoot = path.dirname(manifestPath);
  const manifestModule = await import(
    pathToFileURL(path.join(coursemapper, 'src/lib/scionAdapterManifest.js')).href
  );
  const trainingModule = await import(
    pathToFileURL(path.join(coursemapper, 'scripts/scionAdapterTrainingRun.mjs')).href
  );
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const expectedBase =
    args.tier === 'lite'
      ? manifestModule.SCION_GEMMA4_E2B_BASE
      : {
          modelId: 'google/gemma-4-12B-it-qat-q4_0-unquantized',
          revision: 'b8dea52d5ea56a20e8872f0ee5d25ada7501327e',
          architecture: 'gemma4',
          role: 'instruction',
        };
  const validation = manifestModule.validateScionAdapterManifest(manifest, { expectedBase });
  const fileIssues = [];
  for (const expected of manifest.files || []) {
    const absolute = path.resolve(packageRoot, expected.path || '');
    const relative = path.relative(packageRoot, absolute).replaceAll('\\', '/');
    if (!relative || relative.startsWith('../') || path.isAbsolute(relative)) {
      fileIssues.push(`${expected.path}:path-escape`);
      continue;
    }
    try {
      const stats = await fs.lstat(absolute);
      if (!stats.isFile() || stats.isSymbolicLink()) fileIssues.push(`${expected.path}:not-regular`);
      if (stats.size !== expected.bytes) fileIssues.push(`${expected.path}:bytes`);
      if ((await sha256File(absolute)) !== expected.sha256) fileIssues.push(`${expected.path}:sha256`);
    } catch {
      fileIssues.push(`${expected.path}:unavailable`);
    }
  }
  const dataset = await trainingModule.verifyScionAdapterDatasetForTraining({
    manifestPath: path.resolve(args.datasetManifest),
    lane: 'research',
    sourceRoot: path.resolve(args.repoRoot),
  });
  let training = { valid: true, issues: [], skipped: true };
  if (args.verifyTrainingRun) {
    training = await trainingModule.verifyScionAdapterTrainingRun({
      planPath: path.join(packageRoot, 'training-plan.json'),
      resultPath: path.join(packageRoot, 'training-result.json'),
      datasetManifestPath: path.resolve(args.datasetManifest),
      sourceRoot: path.resolve(args.repoRoot),
    });
  }
  const issues = [...validation.issues, ...fileIssues, ...dataset.issues, ...training.issues];
  const result = {
    status: issues.length === 0 ? 'pass' : 'fail',
    valid: issues.length === 0,
    manifest: { valid: validation.valid, issues: validation.issues },
    files: { valid: fileIssues.length === 0, issues: fileIssues },
    dataset: { valid: dataset.valid, issues: dataset.issues },
    training: { valid: training.valid, issues: training.issues, skipped: training.skipped || false },
    issues: [...new Set(issues)],
  };
  console.log(JSON.stringify(result, null, 2));
  if (!result.valid) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
