// Run with: tsx scripts/build-contracts.ts [--force]
// Compiles local contracts from workspace (aztec-standards removed for HIVE)

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

// Output directory for artifacts (TypeScript wrappers)
const ARTIFACTS_OUTPUT_DIR = 'src/artifacts';

/**
 * Try to run a command
 */
function tryRun(cmd: string, opts: Record<string, unknown> = {}): boolean {
  try {
    const res = spawnSync(cmd, { stdio: 'inherit', shell: true, ...opts });
    return res.status === 0;
  } catch {
    return false;
  }
}

/**
 * Ensure a directory exists
 */
function ensureDir(p: string): void {
  fs.mkdirSync(p, { recursive: true });
}

function getPinnedAztecVersion(projectRoot: string): string | null {
  for (const rcFile of ['.aztecrc', '.gaztecrc']) {
    const rcPath = path.join(projectRoot, rcFile);
    if (fs.existsSync(rcPath)) {
      const version = fs.readFileSync(rcPath, 'utf8').trim();
      if (version) return version;
    }
  }
  return null;
}

function resolveBbBinary(projectRoot: string): string {
  const version = getPinnedAztecVersion(projectRoot);
  const home = process.env.HOME;
  if (version && home) {
    const bbPath = path.join(
      home,
      '.aztec',
      'versions',
      version,
      'node_modules',
      '.bin',
      'bb'
    );
    if (fs.existsSync(bbPath)) {
      return bbPath;
    }
  }
  return 'bb';
}

/**
 * Copy files with optional filter
 */
function copyFiles(
  sourceDir: string,
  targetDir: string,
  forceOverwrite = false,
  filter?: (file: string) => boolean
): number {
  if (!fs.existsSync(sourceDir)) {
    console.log(`⚠️ Source directory ${sourceDir} does not exist`);
    return 0;
  }

  ensureDir(targetDir);
  const files = fs.readdirSync(sourceDir);
  let copiedCount = 0;
  let skippedCount = 0;

  for (const file of files) {
    // Apply filter if provided
    if (filter && !filter(file)) {
      continue;
    }

    const srcPath = path.join(sourceDir, file);
    const dstPath = path.join(targetDir, file);

    if (fs.existsSync(dstPath) && !forceOverwrite) {
      console.log(`   ⏭️ Skipping ${file} (already exists)`);
      skippedCount++;
      continue;
    }

    // Overwrite or copy new
    if (fs.statSync(srcPath).isDirectory()) {
      fs.cpSync(srcPath, dstPath, { recursive: true, force: true });
    } else {
      fs.copyFileSync(srcPath, dstPath);
    }
    console.log(`   ✅ Copied ${file}`);
    copiedCount++;
  }

  console.log(
    `   📊 Copied ${copiedCount} items, skipped ${skippedCount} existing items`
  );
  return copiedCount;
}

/**
 * Strip the __aztec_nr_internals__ prefix from function names in compiled artifact JSONs.
 * Replicates the strip_aztec_nr_prefix.sh script.
 */
function stripAztecNrPrefix(targetDir: string): void {
  if (!fs.existsSync(targetDir)) return;

  const PREFIX = '__aztec_nr_internals__';
  const jsonFiles = fs
    .readdirSync(targetDir)
    .filter((f) => f.endsWith('.json'));

  for (const file of jsonFiles) {
    const filePath = path.join(targetDir, file);
    const artifact = JSON.parse(fs.readFileSync(filePath, 'utf8'));

    if (!Array.isArray(artifact.functions)) continue;

    let modified = false;
    for (const fn of artifact.functions) {
      if (typeof fn.name === 'string' && fn.name.startsWith(PREFIX)) {
        fn.name = fn.name.slice(PREFIX.length);
        modified = true;
      }
    }

    if (modified) {
      fs.writeFileSync(filePath, JSON.stringify(artifact));
      console.log(`   🔧 Stripped __aztec_nr_internals__ prefix from ${file}`);
    }
  }
}

/**
 * Compile local contracts using workspace Nargo.toml at root level
 */
function compileLocalContracts(
  projectRoot: string,
  forceOverwrite: boolean
): boolean {
  const workspaceNargo = path.join(projectRoot, 'Nargo.toml');

  if (!fs.existsSync(workspaceNargo)) {
    console.error(`❌ No workspace Nargo.toml found at ${projectRoot}`);
    return false;
  }

  console.log('\n🔨 Compiling contracts from workspace...');

  // Compile all contracts from workspace root
  tryRun(`cd "${projectRoot}" && gaztec nargo compile`);
  const compiledTarget = path.join(projectRoot, 'target');
  const hasArtifacts =
    fs.existsSync(compiledTarget) &&
    fs.readdirSync(compiledTarget).some((f) => f.endsWith('.json'));
  if (!hasArtifacts) {
    console.error('   ❌ Failed to compile contracts — no artifacts generated');
    return false;
  }

  console.log('   ✅ Contracts compiled successfully');

  // Postprocess: transpile public bytecode (ACIR → AVM) and generate VKs.
  // In v4, `aztec-nargo compile` only produces raw ACIR. `bb aztec_process`
  // transpiles public functions and sets `transpiled: true` in the JSON,
  // which `gaztec codegen` requires.
  console.log('   🔧 Postprocessing contracts (transpile + VK generation)...');
  {
    const bbBinary = resolveBbBinary(projectRoot);
    const bbCmd = [
      `cd "${projectRoot}"`,
      `&& for f in target/*.json; do flags="$flags -i $f"; done;`,
      `"${bbBinary}" aztec_process $flags`,
    ].join(' ');
    if (!tryRun(bbCmd)) {
      console.error('   ❌ Failed to postprocess contracts');
      return false;
    }
  }
  console.log('   ✅ Contracts postprocessed successfully');

  // Strip __aztec_nr_internals__ prefix from function names
  stripAztecNrPrefix(compiledTarget);

  // Copy artifacts from target/ to src/target/ (for codegen) and src/artifacts/
  const targetDir = path.join(projectRoot, 'target');
  const artifactsDir = path.join(projectRoot, ARTIFACTS_OUTPUT_DIR);
  const srcTargetDir = path.join(projectRoot, 'src', 'target');

  if (fs.existsSync(targetDir)) {
    copyFiles(
      targetDir,
      artifactsDir,
      forceOverwrite,
      (file) => file.endsWith('.json') && !file.endsWith('.bak')
    );
    copyFiles(
      targetDir,
      srcTargetDir,
      forceOverwrite,
      (file) => file.endsWith('.json') && !file.endsWith('.bak')
    );
  }

  // Run codegen to generate TypeScript wrappers from JSON
  console.log('   🔧 Generating TypeScript artifacts...');
  if (
    !tryRun(
      `cd "${projectRoot}" && gaztec codegen src/target --outdir src/artifacts -f`
    )
  ) {
    console.warn('   ⚠️ Codegen failed - TS artifacts may be stale');
  } else {
    console.log('   ✅ TypeScript artifacts generated');
  }

  return true;
}

async function main() {
  const forceOverwrite = process.argv.includes('--force');
  const projectRoot = process.cwd();

  console.log(`
╔════════════════════════════════════════════════════════════════╗
║           BUILD CONTRACTS                                      ║
╚════════════════════════════════════════════════════════════════╝
`);

  try {
    // 1) Compile local contracts (ZKML, etc.)
    console.log('\n' + '='.repeat(60));
    console.log('📦 Compile local contracts');
    console.log('='.repeat(60));

    const ok = compileLocalContracts(projectRoot, forceOverwrite);
    if (ok) {
      console.log('\n✅ Build contracts step completed');
    } else {
      console.warn('\n⚠️ Some local contracts failed to compile');
    }

    console.log('\n' + '='.repeat(60));
    console.log('🎉 Build complete!');
    console.log('='.repeat(60) + '\n');
  } catch (err: unknown) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    console.error('\n❌ Build script failed:', errorMessage);
    process.exit(1);
  }
}

main();
