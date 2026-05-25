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
 * Aztec stack version from package.json (no CLI required; safe for CI/local).
 */
function getAztecStackVersion(projectRoot: string): string {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(projectRoot, 'package.json'), 'utf8')
  ) as {
    config?: { aztecVersion?: string };
    dependencies?: Record<string, string>;
  };
  return (
    pkg.config?.aztecVersion ??
    pkg.dependencies?.['@aztec/aztec.js'] ??
    'unknown'
  );
}

/** True when artifacts exist but were never postprocessed (e.g. raw `nargo compile`). */
function artifactsNeedPostprocess(targetDir: string): boolean {
  if (!fs.existsSync(targetDir)) {
    return false;
  }

  const jsonFiles = fs
    .readdirSync(targetDir)
    .filter((f) => f.endsWith('.json') && !f.endsWith('.bak'));

  if (jsonFiles.length === 0) {
    return false;
  }

  return jsonFiles.some((file) => {
    const artifact = JSON.parse(
      fs.readFileSync(path.join(targetDir, file), 'utf8')
    ) as { transpiled?: boolean };
    return artifact.transpiled !== true;
  });
}

/** Remove contract JSONs so `aztec compile` cannot skip postprocess/transpile. */
function clearContractArtifacts(targetDir: string): void {
  if (!fs.existsSync(targetDir)) {
    return;
  }

  for (const file of fs.readdirSync(targetDir)) {
    if (file.endsWith('.json') && !file.endsWith('.bak')) {
      fs.unlinkSync(path.join(targetDir, file));
      console.log(`   🗑️ Removed stale artifact ${file}`);
    }
  }
}

/**
 * Stamp `aztec_version` when compile skipped stamping (fallback only).
 */
function stampAztecVersion(targetDir: string, version: string): void {
  if (!fs.existsSync(targetDir)) {
    return;
  }

  const jsonFiles = fs
    .readdirSync(targetDir)
    .filter((f) => f.endsWith('.json') && !f.endsWith('.bak'));

  for (const file of jsonFiles) {
    const filePath = path.join(targetDir, file);
    const artifact = JSON.parse(fs.readFileSync(filePath, 'utf8')) as {
      aztec_version?: string;
    };

    if (artifact.aztec_version === version) {
      continue;
    }

    artifact.aztec_version = version;
    fs.writeFileSync(filePath, `${JSON.stringify(artifact, null, 2)}\n`);
    console.log(`   🏷️ Stamped aztec_version=${version} in ${file}`);
  }
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
function compileLocalContracts(projectRoot: string): boolean {
  const workspaceNargo = path.join(projectRoot, 'Nargo.toml');

  if (!fs.existsSync(workspaceNargo)) {
    console.error(`❌ No workspace Nargo.toml found at ${projectRoot}`);
    return false;
  }

  const forceRebuild = process.argv.includes('--force');
  const compiledTarget = path.join(projectRoot, 'target');

  if (forceRebuild || artifactsNeedPostprocess(compiledTarget)) {
    if (artifactsNeedPostprocess(compiledTarget)) {
      console.log(
        '   ⚠️ Found non-transpiled artifacts (likely from nargo only); clearing for postprocess...'
      );
    }
    clearContractArtifacts(compiledTarget);
  }

  console.log('\n🔨 Compiling contracts (aztec compile, same as CI)...');

  if (!tryRun(`cd "${projectRoot}" && aztec compile`)) {
    console.error('   ❌ Failed to compile contracts');
    return false;
  }

  const hasArtifacts =
    fs.existsSync(compiledTarget) &&
    fs.readdirSync(compiledTarget).some((f) => f.endsWith('.json'));
  if (!hasArtifacts) {
    console.error('   ❌ Failed to compile contracts — no artifacts generated');
    return false;
  }

  console.log('   ✅ Contracts compiled successfully');

  // Strip __aztec_nr_internals__ prefix from function names
  stripAztecNrPrefix(compiledTarget);

  // Fallback when compile output omits aztec_version (e.g. skipped recompile)
  stampAztecVersion(compiledTarget, getAztecStackVersion(projectRoot));

  const targetDir = path.join(projectRoot, 'target');

  // Run codegen to generate TypeScript wrappers from JSON (same paths as CI)
  console.log('   🔧 Generating TypeScript artifacts...');
  if (
    !tryRun(
      `cd "${projectRoot}" && aztec codegen target --outdir src/artifacts -f`
    )
  ) {
    console.error('   ❌ Codegen failed');
    return false;
  }

  console.log('   ✅ TypeScript artifacts generated');
  return true;
}

async function main() {
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

    const ok = compileLocalContracts(projectRoot);
    if (!ok) {
      console.error('\n❌ Build contracts failed');
      process.exit(1);
    }
    console.log('\n✅ Build contracts step completed');

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
