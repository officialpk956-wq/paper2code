import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';
import { join } from 'path';
import { randomUUID } from 'crypto';
import { tmpdir } from 'os';

const TIMEOUT_MS = 8000;
const MAX_CODE_LENGTH = 20_000;

export interface RunRequest {
  code: string;
  testCases: Array<{
    input: Record<string, unknown>;
    output: unknown;
    visible: boolean;
  }>;
  functionName: string;
  visibleOnly?: boolean;
}

const SCRIPT_TEMPLATE = [
  'import sys',
  'import json',
  'import time',
  'import math',
  '',
  'try:',
  '    import numpy as np',
  '    HAS_NUMPY = True',
  'except ImportError:',
  '    HAS_NUMPY = False',
  '',
  'def _to_comparable(v):',
  '    if HAS_NUMPY and hasattr(v, "tolist"):',
  '        return v.tolist()',
  '    if isinstance(v, (list, tuple)):',
  '        return [_to_comparable(x) for x in v]',
  '    if isinstance(v, float):',
  '        if math.isnan(v) or math.isinf(v):',
  '            return str(v)',
  '    return v',
  '',
  'def _deep_equal(a, b, tol=1e-5):',
  '    a = _to_comparable(a)',
  '    b = _to_comparable(b)',
  '    if isinstance(a, list) and isinstance(b, list):',
  '        if len(a) != len(b):',
  '            return False',
  '        return all(_deep_equal(x, y, tol) for x, y in zip(a, b))',
  '    if isinstance(a, (int, float)) and isinstance(b, (int, float)):',
  '        if abs(float(a) - float(b)) < tol:',
  '            return True',
  '        if a == b:',
  '            return True',
  '        return False',
  '    return a == b',
  '',
  '# ---- USER CODE ----',
  '__USER_CODE__',
  '# ---- END USER CODE ----',
  '',
  'test_cases = json.loads(__TEST_CASES_JSON__)',
  'results = []',
  '',
  'for i, tc in enumerate(test_cases):',
  '    start = time.perf_counter()',
  '    try:',
  '        raw_inputs = tc["input"]',
  '        if HAS_NUMPY:',
  '            inputs = {',
  '                k: np.array(v) if isinstance(v, list) else v',
  '                for k, v in raw_inputs.items()',
  '            }',
  '        else:',
  '            inputs = raw_inputs',
  '        actual = __FUNC_NAME__(**inputs)',
  '        elapsed = (time.perf_counter() - start) * 1000',
  '        actual_cmp = _to_comparable(actual)',
  '        expected = tc["output"]',
  '        passed = _deep_equal(actual_cmp, expected)',
  '        results.append({',
  '            "index": i,',
  '            "passed": passed,',
  '            "actual": actual_cmp,',
  '            "expected": expected,',
  '            "runtime_ms": round(elapsed, 3),',
  '            "error": None,',
  '        })',
  '    except Exception as e:',
  '        elapsed = (time.perf_counter() - start) * 1000',
  '        results.append({',
  '            "index": i,',
  '            "passed": False,',
  '            "actual": None,',
  '            "expected": tc["output"],',
  '            "runtime_ms": round(elapsed, 3),',
  '            "error": str(e),',
  '        })',
  '',
  'print(json.dumps({"results": results, "status": "ok"}))',
].join('\n');

function buildTestScript(
  userCode: string,
  testCases: RunRequest['testCases'],
  functionName: string,
  visibleOnly: boolean,
): string {
  const cases = visibleOnly ? testCases.filter((tc) => tc.visible) : testCases;
  // Wrap JSON in triple-quoted Python string so json.loads can parse it
  // (JSON uses lowercase true/false/null which are not valid Python literals)
  const casesLiteral = `"""${JSON.stringify(cases).replace(/"""/g, '\\"\\"\\"')}"""`;
  return SCRIPT_TEMPLATE
    .replace('__USER_CODE__', userCode)
    .replace('__TEST_CASES_JSON__', casesLiteral)
    .replace('__FUNC_NAME__', functionName);
}

function runScript(scriptPath: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('timeout'));
    }, TIMEOUT_MS);

    exec(
      `python "${scriptPath}"`,
      { timeout: TIMEOUT_MS },
      (err, stdout, stderr) => {
        clearTimeout(timer);
        if (err && !stdout) {
          reject(new Error(stderr || err.message));
          return;
        }
        resolve(stdout.trim());
      },
    );
  });
}

export async function POST(request: NextRequest) {
  let tmpPath: string | null = null;

  try {
    const body = (await request.json()) as RunRequest;
    const { code, testCases, functionName, visibleOnly = true } = body;

    if (!code || typeof code !== 'string') {
      return NextResponse.json({ error: 'code is required' }, { status: 400 });
    }
    if (code.length > MAX_CODE_LENGTH) {
      return NextResponse.json({ error: 'code too long' }, { status: 400 });
    }
    if (!functionName || typeof functionName !== 'string') {
      return NextResponse.json({ error: 'functionName is required' }, { status: 400 });
    }
    // Validate functionName is a safe Python identifier to prevent code injection
    if (!/^[a-zA-Z_][a-zA-Z0-9_]{0,99}$/.test(functionName)) {
      return NextResponse.json({ error: 'functionName must be a valid Python identifier' }, { status: 400 });
    }

    const id = randomUUID().replace(/-/g, '');
    tmpPath = join(tmpdir(), `dojo_${id}.py`);
    const script = buildTestScript(code, testCases, functionName, visibleOnly);
    writeFileSync(tmpPath, script, 'utf8');

    const start = Date.now();
    const raw = await runScript(tmpPath);
    const totalMs = Date.now() - start;

    const parsed = JSON.parse(raw) as { results: unknown[]; status: string };

    return NextResponse.json({ results: parsed.results, totalMs });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message === 'timeout') {
      return NextResponse.json(
        { error: 'Time limit exceeded (8s)', timeout: true },
        { status: 408 },
      );
    }
    return NextResponse.json(
      { error: `Execution error: ${message}` },
      { status: 500 },
    );
  } finally {
    if (tmpPath) {
      try { unlinkSync(tmpPath); } catch {}
    }
  }
}
