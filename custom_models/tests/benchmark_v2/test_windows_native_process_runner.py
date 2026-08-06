from __future__ import annotations
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
NATIVE_RUNNER = PROJECT_ROOT / 'custom_models' / 'docs' / 'benchmark_v2' / 'WINDOWS_NATIVE_PROCESS_RUNNER.ps1'

def _ps_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"

@unittest.skipUnless(shutil.which('powershell.exe'), 'requires Windows PowerShell 5.1')
class WindowsNativeProcessRunnerTests(unittest.TestCase):

    def _write_probe(self, root: Path) -> Path:
        probe = root / 'native_probe.py'
        probe.write_text(textwrap.dedent('\n                import argparse\n                import json\n                import sys\n                from pathlib import Path\n\n                parser = argparse.ArgumentParser()\n                parser.add_argument("--exit-code", type=int, required=True)\n                parser.add_argument("--status", required=True)\n                parser.add_argument("--stderr", default="")\n                parser.add_argument("--report-path")\n                parser.add_argument("values", nargs="*")\n                args = parser.parse_args()\n\n                payload = {\n                    "status": args.status,\n                    "values": args.values,\n                }\n                if args.status == "COMPLETED_WITH_FAILURES":\n                    payload["results"] = [\n                        {\n                            "model_id": f"model_{index:02d}",\n                            "exit_code": 19 if index == 2 else 0,\n                            "per_model_log": f"model_{index:02d}.log",\n                        }\n                        for index in range(26)\n                    ]\n                    payload["failures"] = [payload["results"][2]]\n                if args.report_path:\n                    Path(args.report_path).write_text(\n                        json.dumps(payload, ensure_ascii=False) + "\\n",\n                        encoding="utf-8",\n                    )\n                if args.stderr:\n                    sys.stderr.write(args.stderr)\n                print(json.dumps(payload, ensure_ascii=False))\n                raise SystemExit(args.exit_code)\n                ').lstrip(), encoding='utf-8')
        return probe

    def _run_powershell(self, script: str) -> subprocess.CompletedProcess[str]:
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        return subprocess.run([shutil.which('powershell.exe') or 'powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded], cwd=PROJECT_ROOT, text=True, capture_output=True, encoding='utf-8', errors='replace', check=False)

    @staticmethod
    def _last_json(stdout: str) -> dict:
        for line in reversed(stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise AssertionError(f'no JSON object in PowerShell stdout: {stdout!r}')

    def test_stderr_and_exit_one_are_captured_without_native_command_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = self._write_probe(root)
            report = root / 'final_status.json'
            script = f"\n            $ErrorActionPreference = 'Stop'\n            . {_ps_quote(NATIVE_RUNNER)}\n            $result = Invoke-GyxPythonGate `\n                -Python {_ps_quote(sys.executable)} `\n                -Gate {_ps_quote(probe)} `\n                -Arguments @('--exit-code', '1', '--status', 'COMPLETED_WITH_FAILURES', '--stderr', 'failed_model=persistence', '--report-path', {_ps_quote(report)}) `\n                -WorkingDirectory {_ps_quote(root)} `\n                -LogRoot {_ps_quote(root / 'logs')} `\n                -Label 'formal-run' `\n                -AllowedExitCodes @(0, 1, 4) `\n                -ReportPath {_ps_quote(report)}\n            $result | ConvertTo-Json -Depth 20 -Compress\n            "
            completed = self._run_powershell(script)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = self._last_json(completed.stdout)
            self.assertEqual(result['ExitCode'], 1)
            self.assertEqual(result['Stderr'], 'failed_model=persistence')
            self.assertEqual(result['Json']['status'], 'COMPLETED_WITH_FAILURES')
            self.assertEqual(len(result['Json']['results']), 26)
            self.assertEqual(result['Json']['failures'][0]['model_id'], 'model_02')
            self.assertEqual(result['Json']['results'][-1]['model_id'], 'model_25')
            self.assertTrue(Path(result['LogPath']).is_file())
            self.assertTrue(Path(result['StderrPath']).is_file())
            logged_stdout = json.loads(Path(result['LogPath']).read_text(encoding='utf-8'))
            self.assertEqual(logged_stdout['status'], 'COMPLETED_WITH_FAILURES')
            self.assertEqual(len(logged_stdout['results']), 26)
            self.assertEqual(Path(result['StderrPath']).read_text(encoding='utf-8'), 'failed_model=persistence')
            self.assertNotIn('NativeCommandError', completed.stderr)

    def test_mock_formal_wrapper_fails_only_after_all_26_results_arrive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = self._write_probe(root)
            report = root / 'final_status.json'
            script = f"\n            $ErrorActionPreference = 'Stop'\n            . {_ps_quote(NATIVE_RUNNER)}\n            $result = Invoke-GyxPythonGate `\n                -Python {_ps_quote(sys.executable)} `\n                -Gate {_ps_quote(probe)} `\n                -Arguments @('--exit-code', '1', '--status', 'COMPLETED_WITH_FAILURES', '--stderr', 'failed_model=model_02', '--report-path', {_ps_quote(report)}) `\n                -WorkingDirectory {_ps_quote(root)} `\n                -LogRoot {_ps_quote(root / 'logs')} `\n                -Label 'formal-wrapper' `\n                -AllowedExitCodes @(0, 1, 4) `\n                -ReportPath {_ps_quote(report)}\n            if (@($result.Json.results).Count -ne 26) {{ throw 'incomplete result set' }}\n            if ($result.Json.results[25].model_id -ne 'model_25') {{ throw 'later model missing' }}\n            throw 'formal wrapper summary failure after final report'\n            "
            completed = self._run_powershell(script)
            self.assertNotEqual(completed.returncode, 0)
            payload = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(payload['status'], 'COMPLETED_WITH_FAILURES')
            self.assertEqual(len(payload['results']), 26)
            self.assertEqual(payload['failures'][0]['model_id'], 'model_02')
            self.assertEqual(payload['results'][-1]['model_id'], 'model_25')
            self.assertNotIn('NativeCommandError', completed.stderr)
            self.assertIn('formal wrapper summary failure', completed.stderr)

    def test_stderr_and_exit_zero_still_return_parsed_stdout_json(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = self._write_probe(root)
            script = f"\n            $ErrorActionPreference = 'Stop'\n            . {_ps_quote(NATIVE_RUNNER)}\n            $result = Invoke-GyxPythonGate `\n                -Python {_ps_quote(sys.executable)} `\n                -Gate {_ps_quote(probe)} `\n                -Arguments @('--exit-code', '0', '--status', 'PASS', '--stderr', 'diagnostic warning') `\n                -WorkingDirectory {_ps_quote(root)} `\n                -LogRoot {_ps_quote(root / 'logs')} `\n                -Label 'success-with-stderr' `\n                -AllowedExitCodes @(0)\n            $result | ConvertTo-Json -Depth 20 -Compress\n            "
            completed = self._run_powershell(script)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = self._last_json(completed.stdout)
            self.assertEqual(result['ExitCode'], 0)
            self.assertEqual(result['Json']['status'], 'PASS')
            self.assertEqual(result['Stderr'], 'diagnostic warning')
            self.assertNotIn('NativeCommandError', completed.stderr)

    def test_exit_four_returns_not_ready_report_for_wrapper_decision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = self._write_probe(root)
            report = root / 'final_status.json'
            script = f"\n            $ErrorActionPreference = 'Stop'\n            . {_ps_quote(NATIVE_RUNNER)}\n            $result = Invoke-GyxPythonGate `\n                -Python {_ps_quote(sys.executable)} `\n                -Gate {_ps_quote(probe)} `\n                -Arguments @('--exit-code', '4', '--status', 'NOT_READY', '--stderr', 'readiness incomplete', '--report-path', {_ps_quote(report)}) `\n                -WorkingDirectory {_ps_quote(root)} `\n                -LogRoot {_ps_quote(root / 'logs')} `\n                -Label 'not-ready' `\n                -AllowedExitCodes @(0, 1, 4) `\n                -ReportPath {_ps_quote(report)}\n            $result | ConvertTo-Json -Depth 20 -Compress\n            "
            completed = self._run_powershell(script)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = self._last_json(completed.stdout)
            self.assertEqual(result['ExitCode'], 4)
            self.assertEqual(result['Json']['status'], 'NOT_READY')
            self.assertEqual(result['Stderr'], 'readiness incomplete')
            self.assertNotIn('NativeCommandError', completed.stderr)

    def test_argument_quoting_preserves_spaces_quotes_backslashes_and_unicode(self):
        values = ['', 'space value', 'quote"value', 'trailing\\', 'C:\\Program Files\\模型\\probe.py', '中文参数']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = self._write_probe(root)
            arguments = ['--exit-code', '0', '--status', 'PASS', *values]
            argument_literal = ', '.join((_ps_quote(value) for value in arguments))
            script = f"\n            $ErrorActionPreference = 'Stop'\n            . {_ps_quote(NATIVE_RUNNER)}\n            $result = Invoke-GyxPythonGate `\n                -Python {_ps_quote(sys.executable)} `\n                -Gate {_ps_quote(probe)} `\n                -Arguments @({argument_literal}) `\n                -WorkingDirectory {_ps_quote(root)} `\n                -LogRoot {_ps_quote(root / 'logs')} `\n                -Label 'argv' `\n                -AllowedExitCodes @(0)\n            $result | ConvertTo-Json -Depth 20 -Compress\n            "
            completed = self._run_powershell(script)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = self._last_json(completed.stdout)
            self.assertEqual(result['Json']['values'], values)

    def test_disallowed_exit_is_thrown_only_after_logs_and_json_parse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            probe = self._write_probe(root)
            script = f"\n            $ErrorActionPreference = 'Stop'\n            . {_ps_quote(NATIVE_RUNNER)}\n            try {{\n                Invoke-GyxPythonGate `\n                    -Python {_ps_quote(sys.executable)} `\n                    -Gate {_ps_quote(probe)} `\n                    -Arguments @('--exit-code', '2', '--status', 'FAIL', '--stderr', 'fatal detail') `\n                    -WorkingDirectory {_ps_quote(root)} `\n                    -LogRoot {_ps_quote(root / 'logs')} `\n                    -Label 'disallowed' `\n                    -AllowedExitCodes @(0)\n                [pscustomobject]@{{ Caught = $false }} | ConvertTo-Json -Compress\n            }} catch {{\n                [pscustomobject]@{{ Caught = $true; Message = $_.Exception.Message }} | ConvertTo-Json -Compress\n            }}\n            "
            completed = self._run_powershell(script)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = self._last_json(completed.stdout)
            self.assertTrue(result['Caught'])
            self.assertIn('parsed_status=FAIL', result['Message'])
            stdout_logs = list((root / 'logs').glob('disallowed.*.stdout.log'))
            stderr_logs = list((root / 'logs').glob('disallowed.*.stderr.log'))
            self.assertEqual(len(stdout_logs), 1)
            self.assertEqual(len(stderr_logs), 1)
            self.assertIn('"status": "FAIL"', stdout_logs[0].read_text(encoding='utf-8'))
            self.assertEqual(stderr_logs[0].read_text(encoding='utf-8'), 'fatal detail')
if __name__ == '__main__':
    unittest.main()
