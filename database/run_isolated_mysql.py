"""Run synthetic checks in a task-owned Windows MySQL, without TCP or services.

Download/extract the official ZIP separately into _mysql-integration.
No existing MySQL service, Docker daemon, user DB, or system option file is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from mysql_integration_checks import run_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use-preinitialized-task-data', action='store_true',
                        help='First-run only: use this task\'s already initialized runtime/data.')
    parser.add_argument('--ascii-temp', action='store_true',
                        help='Use and clean a unique task-owned ASCII temporary runtime directory.')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    runtime = source / '_mysql-integration'
    base = runtime / 'mysql-8.4.11-winx64'
    server, client = base / 'bin/mysqld.exe', base / 'bin/mysql.exe'
    if not server.is_file() or not client.is_file():
        raise RuntimeError('Official MySQL 8.4.11 ZIP must be extracted in _mysql-integration first.')
    token = uuid.uuid4().hex[:12]
    if args.ascii_temp and args.use_preinitialized_task_data:
        raise RuntimeError('ASCII temporary runtime requires its own freshly initialized data.')
    temp_parent = Path(tempfile.gettempdir()).resolve()
    run_dir = (temp_parent / ('codex-portfolio-mysql-' + token) if args.ascii_temp
               else runtime / ('run-' + token))
    if args.ascii_temp and not str(run_dir).isascii():
        raise RuntimeError('Temporary runtime path is not ASCII.')
    run_dir.mkdir(parents=False, exist_ok=False)
    (run_dir / 'OWNER.txt').write_text('Task-owned synthetic MySQL integration; no user data.\n', encoding='utf-8')
    uploads = run_dir / 'uploads'
    uploads.mkdir()
    if args.use_preinitialized_task_data:
        data = runtime / 'data'
        marker = runtime / 'preinitialized-data-used.json'
        log = runtime / 'initialize.log'
        if marker.exists() or not data.is_dir() or not log.is_file():
            raise RuntimeError('Preinitialized task data missing or already used; refusing reuse.')
        # This flag is bounded to the exact local task path; no arbitrary datadir input.
        if 'MySQL Server Initialization - end.' not in log.read_text(encoding='utf-8', errors='replace'):
            raise RuntimeError('Task initialization did not complete successfully.')
        marker.write_text(json.dumps({'run_dir': str(run_dir), 'purpose': 'synthetic tests'}), encoding='utf-8')
    else:
        data = run_dir / 'data'
        initialize = [str(server), '--no-defaults', '--initialize-insecure',
                      '--basedir=' + str(base), '--datadir=' + str(data), '--console']
        try:
            with (run_dir / 'initialize.log').open('wb') as stream:
                subprocess.run(initialize, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=True, timeout=300)
        except (subprocess.SubprocessError, OSError):
            shutil.copy2(run_dir / 'initialize.log', source / 'mysql-initialize-failure.log')
            raise
    memory = 'codex_stock_' + token
    command = [str(server), '--no-defaults', '--no-monitor', '--basedir=' + str(base),
               '--datadir=' + str(data), '--console', '--skip-networking', '--mysqlx=OFF',
               '--shared-memory', '--shared-memory-base-name=' + memory,
               '--secure-file-priv=' + str(uploads), '--local-infile=OFF', '--skip-log-bin',
               '--innodb-buffer-pool-size=32M']
    child_env = os.environ.copy()
    # Existing client settings must not influence this isolated test connection.
    child_env.pop('MYSQL_PWD', None)
    connect = [str(client), '--no-defaults', '--no-login-paths', '--protocol=MEMORY',
               '--shared-memory-base-name=' + memory, '--user=root',
               '--default-character-set=utf8mb4', '--batch', '--raw', '--skip-column-names',
               '--connect-timeout=2']
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'run_dir': str(run_dir),
              'datadir': str(data), 'server_command': command, 'client_command': connect,
              'network': 'TCP disabled; X plugin disabled; unique local shared-memory name',
              'source': 'https://cdn.mysql.com/Downloads/MySQL-8.4/mysql-8.4.11-winx64.zip',
              'archive_sha256': 'A492371D687D2BAB088B0062581144A0044B8964BAEFDF4FAA579292B423D25C',
              'checks': [], 'status': 'starting'}
    trace = report['checks']
    process = None
    result = None
    output = ''
    try:
        with (run_dir / 'server.log').open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report['process_id'] = process.pid
            deadline = time.monotonic() + 150
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Owned MySQL exited before readiness; inspect run server.log.')
                attempt = subprocess.run(connect, input='SELECT 1;', text=True, encoding='utf-8',
                                         capture_output=True, env=child_env, timeout=5)
                if attempt.returncode == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Owned MySQL readiness timed out: ' + attempt.stderr)
                time.sleep(1)
            # Password exists only in parent memory/child env, never SQL trace or output files.
            password = secrets.token_hex(24)
            secured = subprocess.run(connect, input="ALTER USER 'root'@'localhost' IDENTIFIED BY '" + password + "';",
                                     text=True, encoding='utf-8', capture_output=True, env=child_env, timeout=15)
            if secured.returncode:
                raise RuntimeError('Could not set isolated test credential; server will be stopped.')
            child_env['MYSQL_PWD'] = password
            identity = subprocess.run(connect, input='SELECT @@version, @@datadir, @@skip_networking, @@shared_memory, @@shared_memory_base_name;',
                                      text=True, encoding='utf-8', capture_output=True, env=child_env, timeout=15)
            if identity.returncode:
                raise RuntimeError('Could not verify isolated MySQL identity: ' + identity.stderr)
            version, actual_data, no_tcp, shared, actual_memory = identity.stdout.strip().split('\t')
            report['server_identity'] = dict(version=version, datadir=actual_data,
                                             skip_networking=no_tcp, shared_memory=shared,
                                             shared_memory_base_name=actual_memory)
            # Windows server strings may expose a non-ASCII parent path in its
            # local encoding. Match the task-owned ASCII suffix as well as the
            # random, freshly generated shared-memory endpoint and child PID.
            expected_suffix = str(data.relative_to(run_dir.parent)).replace('/', '\\').lower().rstrip('\\')
            actual_normalized = actual_data.replace('/', '\\').lower().rstrip('\\')
            if not actual_normalized.endswith(expected_suffix) or no_tcp not in {'1', 'ON'} or shared not in {'1', 'ON'} or actual_memory != memory:
                raise RuntimeError('Server identity/isolation did not match owned task parameters.')
            report['version'] = version
            report['identity_verified'] = True
            result, output = run_checks(client, memory, uploads, source, child_env, trace)
            report.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                          status='passed' if result.wasSuccessful() else 'failed')
    except Exception as exc:
        report['status'] = 'environment_error'
        report['error'] = str(exc)
        output += str(exc) + '\n'
    finally:
        if process is not None and process.poll() is None:
            shutdown = subprocess.run(connect, input='SHUTDOWN;', text=True, encoding='utf-8',
                                      capture_output=True, env=child_env, timeout=15)
            report['shutdown_exit_code'] = shutdown.returncode
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                # This handle is the process created above, never a discovered user server.
                process.terminate()
                process.wait(timeout=15)
                report['forced_owned_process_stop'] = True
        report['server_process_stopped'] = process is None or process.poll() is not None
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        child_env.pop('MYSQL_PWD', None)
        (run_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (run_dir / 'tests.log').write_text(output, encoding='utf-8')
        (source / 'mysql-integration-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (source / 'mysql-integration-tests.log').write_text(output, encoding='utf-8')
        if args.ascii_temp:
            archive_dir = runtime / ('logs-' + token)
            archive_dir.mkdir(exist_ok=False)
            for name in ['initialize.log', 'server.log', 'result.json', 'tests.log']:
                log = run_dir / name
                if log.is_file():
                    shutil.copy2(log, archive_dir / name)
            # Remove only the exact unique directory created by this invocation.
            resolved = run_dir.resolve()
            owned = (resolved.parent == temp_parent and
                     resolved.name == 'codex-portfolio-mysql-' + token and
                     (resolved / 'OWNER.txt').read_text(encoding='utf-8') ==
                     'Task-owned synthetic MySQL integration; no user data.\n')
            if report['server_process_stopped'] and owned:
                shutil.rmtree(resolved)
                report['temporary_runtime_removed'] = not resolved.exists()
            else:
                report['temporary_runtime_removed'] = False
            (source / 'mysql-integration-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            (archive_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(output)
    print(json.dumps({key: report.get(key) for key in ['status','version','tests_run','failures','errors','server_process_stopped','run_dir']}, ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
