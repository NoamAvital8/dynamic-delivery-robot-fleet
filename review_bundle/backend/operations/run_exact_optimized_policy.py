"""Execute an original policy with opt-in exact, shared routing accelerators."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--execution-status', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('original script and arguments required')
    script = Path(command[0]).resolve()
    if script.parent != (ROOT/'scripts').resolve() or not script.is_file():
        parser.error('only original repository policy scripts may be executed')
    from exact_runtime import install, file_hash
    graph = Path(command[command.index('--graph')+1])
    graph_hash = file_hash(graph)

    def publish(key):
        marker = args.cache_dir/'graphs'/f'{graph_hash}.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_name(marker.name+f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps({'graph_sha256':graph_hash,'index_key':key}))
        temporary.replace(marker)
        if args.execution_status is not None:
            data = json.loads(args.execution_status.read_text())
            data.update(phase='simulation', shared_index_key=key)
            temporary = args.execution_status.with_suffix('.tmp')
            temporary.write_text(json.dumps(data))
            temporary.replace(args.execution_status)

    install(args.cache_dir, workers=args.workers, on_index_ready=publish)
    print('execution optimization: exact-shared-float64-index-v1', flush=True)
    sys.argv = [str(script), *command[1:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
