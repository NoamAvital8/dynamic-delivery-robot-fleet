"""Read-only, dependency-free summary of the campaign's currently selected jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

LABELS = {
    'myopic_ab': 'Myopic A+B', 'reactive_insertion': 'Reactive insertion',
    'paper_sa_adapted': 'Paper-adapted', 'full_no_nn': 'Original full, no NN',
    'full_no_idle': 'Original full, no idle', 'full': 'Original full',
    'full_queue_aware': 'Queue-aware + NN', 'full_coordinated_idle': 'Full coordinated + NN',
    'queue_mle_reservation': 'Queue-aware + MLE', 'full_mle_reservation': 'Full coordinated + MLE',
}
CITY_LABELS = {'beijing': 'Beijing', 'sydney': 'Sydney', 'moscow': 'Moscow',
               'johannesburg': 'Johannesburg', 'new_delhi': 'New Delhi', 'paris': 'Paris'}


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, PermissionError):
            if attempt == 2:
                raise
            time.sleep(.05)


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'unsafe campaign identifier: {value!r}')
    return value


def collect(campaign):
    campaign = campaign.resolve()
    manifest = read_json(campaign/'benchmarks/run_manifest.json')
    policies = manifest['policies']
    if len(policies) != len(set(policies)):
        raise ValueError('duplicate policies in active manifest')
    for policy in policies:
        safe_name(policy)
    suitepath = Path(manifest['suite'])
    if not suitepath.is_absolute():
        suitepath = campaign/suitepath
    suite = read_json(suitepath)
    counts = Counter()
    groups = []
    failures = []
    for city, city_data in suite['cities'].items():
        safe_name(city)
        for policy in policies:
            bucket = {'city': city, 'policy': policy, 'completed': 0,
                      'planned': len(city_data['test']), 'loss': 0.,
                      'delivered': 0, 'on_time': 0, 'late': 0}
            for scenario in city_data['test']:
                folder = campaign/'benchmarks'/city/safe_name(scenario['id'])/policy
                statuspath = folder/'status.json'
                if not statuspath.exists():
                    counts['pending'] += 1
                    continue
                status = read_json(statuspath)
                kind = status.get('status', 'unknown')
                counts[kind] += 1
                if kind == 'failed':
                    failures.append({'city': city, 'policy': policy,
                                     'scenario': scenario['id'], 'error': status.get('error', '')})
                if kind != 'complete':
                    continue
                # An output written by a still-running simulator is not counted
                # until its controller has marked it complete and stamped it.
                stamp = read_json(folder/'stamp.json')
                if not stamp.get('signature'):
                    raise ValueError(f'completed result lacks signature: {folder}')
                result = read_json(folder/'result.json')
                loss = float(result['loss_objective'])
                delivered, on_time = int(result['delivered']), int(result['on_time'])
                if (not math.isfinite(loss) or result['orders'] != scenario['orders']
                        or result['scenario_seed'] != scenario['seed']
                        or not 0 <= on_time <= delivered <= result['orders']):
                    raise ValueError(f'invalid completed result: {folder}')
                bucket['completed'] += 1
                bucket['loss'] += loss
                bucket['delivered'] += delivered
                bucket['on_time'] += on_time
                bucket['late'] += delivered - on_time
            groups.append(bucket)
    expected = sum(r['planned'] for r in groups)
    return {'captured_at': datetime.now(ZoneInfo('Asia/Jerusalem')).isoformat(timespec='seconds'),
            'campaign': str(campaign), 'expected': expected, 'counts': dict(counts),
            'cities': list(suite['cities']), 'policies': policies, 'results': groups,
            'failures': failures}


def table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
    def line(row):
        return ' | '.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
    return '\n'.join([line(headers), '-+-'.join('-'*w for w in widths), *(line(row) for row in rows)])


def render(data, deliveries=False):
    counts = data['counts']
    done = counts.get('complete', 0)
    lines = [f"Snapshot (Asia/Jerusalem): {data['captured_at']}",
             f"Completed: {done}/{data['expected']} ({100*done/data['expected']:.1f}%) | "
             f"Running: {counts.get('running', 0)} | Queued: {counts.get('pending', 0)} | "
             f"Failed: {counts.get('failed', 0)}", '',
             'TOTAL LOSS from completed scenarios only; lower is better.',
             'Each cell: loss (completed/planned). Partial totals are NOT comparable across unequal scenario counts.', '']
    cities, policies = data['cities'], data['policies']
    by_key = {(r['city'], r['policy']): r for r in data['results']}
    rows = []
    for policy in policies:
        row = [LABELS.get(policy, policy)]
        for city in cities:
            r = by_key[(city, policy)]
            value = f"{r['loss']:,.0f}" if r['completed'] else '-'
            row.append(f"{value} ({r['completed']}/{r['planned']})")
        rows.append(row)
    lines.append(table(['Policy', *(CITY_LABELS.get(c, c) for c in cities)], rows))
    if deliveries:
        rows = [[CITY_LABELS.get(r['city'], r['city']), LABELS.get(r['policy'], r['policy']),
                 f"{r['completed']}/{r['planned']}", r['delivered'], r['on_time'], r['late']]
                for r in data['results'] if r['completed']]
        if rows:
            lines.extend(['', table(['City', 'Policy', 'Complete', 'Delivered', 'On-time', 'Late'], rows)])
    if data['failures']:
        lines.extend(['', 'Failures:', *(json.dumps(r) for r in data['failures'])])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--deliveries', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    data = collect(args.campaign)
    print(json.dumps(data) if args.json else render(data, args.deliveries))


if __name__ == '__main__':
    main()
