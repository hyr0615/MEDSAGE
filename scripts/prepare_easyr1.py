"""Create a separate EasyR1 working copy and install the MEDSAGE overlay."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--allow-unverified', action='store_true',
                        help='Allow a different EasyR1 interface after manual review.')
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    overlay = Path(__file__).resolve().parents[1] / 'easyr1'
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Destination must be a new directory outside the source tree.')
    if not (source / 'verl/trainer/main.py').is_file():
        raise ValueError('Source is not an EasyR1 source tree.')
    expected = json.loads((overlay / 'compatibility.json').read_text())['files']
    mismatches = []
    for name, digest in expected.items():
        path = source / name
        actual = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() if path.is_file() else None
        if actual != digest:
            mismatches.append(name)
    if mismatches and not args.allow_unverified:
        raise RuntimeError('Unverified EasyR1 interface: ' + ', '.join(mismatches))
    if mismatches:
        print('WARNING: compatibility not verified for ' + ', '.join(mismatches))
    # Copy framework sources and package metadata only.
    shutil.copytree(source / 'verl', destination / 'verl',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.log'))
    for name in ['pyproject.toml', 'setup.py', 'setup.cfg', 'requirements.txt', 'README.md', 'LICENSE']:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    shutil.copytree(overlay / 'verl', destination / 'verl', dirs_exist_ok=True)
    print(f'Prepared EasyR1 working copy: {destination}')


if __name__ == '__main__':
    main()
