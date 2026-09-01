"""
Split the Zenodo slicing dataset (new_dataset/) into train/val/test directories
that the DatanetAPI can read directly.

Each output directory contains:
  - graphs/, routings/, slices/  (Windows directory junctions -> new_dataset/)
  - a subset of the .tar.gz result files (hard-copied)

Split: 60% train / 20% val / 20% test (by tarball, deterministic, seed 42).

Usage:
    python slicing/split_dataset.py [--source new_dataset] [--dest slicing/data]
"""
import argparse
import os
import random
import shutil
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=os.path.join(os.path.dirname(__file__), '..', 'new_dataset'))
    p.add_argument('--dest', default=os.path.join(os.path.dirname(__file__), 'data'))
    return p.parse_args()


def make_junction(link_path, target_path):
    """Create a Windows directory junction (no admin required)."""
    link_path = os.path.abspath(link_path)
    target_path = os.path.abspath(target_path)
    if os.path.exists(link_path):
        print(f"  link already exists: {link_path}")
        return
    if os.name == 'nt':
        # Windows: directory junction (no admin/dev-mode required)
        subprocess.check_call(['cmd', '/c', 'mklink', '/J', link_path, target_path],
                              stdout=subprocess.DEVNULL)
    else:
        # Linux / macOS: plain symlink
        os.symlink(target_path, link_path)


def main():
    args = parse_args()
    source = os.path.abspath(args.source)
    dest = os.path.abspath(args.dest)

    tarballs = sorted([f for f in os.listdir(source) if f.endswith('.tar.gz')])
    print(f"Found {len(tarballs)} tarballs in {source}")

    rng = random.Random(42)
    rng.shuffle(tarballs)

    n = len(tarballs)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    splits = {
        'train': tarballs[:n_train],
        'val': tarballs[n_train:n_train + n_val],
        'test': tarballs[n_train + n_val:],
    }

    shared_dirs = ['graphs', 'routings', 'slices']

    for split_name, split_files in splits.items():
        split_dir = os.path.join(dest, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for d in shared_dirs:
            junction = os.path.join(split_dir, d)
            target = os.path.join(source, d)
            make_junction(junction, target)

        existing = set(f for f in os.listdir(split_dir) if f.endswith('.tar.gz'))
        copied = 0
        for tb in split_files:
            dst_path = os.path.join(split_dir, tb)
            if tb not in existing:
                shutil.copy2(os.path.join(source, tb), dst_path)
                copied += 1

        print(f"  {split_name}: {len(split_files)} tarballs ({copied} newly copied)")

    print("\nDone. Split directories:")
    for split_name in splits:
        print(f"  {os.path.join(dest, split_name)}")


if __name__ == '__main__':
    main()
