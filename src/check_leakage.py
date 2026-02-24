# src/remove_test_duplicates.py
import os
import hashlib
import shutil
from collections import defaultdict
import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def iter_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(root, fn)


def md5_file(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_hash_set(dir_path):
    hashes = set()
    for p in iter_images(dir_path):
        try:
            hashes.add(md5_file(p))
        except Exception as e:
            print(f"[WARN] Could not hash {p}: {e}")
    return hashes


def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["data_dir"]
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
        raise FileNotFoundError("Missing train/ or test/ folder in data_dir")

    print("Building hash set for TRAIN...")
    train_hashes = build_hash_set(train_dir)
    print("Train files hashed:", len(train_hashes))

    quarantine_dir = os.path.join(data_dir, "_quarantine_test_duplicates")
    os.makedirs(quarantine_dir, exist_ok=True)

    removed = 0
    print("Scanning TEST and moving duplicates to quarantine...")
    for p in iter_images(test_dir):
        try:
            h = md5_file(p)
        except Exception as e:
            print(f"[WARN] Could not hash {p}: {e}")
            continue

        if h in train_hashes:
            rel = os.path.relpath(p, test_dir)
            dest = os.path.join(quarantine_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(p, dest)
            print(f"[DUP] Moved: {p}  ->  {dest}")
            removed += 1

    print(f"\nDone. Moved {removed} duplicate test files into:\n{quarantine_dir}")


if __name__ == "__main__":
    main()
