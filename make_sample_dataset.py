"""Build a sample dataset for the attendance portal from the public LFW face dataset.

Output (in ./sample_dataset):
  students/<SAPID>_<Name>/   5-8 registration photos per person (upload these in "Register students")
  classroom/classroom_1.jpg  a grid of held-out photos of some registered people (upload in "Take attendance")
  classroom/classroom_2.jpg  a different subset
  roster.csv                 SAP ID, Name, and who is in which classroom photo (ground truth)

Usage:
  pip install scikit-learn
  python make_sample_dataset.py --people 10
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

OUT = Path(__file__).parent / "sample_dataset"


def grid(images, cols):
    rows = [images[i:i + cols] for i in range(0, len(images), cols)]
    blank = np.full_like(images[0], 255)
    rows = [np.hstack(r + [blank] * (cols - len(r))) for r in rows]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=10)
    ap.add_argument("--min-photos", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    lfw = fetch_lfw_people(min_faces_per_person=args.min_photos, color=True,
                           slice_=None, resize=1.0)
    imgs, labels, names = lfw.images, lfw.target, lfw.target_names
    # sklearn returns float RGB scaled to 0-1 (older versions: 0-255); normalise to 0-255
    if imgs.max() <= 1.0:
        imgs = imgs * 255.0
    chosen = rnd.sample(range(len(names)), min(args.people, len(names)))

    roster, held_out = [], {}
    for n, idx in enumerate(chosen):
        sap = str(590010000 + n + 1)
        name = names[idx]
        photos = [imgs[i] for i in np.where(labels == idx)[0]]
        rnd.shuffle(photos)
        folder = OUT / "students" / f"{sap}_{name.replace(' ', '_')}"
        folder.mkdir(parents=True, exist_ok=True)
        for k, img in enumerate(photos[:7]):
            bgr = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(folder / f"photo_{k + 1}.jpg"), bgr)
        test = np.clip(photos[7], 0, 255).astype(np.uint8)
        held_out[sap] = cv2.cvtColor(test, cv2.COLOR_RGB2BGR)
        roster.append({"SAP ID": sap, "Name": name})

    (OUT / "classroom").mkdir(parents=True, exist_ok=True)
    saps = list(held_out)
    present = {1: saps[: max(2, len(saps) * 2 // 3)], 2: saps[len(saps) // 3:]}
    for num, group in present.items():
        cv2.imwrite(str(OUT / "classroom" / f"classroom_{num}.jpg"),
                    grid([held_out[s] for s in group], cols=4))
    for r in roster:
        r["In classroom_1"] = r["SAP ID"] in present[1]
        r["In classroom_2"] = r["SAP ID"] in present[2]

    with open(OUT / "roster.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(roster[0]))
        w.writeheader()
        w.writerows(roster)
    print(f"Done. Sample dataset written to: {OUT}")


if __name__ == "__main__":
    main()
