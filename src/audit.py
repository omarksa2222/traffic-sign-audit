#!/usr/bin/env python3
"""Reproducible, rule-based visibility audit of GTSRB training images.

Run from the repository root. No classifier is trained. Measurement thresholds
are sample-relative triage settings, not road-safety standards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]
FEATURES = ["brightness", "contrast", "sharpness", "min_dimension"]
LABELS = {
    0: "Speed limit 20 km/h", 1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h", 3: "Speed limit 60 km/h",
    4: "Speed limit 70 km/h", 5: "Speed limit 80 km/h",
    6: "End speed limit 80", 7: "Speed limit 100 km/h",
    8: "Speed limit 120 km/h", 9: "No passing", 10: "No passing >3.5t",
    11: "Right-of-way at intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "No vehicles >3.5t", 17: "No entry",
    18: "General caution", 19: "Dangerous curve left", 20: "Dangerous curve right",
    21: "Double curve", 22: "Bumpy road", 23: "Slippery road", 24: "Road narrows right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End all restrictions", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End no passing", 42: "End no passing >3.5t",
}


def load_annotations(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Read all annotation CSVs, retaining raw values and provenance; return table and diagnostics."""
    files = sorted(data_dir.rglob("GT-*.csv"))
    if not files:
        raise FileNotFoundError(f"No GT-*.csv annotation files found under {data_dir}. Check --data-dir.")
    frames, malformed = [], []
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path, sep=";", dtype=str)
            df.columns = df.columns.str.strip().str.lstrip("\ufeff")
            missing = sorted(set(REQUIRED) - set(df.columns))
            if missing:
                malformed.append({"file": str(csv_path.relative_to(data_dir)), "reason": f"Missing columns: {missing}"})
                continue
            df = df[REQUIRED].copy()
            df["annotation_file"] = csv_path.relative_to(data_dir).as_posix()
            df["annotation_row"] = np.arange(2, len(df) + 2)
            # Dataset files reside alongside the annotations. Store only portable relative paths.
            df["image_path"] = df["Filename"].fillna("").map(
                lambda name: (csv_path.parent / str(name)).relative_to(data_dir).as_posix())
            frames.append(df)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
            malformed.append({"file": str(csv_path), "reason": str(exc)})
    if malformed:
        # Stopping is essential: otherwise reported full-data counts would silently omit files.
        raise ValueError(f"Malformed/unreadable annotation files; no full counts can be reported: {malformed}")
    raw = pd.concat(frames, ignore_index=True)
    duplicate_rows = int(raw.duplicated(subset=REQUIRED + ["annotation_file"]).sum())
    diagnostics = {
        "annotation_files": len(files), "annotation_rows": len(raw),
        "raw_column_dtypes": {k: str(v) for k, v in raw[REQUIRED].dtypes.items()},
        "missing_values": {k: int(v) for k, v in raw[REQUIRED].isna().sum().items()},
        "duplicate_rows_within_file": duplicate_rows,
    }
    for col in REQUIRED[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["class_valid"] = raw["ClassId"].notna() & (raw["ClassId"] >= 0) & (raw["ClassId"] % 1 == 0)
    raw["path_valid"] = raw.apply(lambda row: bool(row["Filename"]) and pd.notna(row["Filename"])
                                  and (data_dir / row["image_path"]).is_file(), axis=1)
    diagnostics["converted_column_dtypes"] = {k: str(v) for k, v in raw[REQUIRED].dtypes.items()}
    diagnostics["invalid_numeric_values"] = {c: int(raw[c].isna().sum()) for c in REQUIRED[1:]}
    diagnostics["invalid_class_rows"] = int((~raw["class_valid"]).sum())
    diagnostics["invalid_image_paths"] = int((~raw["path_valid"]).sum())
    diagnostics["unique_valid_classes"] = int(raw.loc[raw.class_valid, "ClassId"].nunique())
    expected_classes = set(range(43))
    found_classes = set(raw.loc[raw.class_valid, "ClassId"].astype(int))
    diagnostics["missing_expected_training_classes"] = sorted(expected_classes - found_classes)
    diagnostics["full_training_class_coverage"] = expected_classes.issubset(found_classes)
    if diagnostics["missing_expected_training_classes"]:
        print("WARNING: expected GTSRB training classes are missing from the supplied annotations: "
              f"{diagnostics['missing_expected_training_classes']}. Class counts cover only supplied files; "
              "they are not verified full-dataset counts.", file=sys.stderr)
    return raw, diagnostics


def read_image(path: Path) -> np.ndarray | None:
    """Load a BGR image, including Unicode paths unsupported by some OpenCV builds."""
    if not path.is_file():
        return None
    try:
        logging_api = getattr(cv2.utils, "logging", cv2)
        if (sys.platform == "win32" and not str(path).isascii()
                and hasattr(logging_api, "getLogLevel") and hasattr(logging_api, "setLogLevel")):
            # Avoid a misleading warning when the fallback below succeeds. The audit is single-threaded.
            previous_log_level = logging_api.getLogLevel()
            try:
                logging_api.setLogLevel(2)  # Preserve error messages while suppressing the expected warning.
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            finally:
                logging_api.setLogLevel(previous_log_level)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
        # Python opens Unicode paths reliably on Windows; OpenCV still decodes the bytes.
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, ValueError, cv2.error):
        return None


def extract_image_features(path: Path) -> tuple[dict | None, str | None]:
    """Read a BGR image and return grayscale statistics, or a reason on read failure."""
    image = read_image(path)
    if image is None:
        return None, "image could not be read by OpenCV"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    return {"original_width": width, "original_height": height, "min_dimension": min(width, height),
            "brightness": float(np.mean(gray)), "contrast": float(np.std(gray)),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}, None


def assign_review_priority(table: pd.DataFrame, quantile: float) -> tuple[np.ndarray, dict]:
    """Flag any image below a sample percentile for at least one feature; return mask and cutoffs."""
    if not 0 < quantile < 1:
        raise ValueError("Quantile must be strictly between zero and one.")
    values = table[FEATURES].to_numpy(dtype=float)
    cutoffs = np.percentile(values, 100 * quantile, axis=0)
    mask = np.any(values < cutoffs, axis=1)
    return mask, dict(zip(FEATURES, map(float, cutoffs)))


def roi_measurements(path: Path, row: pd.Series) -> tuple[dict | None, str]:
    """Compare full-image measurements with a valid annotated ROI; return reason if unavailable."""
    image = read_image(path)
    if image is None:
        return None, "unreadable image"
    h, w = image.shape[:2]
    cols = ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]
    coords = row[cols].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or not (coords == np.floor(coords)).all():
        return None, "missing or non-integer ROI coordinates"
    x1, y1, x2, y2 = map(int, coords)
    # GTSRB coordinates include the bottom-right pixel; Python slice end is exclusive.
    if not (0 <= x1 <= x2 < w and 0 <= y1 <= y2 < h):
        return None, "ROI outside image or reversed"
    crop = image[y1:y2 + 1, x1:x2 + 1]
    def measures(im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return [float(np.mean(gray)), float(np.std(gray)), float(cv2.Laplacian(gray, cv2.CV_64F).var())]
    whole, region = measures(image), measures(crop)
    return {**{f"full_{k}": v for k, v in zip(FEATURES[:3], whole)},
            **{f"roi_{k}": v for k, v in zip(FEATURES[:3], region)}}, "valid"


def validate() -> None:
    """Exercise read-failure, output columns and Boolean flag invariants with synthetic values."""
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        missing, reason = extract_image_features(root / "not_present.png")
        assert missing is None and reason
        (root / "empty.png").write_bytes(b"")
        assert read_image(root / "empty.png") is None
        for n, val in enumerate((30, 180)):
            image = np.full((20, 30, 3), val, dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            (root / f"{n}.png").write_bytes(encoded.tobytes())
        unicode_path = root / "اختبار" / "صورة.png"
        unicode_path.parent.mkdir()
        unicode_path.write_bytes((root / "0.png").read_bytes())
        unicode_features, unicode_error = extract_image_features(unicode_path)
        assert unicode_error is None and unicode_features == extract_image_features(root / "0.png")[0]
        results = [extract_image_features(root / f"{n}.png")[0] for n in range(2)]
        table = pd.DataFrame(results)
        assert set(FEATURES + ["original_width", "original_height"]).issubset(table.columns)
        flags, _ = assign_review_priority(table, .1)
        assert flags.dtype == np.bool_ and set(flags.tolist()).issubset({True, False})
        roi, status = roi_measurements(root / "0.png", pd.Series({"Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": 29, "Roi.Y2": 19}))
        assert status == "valid" and roi is not None
        unicode_roi, unicode_status = roi_measurements(unicode_path, pd.Series({"Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": 29, "Roi.Y2": 19}))
        assert unicode_status == "valid" and unicode_roi == roi
        _, status = roi_measurements(root / "0.png", pd.Series({"Roi.X1": -1, "Roi.Y1": 0, "Roi.X2": 29, "Roi.Y2": 19}))
        assert status != "valid"
    print("Validation passed: image-read failure, Unicode paths, feature columns, Boolean flags and ROI bounds.")


class TrafficSignAudit:
    """Coordinate annotation validation, image measurements, summaries and figures."""
    def __init__(self, data_dir: Path, output_dir: Path, classes: list[int], sample: int, seed: int):
        self.data_dir, self.output_dir = data_dir, output_dir
        self.classes, self.sample, self.seed = classes, sample, seed
        self.processed, self.figures = output_dir / "processed", output_dir / "figures"
        self.processed.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

    def load(self):
        self.annotations, self.diagnostics = load_annotations(self.data_dir)
        a = self.annotations
        counts = a.loc[a.class_valid].groupby("ClassId").size().rename("annotation_count").reset_index()
        counts["ClassId"] = counts.ClassId.astype(int)
        counts["class_name"] = counts.ClassId.map(LABELS).fillna("Unknown class")
        counts.to_csv(self.processed / "full_class_counts.csv", index=False)
        a.loc[~a.path_valid, ["annotation_file", "annotation_row", "image_path", "ClassId"]].to_csv(
            self.processed / "invalid_paths.csv", index=False)
        missing_classes = set(self.classes) - set(counts.ClassId)
        if missing_classes:
            raise ValueError(f"Selected classes absent from annotations: {sorted(missing_classes)}")

    def extract(self):
        # Sample annotations before checking paths; failed rows are logged, never replaced silently.
        a = self.annotations
        selected = a[a.class_valid & a.ClassId.isin(self.classes)].copy()
        sampled = selected.groupby("ClassId", group_keys=False).sample(n=None, frac=1, random_state=self.seed)
        sampled = sampled.groupby("ClassId", group_keys=False).head(self.sample).copy()
        self.diagnostics.update({"selected_classes": self.classes, "seed": self.seed,
                                 "sample_limit_per_class": self.sample, "sampled_rows": len(sampled)})
        records, skipped, roi_rows, roi_statuses = [], [], [], []
        for _, row in sampled.iterrows():
            features, reason = extract_image_features(self.data_dir / row.image_path) if row.path_valid else (None, "missing image path")
            if features is None:
                skipped.append({"image_path": row.image_path, "class_id": row.ClassId, "reason": reason,
                                "annotation_file": row.annotation_file, "annotation_row": row.annotation_row})
                continue
            record = {"image_path": row.image_path, "annotation_file": row.annotation_file,
                      "annotation_row": int(row.annotation_row), "class_id": int(row.ClassId),
                      "class_name": LABELS.get(int(row.ClassId), f"Class {int(row.ClassId)}"), **features}
            record["annotated_width"] = row.Width
            record["annotated_height"] = row.Height
            record["dimensions_match_annotation"] = bool(row.Width == features["original_width"] and row.Height == features["original_height"])
            # Validate all ROIs, but retain at most ten for an interpretable comparison table.
            roi, status = roi_measurements(self.data_dir / row.image_path, row)
            record["roi_status"] = status
            roi_statuses.append(status)
            if roi is not None and len(roi_rows) < 10:
                roi_rows.append({"image_path": row.image_path, "class_id": int(row.ClassId), **roi})
            records.append(record)
        pd.DataFrame(skipped, columns=["image_path", "class_id", "reason", "annotation_file", "annotation_row"]).to_csv(self.processed / "skipped_images.csv", index=False)
        pd.DataFrame(roi_rows).to_csv(self.processed / "roi_comparison.csv", index=False)
        if not records:
            raise ValueError("No sampled images could be processed; see skipped_images.csv and verify data path.")
        self.table = pd.DataFrame(records)
        self.diagnostics.update({"processed_images": len(records), "skipped_images": len(skipped),
                                 "skip_reasons": pd.Series([x["reason"] for x in skipped], dtype=str).value_counts().to_dict(),
                                 "roi_status_counts": pd.Series(roi_statuses).value_counts().to_dict(),
                                 "annotation_dimension_mismatches": int((~self.table.dimensions_match_annotation).sum())})

    def summarize(self):
        t = self.table
        thresholds = {}
        for name, q in [("strict", .05), ("baseline", .10), ("broad", .15)]:
            t[f"flag_{name}"], thresholds[name] = assign_review_priority(t, q)
        self.thresholds = thresholds
        for metric in FEATURES:
            t[f"low_{metric}"] = t[metric].to_numpy() < thresholds["baseline"][metric]
        t["review_reason_count"] = t[[f"low_{m}" for m in FEATURES]].sum(axis=1)
        t.to_csv(self.processed / "image_features.csv", index=False)
        summary = t.groupby(["class_id", "class_name"]).agg(
            sample_n=("image_path", "size"), mean_width=("original_width", "mean"),
            mean_height=("original_height", "mean"), median_min_dimension=("min_dimension", "median"),
            median_brightness=("brightness", "median"), median_contrast=("contrast", "median"),
            median_sharpness=("sharpness", "median"), flagged_n=("flag_baseline", "sum"),
            strict_rate=("flag_strict", "mean"), baseline_rate=("flag_baseline", "mean"),
            broad_rate=("flag_broad", "mean")).reset_index()
        for col in ["strict", "baseline", "broad"]:
            summary[f"{col}_rank"] = summary[f"{col}_rate"].rank(ascending=False, method="min").astype(int)
        summary.to_csv(self.processed / "class_summary.csv", index=False)
        t[["brightness", "contrast", "sharpness"]].corr(method="spearman").to_csv(self.processed / "spearman_correlations.csv")
        self.summary = summary
        self.diagnostics["thresholds"] = thresholds
        self.diagnostics["flag_rates"] = {n: float(t[f"flag_{n}"].mean()) for n in ("strict", "baseline", "broad")}
        self.diagnostics["feature_definitions"] = {
            "brightness": "mean grayscale intensity (0–255)",
            "contrast": "population standard deviation of grayscale intensity (0–255 intensity units)",
            "sharpness": "variance of CV_64F Laplacian grayscale response (squared response units; proxy only)",
            "min_dimension": "minimum of actual pixel width and height (pixels)"}
        (self.processed / "diagnostics.json").write_text(json.dumps(self.diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")

    def plot(self):
        t, out = self.table, self.figures
        plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .2, "figure.dpi": 130})
        # Select interpretable examples: extreme on independent features, then unflagged.
        chosen = []
        for metric in ["brightness", "sharpness", "contrast", "min_dimension"]:
            for idx in t.sort_values(metric).index:
                if idx not in chosen:
                    chosen.append(idx); break
        for idx in t[~t.flag_baseline].sort_values("sharpness", ascending=False).index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= 6: break
        if len(chosen) < 6:
            chosen.extend([i for i in t.index if i not in chosen][:6-len(chosen)])
        pd.DataFrame(t.loc[chosen, ["image_path", "class_id", "class_name", "brightness", "contrast", "sharpness", "min_dimension", "flag_baseline"]]).to_csv(self.processed / "example_images.csv", index=False)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, idx in zip(axes.flat, chosen):
            row = t.loc[idx]
            image = read_image(self.data_dir / row.image_path)
            if image is None:
                raise OSError(f"Example image could not be reloaded: {row.image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(cv2.resize(image, (160, 160), interpolation=cv2.INTER_NEAREST))
            ax.set_title(f"{row.class_id}: {row.class_name}\n{'REVIEW' if row.flag_baseline else 'not flagged'} | B {row.brightness:.1f}, C {row.contrast:.1f}\nL {row.sharpness:.1f}, min {row.min_dimension}px", fontsize=8)
            ax.set_xlabel("display x (px)"); ax.set_ylabel("display y (px)")
        for ax in list(axes.flat)[len(chosen):]: ax.set_visible(False)
        fig.suptitle("Examples: review flags and measured properties (B=brightness, C=contrast, L=sharpness)")
        fig.savefig(out / "01_contact_sheet.png", bbox_inches="tight"); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, metric, title in zip(axes, ["brightness", "sharpness"], ["Brightness (gray intensity)", "Sharpness (Laplacian variance)"]):
            ax.hist(t[metric], bins=35, color="#4c78a8", edgecolor="white")
            for name, color in [("strict", "#6d8f49"), ("baseline", "#d04a3a"), ("broad", "#9b59b6")]:
                ax.axvline(self.thresholds[name][metric], color=color, linestyle="--", label=f"{name}: {self.thresholds[name][metric]:.1f}")
            ax.set(title=title, xlabel=title, ylabel="Number of images"); ax.legend(fontsize=8)
        fig.suptitle("Sample distributions and project-specific cutoff scenarios")
        fig.savefig(out / "02_distributions.png", bbox_inches="tight"); plt.close(fig)
        groups = [t.loc[t.class_id == c, "sharpness"].values for c in self.classes if (t.class_id == c).any()]
        ids = [c for c in self.classes if (t.class_id == c).any()]
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.boxplot(groups, showfliers=False)
        ax.set_xticks(range(1, len(ids) + 1), [str(c) for c in ids])
        ax.set(title="Sharpness distribution by sign class", xlabel="GTSRB class ID", ylabel="Laplacian variance (squared response units)")
        fig.savefig(out / "03_sharpness_by_class.png", bbox_inches="tight"); plt.close(fig)
        s = self.summary.sort_values("baseline_rate", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.bar(s.class_id.astype(str), s.baseline_rate * 100, color="#d04a3a")
        ax.set(title="Images flagged for human review by class", xlabel="GTSRB class ID", ylabel="Flagged images (%)", ylim=(0, 100))
        for i, row in enumerate(s.itertuples()): ax.text(i, row.baseline_rate*100 + 1, f"{row.baseline_rate*100:.1f}%\n(n={row.sample_n})", ha="center", fontsize=8)
        fig.savefig(out / "04_flag_rate_by_class.png", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.boxplot([t.loc[t.class_id == c, "contrast"].values for c in ids], showfliers=False)
        ax.set_xticks(range(1, len(ids) + 1), [str(c) for c in ids])
        ax.set(title="Contrast distribution by sign class", xlabel="GTSRB class ID", ylabel="Grayscale standard deviation (intensity units)")
        fig.savefig(out / "05_contrast_by_class.png", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.scatter(t["contrast"], t["sharpness"], s=9, alpha=.3, c="#4c78a8")
        ax.set(title="Contrast and sharpness in the selected sample", xlabel="Contrast (grayscale standard deviation)", ylabel="Sharpness (Laplacian variance)")
        fig.savefig(out / "06_contrast_sharpness.png", bbox_inches="tight"); plt.close(fig)

    def run(self):
        self.load(); self.extract(); self.summarize(); self.plot()
        print(f"Processed {len(self.table)} images; skipped {self.diagnostics['skipped_images']}. Results: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/GTSRB/Final_Training/Images"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1, 14, 17, 25])
    parser.add_argument("--sample-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(); return
    if len(set(args.classes)) < 5 or len(args.classes) != len(set(args.classes)):
        parser.error("Choose at least five distinct classes.")
    if not 1 <= args.sample_per_class <= 500:
        parser.error("--sample-per-class must be between 1 and 500.")
    if not args.data_dir.is_dir():
        parser.error(f"Dataset directory not found: {args.data_dir}")
    try:
        TrafficSignAudit(args.data_dir, args.output_dir, args.classes, args.sample_per_class, args.seed).run()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
