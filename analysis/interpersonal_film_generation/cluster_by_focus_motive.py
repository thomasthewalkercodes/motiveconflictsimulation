"""
cluster_by_focus_motive.py

Takes the individual GIFs produced by spiderdiagram_film.py and clusters
them into one GIF per unique parameter-group (everything except
cos_decay.motive_focus), stacking the 8 focus-motive "dialogues" vertically.

Output layout per frame:
  ┌─────────────────────────────┐
  │  focus=0  │  A  │  B        │  row 0
  │  focus=1  │  A  │  B        │  row 1
  │   ...                       │
  │  focus=7  │  A  │  B        │  row 7
  └─────────────────────────────┘

Output files land in the same films/ folder, named:
  <group_base>_cluster_sim<N>.gif
"""

import yaml
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

#########################
#########################
RUN_PREFIX = "ip_bilateral_static_other"  # <- same prefix as spiderdiagram_film.py
SIM = 0  # which simulation index to cluster
#########################
#########################

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
FILMS_DIR = Path(__file__).resolve().parent / "films"
OUT_DIR = FILMS_DIR  # cluster gifs go into the same films/ folder

LABEL_WIDTH = 80  # pixels reserved on the left for a focus-motive label strip
LABEL_BG = (245, 245, 245)
LABEL_FG = (40, 40, 40)


# ── helpers ──────────────────────────────────────────────────────────────────


def load_cfg(run_dir: Path) -> dict:
    yaml_files = list(run_dir.glob("*.yaml"))
    if not yaml_files:
        return {}
    return yaml.full_load(yaml_files[0].read_text())


def group_key(cfg: dict) -> tuple:
    """Hashable key of all params that must be identical within a cluster group.
    Excludes cos_decay.motive_focus (that's the axis we're clustering over)."""
    decay = cfg.get("decay", {})
    cos = decay.get("cos_decay", {})
    pa = cfg.get("person_a", {}).get("influence", {})
    chosen_a = pa.get("chosen_influence", "")
    params_a = pa.get(chosen_a, {})
    pb = cfg.get("person_b", {}).get("influence", {})
    chosen_b = pb.get("chosen_influence", "")
    params_b = pb.get(chosen_b, {})
    return (
        decay.get("chosen_decay"),
        cos.get("amplitude"),
        cos.get("elevation"),
        chosen_a,
        str(params_a.get("motive_focus")),
        params_a.get("conflict_strength"),
        params_a.get("unilateral"),
        chosen_b,
        str(params_b.get("motive_focus")),
        params_b.get("conflict_strength"),
        params_b.get("unilateral"),
    )


def extract_all_frames(gif_path: Path) -> list[Image.Image]:
    """Return every frame of a GIF as a list of RGBA Images."""
    frames = []
    with Image.open(gif_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.convert("RGBA").copy())
    return frames


def make_label_strip(height: int, text: str) -> Image.Image:
    """Thin vertical label strip with rotated text."""
    strip = Image.new("RGBA", (LABEL_WIDTH, height), LABEL_BG + (255,))
    draw = ImageDraw.Draw(strip)
    # Try to get a small font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    # Draw text centred vertically (rotated 90°)
    tmp = Image.new("RGBA", (height, LABEL_WIDTH), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (height - tw) // 2
    ty = (LABEL_WIDTH - th) // 2
    tmp_draw.text((tx, ty), text, font=font, fill=LABEL_FG + (255,))
    strip.paste(tmp.rotate(-90, expand=True), (0, 0))
    return strip


def stack_frames(
    rows_frames: list[list[Image.Image]], focus_labels: list[str]
) -> list[Image.Image]:
    """
    rows_frames : list of n_rows lists, each containing n_frames PIL Images
    focus_labels: list of n_rows label strings
    Returns a list of n_frames composite Images (vertically stacked).
    """
    n_rows = len(rows_frames)
    n_frames = min(len(f) for f in rows_frames)
    if n_frames == 0:
        return []

    row_w, row_h = rows_frames[0][0].size
    total_w = LABEL_WIDTH + row_w
    total_h = row_h * n_rows

    composites = []
    for fi in range(n_frames):
        canvas = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 255))
        for ri, (row_f, label) in enumerate(zip(rows_frames, focus_labels)):
            y_offset = ri * row_h
            # label strip
            strip = make_label_strip(row_h, label)
            canvas.paste(strip, (0, y_offset))
            # gif frame
            canvas.paste(row_f[fi], (LABEL_WIDTH, y_offset))
        composites.append(canvas.convert("RGB"))
    return composites


def save_gif(frames: list[Image.Image], out_path: Path, fps: int = 5) -> None:
    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    run_dirs = sorted(RUNS_DIR.glob(f"{RUN_PREFIX}*"))
    if not run_dirs:
        print(f"No runs found matching prefix '{RUN_PREFIX}' in {RUNS_DIR}")
        return

    # Build groups: key -> list of (run_dir, motive_focus_int)
    groups: dict[tuple, list] = defaultdict(list)
    for run_dir in run_dirs:
        yaml_files = list(run_dir.glob("*.yaml"))
        if not yaml_files:
            continue
        cfg = load_cfg(run_dir)
        raw_focus = cfg.get("decay", {}).get("cos_decay", {}).get("motive_focus", None)
        # Normalise: skip if it's a list/string (old/malformed runs)
        if not isinstance(raw_focus, int):
            continue
        gif_path = FILMS_DIR / f"{run_dir.name}_sim{SIM}.gif"
        if not gif_path.exists():
            # Film not yet generated — skip silently
            continue
        key = group_key(cfg)
        groups[key].append((run_dir, raw_focus, gif_path))

    print(f"Found {len(groups)} group(s) to cluster.")

    for key, members in sorted(groups.items()):
        # Must have exactly the 8 focus motives (0–7)
        foci_present = sorted(set(f for _, f, _ in members))
        if len(foci_present) < 2:
            print(f"  Skipping group with only {len(foci_present)} focus motive(s).")
            continue

        # Sort rows by focus motive value
        members_sorted = sorted(members, key=lambda x: x[1])

        # Load all GIF frames for each row
        rows_frames = []
        focus_labels = []
        valid = True
        for run_dir, focus, gif_path in members_sorted:
            frames = extract_all_frames(gif_path)
            if not frames:
                print(f"  Warning: no frames in {gif_path.name}, skipping group.")
                valid = False
                break
            rows_frames.append(frames)
            focus_labels.append(f"focus={focus}")

        if not valid:
            continue

        # Build composite frames
        composite_frames = stack_frames(rows_frames, focus_labels)
        if not composite_frames:
            continue

        # Name the output after the first run dir (strip run number + timestamp)
        # e.g. ip_bilateral_static_other_00000_2026-03-28_21-48-07  ->  ip_bilateral_static_other
        base_dir_name = members_sorted[0][0].name
        # strip trailing _<timestamp> and _<5digit_num>
        parts = base_dir_name.split("_")
        # parts like ['ip','bilateral','static','other','00000','2026-03-28','21-48-07']
        # We want prefix up to (not including) the 5-digit run number
        prefix_parts = []
        for p in parts:
            if p.isdigit() and len(p) == 5:
                break
            prefix_parts.append(p)
        base = "_".join(prefix_parts)

        # Include a descriptor of what's fixed in this group (A pair, strengths, etc.)
        _, _, _, chosen_a, mf_a, cs_a, uni_a, chosen_b, mf_b, cs_b, uni_b = key
        desc = (
            f"A_mf{mf_a}_cs{cs_a}_uni{int(bool(uni_a))}"
            f"__B_mf{mf_b}_cs{cs_b}_uni{int(bool(uni_b))}"
        )
        out_name = f"{base}_cluster_{desc}_sim{SIM}.gif"
        out_path = OUT_DIR / out_name

        save_gif(composite_frames, out_path)
        print(
            f"  Saved cluster ({len(members_sorted)} rows, "
            f"{len(composite_frames)} frames): {out_name}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
