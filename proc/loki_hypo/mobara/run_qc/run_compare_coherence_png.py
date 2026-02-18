# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image_rgb(path: Path) -> np.ndarray:
	img = Image.open(path)
	if img.mode != 'RGB':
		img = img.convert('RGB')
	return np.asarray(img)


def trim_white_border(
	img_rgb: np.ndarray, *, bg_thresh: int = 250, pad: int = 2
) -> np.ndarray:
	if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
		raise ValueError(f'expected HxWx3 RGB image, got shape={img_rgb.shape}')

	mask = np.any(img_rgb < int(bg_thresh), axis=2)
	if not mask.any():
		return img_rgb

	ys, xs = np.where(mask)
	y0 = max(int(ys.min()) - int(pad), 0)
	y1 = min(int(ys.max()) + int(pad) + 1, img_rgb.shape[0])
	x0 = max(int(xs.min()) - int(pad), 0)
	x1 = min(int(xs.max()) + int(pad) + 1, img_rgb.shape[1])
	return img_rgb[y0:y1, x0:x1]


def save_rgb_png(path: Path, img_rgb: np.ndarray) -> None:
	path = path.expanduser().resolve()
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(img_rgb).save(path)


def list_event_ids(setting_dir: Path) -> list[str]:
	d = setting_dir.expanduser().resolve()
	if not d.is_dir():
		raise NotADirectoryError(f'setting_dir is not a directory: {d}')

	event_ids: list[str] = []
	for p in d.iterdir():
		if p.is_dir() and p.name and p.name[0].isdigit():
			event_ids.append(p.name)
	return sorted(set(event_ids))


def find_overlay_in_event_dir(event_dir: Path, filename: str) -> Path:
	event_dir = event_dir.expanduser().resolve()
	if not event_dir.is_dir():
		raise NotADirectoryError(f'event_dir is not a directory: {event_dir}')

	p0 = event_dir / filename
	if p0.is_file():
		return p0

	cands = sorted(event_dir.rglob(filename))
	if len(cands) == 0:
		raise FileNotFoundError(f"'{filename}' not found under event_dir: {event_dir}")
	if len(cands) > 1:
		preview = '\n'.join(p.as_posix() for p in cands[:50])
		raise RuntimeError(
			f"multiple '{filename}' found under event_dir: {event_dir}\n"
			f'Matches:\n{preview}'
		)
	return cands[0]


def render_compare_one_event(
	event_id: str,
	setting_dirs: list[Path],
	*,
	filename: str,
	out_dir: Path,
	fig_w_per_panel: float = 6.0,
	fig_h: float = 3.2,
	trim_inputs: bool = True,
	bg_thresh: int = 250,
	trim_pad: int = 2,
	wspace: float = 0.01,
	strict: bool = True,
	final_trim: bool = True,
	final_bg_thresh: int = 250,
	final_trim_pad: int = 2,
) -> Path:
	if len(setting_dirs) == 0:
		raise ValueError('setting_dirs must be non-empty')

	images: list[np.ndarray] = []
	labels: list[str] = []
	paths: list[Path] = []

	for sd in setting_dirs:
		sd2 = Path(sd).expanduser().resolve()
		if not sd2.is_dir():
			raise NotADirectoryError(f'setting_dir is not a directory: {sd2}')

		event_dir = sd2 / event_id
		if not event_dir.is_dir():
			msg = f'event_dir not found: {event_dir}'
			if strict:
				raise FileNotFoundError(msg)
			print(f'[WARN] {msg}')
			continue

		png_path = find_overlay_in_event_dir(event_dir, filename=filename)
		img = load_image_rgb(png_path)
		if trim_inputs:
			img = trim_white_border(img, bg_thresh=bg_thresh, pad=trim_pad)

		images.append(img)
		labels.append(sd2.name)
		paths.append(png_path)

	if len(images) == 0:
		raise RuntimeError(f'no images collected for event_id={event_id}')

	n = len(images)
	fig = plt.figure(figsize=(fig_w_per_panel * n, fig_h))

	# suptitle無しで、外側余白ゼロ寄せ
	gs = fig.add_gridspec(
		1,
		n,
		left=0.0,
		right=1.0,
		bottom=0.0,
		top=1.0,
		wspace=float(wspace),
	)

	for i, (img, lbl) in enumerate(zip(images, labels, strict=True)):
		ax = fig.add_subplot(gs[0, i])
		ax.imshow(img)
		ax.axis('off')

	out_dir = Path(out_dir).expanduser().resolve()
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f'{event_id}__{Path(filename).stem}__compare.png'

	# bbox_inches tight で外側余白を削る
	fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.0)
	plt.close(fig)

	# 最後に“完成画像”も白フチ削る（これが一番確実）
	if final_trim:
		img2 = load_image_rgb(out_path)
		img2 = trim_white_border(img2, bg_thresh=final_bg_thresh, pad=final_trim_pad)
		save_rgb_png(out_path, img2)

	return out_path


def render_compare_all_events(
	setting_dirs: list[Path],
	*,
	filename: str,
	out_dir: Path,
	fig_w_per_panel: float = 6.0,
	fig_h: float = 3.2,
	trim_inputs: bool = True,
	bg_thresh: int = 250,
	trim_pad: int = 2,
	wspace: float = 0.01,
	strict: bool = True,
	final_trim: bool = True,
	final_bg_thresh: int = 250,
	final_trim_pad: int = 2,
) -> list[Path]:
	if len(setting_dirs) == 0:
		raise ValueError('setting_dirs must be non-empty')

	all_event_ids: set[str] = set()
	for sd in setting_dirs:
		all_event_ids |= set(list_event_ids(Path(sd)))

	if len(all_event_ids) == 0:
		raise RuntimeError('no event_id directories found (digit-start dirs)')

	out_paths: list[Path] = []
	for event_id in sorted(all_event_ids):
		out_png = render_compare_one_event(
			event_id,
			setting_dirs,
			filename=filename,
			out_dir=out_dir,
			fig_w_per_panel=fig_w_per_panel,
			fig_h=fig_h,
			trim_inputs=trim_inputs,
			bg_thresh=bg_thresh,
			trim_pad=trim_pad,
			wspace=wspace,
			strict=strict,
			final_trim=final_trim,
			final_bg_thresh=final_bg_thresh,
			final_trim_pad=final_trim_pad,
		)
		out_paths.append(out_png)
		print(f'[OK] {out_png}')

	return out_paths


# =========================
# パラメータはここに直書き
# =========================

SETTING_DIRS = [
	Path('/workspace/proc/loki_hypo/mobara/loki_output_mobara_optuna'),
	Path('/workspace/proc/loki_hypo/mobara/loki_output_mobara_eqt'),
	Path('/workspace/proc/loki_hypo/mobara/loki_output_mobara_eqt_trainmiyagi'),
]

FILENAME = 'coherence_xy_overlay_trial0.png'
OUT_DIR = Path('coherence_compares')
# 図の詰め具合
FIG_W_PER_PANEL = 2.5
FIG_H = 3.2
WSPACE = 0.01

# 入力png側の白フチ削り
TRIM_INPUTS = True
BG_THRESH = 250
TRIM_PAD = 2

# 出力（比較png）をさらに最終トリム（これが効く）
FINAL_TRIM = True
FINAL_BG_THRESH = 250
FINAL_TRIM_PAD = 2

STRICT = True


if __name__ == '__main__':
	render_compare_all_events(
		SETTING_DIRS,
		filename=FILENAME,
		out_dir=OUT_DIR,
		fig_w_per_panel=FIG_W_PER_PANEL,
		fig_h=FIG_H,
		trim_inputs=TRIM_INPUTS,
		bg_thresh=BG_THRESH,
		trim_pad=TRIM_PAD,
		wspace=WSPACE,
		strict=STRICT,
		final_trim=FINAL_TRIM,
		final_bg_thresh=FINAL_BG_THRESH,
		final_trim_pad=FINAL_TRIM_PAD,
	)
