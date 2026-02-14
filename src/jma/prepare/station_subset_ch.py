from __future__ import annotations

from pathlib import Path

from jma.prepare.active_channel import write_active_ch_file
from jma.station_reader import read_hinet_channel_table


def read_station_list_txt(path: str | Path) -> list[str]:
	p = Path(path)
	if not p.is_file():
		raise FileNotFoundError(p)

	out: list[str] = []
	seen: set[str] = set()

	for raw in p.read_text(encoding='utf-8').splitlines():
		s = raw.strip()
		if not s or s.startswith('#'):
			continue
		s = s.upper()
		if s in seen:
			continue
		seen.add(s)
		out.append(s)

	if not out:
		raise ValueError(f'empty station list: {p}')

	return out


def _keep_ch_hex_from_station_list(ch_path: Path, keep_stations: list[str]) -> set[str]:
	df = read_hinet_channel_table(ch_path)

	df_sta = df['station'].astype(str).str.strip().str.upper()
	keep_set = set(str(x).strip().upper() for x in keep_stations if str(x).strip())
	if not keep_set:
		raise ValueError('keep_stations is empty after normalization')

	present = set(df_sta.unique().tolist())
	missing = sorted(keep_set - present)
	if missing:
		raise ValueError(
			f'some keep stations not found in {ch_path.name}: {missing[:30]}'
		)

	keep_mask = df_sta.isin(keep_set)
	keep_hex = set(
		df.loc[keep_mask, 'ch_hex'].astype(str).str.strip().str.upper().tolist()
	)
	if not keep_hex:
		raise ValueError(f'no channels selected from {ch_path.name}')

	return keep_hex


def write_station_subset_ch(
	*,
	src_ch_path: str | Path,
	out_ch_path: str | Path,
	keep_stations: list[str],
) -> None:
	src_ch_path = Path(src_ch_path)
	out_ch_path = Path(out_ch_path)

	if not src_ch_path.is_file():
		raise FileNotFoundError(src_ch_path)

	out_ch_path.parent.mkdir(parents=True, exist_ok=True)

	keep_hex = _keep_ch_hex_from_station_list(src_ch_path, keep_stations)
	write_active_ch_file(src_ch_path, keep_ch_hex=keep_hex, out_path=out_ch_path)


def write_station_subset_ch_dir(
	*,
	in_dir: str | Path,
	out_dir: str | Path,
	keep_stations: list[str],
	pattern: str = '*.ch',
	skip_if_exists: bool = True,
) -> None:
	in_dir = Path(in_dir)
	out_dir = Path(out_dir)

	if not in_dir.is_dir():
		raise FileNotFoundError(in_dir)

	out_dir.mkdir(parents=True, exist_ok=True)

	ch_files = sorted(in_dir.glob(pattern))
	if not ch_files:
		raise FileNotFoundError(f'no .ch files under: {in_dir} (pattern={pattern})')

	for ch_path in ch_files:
		out_path = out_dir / ch_path.name
		if skip_if_exists and out_path.is_file():
			continue

		write_station_subset_ch(
			src_ch_path=ch_path,
			out_ch_path=out_path,
			keep_stations=keep_stations,
		)

	print(f'[subset_ch] in={in_dir} -> out={out_dir} files={len(ch_files)}')
