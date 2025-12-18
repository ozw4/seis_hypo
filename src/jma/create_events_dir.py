import json
from pathlib import Path

import pandas as pd


def create_event_input_dir(
	event_row: pd.Series,
	base_input_dir: str | Path,
	*,
	pre_sec: int,
	post_sec: int,
) -> Path:
	base_input_dir = Path(base_input_dir)
	event_id = int(event_row['event_id'])

	# JMAカタログの origin_time を JST のローカル時刻として解釈
	# （レポ内の他のコードと同じく pandas のパーサに統一）
	origin_time = pd.to_datetime(event_row['origin_time'])

	event_dir = base_input_dir / f'{event_id:06d}'
	event_dir.mkdir(parents=True, exist_ok=True)

	waveform_dir = event_dir / 'waveforms' / 'win32'
	waveform_dir.mkdir(parents=True, exist_ok=True)

	event_dict = {
		'event_id': event_id,
		'origin_time': origin_time.isoformat(),  # JST相当のローカル時刻として記録
		'latitude_deg': float(event_row['latitude_deg']),
		'longitude_deg': float(event_row['longitude_deg']),
		'depth_km': float(event_row['depth_km']),
		'mag1': float(event_row['mag1']) if not pd.isna(event_row['mag1']) else None,
		'window': {
			'pre_sec': pre_sec,
			'post_sec': post_sec,
		},
		'source': {
			'catalog_file': 'arrivetime_epicenters_1day.csv',
			'record_type': str(event_row.get('record_type', '')),
			'hypocenter_flag': str(event_row.get('hypocenter_flag', '')),
		},
	}

	with (event_dir / 'event.json').open('w', encoding='utf-8') as f:
		json.dump(event_dict, f, ensure_ascii=False, indent=2)

	return event_dir
