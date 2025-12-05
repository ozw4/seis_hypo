# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_dataframe(csv_path: str) -> pd.DataFrame:
	path = Path(csv_path)
	if not path.is_file():
		raise FileNotFoundError(f'CSV が見つかりません: {path}')
	df = pd.read_csv(path)
	required_columns = [
		'event_id',
		'station_code',
		'phase_name_1',
		'pick_flag_1',
		'max_amplitude_ns',
	]
	missing = [c for c in required_columns if c not in df.columns]
	if missing:
		raise ValueError(f'必要な列がありません: {missing}')
	return df


def plot_event_station_hist(df: pd.DataFrame) -> None:
	event_station_counts = (
		df.groupby('event_id')['station_code'].nunique().sort_values()
	)

	plt.figure(figsize=(8, 6))
	plt.hist(event_station_counts, bins=30, log=True)
	plt.xlabel('Number of stations per event')
	plt.ylabel('Number of events')
	plt.title('Histogram of number of stations per event')
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig('img/arrivetime/event_station_count_hist.png')
	plt.show()


def plot_phase_counts(df: pd.DataFrame) -> None:
	phase_counts = df['phase_name_1'].value_counts().sort_index()

	plt.figure(figsize=(8, 6))
	plt.bar(phase_counts.index.astype(str), phase_counts.values)
	plt.xlabel('phase_name_1')
	plt.ylabel('Occurrences')
	plt.title('phase_name_1 histogram')
	plt.grid(axis='y', alpha=0.3)
	plt.tight_layout()
	plt.savefig('img/arrivetime/phase_name_1_hist.png')
	plt.show()


def plot_pick_flag_counts(df: pd.DataFrame) -> None:
	flag_counts = df['pick_flag_1'].value_counts().sort_index()

	plt.figure(figsize=(6, 5))
	plt.bar(flag_counts.index.astype(str), flag_counts.values)
	plt.xlabel('pick_flag_1')
	plt.ylabel('Occurrences')
	plt.yscale('log')
	plt.title('pick_flag_1 histogram')
	plt.grid(axis='y', alpha=0.3)
	plt.tight_layout()
	plt.savefig('img/arrivetime/pick_flag_1_hist.png')
	plt.show()


def plot_top_station_counts(df: pd.DataFrame, top_n: int = 50) -> None:
	station_counts = df['station_code'].value_counts().head(top_n)

	plt.figure(figsize=(10, 6))
	plt.bar(station_counts.index.astype(str), station_counts.values)
	plt.xlabel('station_code')
	plt.ylabel('Number of picks')
	plt.title(f'Top {top_n} stations by number of picks')
	plt.xticks(rotation=60, ha='right')
	plt.grid(axis='y', alpha=0.3)
	plt.tight_layout()
	plt.savefig(f'img/arrivetime/top_{top_n}_stations_by_picks.png')
	plt.show()


def plot_max_amplitude_hist(df: pd.DataFrame) -> None:
	amp = df['max_amplitude_ns'].dropna()
	if amp.empty:
		raise ValueError('max_amplitude_ns に有効な値がありません')

	plt.figure(figsize=(8, 6))
	plt.hist(amp, bins=50, log=True)
	plt.xlabel('max_amplitude_ns')
	plt.ylabel('number of arrivetime')
	plt.title('max_amplitude_ns histogram')
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig('img/arrivetimemax_amplitude_ns_hist.png')
	plt.show()


def main() -> None:
	csv_path = '/workspace/data/arrivetime/arrivetime_measurements.csv'
	df = load_dataframe(csv_path)

	plot_event_station_hist(df)
	plot_phase_counts(df)
	plot_pick_flag_counts(df)
	plot_top_station_counts(df, top_n=50)
	plot_max_amplitude_hist(df)


if __name__ == '__main__':
	main()
