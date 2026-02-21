from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class JmaPrepareConfig:
	win_event_dir: Path


@dataclass(frozen=True)
class JmaMissingContinuousConfig(JmaPrepareConfig):
	meas_csv: Path
	epi_csv: Path
	pres_csv: Path
	mapping_report_csv: Path
	near0_suggest_csv: Path
	out_missing_csv: Path
	skip_if_no_active_ch: bool = True
	skip_if_done: bool = True
	date_min: str | None = None
	date_max: str | None = None


@dataclass(frozen=True)
class JmaMissingContinuousWaveformConfig(JmaPrepareConfig):
	target_event_dir_names: list[str] = field(default_factory=list)
	out_subdir: str = 'continuous'
	threads: int = 8
	cleanup: bool = True
	skip_if_exists: bool = True
	date_min: str | None = None
	date_max: str | None = None
	skip_if_done: bool = True
	run_tag: str = 'v1'
	max_retry_request_none: int = 5
	threads_ladder: list[int] = field(default_factory=lambda: [8, 4, 2, 1])


@dataclass(frozen=True)
class JmaStep1RescueDownloadConfig(JmaPrepareConfig):
	epi_csv: Path
	tmp_download_dir: Path
	out_rescue_targets_csv: Path
	out_orphan_dirs_csv: Path
	out_rescue_run_csv: Path
	date_min: str | None = None
	date_max: str | None = None
	min_mag: float | None = None
	max_mag: float | None = None
	request_window_min: int = 1
	max_retry_get_event_waveform: int = 1
	retry_sleep_sec: float = 2.0

	skip_if_already_ok: bool = True
	download_run: bool = True
