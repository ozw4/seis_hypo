from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from HinetPy import Client

from jma.download import _supports_station_selection, download_win_for_stations


@dataclass(frozen=True)
class DownloadRetryAttempt:
	threads: int
	try_idx: int
	ladders_used: int
	ladders_total: int
	error: Exception
	message: str
	is_none_retry: bool


@dataclass(frozen=True)
class DownloadRetryResult:
	cnt_path: Path | None
	ch_path: Path | None
	select_used: bool
	threads_used: int
	try_idx: int
	message: str
	error: Exception | None

	@property
	def success(self) -> bool:
		return self.cnt_path is not None and self.ch_path is not None


def download_with_retry(
	client: Client,
	stations: str | Sequence[str],
	when,
	*,
	threads_ladder: Sequence[int],
	max_retry_request_none: int,
	network_code: str = '0101',
	span_min: int = 1,
	outdir: str | Path = '.',
	cleanup: bool = True,
	clear_selection: bool = False,
	skip_if_exists: bool = True,
	use_select: bool | None = None,
	data_name: str | None = None,
	ctable_name: str | None = None,
	retry_on_exception: bool = False,
	raise_on_failure: bool = False,
	none_error_prefix: str = 'Fail to request WIN32 (returned None).',
	on_retry: Callable[[DownloadRetryAttempt], None] | None = None,
) -> DownloadRetryResult:
	ladder = [int(x) for x in threads_ladder]
	if not ladder:
		raise ValueError('threads_ladder must be non-empty')
	if any(x <= 0 for x in ladder):
		raise ValueError(f'invalid threads_ladder: {threads_ladder}')
	if int(max_retry_request_none) <= 0:
		raise ValueError('max_retry_request_none must be >= 1')

	select_guess = (
		_supports_station_selection(network_code)
		if use_select is None
		else bool(use_select)
	)

	last_msg = ''
	last_err: Exception | None = None
	last_threads = 0
	last_try_idx = 0

	for ladder_idx, th in enumerate(ladder, 1):
		for try_idx in range(1, int(max_retry_request_none) + 1):
			last_threads = int(th)
			last_try_idx = int(try_idx)
			try:
				cnt_out, ch_out, select_used = download_win_for_stations(
					client,
					stations=stations,
					when=when,
					network_code=network_code,
					span_min=span_min,
					outdir=outdir,
					threads=int(th),
					cleanup=cleanup,
					clear_selection=clear_selection,
					skip_if_exists=skip_if_exists,
					use_select=use_select,
					data_name=data_name,
					ctable_name=ctable_name,
				)
				return DownloadRetryResult(
					cnt_path=Path(cnt_out),
					ch_path=Path(ch_out),
					select_used=bool(select_used),
					threads_used=last_threads,
					try_idx=last_try_idx,
					message='',
					error=None,
				)
			except ValueError as e:
				msg = str(e)
				last_err = e
				last_msg = msg
				is_none_retry = msg.startswith(none_error_prefix)
				if is_none_retry:
					if on_retry is not None:
						on_retry(
							DownloadRetryAttempt(
								threads=last_threads,
								try_idx=last_try_idx,
								ladders_used=ladder_idx,
								ladders_total=len(ladder),
								error=e,
								message=msg,
								is_none_retry=True,
							)
						)
					continue
				if retry_on_exception:
					if on_retry is not None:
						on_retry(
							DownloadRetryAttempt(
								threads=last_threads,
								try_idx=last_try_idx,
								ladders_used=ladder_idx,
								ladders_total=len(ladder),
								error=e,
								message=msg,
								is_none_retry=False,
							)
						)
					break
				raise
			except Exception as e:
				last_err = e
				last_msg = str(e)
				if retry_on_exception:
					if on_retry is not None:
						on_retry(
							DownloadRetryAttempt(
								threads=last_threads,
								try_idx=last_try_idx,
								ladders_used=ladder_idx,
								ladders_total=len(ladder),
								error=e,
								message=last_msg,
								is_none_retry=False,
							)
						)
					break
				raise

	result = DownloadRetryResult(
		cnt_path=None,
		ch_path=None,
		select_used=bool(select_guess),
		threads_used=last_threads,
		try_idx=last_try_idx,
		message=last_msg,
		error=last_err,
	)
	if raise_on_failure:
		if last_err is not None:
			raise last_err
		raise RuntimeError('download failed (no outputs)')
	return result
