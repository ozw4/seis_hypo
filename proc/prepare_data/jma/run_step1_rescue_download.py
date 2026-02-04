# %%
# proc/prepare_data/jma/run_step1_rescue_download.py
from __future__ import annotations

from pathlib import Path

from common.load_config import load_config
from jma.prepare.config import JmaStep1RescueDownloadConfig
from jma.prepare.step1_rescue import run_step1_rescue_download

# =========================
# 設定（YAML から読み込む）
# =========================

YAML_PATH = Path(__file__).resolve().parent / 'config' / 'step1_rescue_download.yaml'
PRESET = 'sample'


# =========================
# 実装
# =========================


def main() -> None:
	cfg = load_config(JmaStep1RescueDownloadConfig, YAML_PATH, PRESET)
	run_step1_rescue_download(cfg)


if __name__ == '__main__':
	main()

# %%
