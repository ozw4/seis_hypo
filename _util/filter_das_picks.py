from pathlib import Path

import numpy as np
import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} に必要な列がありません: {missing}")


def filter_and_decimate_das_picks(
    df_events: pd.DataFrame,
    df_picks: pd.DataFrame,
    *,
    dt_sec: float,
    fiber_spacing_m: float = 1.0,
    channel_start: int = 200,
    win_half_samples: int = 500,
    residual_thresh_s: float = 0.05,
    spacing_m: float = 100.0,
) -> pd.DataFrame:
    """
    RANSAC で求めた P 波到達カーブ (t = a + s x) と DAS ピックを組み合わせて:

      1. 直線に整合的 (|残差| <= residual_thresh_s) なピックだけ残す
      2. offset を spacing_m ごとにビニングし、各ビンで w_conf 最大の 1 チャネルに間引く

    Parameters
    ----------
    df_events : events_summary_*.csv を読んだ DataFrame
        必須列: ['event_id', 'slowness_s_per_m', 'intercept_s']
    df_picks : das_picks_*.csv を読んだ DataFrame
        必須列: ['event_id', 'peak_index', 'channel', 'sample_index', 'w_conf']
    dt_sec : float
        サンプリング間隔 (例: 0.01)
    fiber_spacing_m : float, default 1.0
        1 チャネルあたりの DAS 空間間隔 [m]
    channel_start : int, default 200
        use_ch_range.start と同じ値を入れる
    win_half_samples : int, default 500
        検出窓の左右ハーフ幅 (コード中で idx-500:idx+500 なので 500)
    residual_thresh_s : float, default 0.05
        直線からの許容残差 [s]
    spacing_m : float, default 100.0
        この間隔ごとにチャネルを 1 本に間引く [m]

    Returns
    -------
    df_out : DataFrame
        フィルタ + 間引き後の DAS ピック。
        列は元の df_picks に加えて:
          - offset_m
          - t_obs_sec
          - t_pred_sec
          - resid_sec
          - fit_bin (spacing_m ごとのビン番号)
    """
    validate_columns(
        df_events,
        ["event_id", "slowness_s_per_m", "intercept_s"],
        "df_events",
    )
    validate_columns(
        df_picks,
        ["event_id", "peak_index", "channel", "sample_index"],
        "df_picks",
    )

    if "w_conf" not in df_picks.columns:
        df_picks = df_picks.copy()
        df_picks["w_conf"] = 1.0

    # event_id でマージして RANSAC パラメータ a,s をくっつける
    ev_cols = ["event_id", "slowness_s_per_m", "intercept_s"]
    df = df_picks.merge(df_events[ev_cols], on="event_id", how="inner")

    # offset [m] を復元: channel_start チャネルを offset=0 とする
    df["offset_m"] = (df["channel"].astype(int) - int(channel_start)) * float(
        fiber_spacing_m
    )

    # 窓内ローカル時間 t_obs_sec を復元
    #   窓開始サンプル = peak_index - win_half_samples
    #   t_obs_sec = (sample_index - 窓開始) * dt
    peak_idx = df["peak_index"].astype(int)
    sample_idx = df["sample_index"].astype(int)
    win_start = peak_idx - int(win_half_samples)
    df["t_obs_sec"] = (sample_idx - win_start) * float(dt_sec)

    # RANSAC 直線からの予測時間と残差
    s = df["slowness_s_per_m"].astype(float)
    a = df["intercept_s"].astype(float)
    x = df["offset_m"].astype(float)

    df["t_pred_sec"] = a + s * x
    df["resid_sec"] = df["t_obs_sec"] - df["t_pred_sec"]

    # 1. 残差でフィルタリング
    m_resid = df["resid_sec"].abs() <= float(residual_thresh_s)
    df_filt = df[m_resid].copy()

    if df_filt.empty:
        return df_filt

    # 2. spacing_m ごとにビニングして w_conf 最大の 1 本を採用
    spacing_m = float(spacing_m)
    df_filt["fit_bin"] = np.round(df_filt["offset_m"] / spacing_m).astype(int)

    # イベントごと + ビンごとに w_conf 最大の行 index を取る
    grp = df_filt.groupby(["event_id", "fit_bin"], sort=False)["w_conf"]
    idx_best = grp.idxmax()
    df_out = df_filt.loc[idx_best].sort_values(
        ["event_id", "offset_m"], ignore_index=True
    )

    return df_out


if __name__ == "__main__":
    # 例: さっき保存した CSV から読む場合
    events_csv = Path("events_summary_20200215_20200301.csv")
    picks_csv = Path("das_picks_20200215_20200301.csv")

    df_events = pd.read_csv(events_csv)
    df_picks = pd.read_csv(picks_csv)

    df_filtered = filter_and_decimate_das_picks(
        df_events,
        df_picks,
        dt_sec=0.01,
        fiber_spacing_m=1.0,
        channel_start=200,        # use_ch_range.start と合わせる
        win_half_samples=500,     # idx-500:idx+500 に合わせる
        residual_thresh_s=0.05,   # ±0.05s 以内を「整合的」とみなす
        spacing_m=100.0,          # 100m ごとに 1 チャネルに間引き
    )

    df_filtered.to_csv(
        "das_picks_filtered_decimated.csv",
        index=False,
    )
    print(f"filtered+decimated DAS picks: {len(df_filtered)} rows")
