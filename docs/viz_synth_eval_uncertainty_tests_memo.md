# 技術メモ: `viz.hypo.synth_eval` 不確実性楕円プロット追加に対するテストの保証事項

## 対象パッチ
- `src/viz/hypo/synth_eval.py`
  - 既存 `save_true_pred_xyz_3view()` を **図生成**と**仕上げ**に分割
    - `_build_true_pred_xyz_3view_figure()`
    - `_finalize_true_pred_xyz_3view()`
  - 1σ投影楕円の描画機能を追加
    - `_ellipse_axis_aligned_halfwidth()`
    - `_ellipse_polyline()`
    - `_clip_ab()`
    - `save_true_pred_xyz_3view_with_uncertainty()`
- `src/qc/hypo/synth_eval.py`
  - QC出力に `xy_true_vs_hyp_uncertainty.png` を追加（列が揃わない場合はスキップ）

## テストファイル
- `tests/test_viz_hypo_synth_eval_uncertainty_plot.py`

> 目的は「描画の見た目」ではなく、**壊れやすい契約（入力整合・列要件・mask連動・楕円レイヤ追加・skip条件）**を最小限のコストで担保すること。

---

## 1. `save_true_pred_xyz_3view()` 回帰テスト

### テスト: `test_save_true_pred_xyz_3view_still_saves_png`
**保証していること**
- 既存API `save_true_pred_xyz_3view(true_xyz_m, pred_xyz_m, out_png, ...)` が分割リファクタ後も例外なく実行できる
- PNGファイルが実際に生成される（ファイル存在 + 0バイトでない）

**実装メモ**
- `viz.hypo.synth_eval.save_figure` を monkeypatch して `fig.savefig()` でPNGを出しつつ、`fig` を捕捉する（テスト内検証用）

**注意（warning）**
- `tight_layout` が空Axes（凡例用）を含む図で警告を出す場合がある
  → テストでは `pytest.warns(UserWarning, match="tight_layout")` で **想定内として明示**（ノイズ抑制）

---

## 2. 楕円ポリライン生成の入力制約

### テスト: `test_ellipse_polyline_min_points_and_shape`
**保証していること**
- `_ellipse_polyline(..., n_points)` は `n_points < 20` で `ValueError` を返す（過度に粗いポリラインを許可しない）
- 正常系では `(n_points, 2)` の配列を返し、全要素がfiniteである

---

## 3. `save_true_pred_xyz_3view_with_uncertainty()` 入力検証（fail-fast）

### テスト: `test_save_true_pred_xyz_3view_with_uncertainty_validates_inputs`
**保証していること**
- `true_xyz_m` と `pred_xyz_m` のイベント数と `df_eval` の行数が一致しない場合に `ValueError`
- `df_eval` に不確実性列（`ell_*` 9列）が揃わない場合に `KeyError`
- `sigma_scale_sec <= 0`（または非finite）で `ValueError`

**意図**
- QCパイプラインで「不整合を黙って描画」しない（早期に原因を顕在化）

---

## 4. 楕円レイヤ追加と凡例の挙動（OK/POOR分岐）

### テスト: `test_save_true_pred_xyz_3view_with_uncertainty_adds_ellipses_and_legend`
**入力設計**
- 2イベントを用意し、`ell_s1_km` のmaxを操作して
  - event0: ok（`ell_3d_max_km <= poor_thresh_km`）
  - event1: poor（`ell_3d_max_km > poor_thresh_km`）
- poor側は `clip_km` により半径がクリップされる経路も同時に通す（極端値で描画が破綻しないことの間接確認）

**保証していること**
- XY/XZ/YZ の3断面Axesそれぞれに **LineCollection が2つ**（ok用・poor用）追加される
  - つまり、分類ロジックとレイヤ追加が機能している
- 凡例（空Axes）に
  - `1σ ellipse`
  - `1σ ellipse (poor)`
  のラベルが含まれる（handles追加が機能している）
- 出力PNGが生成される

**注意（warning）**
- 図構成上 `tight_layout` 警告が出ることがあるため、テストでは明示的に受け止める

---

## 5. QC統合: 不確実性プロットのスキップ条件

### テスト: `test_run_qc_skips_uncertainty_when_missing_columns`
**保証していること**
- `eval_metrics.csv` に `ell_*` 9列が揃わない場合
  - `save_true_pred_xyz_3view_with_uncertainty` を呼ばずに **スキップ**する
- 既存の `xy_true_vs_hyp.png` は生成される（回帰防止）

**実装メモ**
- `save_hist / save_dxdy_scatter / save_true_pred_xyz_3view` は軽量化のためPNGをtouchするダミーに差し替え
- `save_true_pred_xyz_3view_with_uncertainty` は呼ばれたら即failするスタブにして、**未呼び出し**を検証

---

## 6. QC統合: mask連動と `df_plot` の整合

### テスト: `test_run_qc_passes_masked_df_to_uncertainty`
**入力設計**
- `x_m_hyp` に `NaN` を混ぜ、`mask = finite(true)&finite(pred)` により 1イベントが除外される状況を作る

**保証していること**
- 不確実性プロット呼び出し時に渡される
  - `true_xyz_m`, `pred_xyz_m`, `df_plot`
  の長さが一致する（mask適用が同期している）
- `df_plot` が `reset_index(drop=True)` 済みで、残った行の index が `[0]` になる（QC側の期待通り）
- `xy_true_vs_hyp_uncertainty.png` が生成される

---

## テスト範囲外（別途担保が必要な領域）
- 出力画像の見た目（楕円の回転方向や上下反転などの視覚的妥当性）
- `pad_x/pad_y/pad_z` による軸レンジ拡張が常に十分か（必要なら `x_range/y_range/z_range` の増分を直接検証）
- 大量イベント時の性能（LineCollectionの生成コスト）

---

## まとめ
本テスト群は、今回の追加機能に対して次を最低限保証する:
1. 既存3view描画APIの回帰なし（PNG出力）
2. 楕円描画の入力制約とfail-fast検証
3. 不確実性列が揃う場合のみ楕円レイヤを追加し、凡例に反映
4. QC側でのスキップ条件（列不足）と、mask適用後の `df_plot` 同期が正しい
