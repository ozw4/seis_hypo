# HypoInverse 震源不確実性（誤差楕円）可視化　技術メモ（作業記録）

## 目的
- HypoInverse（v1.4）出力に含まれる震源決定不確実性の可視化
- 合成走時データに対する HypoInverse 震源決定結果の評価図に、投影誤差楕円（1σ）重畳
- 既存パイプライン（`run_synth_eval → evaluate → run_qc → viz`）の維持、成果物追加のみ

---

## 前提・設計方針
- HypoInverse の誤差スケールに関する方針
  - `RDERR`（ERR コマンド）による参照ピック誤差スケール
  - `ERCOF`（ERC コマンド）による RMS 由来の付加スケール
- 本実装での固定条件
  - `ERR 1.0`（RDERR=1.0 秒仮定、基準スケール）
  - `ERC 0`（ERCOF=0、RMS 起因の付加スケール無効化）
  - 可視化側で `sigma_scale_sec` により秒相当スケール変更（線形スケール）

---

## 成果物（アウトプット）
- 3-view 図（True vs HypoInverse）に投影 1σ 楕円重畳した図の生成
  - `runs/<outputs_dir>/xy_true_vs_hyp_uncertainty.png`
- 不確実性描画設定メタの保存
  - `runs/<outputs_dir>/uncertainty_plot_meta.txt`
- 実行時設定のスナップショット保存
  - `runs/<outputs_dir>/config_used.yaml`

---

## 変更概要（方針A〜E）

### 方針A：HypoInverse cmd への ERR/ERC 強制注入（基準スケール固定）
- 目的
  - テンプレート cmd の内容に依存しない `ERR 1.0` / `ERC 0` の保証
  - `LOC` より前で有効となる配置の保証
- 追加ファイル
  - `src/hypo/hypoinverse_cmd.py`
    - `cmd_token(line)`：コマンド判定（コメント・空行除外）
    - `force_err_erc(lines)`：`ERR 1.0` / `ERC 0` の置換・不足時挿入・LOC 前重複排除・fail-fast 検証
- 既存改修
  - `src/hypo/synth_eval/hypoinverse_runner.py`
    - CRH 経路（`patch_cmd_template`）末尾で `force_err_erc` 適用
    - CRE 経路（`patch_cmd_template_for_cre`）末尾で `force_err_erc` 適用
    - `cmd_token` 利用による頑健化（コメント除外・先頭トークン判定）

### 方針B：`.prt` から ERROR ELLIPSE / EIGENVALUES パース（イベント単位）
- 目的
  - HypoInverse `.prt` に出力される誤差楕円体情報の機械可読化
  - `eval_metrics.csv` への誤差楕円体列追加
- 既存改修
  - `src/hypo/hypoinverse_prt.py`
    - 1パス走査方式への移行（イベント対応付けの堅牢化）
      - pending バッファ：`pending_error_ellipse` / `pending_eigenvalues`
      - summary 行出現時の record 確定、pending の付与、リセット
      - `NSTA NPHS` ブロック値の直近 record への更新
    - 追加パーサ
      - `parse_error_ellipse_line(line)`：`<SERR AZ DIP>` ×3 抽出、float/int 変換、欠損時 fail-fast
      - `parse_eigenvalues_block(lines, i)`：`EIGENVALUES` + 次行 `(a b c d)` の4値抽出（任意、無い場合 None）
    - 必須キー検証
      - NSTA 系必須キーの欠損検出
      - 誤差楕円体必須キーの欠損検出
- 共通カラム集合の導入
  - `ELLIPSE_COLS` 参照への統一（後述）

### 方針C：誤差楕円体 → 共分散 → 断面投影楕円（数式ユーティリティ）
- 目的
  - ERROR ELLIPSE（主軸長・方位・傾斜）から 3D 共分散行列の復元
  - XY / XZ / YZ 断面への周辺化（部分行列抽出）と 2D 楕円パラメータ化
- 追加ファイル
  - `src/hypo/uncertainty_ellipsoid.py`
    - `ELLIPSE_COLS`：誤差楕円体入力列の共通定義
    - `unit_vector_from_az_dip(az_deg, dip_deg)`
      - 座標系固定：`X=East, Y=North, Z=Depth (down +)`
      - AZ 定義固定：北基準・東回り（0=N, 90=E）
      - DIP 定義固定：水平から下向き（0=水平, 90=鉛直下）
    - `error_ellipse_to_cov_xyz_km2(...)`
      - Σ = U diag(s^2) U^T の構成
      - 直交性チェック（fail-fast、補正なし）
      - 対称化、微小負固有値クランプ（FP誤差対策）
      - `sigma_scale_sec` による外部スケール（Σ *= σ^2）
    - `cov_xyz_to_cov_2d_km2(cov_xyz, plane)`
      - `xy`: [X,Y]
      - `xz`: [X,Z]
      - `yz`: [Z,Y]（描画仕様に合わせた固定順：横=Depth, 縦=Y）
    - `cov_2d_to_ellipse_params(cov_2d)`
      - 2×2 共分散の固有分解
      - `a_km, b_km`（1σ半径）、`theta_rad`（+x から反時計回り）
    - `projected_ellipses_from_record(rec)`
      - `eval_metrics` 行（Mapping/Series）から XY/XZ/YZ の (a,b,theta) と `ell_3d_max_km` 生成

### 方針D：3-view 図への楕円重畳（LineCollection）
- 目的
  - 既存 3-view 図（True/Pred/Station）への誤差楕円（線のみ）の重畳
  - 多イベント重畳の高速描画
  - 拘束不足（poor）色分け・最大サイズクリップ・見切れ防止
- 既存改修
  - `src/viz/hypo/synth_eval.py`
    - 既存 `save_true_pred_xyz_3view` の分割
      - `_build_true_pred_xyz_3view_figure(...)`：図・Axes・整形済みデータ・凡例ハンドル群の生成
      - `_finalize_true_pred_xyz_3view(...)`：軸同期・タイトル・凡例・保存処理
      - `save_true_pred_xyz_3view(...)`：従来 API の維持（内部で build/finalize 呼び出し）
    - 追加 `save_true_pred_xyz_3view_with_uncertainty(...)`
      - `df_eval` の `ELLIPSE_COLS` を検証
      - `projected_ellipses_from_record` により各イベント楕円計算
      - 中心点：推定震源（Pred）
      - 断面定義
        - XY：`(x, y)`
        - XZ：`(x, z)`
        - YZ：`(z, y)`（横=Depth, 縦=Y）
      - クリップ：`max(a,b) > clip_km` で等比縮小
      - 拘束不足判定：`ell_3d_max_km > poor_thresh_km`
      - 軸範囲パディング：回転楕円の軸平行半幅推定値による pad_x/pad_y/pad_z 拡張
      - LineCollection による一括描画（ok / poor 別コレクション）
      - 凡例追加：`1σ ellipse` / `1σ ellipse (poor)`
- QC 統合（初版）
  - `src/qc/hypo/synth_eval.py`
    - `xy_true_vs_hyp_uncertainty.png` 生成の追加
    - `df_plot` の mask（finite true/pred）の適用

### 方針E：実行導線整備・再現性・設定化・fail-fast
- 目的
  - `proc/.../run_synth_eval.py` 実行のみで成果物一式生成
  - 必須成果物欠落の早期検出
  - 設定・生成物・描画条件の記録
- 共通カラム定義の適用
  - `src/hypo/uncertainty_ellipsoid.py`：`ELLIPSE_COLS` 追加
  - `src/hypo/hypoinverse_prt.py`：必須楕円キー検証を `ELLIPSE_COLS` に統一
  - `src/viz/hypo/synth_eval.py`：楕円列存在チェックを `ELLIPSE_COLS` に統一
- pipeline 側の強化
  - `src/hypo/synth_eval/pipeline.py`
    - `config_used.yaml` の保存（`_write_config_snapshot`、`copy2`）
    - 主要生成物の `[OK] wrote:` ログ追加
    - 必須成果物チェック（fail-fast）：`.prt` / `.sum` / `.arc`
    - `.prt` パース結果の楕円列チェック（fail-fast）：`ELLIPSE_COLS` 欠落時例外
- QC 側の設定化・ログ・メタ保存
  - `src/qc/hypo/synth_eval.py`
    - `uncertainty_plot` 設定追加（dataclass）
    - `[INFO] uncertainty_plot: ...` ログ出力
    - `uncertainty_plot_meta.txt` 保存（ERR/ERC 条件含む）
    - 既存 QC 図出力の `[OK] wrote:` ログ追加
- テスト更新
  - `tests/test_hypo_synth_eval_pipeline_orchestration_cre_crh.py`
    - `eval_df` スタブに `ell_*` 追加
    - HypoInverse 実行スタブで `.sum` / `hypoinverse_run_out.arc` 生成追加

---

## 設定（YAML）
- QC 用 `uncertainty_plot` 設定ブロック
```yaml
uncertainty_plot:
  enabled: true
  sigma_scale_sec: 1.0
  poor_thresh_km: 5.0
  clip_km: 10.0
  n_ellipse_points: 100
  ellipse_lw: 0.8
  ellipse_alpha: 0.85
```
- 意味
  - `sigma_scale_sec`：1σ 秒相当スケール（楕円半径に線形適用）
  - `poor_thresh_km`：拘束不足判定（3D最大主軸長ベース）
  - `clip_km`：描画最大半径の上限（等比縮小）
  - `n_ellipse_points`：楕円ポリライン分割数
  - `ellipse_lw` / `ellipse_alpha`：描画スタイル

---

## 主要ファイル一覧
- cmd パッチ
  - `src/hypo/hypoinverse_cmd.py`
  - `src/hypo/synth_eval/hypoinverse_runner.py`
- `.prt` パース
  - `src/hypo/hypoinverse_prt.py`
- 楕円体・共分散・投影ユーティリティ
  - `src/hypo/uncertainty_ellipsoid.py`
- 可視化
  - `src/viz/hypo/synth_eval.py`
- QC オーケストレーション
  - `src/qc/hypo/synth_eval.py`
- パイプライン
  - `src/hypo/synth_eval/pipeline.py`

---

## 生成物（run_dir 配下）
- 実行設定
  - `config_used.yaml`
- 入力（合成）
  - `station_synth.csv`
  - `stations_synth.sta`
  - `hypoinverse_input.arc`
  - `synth.cmd`
  - `P.crh`, `S.crh` または `P.cre`, `S.cre`（条件により）
- HypoInverse 出力
  - `hypoinverse_run.prt`
  - `hypoinverse_run.sum`
  - `hypoinverse_run_out.arc`
- 評価・QC
  - `eval_metrics.csv`（`ELLIPSE_COLS` 含む）
  - `eval_stats.csv`
  - `xy_true_vs_hyp.png`
  - `xy_true_vs_hyp_uncertainty.png`
  - `uncertainty_plot_meta.txt`
  - 既存 QC 図（hist/scatter 等）

---

## 実行確認観点
- `synth.cmd` 内の `ERR 1.0` / `ERC 0` の存在、`LOC` より前の配置
- `.prt` 内の `ERROR ELLIPSE` 行のイベント数一致
- `eval_metrics.csv` に `ELLIPSE_COLS` が欠損なく存在
- `xy_true_vs_hyp_uncertainty.png` の生成、楕円の中心が Pred であること
- YZ 断面の軸定義（横=Depth, 縦=Y）一致
- `uncertainty_plot_meta.txt` と `[INFO]` ログによる描画条件追跡性

---

## 備考
- ERR=1.0 秒仮定時の楕円肥大化：仕様通り（秒相当スケールの大きさ）
- 現実的スケール表示：`sigma_scale_sec` を 0.01〜0.1 程度へ変更する運用
- 拘束不足判定の現仕様：3D 最大主軸長（`ell_3d_max_km`）閾値比較による分類
