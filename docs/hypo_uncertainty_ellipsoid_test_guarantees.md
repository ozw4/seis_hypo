# 技術メモ: `hypo.uncertainty_ellipsoid` テストの保証事項

## 対象

- 実装: `src/hypo/uncertainty_ellipsoid.py`
- テスト: `tests/test_hypo_uncertainty_ellipsoid.py`

本モジュールは HypoInverse の **ERROR ELLIPSE**（主軸長 SERR と方位 AZ・傾斜 DIP）から、局所座標系 **(X=East, Y=North, Z=Depth[down+])** における 3D 共分散 **Σ_xyz (km^2)** を構築し、断面（XY/XZ/YZ）へ周辺化（部分行列抽出）した 2D 共分散から **投影 1σ 楕円パラメータ (a_km, b_km, theta_rad)** を計算する。

---

## このテスト群が担保すること（サマリ）

- **座標定義の固定**: AZ（北基準・東回り）、DIP（水平から下向き）→ (E,N,Down) が仕様どおり。
- **投影断面の順序固定**: `yz` は **(Z, Y)** の順序（x=Depth, y=North に合わせる）。
- **スケール則の固定**:
  - 共分散は **σ^2 スケール**（`sigma_scale_sec` の二乗）
  - 半径は **σ スケール**（`a,b,ell_3d_max` が線形）
- **fail-fast**: 非有限値、負の SERR、非直交主軸、非PSD共分散（許容 `tol` 超）を黙殺しない。
- **下流統合の前提**: Mapping/Series からのラッパ呼び出しが直接呼び出しと同値。

---

## テストケース別の保証事項

### 1) `unit_vector_from_az_dip()`（座標系の核）

**該当テスト**
- `test_unit_vector_from_az_dip_cardinals_and_norm`
- `test_unit_vector_from_az_dip_rejects_non_finite`

**保証していること**
- 代表角で方向が一致：
  - `(az=0, dip=0)` → `[0,1,0]`（North）
  - `(az=90, dip=0)` → `[1,0,0]`（East）
  - `(dip=90)` → `[0,0,1]`（Down、az非依存）
- `||u|| = 1`（正規化が保証）
- `NaN/Inf` 入力は `ValueError`（fail-fast）

**補足（今回の warning 対応）**
- `Inf` 入力で `sin/cos` が先に走って numpy warning が出ないよう、三角関数計算前に有限性チェックして `ValueError` を投げる修正が入った前提でテストが成立する。

---

### 2) `error_ellipse_to_cov_xyz_km2()`（3D共分散の正しさ）

**該当テスト**
- `test_error_ellipse_to_cov_xyz_km2_diag_basis_and_sigma_scale`
- `test_error_ellipse_to_cov_xyz_km2_rejects_invalid_sigma`
- `test_error_ellipse_to_cov_xyz_km2_rejects_invalid_serr`
- `test_error_ellipse_to_cov_xyz_km2_rejects_non_orthogonal_axes`

**保証していること**
- 主軸が E/N/Down に一致する場合、`Σ_xyz = diag([s1^2,s2^2,s3^2])`（基本式の健全性）
- `sigma_scale_sec` が **共分散に対して二乗で効く**（σ=2 → Σが4倍）
- 入力が不正なら即 `ValueError`：
  - `sigma_scale_sec<=0` / `NaN/Inf`
  - `SERR<0` / `NaN/Inf`
- 主軸が非直交なら即 `ValueError`（補正しない）

---

### 3) `cov_xyz_to_cov_2d_km2()`（断面抽出の順序）

**該当テスト**
- `test_cov_xyz_to_cov_2d_km2_plane_extraction_order_is_fixed`
- `test_cov_xyz_to_cov_2d_km2_rejects_bad_shape_and_plane`

**保証していること**
- 平面抽出のインデックスが仕様通り：
  - `xy` → (X,Y) = (E,N)
  - `xz` → (X,Z) = (E,Down)
  - `yz` → (Z,Y) = (Down,North)（プロット軸に合わせた固定順）
- 3x3 以外、未知 plane は `ValueError`

---

### 4) `cov_2d_to_ellipse_params()`（2D楕円の半径と角度）

**該当テスト**
- `test_cov_2d_to_ellipse_params_diagonal_cases`
- `test_cov_2d_to_ellipse_params_rotated_45deg`
- `test_cov_2d_to_ellipse_params_clamps_small_negative_eigs`
- `test_cov_2d_to_ellipse_params_rejects_large_negative_eigs_and_bad_shape`

**保証していること**
- 対角共分散で `a,b = sqrt(固有値)`、`theta` が主軸方向になる
- 回転ケース（45°）でも `theta` が **mod π** で整合（主軸方向の不定性を許容しつつ方向は保証）
- 小さい負固有値（FP誤差相当）は `tol` 内なら 0 にクランプして継続
- `tol` を超える負固有値は `ValueError`（非PSDを通さない）
- 2x2 以外は `ValueError`

---

### 5) `projected_ellipses_from_error_ellipse()`（一気通貫の整合）

**該当テスト**
- `test_projected_ellipses_from_error_ellipse_axis_aligned_consistency`
- `test_projected_ellipses_from_error_ellipse_sigma_scales_radii_linearly`

**保証していること**
- 軸一致ケースで、投影楕円の半径 (a,b) が各断面で期待通り：
  - XY → `{s1,s2}`
  - XZ → `{s1,s3}`
  - YZ → `{s3,s2}`（Z,Y順）
- `ell_3d_max_km = max(s1,s2,s3) * sigma_scale_sec`
- 半径が `sigma_scale_sec` に **線形**に比例（σ=2 → a,bが2倍）

---

### 6) `projected_ellipses_from_record()`（ラッパ同値性）

**該当テスト**
- `test_projected_ellipses_from_record_matches_direct_call`

**保証していること**
- `rec`（dict/Series互換 Mapping）から `ell_*` 9キーを読む経路が、直接 `projected_ellipses_from_error_ellipse()` を呼ぶ結果と完全一致
- 下流（`eval_metrics.csv` や DataFrame 行）からの使用が安全

---

## このテストの範囲外（別で担保すべきもの）

- 実PRT由来の軸が常に直交性条件を満たすか（入力品質の問題）
- クリッピング・色分け・Z反転など **描画仕様**の正しさ（可視化側E2Eで担保）
- PRTパースの正しさ（`hypoinverse_prt` 側テストで担保済み）
