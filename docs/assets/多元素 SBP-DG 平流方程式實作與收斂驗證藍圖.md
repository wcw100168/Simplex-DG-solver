# 多元素 SBP-DG 平流方程式實作與收斂驗證藍圖 (基於 Chen & Shu, 2020)

**Status:** 🚧 **PLANNING PHASE** **Target:** 將現有的標準 Nodal DG（依賴投影密集逆質量矩陣與 Split-Form）重構為 **SBP-DG (Summation-by-Parts DG)** 架構。導入對角質量矩陣、通量差分 (Flux Differencing) 以及熵穩定 SATs (Simultaneous Approximation Terms) 邊界交換。
**Reference:** *Chen, T., & Shu, C.-W. (2020). Review of Entropy Stable Discontinuous Galerkin Methods...*

---

## 🎯 核心設計理念與數學轉換 (Core Paradigm Shift)

為了嚴格遵循 Chen & Shu (2020) 的 SBP-DG 哲學，必須對原 `Multi_Element_Advection_Convergence.ipynb` 進行三大核心機制的徹底轉換：

1. **從「密集投影」回歸「對角逆質量矩陣」**：
   - **棄用**：`M_inv_projected = V_nodal @ np.linalg.inv(M_modal) @ V_nodal.T`。
   - **採用**：純對角的積分權重反轉 `W_inv = 1.0 / weights_2d`（即論文中的 $(M^\kappa)^{-1}$）。SBP 的核心基礎在於對角範數 (Diagonal-norm SBP property)。

2. **從「Split-Form」轉向「通量差分 (Flux Differencing)」**：
   - 原有的體積項利用乘積法則拆解（如 `D_x(u*q) + D_y(v*q)` 等）來提升穩定性。
   - SBP-DG 必須使用通量差分：$2 \sum D_m^\kappa \circ F_{m,S}(\vec{u^\kappa}, \vec{u^\kappa}) \vec{1^\kappa}$。透過 Hadamard Product (逐項相乘) 結合熵保守通量 (Entropy Conservative Flux) 來抵銷對角矩陣帶來的混疊誤差 (Aliasing)。

3. **從「標準迎風/懲罰項」轉向「熵穩定 SATs (Simultaneous Approximation Terms)」**：
   - 介面交換將嚴格遵守 SAT 格式：$\text{SAT} = (M^\kappa)^{-1} \sum (R^{\gamma\kappa})^T B^\gamma (\vec{f_n^{\gamma\kappa}} - \vec{f_n^{\gamma\kappa,\kappa}})$。

---

## 🏗️ 實作修改階段清單 (Implementation Blueprint)

### 第一階段：SBP 算子與對角矩陣初始化 (CELL 2 修改)
原程式碼中的 `initialize_multi_element_dg` 函數需要大幅修改。

1. **移除密集投影矩陣**：
   - 刪除 `M_modal` 與 `M_inv_projected` 的計算。
   - 直接將體積積分權重 `weights_2d` 作為對角質量矩陣 $M^\kappa$。
2. **建構 SBP 微分矩陣 (Critical API)**：
   - 原本的 `D_r_ref, D_s_ref` 若只是單純的 Nodal Galerkin 微分矩陣，可能會破壞 SBP 特性 $M D + D^T M = E$。
   - **修改要求**：需要引入（或確保現有函數具備）根據論文 Theorem 3.1 生成嚴格 SBP 微分矩陣的 API：
     $D_m^\kappa = \frac{1}{2} (M^\kappa)^{-1} \sum n_m^{\gamma\kappa} (R + V P)^T B (R - V P) + V \hat{D}_m P$
   - *注意：若您原本使用的 Nodal 節點與權重已經滿足 Gauss-Lobatto 類型的 SBP 特性，則需驗證 $M D + D^T M$ 是否精確等於邊界矩陣 $E$。*

### 第二階段：通量差分體積項實作 (CELL 3 - Volume Term 修改)
重構 `compute_rhs_vectorized` 函數中的 VOLUME TERM 區塊。

1. **實作熵保守通量 (Entropy Conservative Flux)**：
   - 新增 API：`entropy_conservative_flux_x(qL, qR)` 與 `_y(qL, qR)`。對於線性平流方程式，這通常是對稱平均（如 $\frac{u_L + u_R}{2}$），但在未來擴充到 SWE 時會變得非常複雜。
2. **建構通量差分 (Flux Differencing) 矩陣運算**：
   - 廢除原本的 `term1`, `term2`, `term3`。
   - **實作邏輯**：針對每一個元素，計算所有節點兩兩之間的通量矩陣 $F_{m,S}$（形狀為 `Np x Np`），然後與 SBP 微分矩陣 `D_r_SBP` 進行 Hadamard Product `*`。
   - 數學式：$2 \times (D_r \circ F_{r,S}) \cdot \vec{1}$。

### 第三階段：SATs 介面通量實作 (CELL 3 - Surface Term 修改)
重構 `compute_rhs_vectorized` 函數中的 SURFACE TERM 區塊。

1. **保留現有拓樸提取，但修改組裝邏輯**：
   - 原有的 `q_M` 與 `q_P` 提取邏輯完全保留（非常適合 SATs 運算）。
   - 計算介面數值通量 $\hat{f}_n(q_M, q_P)$，需確保其為「熵穩定通量 (Entropy Stable Flux)」。
2. **SAT 矩陣-向量乘法組裝**：
   - 原本的 `surface_integral = E.T @ scaled_penalty` 與 SAT 的 $(R^{\gamma\kappa})^T B^\gamma (\dots)$ 結構上等價，但必須確保 `scaled_penalty` 精確對應於 $(\vec{f_n^{\gamma\kappa}} - \hat{f}_n)$。
   - **最終 SAT 組裝**：
     ```python
     # 嚴格使用對角逆質量矩陣
     W_inv = 1.0 / weights_2d
     surface_term = (1.0 / J[np.newaxis, :]) * W_inv[:, np.newaxis] * surface_integral
     ```
   - 確保此處**絕對不使用** `M_inv_projected @ surface_integral`。

---

## ⚠️ 需格外注意的 API 與潛在陷阱 (Critical APIs & Pitfalls)

1. **`numpy` 廣播與記憶體爆炸陷阱 (Hadamard Product of Flux Differencing)**
   - **危險**：通量差分需要計算每個元素內任意兩個節點 $j, l$ 的 $f_{m,S}(u_j, u_l)$。如果直接寫成全域的 4D 張量 `(Np, Np, K)` 來向量化，當 `K` (元素數量) 或 `Np` 很大時，會瞬間引發 **Out of Memory (OOM)** 崩潰。
   - **對策**：設計通量差分 API 時，如果 Python 原生的 Numpy 廣播負擔太重，必須考慮使用 JAX 的 `vmap`，或是將 `(D \circ F) * 1` 的運算寫成巧妙的矩陣乘法展開（例如利用線性方程式的特性簡化，或利用 Numba/Cython 進行 JIT 編譯，避免生成完整的 `Np x Np x K` 密集張量）。

2. **SBP 微分矩陣的 API 依賴**
   - 論文強烈依賴 $D_r, D_s$ 是 SBP 算子。請檢查您程式庫中 `build_differentiation_matrices` 生成的矩陣是否滿足對角範數 SBP 定義。若不滿足，直接換成 $(M^\kappa)^{-1}$ 必定會導致如之前一樣的高頻雜訊崩潰。必須實作或調用專門生成 SBP Operator 的 API。

3. **邊界提取算子 $R$ (`E` 矩陣) 的定義一致性**
   - 論文中的 $R$ (Extrapolation matrix) 對應於您程式碼中的 `E` 矩陣。在 SBP 框架中，如果內部積分點與邊界點完全重合 (Collocated surface nodes)，$R$ 只是一個簡單的 Boolean 提取矩陣（如您當前的實作）。請確保這一點在切換到 SBP 時維持不變，否則 SAT 項的權重 $B^\gamma$ (`weights_1d`) 乘法會出現尺度錯誤。
