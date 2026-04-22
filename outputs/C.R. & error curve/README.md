# 2D Advection Results (DG Method)

This repository presents numerical results for 2D linear advection using a Discontinuous Galerkin (DG) scheme under different CFL conditions and velocity fields.

---

## 🔧 Numerical Setup

* Polynomial degree: `k = 4`
* Order parameter: `N = k + 1`
* Flux type: `upwind`
* LF dissipation parameter: `alpha_LF = 1.0`
* Mass matrix: `M_inv_projected`
* Boundary condition: **Periodic boundary exchange (週期性邊界交換)**

---

## 📁 Case 1: FinalTime = 1, CFL = 0.05

### 📌 Diagonal Advection (xy)

* Velocity field: **(u, v) = (1, 1)**
* Exact solution:

```python
q_expr = sp.sin(2 * np.pi * (x + y - 2 * t))
```

#### Results

##### x-direction

![img](FinalTime=1CFL=005/x/2026-04-20_22.42.39.png)
![img](FinalTime=1CFL=005/x/image.png)
![img](FinalTime=1CFL=005/x/image2.png)

##### xy (diagonal)

![img](FinalTime=1CFL=005/xy/2026-04-22_15.05.33.png)
![img](FinalTime=1CFL=005/xy/image.png)
![img](FinalTime=1CFL=005/xy/image2.png)

##### y-direction

* Velocity field: **(u, v) = (0, 1)**
* Exact solution:

```python
q_expr = sp.sin(2 * np.pi * (y - t))
```

![img](FinalTime=1CFL=005/y/2026-04-22_14.17.17.png)
![img](FinalTime=1CFL=005/y/image.png)
![img](FinalTime=1CFL=005/y/image2.png)

---

## 📁 Case 2: FinalTime = 20, CFL = 1

### 📌 x-direction Advection

* Velocity field: **(u, v) = (1, 0)**
* Exact solution:

```python
q_expr = sp.sin(2 * np.pi * (x - t))
```

#### Results

##### x-direction

![img](FinalTime=20CFL=1/x/2026-04-22_13.06.46.png)
![img](FinalTime=20CFL=1/x/image.png)
![img](FinalTime=20CFL=1/x/image2.png)

##### xy

![img](FinalTime=20CFL=1/xy/2026-04-21_21.32.07.png)
![img](FinalTime=20CFL=1/xy/image.png)
![img](FinalTime=20CFL=1/xy/image2.png)

##### y-direction

![img](FinalTime=20CFL=1/y/2026-04-22_13.47.41.png)
![img](FinalTime=20CFL=1/y/image.png)
![img](FinalTime=20CFL=1/y/image2.png)

---

## 🧠 Notes

* The folder `xy` corresponds to **diagonal wave propagation**:

  ```
  q = sin(2π(x + y - 2t))
  ```

* Different folders (`x`, `y`, `xy`) represent wave propagation along:

  * x-axis
  * y-axis
  * diagonal direction

* All simulations use **periodic boundary conditions**.

---

## 📊 Summary

| Case | CFL  | Final Time | Direction | Velocity |
| ---- | ---- | ---------- | --------- | -------- |
| 1    | 0.05 | 1          | xy        | (1,1)    |
| 1    | 0.05 | 1          | y         | (0,1)    |
| 2    | 1    | 20         | x         | (1,0)    |

---
