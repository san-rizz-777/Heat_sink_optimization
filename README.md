# 🔥 Heat Sink Optimization for Data Center Cooling

### Author: Sanskar Gunde (MM24B005)

---

## 🧠 Project Overview
This project focuses on **optimizing the design of a heat sink** used for air-cooled data center systems. The goal is to achieve **efficient thermal performance** while minimizing **fan power consumption** and **material costs**.

The project models **conductive heat transfer** through fins and **convective cooling** to air, followed by optimization of fin geometry and airflow velocity using a **Differential Evolution (DE)** algorithm.

---

## ⚙️ Problem Description
A heat sink’s efficiency depends on:
- Fin **height**, **thickness**, **spacing**, and **count**
- **Airflow velocity** over the fins
- **Material** thermal conductivity and cost

The optimization problem aims to **minimize the base temperature** of the heat sink while **reducing energy and material costs**.

---

## 🧩 Methodology

### 1. Thermal Modeling
- **Conduction** modeled through fin bodies using Fourier’s law.  
- **Convection** modeled at fin surfaces using Newton’s law of cooling.  
- Combined steady-state equations yield fin efficiency and overall heat transfer rate.

### 2. Optimization
- Uses **Differential Evolution (scipy.optimize.differential_evolution)**.  
- Objective function combines:
  - Base temperature minimization  
  - Power consumption and material cost penalties  
- No gradient required → ideal for multi-variable, nonlinear optimization.

### 3. Visualization
- Temperature distributions plotted with **Matplotlib**.
- Trade-off curves show relationships between temperature, power, and cost.

---

## 🛠️ Technologies Used
| Library | Purpose |
|----------|----------|
| **NumPy** | Numerical computation and matrix operations |
| **SciPy** | Differential Evolution optimization |
| **Matplotlib** | Visualization and performance plotting |
| **Python** | Implementation and simulation environment |

---

## 📁 Repository Structure
Heat_sink_optimization/
├── heat_transfer.py # Conductive and convective heat models
├── fluid_dynamics.py # Airflow and fan power models
├── optimization.py # Differential Evolution implementation
├── visualization.py # Graphs and heat maps
├── validation.py # Sanity checks and parameter validation
├── results/ # Generated result plots
└── main.py # Entry point for running the full simulation


---

##  How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/san-rizz-777/Heat_sink_optimization.git
   cd Heat_sink_optimization
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the simulation:
   ```bash
   python main.py

---
## Output Summary
# After optimization, the code outputs:
- Optimum fin geometry and air velocity
- Reduced base temperature
- Fan power vs. temperature trade-off plots
- Convergence curve of Differential Evolution

 ---
 
## 📚 References
- Incropera, F. P. & DeWitt, D. P. Fundamentals of Heat and Mass Transfer
- SciPy Documentation — Differential Evolution
- Bejan, A. Convection Heat Transfer
