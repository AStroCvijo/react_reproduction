# ReAct reproduction
Reproduction of ICLR 2023 paper ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/pdf/2210.03629).
Developed as part of an application for EEML 2025, Sarajevo, Bosnia and Herzegovina.

## ⚡**Quickstart**

### **Prerequisites**
- Python 3.9+
- Conda (for environment management)

### 🛠️**Setup**
Clone the repository and install dependencies:

```bash
git clone https://github.com/AStroCvijo/react_reproduction.git
cd react_reproduction
conda create --name react python=3.9
conda activate react
conda install -c conda-forge libstdcxx-ng
pip install -r requirements.txt
```
---

## 🖥️ Scripts for Running Experiments

### **🔍 FEVER Dataset**
| Name                 | Description                                   |  Command                  |
|------------------------|-----------------------------------------------|----------------------------------|
| Standard         | Standard inference (no reasoning/acting)      | `./scripts/fever/standard.sh`    |
| CoT               | Chain-of-Thought (CoT)                        | `./scripts/fever/cot.sh`         |
| CoT-SC            | CoT with self-consistency (21 samples)        | `./scripts/fever/cot_sc.sh`      |
| Act               | Action-only (no reasoning)                    | `./scripts/fever/act.sh`         |
| ReAct             | ReAct (reasoning + acting)                    | `./scripts/fever/react.sh`       |
| CoT-SC -> ReAct      | CoT with self-consistency and ReAct hybrid    | `./scripts/fever/cot_sc_react.sh`|
| ReAct ->  CoT-SC   | ReAct and CoT with self-consistency hybrid    | `./scripts/fever/react_cot_sc.sh`|


### **🍲 HotpotQA Dataset**
| Name                 | Description                                   |  Command                  |
|------------------------|-----------------------------------------------|----------------------------------|
| Standard         | Standard inference (no reasoning/acting)   | `./scripts/hotpotqa/standard.sh` |
| CoT               | Chain-of-Thought (CoT)                     | `./scripts/hotpotqa/cot.sh`      |
| CoT-SC           | CoT with self-consistency (21 samples)     | `./scripts/hotpotqa/cot_sc.sh`   |
| Act                | Action-only (no reasoning)                 | `./scripts/hotpotqa/act.sh`      |
| ReAct             | ReAct (reasoning + acting)                 | `./scripts/hotpotqa/react.sh`    |
| CoT-SC -> ReAct      | CoT with self-consistency and ReAct hybrid | `./scripts/hotpotqa/cot_sc_react.sh`|
| ReAct ->  CoT-SC      | ReAct and CoT with self-consistency hybrid | `./scripts/hotpotqa/react_cot_sc.sh`|


### **🏠 ALFWorld Dataset**
| Name          | Description                          |  Command               |
|-----------------|--------------------------------------|-------------------------------|
| Act        | Action-only (no reasoning)           | `./scripts/alfworld/act.sh`   |
| ReAct      | ReAct (reasoning + acting)          | `./scripts/alfworld/react.sh` |


### **🛍️ WebShop Dataset**
| Name          | Description                          |  Command               |
|-----------------|--------------------------------------|-------------------------------|
| Act        | Action-only (no reasoning)              | `./scripts/webshop/act.sh`    |
| ReAct      | ReAct (reasoning + acting)             | `./scripts/webshop/react.sh`  |

---

## 📖**Arguments Guide**

| Argument         | Description                                           | Default    | Options                                                             |
|------------------|-------------------------------------------------------|------------|---------------------------------------------------------------------|
| -ds, --data_set  | Dataset selection                                    | FEVER      | FEVER, HotpotQA, ALFWorld, WebShop                                  |
| -ps, --prompt_style | Prompt style to use                                 | ReAct      | ReAct, Act, CoT, Standard, CoT-SC-ReAct, ReAct-CoT-SC                |
| -ns, --num_samples | Number of samples to generate                       | 1          | Any positive integer                                                 |
| -t, --tempreture  | Temperature setting for response variability        | 0.0        | Any float value (0.0 to 1.0)                                         |
