# Unified Recommendation Unlearning Framework

This repository provides a unified and corrected implementation of three state-of-the-art Recommendation Unlearning (RU) frameworks: **RecEraser**, **UltraRE**, and **CURE4Rec**. We have integrated these into a structured codebase to facilitate fair benchmarking and resolve inconsistencies found in original implementations.

## 🚀 Overview

Our framework bridges the gaps in existing codebases—specifically regarding missing aggregation stages and misaligned data partitioning strategies.

* **RecEraser (WWW '22):** A general framework for collaborative unlearning using data shards and adaptive aggregation.
* **UltraRE (NIPS '23):** Enhancing unlearning via error decomposition. **Fixed:** Added the missing Stage III model aggregation.
* **CURE4Rec (NIPS '24):** A benchmark for unlearning with deeper influence. **Fixed:** Aligned Stage III with Attention-based aggregation for architectural consistency.

### 🛠 Key Improvements

* **Standardized Stage III:** We utilize **Attention-based aggregation** across all models to ensure stable utility.
* **Partitioning Alignment:** We implement **Interaction-Based Partitioning (InBP)** across all methods to allow for a direct comparison, aligning with the core RecEraser methodology.
* **Evaluation Protocol:** We employ a **Leave-One-Out** evaluation strategy for more granular precision.

---

## 📊 Experimental Results (NeuMF on ml-1m)

The following results demonstrate performance on the `ml-1m` dataset using a **NeuMF** backbone under the **InBP** setting.

| Method | Metric | 0% Unlearn | 5% Unlearn | 10% Unlearn |
| --- | --- | --- | --- | --- |
| **Retrain** | NDCG / HR | 0.3122 / 0.5885 | 0.3063 / 0.5852 | 0.3114 / 0.5899 |
| **SISA** | NDCG / HR | 0.2378 / 0.4758 | 0.2360 / 0.4665 | 0.2378 / 0.4687 |
| **RecEraser** | NDCG / HR | 0.2533 / 0.4950 | 0.2684 / 0.5060 | 0.2613 / 0.5058 |
| **UltraEraser** | NDCG / HR | **0.2688** / **0.5205** | 0.2655 / **0.5155** | **0.2724** / **0.5145** |

### 💡 Key Insight: The InBP vs. UBP Stability Trade-off

Our benchmarking reveals a critical nuance in how data partitioning affects unlearning utility:

* **Partitioning vs. Stability:** While the original NIPS papers demonstrate that **UltraRE** outperforms **RecEraser** under **User-Based Partitioning (UBP)**, our results show that **Interaction-Based Partitioning (InBP)** introduces interaction-level variance that can destabilize the **Optimal Transport (OT) clustering** process.
* **Backbone Performance:** Due to this instability, UltraRE initially exhibits slightly lower utility than RecEraser on `WMF` and `LightGCN` when utilizing InBP.
* **Regularization as a Stabilizer:** However, in the case of `NeuMF`, we found that carefully tuning the **regularization (reg)** within the **OT-cluster** effectively "damps" this InBP-induced instability, allowing UltraRE to recover its performance edge.

---

## 💻 Usage

To run the unlearning process (e.g., 5% random deletion on MovieLens-1M):

```bash
python main.py \
  --epoch 100 \
  --dataset ml-1m \
  --model neumf \
  --group 10 \
  --learn receraser \
  --deltype random \
  --delper 5 \
  --verbose 2

```

**Key Arguments:**

* `--learn`: Framework choice (`receraser`, `ultrare`, `sisa`, `retrain`).
* `--model`: Backbone choice (`neumf`, `wmf`, `lightgcn`).
* `--delper`: Percentage of interactions to unlearn (0, 5, 10).

---

## 📝 Citation

If you find this unified framework useful, please cite the original works:

```bibtex
@inproceedings{receraser2022,
  title={Recommendation Unlearning},
  author={Chen, Chong and Sun, Fei and Zhang, Min and Ding, Bolin},
  booktitle={Proceedings of the ACM Web Conference 2022},
  year={2022}
}

@inproceedings{ultrare2023,
  title={UltraRE: Enhancing RecEraser for Recommendation Unlearning via Error Decomposition},
  booktitle={NIPS},
  year={2023}
}

@inproceedings{cure4rec2024,
  title={CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influence},
  booktitle={NIPS},
  year={2024}
}

We are open to academic discussion and further contributions to the Recommendation Unlearning community.
```
