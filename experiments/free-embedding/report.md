# Free-embedding optimization, short budget

Small-scale rerun of the free-embedding protocol behind Table 6 of Weller et al., LIMIT (ICLR 2026). Query and document vectors are fit directly with full-batch InfoNCE. The paper's critical-n is the largest n at which unit-norm embeddings of dimension d still reach perfect top-2 accuracy on all C(n, 2) queries.

## Budget

This run uses Adam (lr 0.01), temperature 0.1, at most **3000 steps**, and early stopping when the loss fails to improve by 1e-05 for **400 steps**. The paper used the same optimizer, learning rate, and temperature, with up to **100000 steps** and patience **1000**. The step budget here is shorter than the paper, so missing Recall@2 = 1.0 at the published critical-n is a short-budget outcome, not a new estimate of that critical value. Recall@2 is measured after every step; the paper logged it every 50 steps and stopped once it hit 1.0.

Device: `cuda:0` (NVIDIA RTX PRO 6000 Blackwell Server Edition). Torch `2.14.1+cu130` float32. Seed 42. Wall time 11.0 s.

## Free embeddings versus Table 6

Paper critical-n (perfect Recall@2): **d=4 → 10**, **d=8 → 28**, d=12 → 47. This grid does not include d=12, and for d=8 it stops at n=24, which is below the published critical value 28. Every free-embedding run uses the full set of C(n, 2) queries. None is capped at 800, so these rows use the same query set as the paper's protocol.

| d | n | C(n, 2) | vs paper critical-n | best Recall@2 | final Recall@2 | queries perfect | perfect? | steps | stop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 6 | 15 | below critical (critical 10) | 1.0000 | 1.0000 | 15/15 | yes | 215 | perfect_recall |
| 4 | 8 | 28 | below critical (critical 10) | 1.0000 | 1.0000 | 28/28 | yes | 624 | perfect_recall |
| 4 | 10 | 45 | at critical (critical 10) | 0.9444 | 0.9444 | 40/45 | no | 2323 | early_stop_loss |
| 4 | 12 | 66 | above critical (critical 10) | 0.9015 | 0.8788 | 53/66 | no | 1478 | early_stop_loss |
| 4 | 16 | 120 | above critical (critical 10) | 0.8167 | 0.8000 | 82/120 | no | 3000 | max_steps |
| 8 | 10 | 45 | below critical (critical 28) | 1.0000 | 1.0000 | 45/45 | yes | 67 | perfect_recall |
| 8 | 16 | 120 | below critical (critical 28) | 1.0000 | 1.0000 | 120/120 | yes | 251 | perfect_recall |
| 8 | 24 | 276 | below critical (critical 28) | 1.0000 | 1.0000 | 276/276 | yes | 376 | perfect_recall |

Perfect Recall@2 under this budget: d=4 n=6, d=4 n=8, d=8 n=10, d=8 n=16, d=8 n=24.
Short of perfect: d=4 n=10 best 0.9444, d=4 n=12 best 0.9015, d=4 n=16 best 0.8167.

## Moment-curve control

Documents are frozen on the 4-d moment curve `t_i = -1 + 2*(i + 0.5)/n  for i = 0..n-1` and are left unnormalized. Only query vectors are trained, with the same Adam budget, unit-norm projection, and InfoNCE objective. For n=16 every pair is a query (120 queries). For n=46 the run uses 200 random pairs (seed 43), not all 1035, so that row is a subsample and is not a critical-n measurement.

A closed-form query exists for each pair: the degree-4 polynomial `-(x-t_a)^2 (x-t_b)^2`, with the constant term dropped because it does not change inner-product order. Its Recall@2 checks that this document geometry can represent the labels before any training.

| n | queries | total pairs | closed-form Recall@2 | init Recall@2 | best Recall@2 | final Recall@2 | best queries perfect | perfect? | steps | stop | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 120 | 120 | 1.0000 | 0.1458 | 0.5042 | 0.5000 | 34/120 | no | 3000 | max_steps | 2.1 |
| 46 | 200 | 1035 | 1.0000 | 0.0475 | 0.1800 | 0.1725 | 5/200 | no | 2923 | early_stop_loss | 2.0 |

Closed-form queries score Recall@2 = 1.0 on both controls, so these frozen document positions can place both gold documents strictly in the top 2. The query-only Adam run does not. It finishes at a lower InfoNCE loss than those perfect queries:

- n=16: closed-form loss 2.1987 (Recall@2 1.0000, min score gap 4.36e-05), Adam final loss 1.8958 (Recall@2 0.5000, 33/120 queries perfect).
- n=46: closed-form loss 3.2480 (Recall@2 1.0000, min score gap 1.17e-06), Adam final loss 2.9098 (Recall@2 0.1725, 5/200 queries perfect).

InfoNCE at temperature 0.1 finishes at a lower loss than the perfect ranking. The smallest closed-form gaps sit far below the temperature, so a perfect ranking can still look almost tied to the softmax. A query that lifts one gold well above the other can score a lower loss and leave a negative inside the top 2.

On n=16 the monitored loss is 1.8981 at step 1501 and 1.8958 at the end (max_steps, 3000 steps).

On n=46 the monitored loss is 2.9133 at step 1501 and 2.9098 at the end (early_stop_loss, 2923 steps).

That plateau is already in place well before the paper's 100000-step budget. On this trajectory, further steps of the same optimizer are not approaching Recall@2 = 1.0.

## Reading

Under this shorter budget, free embeddings reach perfect Recall@2 for d=4 n=6 (best Recall@2 1.0000, 15/15 queries), d=4 n=8 (best Recall@2 1.0000, 28/28 queries), d=8 n=10 (best Recall@2 1.0000, 45/45 queries), d=8 n=16 (best Recall@2 1.0000, 120/120 queries), d=8 n=24 (best Recall@2 1.0000, 276/276 queries). They miss perfect Recall@2 for d=4 n=10 (best Recall@2 0.9444, 40/45 queries), d=4 n=12 (best Recall@2 0.9015, 53/66 queries), d=4 n=16 (best Recall@2 0.8167, 82/120 queries). The d=8 grid stops at n=24, below the paper's critical n=28, which was not run. d=12 was not run.

Freezing documents on the 4-d moment curve does not make the same optimizer succeed (n=16 best Recall@2 0.5042 (34/120 queries); n=46 best Recall@2 0.1800 (5/200 queries)). Closed-form queries on those same documents score Recall@2 1.0. The step budget is shorter than the paper: 3000 Adam steps and patience 400, against 100000 steps and patience 1000.

