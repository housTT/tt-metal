| gate | bf8_act | bf8w_hifi3_erf | threshold |
|---|---|---|---|
| E2 accuracy | 0.3605 (delta -0.0010, pass) | 0.3590 (delta -0.0025, pass) | within 0.01 of CPU fp32 0.3615 |
| E2 soft_accuracy | 0.3312 (delta -0.0003, pass) | 0.3315 (delta +0.0000, pass) | within 0.015 of CPU fp32 0.3315 |
| E2 brier_vs_soft | 0.3133 (delta -0.0022, pass) | 0.3109 (delta -0.0046, pass) | within 0.015 of CPU fp32 0.3155 |
| E2 ece | 0.1730 (delta -0.0017, pass) | 0.1716 (delta -0.0031, pass) | within 0.015 of CPU fp32 0.1747 |
| E2 score_mae | 0.6903 (delta -0.0034, pass) | 0.6892 (delta -0.0045, pass) | within 0.015 of CPU fp32 0.6937 |
| E1 tensor path confident agreement | 99.50 percent (401 of 403) | 100.00 percent (403 of 403) | >= 98 percent |
| E1 tensor path argmax agreement | 97.34 percent (475 of 488) | 97.54 percent (476 of 488) | >= 95 percent |
| E1 wire path confident agreement | 99.50 percent (401 of 403) | 100.00 percent (403 of 403) | >= 98 percent |
| E1 wire path argmax agreement | 97.34 percent (475 of 488) | 97.54 percent (476 of 488) | >= 95 percent |
| confirmation | pass | pass | all of the above |
| stage 6 alone versus in batch (candidate bf8_act) | max abs delta p 0.0381, same argmax 16 of 16 (FAIL) | | <= 0.01 and the same argmax |
