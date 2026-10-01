| variant | gate_up_dtype | down_dtype | proj_dtype | fidelity | gdn_fp32_state | matmul_policy | warm_row_s | warm_ms_per_token | acc | brier | ece | argmax_flips | flips_margin | flips_neartie | max_dp | mean_dp | min_pcc | traced_tail128_ms | traced_state2048_ms | cache_gb | status | pareto | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_policy_on | bfp4 | bfp8 | bfp8 | LoFi | 0 | True | 0.382 | 0.700 | 0.8213 | 0.2481 | 0.0822 | 0 | 0 | 0 | 0.1024 | 0.0366 | 0.9768 | 105.0 | 898.7 | 8.86 | ok |  |  |
| baseline | bfp4 | bfp8 | bfp8 | LoFi | 0 | False | 0.577 | 1.055 | 0.8213 | 0.2481 | 0.0856 | 0 | 0 | 0 | 0.1024 | 0.0366 | 0.9768 | 107.0 | 1510.2 | 8.86 | ok | yes |  |
| all_bfp8_gdnfp32 | bfp8 | bfp8 | bfp8 | LoFi | 1 | False | 0.579 | 1.059 | 0.8316 | 0.2429 | 0.0779 | 1 | 0 | 1 | 0.0876 | 0.0266 | 0.9873 |  |  | 10.47 | ok | yes |  |
| mlp_bfp8 | bfp8 | bfp8 | bfp8 | LoFi | 0 | False | 0.579 | 1.059 | 0.8316 | 0.2428 | 0.0778 | 1 | 0 | 1 | 0.0876 | 0.0264 | 0.9873 | 107.0 | 1519.8 | 10.47 | ok | yes | yes |
| all_bfp8_hifi2 | bfp8 | bfp8 | bfp8 | HiFi2 | 0 | False | 0.598 | 1.094 | 0.8247 | 0.2435 | 0.0713 | 0 | 0 | 0 | 0.0894 | 0.0271 | 0.9883 |  |  | 10.47 | ok |  |  |
| all_bf16 | bf16 | bf16 | bf16 | HiFi2 | 0 | False | 0.604 | 1.105 | 0.8247 | 0.2439 | 0.0677 | 1 | 0 | 1 | 0.0661 | 0.0243 | 0.9893 |  |  | 16.95 | ok |  |  |
| mlp_bf16 | bf16 | bf16 | bfp8 | HiFi2 | 0 | False | 0.604 | 1.106 | 0.8316 | 0.2433 | 0.0725 | 0 | 0 | 0 | 0.0722 | 0.0246 | 0.9888 | 114.0 | 1563.7 | 15.0 | ok |  |  |
| mlp_bfp8_policy_on | bfp8 | bfp8 | bfp8 | LoFi | 0 | True |  |  |  |  |  |  |  |  |  |  |  |  |  | 10.47 | error: RuntimeError: TT_THROW @ /home/hous/dev/kev/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2682: tt::exception |  |  |
| all_bfp8_hifi2_policy_on | bfp8 | bfp8 | bfp8 | HiFi2 | 0 | True |  |  |  |  |  |  |  |  |  |  |  |  |  | 10.47 | error: RuntimeError: TT_THROW @ /home/hous/dev/kev/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2682: tt::exception |  |  |
| all_bfp8_gdnfp32_policy_on | bfp8 | bfp8 | bfp8 | LoFi | 1 | True |  |  |  |  |  |  |  |  |  |  |  |  |  | 10.47 | error: RuntimeError: TT_THROW @ /home/hous/dev/kev/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2682: tt::exception |  |  |
