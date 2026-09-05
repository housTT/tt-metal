# Alternating reader confirmation

Each device sample is one signposted replay. Host medians use five warmed windows of 64 replays per candidate, collected without the profiler. The three N=4096 output roles use common per_core_N=18 to make reader3 legal; production reader2 keeps per_core_N=16, whose passing controls are separately archived. Physical bandwidth includes BFP4 tile headers and per-bank reader padding. Runtime attributes, execution identities, source hashes and compressed operation CSVs are retained beside the source device accounting JSONs.

All 66 profiled and 66 unprofiled projection cases pass real-input PCC and exact eager/trace equality. The selected reader count wins both device matmul and unprofiled trace medians for every role.

| Layer kind | Role | Readers | Device matmul us | Unprofiled trace us | Physical GB/s | Peak % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear_attention | gdn_packed | 1 | 69.737 | 80.431 | 279.1 | 54.5 |
| linear_attention | gdn_packed | 2 | 53.281 | 64.615 | 376.4 | 73.5 |
| linear_attention | gdn_packed | **3** | 46.763 | 58.123 | 416.2 | 81.3 |
| linear_attention | gdn_z_epilogue | 1 | 49.053 | 58.758 | 192.4 | 37.6 |
| linear_attention | gdn_z_epilogue | **2** | 34.090 | 45.020 | 276.8 | 54.1 |
| linear_attention | gdn_z_epilogue | 3 | 41.856 | 52.948 | 253.7 | 49.5 |
| linear_attention | gdn_out | 1 | 34.640 | 42.867 | 272.4 | 53.2 |
| linear_attention | gdn_out | **2** | 26.134 | 35.854 | 361.1 | 70.5 |
| linear_attention | gdn_out | 3 | 36.364 | 45.793 | 292.0 | 57.0 |
| linear_attention | gate_proj | 1 | 106.358 | 116.897 | 266.2 | 52.0 |
| linear_attention | gate_proj | 2 | 66.409 | 77.521 | 426.3 | 83.3 |
| linear_attention | gate_proj | **3** | 63.485 | 73.568 | 446.0 | 87.1 |
| linear_attention | up_proj | 1 | 106.317 | 116.895 | 266.3 | 52.0 |
| linear_attention | up_proj | 2 | 66.721 | 77.554 | 424.3 | 82.9 |
| linear_attention | up_proj | **3** | 63.502 | 73.595 | 445.8 | 87.1 |
| linear_attention | down_proj | 1 | 96.850 | 106.606 | 292.3 | 57.1 |
| linear_attention | down_proj | **2** | 61.879 | 71.501 | 457.5 | 89.4 |
| linear_attention | down_proj | 3 | 67.876 | 76.597 | 469.3 | 91.7 |
| full_attention | qkvg | 1 | 91.986 | 102.595 | 256.5 | 50.1 |
| full_attention | qkvg | 2 | 63.953 | 74.943 | 368.9 | 72.1 |
| full_attention | qkvg | **3** | 58.010 | 69.961 | 427.1 | 83.4 |
| full_attention | o_proj | 1 | 34.309 | 42.551 | 275.1 | 53.7 |
| full_attention | o_proj | **2** | 26.026 | 35.691 | 362.6 | 70.8 |
| full_attention | o_proj | 3 | 36.137 | 45.502 | 293.8 | 57.4 |
| full_attention | gate_proj | 1 | 106.269 | 116.932 | 266.4 | 52.0 |
| full_attention | gate_proj | 2 | 66.288 | 77.553 | 427.1 | 83.4 |
| full_attention | gate_proj | **3** | 63.717 | 73.732 | 444.4 | 86.8 |
| full_attention | up_proj | 1 | 106.281 | 116.989 | 266.4 | 52.0 |
| full_attention | up_proj | 2 | 66.368 | 77.561 | 426.6 | 83.3 |
| full_attention | up_proj | **3** | 63.812 | 73.594 | 443.7 | 86.7 |
| full_attention | down_proj | 1 | 96.633 | 106.726 | 293.0 | 57.2 |
| full_attention | down_proj | **2** | 62.490 | 71.484 | 453.1 | 88.5 |
| full_attention | down_proj | 3 | 68.404 | 76.813 | 465.6 | 90.9 |
