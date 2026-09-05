# Local projection geometry search

Real decoder-produced TP-local inputs; warmed trace includes the row projection's collective.
DRAM-sharded family uses reader1: reader2/3 are blocked by the verified mesh API call.
PCC is against the original local geometry, not a replacement for whole-layer HF gates.

| Artifact | Family | Weight dtype | Layer | Role | K × N per device | Input/compute cores | K-block tiles | Median μs | Min PCC | Exact original | Exact replay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 1 | 110.126 | 0.99981728 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 2 | 63.963 | 0.99989629 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 4 | 41.468 | 0.99991751 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 8 | 32.328 | 0.99995652 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 16 | 30.044 | 1.00000000 | True | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 32 | 29.958 | 0.99993065 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 64 | 30.923 | 0.99978541 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 128 | 34.577 | 0.99978541 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 1 | 121.489 | 0.99967132 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 2 | 69.759 | 0.99971690 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 4 | 44.509 | 0.99975927 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 8 | 32.824 | 0.99979887 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 16 | 26.214 | 0.99985211 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 32 | 24.325 | 1.00000000 | True | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 64 | 24.729 | 0.99971557 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 128 | 26.569 | 0.99971557 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 1 | 56.562 | 0.99999796 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 2 | 43.885 | 0.99999885 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 4 | 37.719 | 0.99999931 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 8 | 35.890 | 1.00000000 | True | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 16 | 36.566 | 0.99999949 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 32 | 37.285 | 0.99999949 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 1 | 118.205 | 0.99987108 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 2 | 67.007 | 0.99991031 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 4 | 41.994 | 1.00000000 | True | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 8 | 36.735 | 0.99994370 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 16 | 37.351 | 0.99990779 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 32 | 39.997 | 0.99981037 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 64 | 41.699 | 0.99956759 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 128 | 43.977 | 0.99956759 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 1 | 118.499 | 0.99984121 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 2 | 66.939 | 0.99989065 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 4 | 41.970 | 0.99991290 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 8 | 37.240 | 0.99994197 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 16 | 37.833 | 1.00000000 | True | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 32 | 39.557 | 0.99989324 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 64 | 41.792 | 0.99962674 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 128 | 44.019 | 0.99962674 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 1 | 108.932 | 0.99991994 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 2 | 70.475 | 0.99993958 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 3 | 58.075 | 0.99995010 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 4 | 53.521 | 0.99995594 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 6 | 51.787 | 0.99996779 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 8 | 51.951 | 0.99997139 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 12 | 53.259 | 1.00000000 | True | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 16 | 54.747 | 0.99996670 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 24 | 56.536 | 0.99993431 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 32 | 57.364 | 0.99990334 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 48 | 58.274 | 0.99985000 | False | True |
| geometry_interleaved_layer0.json | interleaved | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 96 | 60.398 | 0.99985000 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 1 | 114.436 | 0.99995974 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 2 | 66.362 | 1.00000000 | True | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 4 | 42.596 | 0.99997652 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 8 | 35.187 | 0.99997348 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 16 | 33.810 | 0.99996324 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 32 | 35.397 | 0.99993051 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 64 | 39.528 | 0.99984867 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 128 | 41.878 | 0.99984867 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 1 | 55.100 | 0.99998458 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 2 | 42.317 | 0.99999109 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 4 | 36.930 | 0.99999309 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 8 | 35.957 | 1.00000000 | True | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 16 | 36.562 | 0.99998965 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 32 | 37.330 | 0.99998965 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 1 | 118.229 | 0.99985044 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 2 | 66.943 | 0.99991056 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 4 | 41.991 | 1.00000000 | True | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 8 | 36.860 | 0.99993611 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 16 | 37.983 | 0.99989128 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 32 | 39.530 | 0.99978591 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 64 | 41.668 | 0.99949751 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 128 | 43.995 | 0.99949751 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 1 | 118.665 | 0.99983339 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 2 | 66.934 | 0.99987535 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 4 | 42.058 | 0.99990764 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 8 | 36.861 | 0.99993264 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 16 | 37.570 | 1.00000000 | True | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 32 | 39.949 | 0.99988163 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 64 | 41.790 | 0.99954742 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 128 | 43.989 | 0.99954742 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 1 | 109.170 | 0.99988414 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 2 | 70.501 | 0.99992095 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 3 | 58.085 | 0.99993153 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 4 | 53.571 | 0.99993987 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 6 | 51.831 | 0.99995394 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 8 | 51.959 | 0.99995679 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 12 | 53.396 | 1.00000000 | True | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 16 | 54.758 | 0.99995175 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 24 | 56.563 | 0.99992236 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 32 | 57.387 | 0.99986216 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 48 | 58.316 | 0.99974473 | False | True |
| geometry_interleaved_layer3.json | interleaved | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 96 | 60.443 | 0.99974473 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 4 | 1 | 98.283 | 0.99991130 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 4 | 2 | 56.366 | 0.99995037 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 4 | 4 | 36.584 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 4 | 8 | 30.799 | 0.99995829 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 4 | 16 | 29.173 | 0.99991751 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 4 | 32 | 30.575 | 0.99980428 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 8 | 1 | 98.502 | 0.99991130 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 8 | 2 | 56.701 | 0.99995037 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 8 | 4 | 37.022 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 8 | 8 | 31.123 | 0.99995829 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 8 | 16 | 29.745 | 0.99991751 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 16 | 1 | 98.580 | 0.99991130 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 16 | 2 | 56.820 | 0.99995037 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 16 | 4 | 37.096 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 16 | 8 | 31.305 | 0.99995829 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 1 | 96.362 | 0.99991130 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 2 | 54.427 | 0.99995037 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 32 | 4 | 34.707 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 64 | 1 | 100.307 | 0.99991130 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_packed | 4096 × 2112 | 64 | 2 | 58.501 | 0.99995037 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 4 | 1 | 101.424 | 0.99984954 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 4 | 2 | 59.397 | 0.99989535 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 4 | 4 | 38.573 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 4 | 8 | 28.269 | 0.99991231 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 4 | 16 | 24.780 | 0.99987521 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 4 | 32 | 23.586 | 0.99975927 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 8 | 1 | 101.906 | 0.99984954 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 8 | 2 | 59.926 | 0.99989535 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 8 | 4 | 39.102 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 8 | 8 | 28.908 | 0.99991231 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 8 | 16 | 25.535 | 0.99987521 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 16 | 1 | 101.981 | 0.99984954 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 16 | 2 | 60.098 | 0.99989535 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 16 | 4 | 39.281 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 16 | 8 | 29.078 | 0.99991231 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 1 | 99.762 | 0.99984954 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 2 | 57.686 | 0.99989535 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 32 | 4 | 36.912 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 64 | 1 | 103.778 | 0.99984954 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_z_epilogue | 4096 × 1024 | 64 | 2 | 62.072 | 0.99989535 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 4 | 1 | 55.617 | 0.99999937 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 4 | 2 | 45.578 | 0.99999928 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 4 | 4 | 42.617 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 4 | 8 | 41.699 | 0.99999955 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 8 | 1 | 55.484 | 0.99999937 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 8 | 2 | 45.559 | 0.99999928 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 8 | 4 | 42.500 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 16 | 1 | 55.635 | 0.99999937 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 16 | 2 | 45.670 | 0.99999928 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gdn_out | 1024 × 4096 | 32 | 1 | 56.077 | 0.99999937 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 4 | 1 | 97.795 | 0.99986986 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 4 | 2 | 55.918 | 0.99992223 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 4 | 4 | 40.272 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 4 | 8 | 35.438 | 0.99994016 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 4 | 16 | 34.049 | 0.99990951 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 4 | 32 | 35.724 | 0.99981675 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 8 | 1 | 95.172 | 0.99986986 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 8 | 2 | 53.480 | 0.99992223 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 8 | 4 | 37.668 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 8 | 8 | 33.234 | 0.99994016 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 8 | 16 | 32.070 | 0.99990951 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 16 | 1 | 97.674 | 0.99986986 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 16 | 2 | 55.877 | 0.99992223 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 16 | 4 | 40.262 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 16 | 8 | 35.759 | 0.99994016 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 1 | 98.409 | 0.99986986 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 2 | 56.495 | 0.99992223 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 32 | 4 | 40.823 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 64 | 1 | 99.540 | 0.99986986 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | gate_proj | 4096 × 3072 | 64 | 2 | 57.803 | 0.99992223 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 4 | 1 | 97.822 | 0.99987695 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 4 | 2 | 55.959 | 0.99992514 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 4 | 4 | 40.140 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 4 | 8 | 35.325 | 0.99994902 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 4 | 16 | 34.044 | 0.99991889 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 4 | 32 | 35.739 | 0.99982695 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 8 | 1 | 95.149 | 0.99987695 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 8 | 2 | 53.475 | 0.99992514 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 8 | 4 | 37.636 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 8 | 8 | 33.164 | 0.99994902 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 8 | 16 | 32.043 | 0.99991889 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 16 | 1 | 97.595 | 0.99987695 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 16 | 2 | 55.839 | 0.99992514 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 16 | 4 | 40.198 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 16 | 8 | 35.667 | 0.99994902 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 1 | 98.471 | 0.99987695 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 2 | 56.507 | 0.99992514 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 32 | 4 | 40.845 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 64 | 1 | 99.559 | 0.99987695 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | up_proj | 4096 × 3072 | 64 | 2 | 57.836 | 0.99992514 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 1 | 98.696 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 2 | 68.846 | 0.99995784 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 3 | 63.423 | 0.99996475 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 4 | 60.718 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 6 | 58.279 | 0.99997040 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 8 | 57.243 | 0.99996766 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 12 | 56.509 | 0.99995794 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 4 | 24 | 58.024 | 0.99989990 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 8 | 1 | 95.978 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 8 | 2 | 66.507 | 0.99995784 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 8 | 3 | 61.219 | 0.99996475 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 8 | 4 | 58.498 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 8 | 6 | 56.062 | 0.99997040 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 8 | 12 | 54.624 | 0.99995794 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 12 | 1 | 98.131 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 12 | 2 | 68.281 | 0.99995784 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 12 | 4 | 60.059 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 12 | 8 | 56.967 | 0.99996766 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 16 | 1 | 98.535 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 16 | 2 | 68.853 | 0.99995784 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 16 | 3 | 63.468 | 0.99996475 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 16 | 6 | 58.400 | 0.99997040 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 24 | 1 | 98.942 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 24 | 2 | 69.222 | 0.99995784 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 24 | 4 | 61.039 | 1.00000000 | True | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 1 | 98.999 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 32 | 3 | 63.882 | 0.99996475 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 48 | 1 | 99.775 | 0.99993279 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 48 | 2 | 70.263 | 0.99995784 | False | True |
| geometry_layer0.json | dram | BFP4 (historical) | 0 | down_proj | 3072 × 4096 | 96 | 1 | 104.881 | 0.99993279 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 4 | 1 | 98.513 | 0.99995974 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 4 | 2 | 56.564 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 4 | 4 | 37.509 | 0.99997652 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 4 | 8 | 32.637 | 0.99997348 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 4 | 16 | 31.127 | 0.99996324 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 4 | 32 | 32.344 | 0.99993051 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 8 | 1 | 98.583 | 0.99995974 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 8 | 2 | 56.762 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 8 | 4 | 37.653 | 0.99997652 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 8 | 8 | 32.917 | 0.99997348 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 8 | 16 | 31.691 | 0.99996324 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 16 | 1 | 98.726 | 0.99995974 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 16 | 2 | 56.906 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 16 | 4 | 37.724 | 0.99997652 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 16 | 8 | 33.148 | 0.99997348 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 1 | 96.475 | 0.99995974 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 2 | 54.560 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 32 | 4 | 35.444 | 0.99997652 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 64 | 1 | 100.521 | 0.99995974 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | qkvg | 4096 × 2560 | 64 | 2 | 58.755 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 4 | 1 | 54.801 | 0.99998992 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 4 | 2 | 45.067 | 0.99999371 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 4 | 4 | 42.968 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 4 | 8 | 41.598 | 0.99999309 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 8 | 1 | 54.652 | 0.99998992 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 8 | 2 | 45.079 | 0.99999371 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 8 | 4 | 42.230 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 16 | 1 | 54.779 | 0.99998992 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 16 | 2 | 45.127 | 0.99999371 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | o_proj | 1024 × 4096 | 32 | 1 | 55.363 | 0.99998992 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 4 | 1 | 97.782 | 0.99984722 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 4 | 2 | 55.826 | 0.99990938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 4 | 4 | 39.953 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 4 | 8 | 35.334 | 0.99993283 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 4 | 16 | 34.167 | 0.99989093 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 4 | 32 | 35.840 | 0.99978446 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 8 | 1 | 95.178 | 0.99984722 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 8 | 2 | 53.413 | 0.99990938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 8 | 4 | 37.637 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 8 | 8 | 33.187 | 0.99993283 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 8 | 16 | 32.052 | 0.99989093 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 16 | 1 | 97.667 | 0.99984722 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 16 | 2 | 55.887 | 0.99990938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 16 | 4 | 40.085 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 16 | 8 | 35.645 | 0.99993283 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 1 | 98.413 | 0.99984722 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 2 | 56.506 | 0.99990938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 32 | 4 | 40.951 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 64 | 1 | 99.522 | 0.99984722 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | gate_proj | 4096 × 3072 | 64 | 2 | 57.853 | 0.99990938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 4 | 1 | 97.879 | 0.99987487 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 4 | 2 | 55.976 | 0.99991938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 4 | 4 | 40.244 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 4 | 8 | 35.313 | 0.99994374 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 4 | 16 | 34.021 | 0.99990986 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 4 | 32 | 35.783 | 0.99980487 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 8 | 1 | 95.189 | 0.99987487 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 8 | 2 | 53.408 | 0.99991938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 8 | 4 | 37.690 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 8 | 8 | 33.155 | 0.99994374 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 8 | 16 | 32.045 | 0.99990986 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 16 | 1 | 97.578 | 0.99987487 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 16 | 2 | 55.822 | 0.99991938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 16 | 4 | 40.127 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 16 | 8 | 35.743 | 0.99994374 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 1 | 98.469 | 0.99987487 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 2 | 56.505 | 0.99991938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 32 | 4 | 40.777 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 64 | 1 | 99.606 | 0.99987487 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | up_proj | 4096 × 3072 | 64 | 2 | 57.848 | 0.99991938 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 1 | 98.687 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 2 | 68.825 | 0.99994372 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 3 | 63.435 | 0.99995177 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 4 | 60.756 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 6 | 58.267 | 0.99995860 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 8 | 57.228 | 0.99995500 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 12 | 56.556 | 0.99993946 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 4 | 24 | 57.992 | 0.99988234 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 8 | 1 | 96.011 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 8 | 2 | 66.526 | 0.99994372 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 8 | 3 | 61.268 | 0.99995177 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 8 | 4 | 58.528 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 8 | 6 | 56.066 | 0.99995860 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 8 | 12 | 54.616 | 0.99993946 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 12 | 1 | 98.092 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 12 | 2 | 68.214 | 0.99994372 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 12 | 4 | 60.072 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 12 | 8 | 56.879 | 0.99995500 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 16 | 1 | 98.552 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 16 | 2 | 68.975 | 0.99994372 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 16 | 3 | 63.535 | 0.99995177 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 16 | 6 | 58.443 | 0.99995860 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 24 | 1 | 98.979 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 24 | 2 | 69.195 | 0.99994372 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 24 | 4 | 60.999 | 1.00000000 | True | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 1 | 98.982 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 32 | 3 | 63.893 | 0.99995177 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 48 | 1 | 99.751 | 0.99990855 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 48 | 2 | 70.258 | 0.99994372 | False | True |
| geometry_layer3.json | dram | BFP4 (historical) | 3 | down_proj | 3072 × 4096 | 96 | 1 | 104.799 | 0.99990855 | False | True |
| geometry_qkvg8_dram4_b32_corrected_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 32 | 47.714 | 0.99994194 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 1 | 98.520 | 0.99994498 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 2 | 56.595 | 0.99996197 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 4 | 44.269 | 0.99997529 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 8 | 44.585 | 0.99998327 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 16 | 45.648 | 1.00000000 | True | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 8 | 1 | 98.647 | 0.99994498 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 8 | 2 | 56.848 | 0.99996197 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 8 | 4 | 44.667 | 0.99997529 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 8 | 8 | 45.112 | 0.99998327 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 8 | 16 | 46.186 | 1.00000000 | True | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 16 | 1 | 98.769 | 0.99994498 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 16 | 2 | 57.027 | 0.99996197 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 16 | 4 | 44.945 | 0.99997529 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 16 | 8 | 45.366 | 0.99998327 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 1 | 96.532 | 0.99994498 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 2 | 54.684 | 0.99996197 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 4 | 42.535 | 0.99997529 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 1 | 100.472 | 0.99994498 | False | True |
| geometry_qkvg8_dram_capture_v2_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 2 | 58.822 | 0.99996197 | False | True |
| geometry_qkvg8_dram_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 1 | 98.514 | 0.99994498 | False | True |
| geometry_qkvg8_dram_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 2 | 56.638 | 0.99996197 | False | True |
| geometry_qkvg8_dram_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 4 | 44.269 | 0.99997529 | False | True |
| geometry_qkvg8_dram_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 8 | 44.576 | 0.99998327 | False | True |
| geometry_qkvg8_dram_layer3.json | dram | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 4 | 16 | 45.859 | 1.00000000 | True | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 1 | 173.418 | 0.99995815 | False | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 2 | 94.537 | 0.99997682 | False | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 4 | 56.636 | 1.00000000 | True | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 8 | 45.307 | 0.99998598 | False | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 16 | 46.213 | 0.99997529 | False | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 32 | 48.130 | 0.99994194 | False | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 64 | 49.473 | 0.99986102 | False | True |
| geometry_qkvg8_grid110_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 110 | 128 | 49.987 | 0.99986102 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 1 | 127.024 | 0.99995815 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 2 | 71.368 | 0.99997682 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 4 | 47.261 | 1.00000000 | True | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 8 | 48.725 | 0.99998598 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 16 | 49.963 | 0.99997529 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 32 | 50.276 | 0.99994194 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 64 | 49.200 | 0.99986102 | False | True |
| geometry_qkvg8_grid64_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 64 | 128 | 50.270 | 0.99986102 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 1 | 114.729 | 0.99994498 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 2 | 66.239 | 0.99996197 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 4 | 50.400 | 0.99997529 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 8 | 49.427 | 0.99998327 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 16 | 50.211 | 1.00000000 | True | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 32 | 50.829 | 0.99997311 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 64 | 53.670 | 0.99990516 | False | True |
| geometry_qkvg8_wide_capture_v2_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 128 | 53.753 | 0.99990516 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 1 | 114.785 | 0.99994498 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 2 | 66.241 | 0.99996197 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 4 | 50.355 | 0.99997529 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 8 | 49.489 | 0.99998327 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 16 | 50.160 | 1.00000000 | True | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 32 | 50.855 | 0.99997311 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 64 | 52.664 | 0.99990516 | False | True |
| geometry_qkvg8_wide_layer3.json | interleaved | DataType.BFLOAT8_B | 3 | qkvg | 4096 × 2560 | 32 | 128 | 53.703 | 0.99990516 | False | True |
