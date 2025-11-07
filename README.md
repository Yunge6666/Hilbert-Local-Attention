Pytorch version should be >= 2.7.0!

**attn_reshape.ipynb** includes timing tests. After installing the required libraries, you can run it directly on a GPU. Because FlexAttention support varies across different hardware architectures, different GPUs may achieve different acceleration results. Furthermore, due to the volatility of CUDA timing, it is recommended not to rely solely on single measurements.

Pretrained weighs for HNT-mini and HWT-tiny are listed below:

| Image resolution | Model    | Top-1 Acc % | Top-5 Acc % | Checkpoint                                                                                        |
| ---------------- | -------- | ----------- | ----------- | ------------------------------------------------------------------------------------------------- |
| 224*224          | HWT-tiny | 81.0        | 95.5        | [GoogleDrive](https://drive.google.com/file/d/1utPyBWCc-wKIg7GMdLuvB940Zw2_2BKx/view?usp=drive_link) |
| 256*256          | HWT-tiny | 81.5        | 95.7        | [GoogleDrive](https://drive.google.com/file/d/1WYLXTQNh9d27EJ_fBZaq1a2Nq0ltv_A-/view?usp=drive_link) |
| 224*224          | HNT-mini | 81.6        | 95.6        | [GoogleDrive](https://drive.google.com/file/d/1uEflG5N_QhH_nd8if4Be-eQbOBbWF83_/view?usp=drive_link) |
