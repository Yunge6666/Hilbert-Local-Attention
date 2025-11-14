# Overview📑 
### HILBERT-GUIDED BLOCK-SPARSE LOCAL ATTENTION     [<ins>Arkiv<ins>](https://arxiv.org/abs/2511.05832) 
This work proposes a novel method for constructing windows and neighborhoods based on the Hilbert curve. Image tokens are first reordered along a Hilbert curve, and windows and neighborhoods are then formed on the reordered 1D sequence. From a block-sparse perspective, this strategy significantly increases block sparsity and can be combined with existing block-sparse kernels to improve the efficiency of 2D local attention.

![Hilbert Local Attention Patterns](hilbertlocal.png "Hilbert local attention vs. Regular local attention")

Experiments show that the proposed Hilbert Window Attention and Hilbert Slide Attention can accelerate window attention and slide attention by about $4\times$ and $18\times$, respectively. To assess practicality, the strategy is instantiated as the Hilbert Window Transformer and the Hilbert Neighborhood Transformer, both of which achieve end-to-end speedups with minimal accuracy loss.

---

# Usage🛠️

Pytorch version should be >= 2.7.0!

**attn_reshape.ipynb** includes timing tests. After installing the required libraries, you can run it directly on a GPU. Because FlexAttention support varies across different hardware architectures, different GPUs may achieve different acceleration results. Furthermore, due to the volatility of CUDA timing, it is recommended not to rely solely on single measurements.

Pretrained weights for HWT-tiny are listed below:

| Image resolution | Model    | Top-1 Acc % | Top-5 Acc % | Checkpoint                                                                                        |
| ---------------- | -------- | ----------- | ----------- | ------------------------------------------------------------------------------------------------- |
| 224*224          | HWT-tiny | 81.0        | 95.5        | [GoogleDrive](https://drive.google.com/file/d/1utPyBWCc-wKIg7GMdLuvB940Zw2_2BKx/view?usp=drive_link) |
| 256*256          | HWT-tiny | 81.5        | 95.7        | [GoogleDrive](https://drive.google.com/file/d/1WYLXTQNh9d27EJ_fBZaq1a2Nq0ltv_A-/view?usp=drive_link) |

More pretrained weights are on the way...

---

# Acknowledgement🤝

We sincerely thank the following outstanding works for providing the foundation upon which our code is built.

* [FlexAttention](https://github.com/meta-pytorch/attention-gym)
* [Swin transformer](https://github.com/microsoft/Swin-Transformer)
* [NAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
