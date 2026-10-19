# License scope and acknowledgements

The root [MIT license](LICENSE), copyright (c) 2026 **Boyu Teng**, applies to project-owned code in `scripts/`, `matlab/`, `baseline/`, project tests, configuration, documentation and reference results. The associated paper is by **Boyu Teng and Xiaojun Yuan**. Existing attribution comments are retained.

**This distribution includes dependencies with different terms. It is not an entirely MIT-licensed distribution.** In particular, the bundled QuaDRiGa implementation and NVIDIA-derived operations carry noncommercial restrictions. The root MIT license does not replace those terms or grant rights in upstream third-party material. MATLAB and CUDA toolkits are external dependencies and are not distributed here.

| Component | Attribution and license | Scope |
| --- | --- | --- |
| Score-SDE / NCSN++ | Copyright 2020 The Google Research Authors; [Apache-2.0](licenses/Apache-2.0.txt). [Upstream](https://github.com/yang-song/score_sde_pytorch) | `third_party/score_sde/sde_1st` and `sde_2nd`, except the specific files below |
| Exponential moving average | Samuel G. Fadel, 2019; [MIT](licenses/pytorch_ema-MIT.txt). [Upstream](https://github.com/fadel/pytorch_ema) | `models/ema.py` in both trees; existing TensorFlow/Google-derived comments are also retained |
| StyleGAN2 PyTorch operation wrappers | Kim Seonghyeon, 2019; [MIT](licenses/stylegan2-pytorch-MIT.txt). [Upstream](https://github.com/rosinality/stylegan2-pytorch) | Python wrappers in both `op/` directories, subject to the NVIDIA-derived component terms below |
| NVIDIA StyleGAN2 kernels and ported resampling functions | NVIDIA Corporation, 2019; [NVIDIA Source Code License-NC](licenses/NVIDIA-Source-Code-License-NC.txt). [Upstream license](https://nvlabs.github.io/stylegan2/license.html) | `op/*.cu`, related operation code, and the functions identified as ported from StyleGAN2 in `models/up_or_down_sampling.py` |
| Third-Party Modified Version of the Fraunhofer QuaDRiGa | Fraunhofer-Gesellschaft / Fraunhofer HHI, 2011–2019; [original complete license](third_party/quadriga_src/QuaDRiGa_License.txt) | `third_party/quadriga_src/`; version 2.6.1-0 snapshot and project channel adaptations; noncommercial use conditions apply |

Score-SDE research credit: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon and Ben Poole, *Score-Based Generative Modeling through Stochastic Differential Equations*. Architecture/loss source comments remain in place.

## Project modifications

Packaging dated **2026-10-18** preserves the experimental NCSN++ architecture, the first- and second-order loss computations, and the CUDA kernel implementations. Configuration files now load the portable channel settings. `op/fused_act.py` and `op/upfirdn2d.py` defer extension compilation until a CUDA operation; their original CPU implementation is used for CPU checks. Both modifications are marked in the files.

The included QuaDRiGa tree is designated **Third-Party Modified Version of the Fraunhofer QuaDRiGa**. It contains the project snapshot with the custom aggregate CDL channel generation functions. On 2026-10-18 it was packaged with these experiment-specific adaptations and this notice; unrelated experiment entry points were omitted. No additional numerical changes were made during packaging. Earlier project modification dates are not recorded by the source snapshot. The required directory name `quadriga_src` is preserved. See its local `NOTICE.md` and the source manifest.

## Non-official MDVALSE implementation

The code provider, Boyu Teng, identifies `baseline/M2VALSE_tby2.m` as their **non-official implementation** used in these experiments. It is included as project code under the root MIT license. Its existing header crediting Qi Zhang and Jiang Zhu and its reference to Mihai-Alin Badiu is preserved verbatim (including the historical spelling in that header). This is not an official release of those authors' repository and does not assert an MIT license for their separate upstream code.

Algorithm references: [MDVALSE by Qi Zhang, Jiang Zhu, Ning Zhang and Zhiwei Xu](https://arxiv.org/abs/1912.11759); [authors' MDVALSE repository](https://github.com/RiverZhu/MDVALSE); VALSE by Mihai-Alin Badiu, Thomas Lundgaard Hansen and Bernard Henri Fleury, [DOI 10.1109/TSP.2017.2655489](https://doi.org/10.1109/TSP.2017.2655489).

The bundled full license texts and original notices govern their respective components. No author or institution endorses this packaging by being credited here.
