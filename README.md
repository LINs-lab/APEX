<h1 align="center"><sub><sup>Self-Adversarial One Step Generation via Condition Shifting</sup></sub></h1>

<p align="center">
  <a href="https://zhenglin-cheng.com/" target="_blank">Deyuan&nbsp;Liu</a><sup>*</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=-8XvRRIAAAAJ" target="_blank">Peng&nbsp;Sun</a><sup>*</sup> &ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Yansen&nbsp;Han</a> &ensp; <b>&middot;</b> &ensp;
  <a href="https://zhenglin-cheng.com/" target="_blank">Zhenglin&nbsp;Cheng</a> &ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Chuyan&nbsp;Chen</a> &ensp; <b>&middot;</b> &ensp;
  <a href="https://lins-lab.github.io/" target="_blank">Tao&nbsp;Lin</a>
</p>

<div align="center">

[![Github Repo](https://img.shields.io/badge/LINs--lab%2FAPEX-black?logo=github)](https://github.com/LINs-lab/APEX)&#160;
[![Github Repo](https://img.shields.io/badge/inclusionAI%2FTwinFlow-black?logo=github)](https://github.com/inclusionAI/TwinFlow)&#160;
<a href="https://arxiv.org/abs/2604.xxxxx" target="_blank"><img src="https://img.shields.io/badge/Paper-b5212f.svg?logo=arxiv" height="21px"></a>
[![WeChat](https://img.shields.io/badge/WeChat-green?logo=wechat&amp&color=white)](./assets/wechat.png)

</div>

**Join the WeChat Group, feel free to reach out anytime if you have any questions!👇**

<p>👇 WeChat Group QR Code/微信群二维码 👇</p>

- Technical Discussion Group 1 is full, please join Technical Discussion Group 2

| Technical Discussion Group/技术讨论群 | Model Users Discussion Group/AIGC模型使用讨论群 |
|----------------------------|------------------------------------------|
| <img src="./assets/wechat.png" style="width: 70%;" /> | <img src="./assets/wechat2.png" style="width: 70%;" /> |

</details>

## 🧭 Table of Contents

- [🔥 Codebase Usage 🔥](src/README.md)

## 📰 News

- We release experimental version of faster Z-Image-Turbo!

## 💪 Open-source Plans

- [x] Release faster experimental version of Z-Image-Turbo.
- [ ] Release APEX-v0 version of Z-Image.
- [ ] Release APEX-v0 version of Qwen-Image-2512.
- [ ] Release large-scale training code.

## APEX

### APEX-Qwen-Image-2512 Visualization

<div align="center">
  <img src="assets/apex.jpg" width="1000" />
  <p style="margin-top: 8px; font-size: 14px; color: #666; font-weight: bold;">
    2-NFE visualization of Qwen-Image-2512
  </p>
</div>


### Inference Demo

**For ComfyUI users, please see https://github.com/smthemex/ComfyUI_TwinFlow.**

Install the latest diffusers:

```bash
pip install git+https://github.com/huggingface/diffusers
```

Run inference demo `inference.py`:

```python
python inference.py
```

We recommend to sample for 2~4 NFEs:

```python
# 4 NFE config
sampler_config = {
    "sampling_steps": 4,
    "stochast_ratio": 1.0,
    "extrapol_ratio": 0.0,
    "sampling_order": 1,
    "time_dist_ctrl": [1.0, 1.0, 1.0],
    "rfba_gap_steps": [0.001, 0.5],
}

# 2 NFE config
sampler_config = {
    "sampling_steps": 2,
    "stochast_ratio": 1.0,
    "extrapol_ratio": 0.0,
    "sampling_order": 1,
    "time_dist_ctrl": [1.0, 1.0, 1.0],
    "rfba_gap_steps": [0.001, 0.6],
}
```

## 📖 Citation

```bibtex
@misc{liu2026selfadversarialstepgenerationcondition,
      title={Self-Adversarial One Step Generation via Condition Shifting}, 
      author={Deyuan Liu and Peng Sun and Yansen Han and Zhenglin Cheng and Chuyan Chen and Tao Lin},
      year={2026},
      eprint={2604.12322},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.12322}, 
}


@article{cheng2025twinflow,
  title={TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows},
  author={Cheng, Zhenglin and Sun, Peng and Li, Jianguo and Lin, Tao},
  journal={arXiv preprint arXiv:2512.05150},
  year={2025}
}

@misc{sun2025anystep,
  author = {Sun, Peng and Lin, Tao},
  note   = {GitHub repository},
  title  = {Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation},
  url    = {https://github.com/LINs-lab/RCGM},
  year   = {2025}
}

@article{sun2025unified,
  title = {Unified continuous generative models},
  author = {Sun, Peng and Jiang, Yi and Lin, Tao},
  journal = {arXiv preprint arXiv:2505.07447},
  year = {2025},
  url = {https://arxiv.org/abs/2505.07447},
  archiveprefix = {arXiv},
  eprint = {2505.07447},
  primaryclass = {cs.LG}
}
```

## 🤗 Acknowledgement

APEX is built upon [TwinFlow](https://github.com/inclusionAI/TwinFlow), [RCGM](https://github.com/LINs-lab/RCGM) and [UCGM](https://github.com/LINs-lab/UCGM).

Note: The [LINs Lab](https://lins-lab.github.io/) has openings for PhD students for the Fall 2026/2027 intake. Interested candidates are encouraged to reach out.
