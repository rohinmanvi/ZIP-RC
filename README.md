# ZIP-RC
Official repository for ["Zero-Overhead Introspection for Adaptive Test-Time Compute"](https://rohinmanvi.github.io/ZIP-RC/).

ZIP-RC equips an LLM with zero-overhead introspective predictions of a joint distribution over:
(i) final reward (e.g., correctness) and (ii) remaining generation length.

It does this by reserving a contiguous slice of vocabulary logits for an auxiliary head and reading those logits in the same forward pass used for next-token prediction. During decoding, those reserved tokens must be masked so they are never sampled.

<p align="center">
  <a href="assets/ZIP-RC.jpg">
    <img src="assets/ZIP-RC.jpg" width="900" alt="ZIP-RC overview: zero-overhead joint reward-cost prediction via reserved vocabulary logits." />
  </a>
</p>

## Links
- Project page: https://rohinmanvi.github.io/ZIP-RC/
- Paper (PDF): https://arxiv.org/pdf/2512.01457.pdf
- arXiv: https://arxiv.org/abs/2512.01457

## Quickstart
Create the conda environment:

```bash
conda env create -f environment.yml
conda activate zip
```

Run a tiny end-to-end smoke test (downloads Hugging Face models/datasets; GPU required):

```bash
bash scripts/smoke_test.sh
```

For the full training pipeline + script flags, see `docs/pipeline.md`.

## Citation
If you find ZIP-RC useful, please cite:
```bibtex
@misc{manvi2025zerooverheadintrospectionadaptivetesttime,
      title={Zero-Overhead Introspection for Adaptive Test-Time Compute},
      author={Rohin Manvi and Joey Hong and Tim Seyde and Maxime Labonne and Mathias Lechner and Sergey Levine},
      year={2025},
      eprint={2512.01457},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.01457},
}
```
