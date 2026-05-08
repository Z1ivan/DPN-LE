# DPN-LE

Official implementation of **DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models**.

DPN-LE is a training-free method for personality control in LLMs. It contrasts MLP activations from high-trait and low-trait samples, selects sparse trait-exclusive neurons with Cohen's d and activation-magnitude filtering, and applies inference-time interventions at the input of `down_proj`.

> **⚠️ Note**: Due to server migration, some code or data files may be missing or contain errors. If you encounter any issues, please don't hesitate to open an issue or contact us. We're actively maintaining this repository and will address problems promptly.

## Install

```bash
cd DPN-LE
pip install -e .
```

The package dependencies include `xlrd` and `openpyxl` for Excel-based IPIP-NEO item keys.
For paper-aligned general-capability experiments, install the optional vLLM extra:

```bash
pip install -e '.[vllm]'
```

For GPT-based PersonalityBench scoring, install the optional scoring extra:

```bash
pip install -e '.[scoring]'
```

## Repository Layout

```text
src/dpn_le/                         # reusable DPN-LE package
scripts/build_steering_data.py      # activation extraction + steering-data construction
experiments/personality_bench/      # PersonalityBench generation and GPT-based scoring
experiments/ipip_neo/               # IPIP-NEO-300 single-trait inference/evaluation
experiments/general_capability/     # DPN-LE capability checks on GSM8K/HotpotQA/TriviaQA
experiments/npti_general_capability/# preliminary NPTI capability checks
data/personalitybench/              # NPTI/PersonalityBench data used for neuron search/test
data/ipip_neo/                      # PAPI/IPIP-NEO-300 test files
data/general_capability/            # benchmark subsets used by capability scripts
data/npti_neuron_results/           # NPTI neuron dictionaries for the NPTI baseline
```

## Data Preparation

### PersonalityBench Dataset

The PersonalityBench dataset is **included** in this repository at `data/personalitybench/`:

```text
data/personalitybench/
├── description.json       # 80 high-trait + 80 low-trait descriptions per trait
├── search/                # ~36,000 questions per trait for neuron search
│   ├── Agreeableness.json
│   ├── Conscientiousness.json
│   ├── Extraversion.json
│   ├── Neuroticism.json
│   └── Openness.json
└── test/                  # ~89-100 test questions per trait
    ├── Agreeableness.json
    ├── Conscientiousness.json
    ├── Extraversion.json
    ├── Neuroticism.json
    └── Openness.json
```

**Citation**: If you use PersonalityBench, please cite the NPTI paper:

```bibtex
@inproceedings{deng2025neuron,
  title={Neuron based personality trait induction in large language models},
  author={Deng, Jia and Tang, Tianyi and Yin, Yanbin and Zhao, Xin and Wen, Ji-Rong and others},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={85059--85083},
  year={2025}
}
```

### Steering Data

You need to generate steering data before running experiments. See [data/steering_data/README.md](data/steering_data/README.md) for detailed instructions.

**Quick start** (requires GPU):

To rebuild it from PersonalityBench yourself (requires GPU):

```bash
python scripts/build_steering_data.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --personalitybench_dir data/personalitybench \
  --output_dir data/steering_data/llama3_8b \
  --traits all \
  --num_samples 1000 \
  --batch_size 8 \
  --use_chat_template
```

This generates steering vectors and neuron indices for all 5 Big Five traits:

```text
data/steering_data/llama3_8b/
├── activations/           # Intermediate MLP activations (can be deleted after)
└── steering_data/         # Final steering data (required for inference)
    ├── layer12_Agreeableness_steering_data.pt
    ├── layer12_Agreeableness_steering_data.json
    └── ... (5 traits × 20 layers = 100 layer-trait configs)
```

**Time estimate**: 2-4 hours on a single GPU (16GB+ VRAM) for all traits.

### IPIP-NEO-300 Data

The IPIP-NEO-300 files required by the generalization experiment are included:

```text
data/ipip_neo/
├── Test-set.json
├── mpi_300_split.json
└── IPIP-NEO-ItemKey.xls
```

These files are adapted from the public PAlign/PAPI release. The included
test set uses de-identified `case` ids and questionnaire/demographic fields
from the source release; it does not contain names, emails, API keys, or local
machine paths. If you use the PAPI/IPIP data, also cite the PAlign paper:

```bibtex
@inproceedings{zhu2025personality,
  title={Personality alignment of large language models},
  author={Zhu, Minjun and Weng, Yixuan and Yang, Linyi and Zhang, Yue},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={14206--14255},
  year={2025}
}
```

The general-capability evaluation files used in the paper are also included:

```text
data/general_capability/
├── gsm8k_test.json
├── hotpotqa_validation.json
└── triviaqa_validation.json
```

The NPTI neuron dictionaries used by the paper's preliminary NPTI capability
experiment are included at `data/npti_neuron_results/`. They are copied from
the original NPTI release and are used only by
`experiments/npti_general_capability/evaluate_vllm.py`.

## Use In Python

```python
from dpn_le import DPNLEInference, get_model_config

model = "meta-llama/Meta-Llama-3-8B-Instruct"
inference = DPNLEInference(model, get_model_config(model))

inference.apply_steering(
    "data/steering_data/llama3_8b/steering_data",
    trait="Neuroticism",
    gamma=1.0,
    direction="increase",
    method="weighted",
    neuron_mode="both",
)

answers = inference.generate_questions([
    "How do you usually react when something goes wrong?"
])
```

`method="linear"` is DPN-LE. `method="weighted"` is DPN-LE_w.
`neuron_mode="both"` uses high- and low-exclusive neurons together and flips the vector for `decrease`, matching the paper and general-capability experiments. `neuron_mode="directional"` uses high neurons for `increase` and low neurons for `decrease`, matching the IPIP-NEO single-trait script.

### Other Models

Built-in configs are provided for LLaMA-3-8B-Instruct and Qwen2.5-7B-Instruct.
For other decoder-only models with an MLP `down_proj` input, pass a custom
configuration:

```python
from dpn_le import DPNLEInference, create_custom_config

config = create_custom_config(
    model_name="your/model",
    num_layers=32,
    intermediate_size=11008,
    separation_start_layer=12,
)
inference = DPNLEInference("your/model", config)
```

## Experiments

PersonalityBench generation:

```bash
python experiments/personality_bench/evaluate.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data data/personalitybench/test/Neuroticism.json \
  --steering_data_dir data/steering_data/llama3_8b/steering_data \
  --trait Neuroticism \
  --direction increase \
  --method weighted \
  --neuron_mode both \
  --gamma 1.0 \
  --output outputs/personalitybench_neuroticism.json
```

IPIP-NEO-300 single-trait evaluation:

```bash
python experiments/ipip_neo/evaluate.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --steering_data_dir data/steering_data/llama3_8b/steering_data \
  --data_dir data/ipip_neo \
  --output outputs/ipip_neo_weighted.json \
  --method weighted \
  --gamma 1.0 \
  --quantile 0.995 \
  --cohens_d 0.8 \
  --threshold 2.8 \
  --neuron_mode directional \
  --use_chat_template
```

General capability evaluation:

```bash
python experiments/general_capability/evaluate_vllm.py \
  --benchmark gsm8k \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --steering_data_dir data/steering_data/llama3_8b/steering_data \
  --trait Neuroticism \
  --direction increase \
  --method weighted \
  --neuron_mode both \
  --gamma 0.8 \
  --output outputs/gsm8k_neuroticism.json
```

HotpotQA and TriviaQA use the same arguments with
`--benchmark hotpotqa` and `--benchmark triviaqa`. The script uses the included
benchmark files by default: GSM8K full test set, and the first 1,000 examples
for HotpotQA and TriviaQA.

NPTI preliminary general-capability evaluation:

```bash
python experiments/npti_general_capability/evaluate_vllm.py \
  --benchmark gsm8k \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --neuron_dir data/npti_neuron_results \
  --trait Neuroticism \
  --direction increase \
  --gamma 1.4 \
  --output outputs/npti_gsm8k_neuroticism.json
```

The evaluation entry points are separated intentionally:

- `experiments/personality_bench/evaluate.py` generates DPN-LE responses for
  PersonalityBench.
- `experiments/personality_bench/score_with_gpt.py` reproduces the NPTI-style
  GPT scoring for trait strength and fluency.
- `experiments/ipip_neo/evaluate.py` runs the IPIP-NEO-300 single-trait
  protocol. It uses the 120 train items only to estimate a subject's trait
  direction and performs inference on held-out items; it does not train model
  weights or probes.
- `experiments/general_capability/evaluate_vllm.py` evaluates DPN-LE on
  GSM8K, HotpotQA, and TriviaQA with the same metrics as the NPTI-style
  capability checks.
- `experiments/npti_general_capability/evaluate_vllm.py` runs the preliminary
  NPTI baseline using the included NPTI neuron dictionaries.

## Data Notes

PersonalityBench data should be cited with NPTI, and IPIP/PAPI data should be
cited with PAlign/PAPI as shown above. The IPIP inventory itself is public
domain, but the bundled test responses come from the public PAlign/PAPI test
release and should retain that attribution.

## Citation

```bibtex
@article{zheng2026dpn,
  title={DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models},
  author={Zheng, Lifan and Yang, Xue and Chen, Jiawei and Wu, Chenyan and Zhang, Jingyuan and Kong, Fanheng and Zeng, Xinyi and Chen, Xiang and Tian, Yu},
  journal={arXiv preprint arXiv:2604.27929},
  year={2026}
}
```

## License

MIT.
