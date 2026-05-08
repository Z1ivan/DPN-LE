# Steering Data

This directory is intentionally empty in the source release. DPN-LE steering
data depends on the target model, trait, selected layers, and neuron-selection
thresholds, so it should be generated locally.

Build steering data from the included PersonalityBench files:

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

The generated layout is:

```text
data/steering_data/<model_id>/
├── activations/
└── steering_data/
    ├── layer12_Agreeableness_steering_data.pt
    ├── layer12_Agreeableness_steering_data.json
    └── ...
```

The `.pt` files are required for inference and evaluation. The activation
caches under `activations/` are intermediate files and can be deleted after
the steering data has been generated.
