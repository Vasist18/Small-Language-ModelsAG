# Evaluating Cost–Accuracy Trade-offs of Agentic Scaffolds for Small Language Models

Code and per-question outputs for the paper **"Evaluating Cost–Accuracy Trade-offs of Agentic Scaffolds for Small Language Models"** by Kuntapalli Vasist and G V S Sathwik.

We compare six inference strategies on a fixed random subset of 500 GSM8K test problems, using three instruction-tuned models from the Qwen2.5 family (0.5B, 1.5B and 3B), and report accuracy together with token cost.

## Contents

| File / folder | Description |
|---|---|
| `agentic_scaffold_cost_accuracy.ipynb` | Colab notebook that runs all experiments and the analysis (tables, figure, paired tests, error analysis). |
| `results/` | Per-question outputs, one JSON file per model and strategy, named `<model>__<strategy>.json` (for example `Qwen2.5-1.5B-Instruct__cot.json`). |
| `LICENSE` | License for the code. |

## Strategies

| Name in files | Strategy |
|---|---|
| `direct` | Final number only, no reasoning |
| `cot` | Chain-of-thought, greedy decoding |
| `pal` | Program-aided reasoning with a Python tool, greedy decoding |
| `pal_repair` | PAL with up to two repair rounds from execution errors |
| `cot_sc` | CoT self-consistency, five samples, majority vote |
| `pal_sc` | PAL self-consistency, five samples, majority vote |

The exploratory agree-gate is computed in the analysis cell from the logged `cot`, `pal` and `cot_sc` runs, so it needs no extra generation and has no separate file.

## Output format

Each JSON file is a list with one record per question. Records include the question index (`qid`), the gold answer (`gold`), the predicted answer (`pred`), whether it was correct (`correct`), and token counts (`prompt_tokens`, `gen_tokens`, `total_tokens`). Self-consistency records also include the individual sample answers (`votes`) and whether any sample was correct (`oracle`).

## Reproducing the results

**Requirements:** a Google Colab runtime with a **T4 GPU** and a free Kaggle account.

1. Open the notebook in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` under **Secrets** (key icon in the left sidebar). Otherwise you will be prompted to log in.
3. Optionally set `SMOKE = True` in the config cell for a quick pipeline test.
4. Select **Runtime → Run all**.

The notebook downloads GSM8K through `kagglehub` (`thedevastator/grade-school-math-8k-q-a`), runs the three models in turn, and saves a checkpoint after each strategy, so an interrupted run can resume.

Approximate T4 time for 500 questions across all six strategies: 0.5B about 25 minutes, 1.5B about 55 minutes, 3B about 2 hours.

**Settings used in the paper:** 500 questions drawn with seed 42; greedy decoding for single-sample strategies; temperature 0.7, top-p 0.95 and five samples for self-consistency; batch size 16; maximum new tokens 24 (Direct), 512 (CoT) and 320 (PAL and repair); up to two repair rounds.

**Software versions:** Python 3.13.15, PyTorch 2.11.0+cu130, transformers 5.18.0, kagglehub 1.0.2.

## Safety note

The notebook executes model-written Python code in a separate process with a 5-second time limit and a 2 GB memory limit. This is not a complete security boundary. Run it only in a disposable environment such as Colab, and never on a machine that holds secrets.

## Data and licensing

- GSM8K is released under the MIT license (Cobbe et al., 2021). The copy used here is the Kaggle mirror `thedevastator/grade-school-math-8k-q-a`.
- Model weights are obtained from Hugging Face (Qwen2.5-0.5B, 1.5B and 3B-Instruct) and are subject to their own licenses.
- The code in this repository is released under the license in `LICENSE`.

## Citation

If you use this work, please cite the paper and this repository.

```
Vasist, K., & Sathwik, G V S. Evaluating Cost–Accuracy Trade-offs of Agentic
Scaffolds for Small Language Models. [Journal/venue, year].
Code and data: [DOI].
```

## Contact

Kuntapalli Vasist — learningwithvashistkuntapalli@gmail.com
