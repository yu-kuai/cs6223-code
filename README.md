# Can an AI find its own mistakes?

CS6223 course project. An AI model **tests itself**: it writes small Python programs that change a test question
in a way that should *not* change the right answer, then asks itself the changed question. If its answer changes,
that is a failure (a bug in the model).

Example (Qwen3-32B, MMLU):

| | Question | Options | Model answer |
|---|---|---|---|
| Original | The complex proposition fallacy involves … | A = *including more than one claim …* (correct) | **A** ✔ |
| Changed (options shuffled) | same question | the correct option is now **C** | **A** ✘ |

The model kept the letter, not the content. That is a real failure, found automatically.

* Text: Qwen3 **1.7B / 8B / 32B** on **MMLU** (1,140 questions)
* Images: Qwen3-VL **2B / 8B / 32B** on **ScienceQA** image questions (2,017 questions)
* The tested model is also the one that writes the change programs ("self-testing").

Paper (LaTeX): https://github.com/yu-kuai/cs6223-report. This repository holds the code only.

## How the loop works

```
 seed programs ──► model writes / repairs change programs (Python, run in a sandbox)
                         │
                         ▼
        change a question ──► fixed checks (code) ──► AI judge (GLM): "is the answer still the same?"
                         │
                         ▼
        ask the model the original and changed question ──► answer changed? = failure found
                         │
                         ▼
        feedback: programs that find real failures get more tests; programs that make
        unfair changes get repaired or retired
```

The programs **never see the correct answer** (so they can't cheat by changing it on purpose).

## Main results (3 runs each, 5,000 model calls per run)

Number of **different questions** with a real failure, checked by a second AI grader (DeepSeek).
"Static" = only our hand-written programs, no evolution. "Guarded" = our final self-evolving version.

| Model | Static | Guarded |
|---|---|---|
| Text 1.7B | 99 | 100 |
| Text 8B | 49 | 59 |
| Text 32B | 33 | 39 |
| Image 2B | 40 | ~20 * |
| Image 8B | 29 | ~13 * |
| Image 32B | 16 | ~37 * |

\* after hand-checking a sample: the grader accepted many images that were too damaged to answer.

Short version: the self-evolving loop finds a few more real failures on text, and clearly more on the
strongest image model (32B), but for images it also produces many "broken" tests (pictures destroyed by noise).
See the report for details.

## Repository layout

| Path | What |
|---|---|
| `src/fuzz_loop.py` | The main loop (program writing, repair, testing, feedback) |
| `src/prompts.py` | All prompts for the program writer and the judge |
| `src/sandbox.py`, `src/sandbox_worker.py` | Run model-written programs safely (time limit, no network) |
| `src/llm_client.py` | vLLM (local models) and API (GLM / DeepSeek) client, call budget |
| `src/judge_api_grade.py` | Second grader (DeepSeek) for all found failures |
| `src/compare_v2.py` | Builds the result tables (`--task vqa`, `--strict`) |
| `src/debatable.py` | Detects "trick" changes (double negation, broken-up question) |
| `src/build_*_label_sample.py` | Samples for spot-check labeling |
| `programs/seed*/` | Hand-written seed programs (text, image; `_hidden` = answer-free versions) |
| `scripts/` | Data preparation and Slurm job scripts |

## How to run

Needs a Slurm GPU cluster, vLLM, and an OpenAI-compatible API for the judge/grader.
The Slurm scripts contain the paths of our cluster (`PROJ` in `scripts/slurm/common.sh` and the `#SBATCH -o` log lines); change them to your own.

1. API settings (never commit this file):
   ```
   ~/.config/soclaas/soclaas.env
   SOCLAAS_BASE_URL=...
   SOCLAAS_API_KEY=...
   ```
2. Data: `python scripts/prepare_mmlu.py`, `python scripts/prepare_scienceqa.py`
3. Baseline answers: `scripts/slurm/10_baseline.sbatch`, `scripts/slurm/11_baseline_vqa.sbatch`
4. One fuzzing run ("Guarded" settings), e.g. 8B text, seed 0:
   ```
   RUN_SUFFIX=_v4 scripts/submit_fuzz.sh 8b full 5000 \
     --judge-api x-test-1 --hide-answer --fixed-checks --v4-checks --max-share 0.25 \
     --test-repairs --rich-repair --bold --max-programs 12 --seed 0
   ```
   Use `vl-2b / vl-8b / vl-32b` for images. Use strategy `fixed` for the Static baseline.
5. Grade: `python src/judge_api_grade.py <run_dir> --model x-test-2 --tag deepseek`
6. Tables: `python src/compare_v2.py`, `python src/compare_v2.py --strict`, `python src/compare_v2.py --task vqa`

Data, model weights, run logs and results are not in the repository (see `.gitignore`). Data is re-created in step 2; results in steps 4–6.
