# Changelog

All notable changes to FinAlign are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-30

First release: the full pipeline from announcements to a merged, served model.

### Added

- **Announcement data.** An offline generator of fictional listed companies' periodic reports, earnings forecasts, dividend plans and buyback updates for 2021 to 2024, each with structured facts for exact grading; and a cninfo crawler that downloads real PDFs and turns the key-figures table into the same row format and facts.
- **Cleaning and splits.** Full-width and thousands-separator normalisation, page-header removal, MD5 exact deduplication, and MinHash-LSH near-duplicate detection that also requires the two documents to disclose the same figures. Companies are split into train, dev and test; periods into history and recent.
- **Training data.** A continued-pretraining corpus, 13,600 multi-turn SFT conversations that put the question, three retrieved evidence chunks and a cited answer together, and 4,000 preference pairs whose rejected answers carry one period, unit, number, citation or unsupported-answer error checked by the grader.
- **Training.** Continued pretraining, LoRA SFT with loss on assistant turns only, and DPO from the merged SFT weights against a frozen SFT copy (or the same model with its adapters switched off), with an optional NLL term on the chosen answer. `--backend hf` runs every stage on a transformers model such as Qwen3-8B-Base.
- **On-policy negatives.** `scripts/mine_policy_negatives.py` samples the SFT model on preference prompts and keeps answers the grader marks wrong, mixed with an equal number of rule-based pairs. Each mixed pair gets its evidence in a random order with the citations renumbered to match, so the position of a citation carries no preference signal (`--permute_evidence 0` keeps the original order).
- **Evaluation.** Numeric, unit, period, citation and refusal accuracy on 500 answerable and 50 unanswerable questions about unseen companies' 2024 reports, with numeric errors traced to evidence from other periods.
- **Serving.** Adapter merging and an OpenAI-compatible API that retrieves announcement evidence itself or takes evidence passed by an upstream RAG pipeline.
- `--group_by_length` batches SFT samples of similar length; `--init_lora` and `--skip_samples` continue SFT from an adapter over the rest of the data.
