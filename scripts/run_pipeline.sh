#!/usr/bin/env bash
# FinAlign 全流程：数据 → CPT → LoRA SFT → on-policy 负例挖掘 → DPO → 评测 → 合并
#   PROFILE=gpu bash scripts/run_pipeline.sh    全量配置（单卡 GPU）
#   PROFILE=cpu bash scripts/run_pipeline.sh    CPU 缩量配置（本仓库 README 中的结果即用此配置跑出）
# hf 后端（Qwen3-8B-Base）：BACKEND_ARGS="--backend hf --hf_path /path/to/Qwen3-8B-Base --gradient_checkpointing 1" PROFILE=gpu bash scripts/run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PROFILE=${PROFILE:-gpu}
BACKEND_ARGS=${BACKEND_ARGS:-}
mkdir -p logs out results

if [ "$PROFILE" = "cpu" ]; then
  CPT_ARGS="--batch_size 16 --max_steps 120 --learning_rate 2e-4"
  # SFT 分两段走完 13,600 条会话：第一段 6,000 条后先评测一次（README 中的 sft_6k），第二段从该 LoRA 续训剩余 7,600 条
  SFT1_ARGS="--batch_size 16 --epochs 1 --max_samples 6000 --learning_rate 5e-4"
  SFT2_ARGS="--batch_size 16 --epochs 1 --skip_samples 6000 --learning_rate 3e-4 --group_by_length 1"
  MINE_ARGS="--num_prompts 1200 --samples 2 --temperature 0.8"
  DPO_ARGS="--batch_size 8 --learning_rate 3e-5 --beta 0.1 --sft_alpha 1.0"
else
  CPT_ARGS="--batch_size 32 --epochs 1 --learning_rate 2e-4"
  SFT_ARGS="--batch_size 32 --epochs 2 --learning_rate 5e-4 --group_by_length 1"
  MINE_ARGS="--num_prompts 2000 --samples 2 --temperature 0.8"   # 剩余 2,000 个 prompt 的规则扰动对用于混合
  DPO_ARGS="--batch_size 16 --epochs 1 --learning_rate 3e-5 --beta 0.1 --sft_alpha 1.0"
fi
if [[ "$BACKEND_ARGS" == *"--backend hf"* ]]; then
  BASE=base       # hf 后端从 --hf_path 指定的基座出发
else
  BASE=out/pretrain_768.pth
  [ -f "$BASE" ] || curl -L -o "$BASE" https://www.modelscope.cn/models/gongjy/minimind-3-pytorch/resolve/master/pretrain_768.pth
fi

# ---------- 1. 数据 ----------
if [ ! -f data/dpo/dpo_fin.jsonl ]; then
  python data_pipeline/synth_announcements.py --num_companies 300 --out data/raw/announcements.jsonl
  python data_pipeline/clean_dedup.py --in data/raw/announcements.jsonl --out data/processed/docs.jsonl
  python data_pipeline/build_chunks.py --docs data/processed/docs.jsonl
  python data_pipeline/build_cpt.py --out data/cpt/pretrain_fin.jsonl
  python data_pipeline/build_qa.py --mode sft --num 13600 --out data/sft/sft_fin.jsonl
  python data_pipeline/build_qa.py --mode dev --num 300 --out data/sft/dev_fin.jsonl
  python data_pipeline/build_qa.py --mode eval --num 500 --num_refusal 50 --out data/eval/finqa_test.jsonl
  python data_pipeline/build_dpo.py --num 4000 --out data/dpo/dpo_fin.jsonl
fi

# ---------- 2. 领域继续预训练 ----------
python trainer/train_pretrain.py $BACKEND_ARGS --from_weight $BASE --save_weight cpt_fin $CPT_ARGS 2>&1 | tee logs/cpt.log

# ---------- 3. LoRA SFT + 评测 ----------
if [ "$PROFILE" = "cpu" ]; then
  python trainer/train_lora.py $BACKEND_ARGS --from_weight cpt_fin --lora_name lora_sft6k $SFT1_ARGS 2>&1 | tee logs/sft_6k.log
  python evaluation/eval_finqa.py $BACKEND_ARGS --from_weight cpt_fin --loras lora_sft6k --tag sft_6k 2>&1 | tee logs/eval_sft_6k.log
  python trainer/train_lora.py $BACKEND_ARGS --from_weight cpt_fin --init_lora lora_sft6k --lora_name lora_sft $SFT2_ARGS 2>&1 | tee logs/sft_stage2.log
else
  python trainer/train_lora.py $BACKEND_ARGS --from_weight cpt_fin --lora_name lora_sft $SFT_ARGS 2>&1 | tee logs/sft.log
fi
python evaluation/eval_finqa.py $BACKEND_ARGS --from_weight cpt_fin --loras lora_sft --tag sft 2>&1 | tee logs/eval_sft.log

# ---------- 4. on-policy 负例挖掘：SFT 模型采样判错的回答作为 rejected，与等量规则扰动偏好对混合 ----------
python scripts/mine_policy_negatives.py $BACKEND_ARGS --from_weight cpt_fin --loras lora_sft $MINE_ARGS \
  --out data/dpo/dpo_fin_onpolicy.jsonl --mix_out data/dpo/dpo_fin_mix.jsonl 2>&1 | tee logs/mine.log

# ---------- 5. DPO（SFT 权重初始化策略模型，冻结 SFT 副本为参考模型，chosen NLL 正则）+ 评测 ----------
python trainer/train_dpo.py $BACKEND_ARGS --from_weight cpt_fin --sft_lora lora_sft --lora_name lora_dpo \
  --data_path data/dpo/dpo_fin_mix.jsonl $DPO_ARGS 2>&1 | tee logs/dpo.log
python evaluation/eval_finqa.py $BACKEND_ARGS --from_weight cpt_fin --loras lora_sft,lora_dpo --tag sft_dpo 2>&1 | tee logs/eval_dpo.log

# ---------- 6. 适配器合并（推理部署 / RAG 集成） ----------
python scripts/merge_lora.py $BACKEND_ARGS --from_weight cpt_fin --loras lora_sft,lora_dpo --save_weight finalign
python scripts/compare_results.py results/sft results/sft_dpo | tee results/summary.md
