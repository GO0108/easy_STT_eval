#!/bin/bash

MODELS=(
    # "whisper:openai/whisper-tiny"
    # "whisper:openai/whisper-base"
    # "whisper:openai/whisper-small"
    # "whisper:openai/whisper-medium"
    # "whisper:openai/whisper-large-v2"
    # "whisper:openai/whisper-large-v3"
    # "whisper:distil-whisper/distil-large-v2"
    "google:google_stt",
    "api_whisper:whisper-1"
)
DATASET_TYPE="huggingface"
# DATASET_NAME="parler-tts/mls_eng"
DATASET_NAME="hf-internal-testing/librispeech_asr_dummy"
SPLIT="validation"
AUDIO_COLUMN="audio"
ID_COLUMN="id"
OUTPUT_DIR="results"
DEVICE="cpu"

mkdir -p $OUTPUT_DIR

for MODEL in "${MODELS[@]}"; do
    echo "Testando modelo: $MODEL"
    python main.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $DATASET_NAME \
        --split $SPLIT \
        --audio_column $AUDIO_COLUMN \
        --id_column $ID_COLUMN \
        --output_dir $OUTPUT_DIR \
        --models $MODEL \
        --device $DEVICE
done

echo "Testes concluídos!"