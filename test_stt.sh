#!/bin/bash

MODELS=(
    # "whisper:openai/whisper-tiny"
    # "whisper:openai/whisper-base"
    # "whisper:openai/whisper-small"
    # "whisper:openai/whisper-medium"
    # "whisper:openai/whisper-large-v2"
    # "whisper:openai/whisper-large-v3"
    # "whisper:distil-whisper/distil-large-v2"
    # "whisper:openai/whisper-large-v3-turbo"

    # "faster_whisper:tiny.en"
    # "faster_whisper:base.en"
    # "faster_whisper:small.en"
    # "faster_whisper:distil-small.en"
    # "faster_whisper:medium.en"
    # "faster_whisper:distil-medium.en"

    # "faster_whisper:tiny"
    # "faster_whisper:base"
    # "faster_whisper:small"
    # "faster_whisper:medium"
    # "faster_whisper:large-v2"
    # "faster_whisper:large-v3"
    # "faster_whisper:distil-large-v2"
    # "faster_whisper:large-v3-turbo"
    # "faster_whisper:turbo"

    # "whisper_s2t:tiny"
    # "whisper_s2t:base"
    # "whisper_s2t:small"
    # "whisper_s2t:medium"
    # "whisper_s2t:large-v2"
    # "whisper_s2t:large-v3"
    # "whisper_s2t:distil-large-v2"
    # "whisper_s2t:large-v3-turbo"
    # "whisper_s2t:turbo"

    # "wav2vec_conformer:facebook/wav2vec2-conformer-rel-pos-large"
    "google:google_stt"
    # "api_whisper:whisper-1"

    # "wav2vec:facebook/wav2vec2-base"
    # "wav2vec:facebook/wav2vec2-base-960h"
    # "wav2vec:facebook/wav2vec2-large"
    # "wav2vec:facebook/wav2vec2-large-960h"
    # "wav2vec:facebook/wav2vec2-large-xlsr-53"
    # "wav2vec:facebook/wav2vec2-xls-r-300m"

    # "nvidia:nvidia/canary-1b"
    # "nvidia:nvidia/parakeet-ctc-0.6b"
    # "nvidia:nvidia/parakeet-ctc-1.1b"
    # "nvidia:nvidia/parakeet-rnnt-0.6b"
    # "nvidia:nvidia/parakeet-rnnt-1.1b"
    # "nvidia:nvidia/parakeet-tdt-1.1b"
    # "nvidia:nvidia/parakeet-tdt_ctc-1.1b"
    # "nvidia:nvidia/parakeet-tdt_ctc-110m"
    # "nvidia:nvidia/stt_en_conformer_ctc_small"
    # "nvidia:nvidia/stt_en_conformer_ctc_large"
    # "nvidia:nvidia/stt_en_conformer_transducer_large"
    # "nvidia:nvidia/stt_en_conformer_transducer_xlarge"
    # "nvidia:nvidia/stt_en_fastconformer_ctc_large"
    # "nvidia:nvidia/stt_en_fastconformer_ctc_xlarge"
    # "nvidia:nvidia/stt_en_fastconformer_ctc_xxlarge"
    # "nvidia:nvidia/stt_en_fastconformer_transducer_large"
    # "nvidia:nvidia/stt_en_fastconformer_transducer_xlarge"
    # "nvidia:nvidia/stt_en_fastconformer_transducer_xxlarge"
    # "nvidia:nvidia/stt_en_fastconformer_hybrid_large_pc"
    # "nvidia:nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
    # "nvidia:stt_en_jasper10x5dr"
    # "nvidia:stt_en_quartznet15x5"
    # "nvidia:stt_en_contextnet_256"
    # "nvidia:stt_en_contextnet_512"
    # "nvidia:stt_en_contextnet_1024"
    # "nvidia:stt_en_contextnet_256_mls"
    # "nvidia:stt_en_contextnet_512_mls"
    # "nvidia:stt_en_contextnet_1024_mls"
    # "nvidia:nvidia/stt_en_citrinet_256_ls"
    # "nvidia:nvidia/stt_en_citrinet_384_ls"
    # "nvidia:nvidia/stt_en_citrinet_512_ls"
    # "nvidia:nvidia/stt_en_citrinet_768_ls"
    # "nvidia:nvidia/stt_en_citrinet_1024_ls"
    # "nvidia:nvidia/stt_en_citrinet_1024_gamma_0_25"

    # "nvidia:stt_en_squeezeformer_ctc_large_ls"
    # "nvidia:stt_en_squeezeformer_ctc_small_ls"
    # "nvidia:stt_en_squeezeformer_ctc_medium_ls"
    # "nvidia:stt_en_squeezeformer_ctc_medium_large_ls"
    # "nvidia:stt_en_squeezeformer_ctc_small_medium_ls"
    # "nvidia:stt_en_squeezeformer_ctc_xsmall_ls"
    # "nvidia:nvidia/parakeet-tdt-0.6b-v2"



)
DATASET_TYPE="local"
# DATASET_NAME="parler-tts/mls_eng"
DATASET_NAME="/workspace/data"
SPLIT="validation"
AUDIO_COLUMN="audio"
METADATA_PATH="/workspace/metadata_b.csv"
ID_COLUMN="id"
OUTPUT_DIR="results"
DEVICE="cpu"

mkdir -p $OUTPUT_DIR

for MODEL in "${MODELS[@]}"; do
    echo "Testando modelo: $MODEL"
    python main.py \
        --dataset_type $DATASET_TYPE \
        --dataset_name $DATASET_NAME \
        --metadata_path $METADATA_PATH \
        --split $SPLIT \
        --audio_column $AUDIO_COLUMN \
        --id_column $ID_COLUMN \
        --output_dir $OUTPUT_DIR \
        --models $MODEL \
        --device $DEVICE
done

echo "Testes concluídos!"