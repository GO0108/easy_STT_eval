
from models import WhisperModel, Wav2VecModel, NvidiaModel, GoogleSpeechRecognitionModel, APIWhisperModel, Wav2VecConformerModel, FasterWhisperModel, WhisperS2TModel
import os
import pandas as pd
from tqdm import tqdm
import torch
from data import DatasetLoader
from data_utils import transcribe_audio_files, transcribe_audio_arrays, save_results
import argparse
from eval import calculate_metrics


def get_model_instance(model_type, model_id, device):
    """
    Retorna uma instância do modelo com base no tipo.
    """
    if model_type == "whisper":
        return WhisperModel(model_id=model_id, device=device)
    if model_type == "faster_whisper":
        return FasterWhisperModel(model_id=model_id, device=device)
    elif model_type == "wav2vec":
        return Wav2VecModel(model_id=model_id, device=device)
    elif model_type == "wav2vec_conformer":
        return Wav2VecConformerModel(model_id=model_id, device=device)
    elif model_type == "whisper_s2t":
        return WhisperS2TModel(model_id=model_id, device=device)
    elif model_type == "nvidia":
        return NvidiaModel(model_id=model_id, device=device)
    elif model_type == "google":
        return GoogleSpeechRecognitionModel()
    elif model_type == "api_whisper":
        return APIWhisperModel(model_id=model_id)  
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Script para transcrição de áudio usando diferentes modelos de ASR.")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["local", "huggingface"],
                        help="Tipo de dataset: 'local' para arquivos locais ou 'huggingface' para datasets do Hugging Face.")
    
    parser.add_argument("--metadata_path", type=str, help="Caminho para o arquivo de metadados (para datasets locais).")
    parser.add_argument("--dataset_name", type=str, help="Nome do dataset no Hugging Face (para datasets do Hugging Face).")
    parser.add_argument("--split", type=str, default="test", help="Divisão do dataset (train, test, validation).")
    parser.add_argument("--audio_column", type=str, default="audio", help="Coluna do dataset contendo os arrays de áudio.")
    parser.add_argument("--id_column", type=str, default="path", help="Coluna do dataset contendo os IDs dos áudios.")
    parser.add_argument("--reference_column", type=str, default="text", help="Coluna do dataset contendo as referências.")
    
    parser.add_argument("--output_dir", type=str, default="results", help="Diretório para salvar os resultados.")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Lista de modelos no formato 'tipo:model_id'.")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo a ser usado (cpu ou cuda).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Carregar os dados
    if args.dataset_type == "local":
        if not args.metadata_path:
            raise ValueError("O argumento '--metadata_path' é obrigatório para datasets locais.")
        audio_data = DatasetLoader.load_local_metadata(args.metadata_path)
        audio_path = os.path.dirname(args.metadata_path)  # Diretório base dos arquivos de áudio
    elif args.dataset_type == "huggingface":
        if not args.dataset_name:
            raise ValueError("O argumento '--dataset_name' é obrigatório para datasets do Hugging Face.")
        dataset = DatasetLoader.load_huggingface_dataset(args.dataset_name, args.split, args.audio_column)
        audio_data = DatasetLoader.get_audio_data(dataset, args.audio_column, args.id_column)

    # Processar cada modelo
    for model_entry in args.models:
        model_type, model_id = model_entry.split(":")
        print(f"Usando o modelo: {model_id} (tipo: {model_type})")

        model = get_model_instance(model_type, model_id, args.device)

        # Transcrever áudios
        if args.dataset_type == "local":
            data = transcribe_audio_files(model, audio_data, audio_path)
        elif args.dataset_type == "huggingface":
            data = transcribe_audio_arrays(model, audio_data)

        # Salvar resultados
        output_file = os.path.join(args.output_dir, f"{model_type}_{model_id.split('/')[-1]}.csv")
        save_results(data, output_file)
        torch.cuda.empty_cache()

    print("Processamento concluído!")
    calculate_metrics(args.output_dir, dataset)



if __name__ == "__main__":
    main()