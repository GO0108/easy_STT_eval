from data import DatasetLoader
import pandas as pd
import time
from tqdm import tqdm


def load_metadata(metadata_path):
    """
    Carrega o arquivo de metadados e retorna a lista de áudios.
    """
    test_metadata = pd.read_csv(metadata_path)
    audios = list(test_metadata.audio_segmentado)
    return audios


def transcribe_audio_files(model, audio_files, audio_path):
    """
    Transcreve uma lista de arquivos de áudio usando o modelo fornecido.
    """
    data = []
    total_files = len(audio_files)

    for audio_file in tqdm(audio_files, total=total_files, desc=f"Processing {model.model_id}"):
        inicio = time.time()
        transcription = model.transcribe(audio_path + audio_file)
        fim = time.time()

        tempo_execucao = fim - inicio
        data.append({
            'audio_file': audio_file,
            'tempo': tempo_execucao,
            'transcricao': transcription
        })

    return data

def transcribe_audio_arrays(model, audio_data):
    """
    Transcreve uma lista de arrays de áudio usando o modelo fornecido.
    """
    data = []
    total_files = len(audio_data)
    for id, reference, audio_array, sampling_rate in tqdm(audio_data, total=total_files, desc=f"Processing {model.model_id}"):
        inicio = time.time()
        transcription = model.transcribe((audio_array, sampling_rate))
        fim = time.time()

        tempo_execucao = fim - inicio
        data.append({
            'audio_id': id,  
            'referencia': reference,
            'tempo': tempo_execucao,
            'transcricao': transcription
        })

    return data

def save_results(data, output_path):
    """
    Salva os resultados em um arquivo CSV.
    """
    dataset = pd.DataFrame(data)
    dataset.to_csv(output_path, index=False)  # Salva o DataFrame no arquivo CSV