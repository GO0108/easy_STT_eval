from datasets import load_dataset
import pandas as pd
import os


class DatasetLoader:
    """
    Classe para carregar datasets de diferentes fontes.
    """
    @staticmethod
    def load_local_metadata(metadata_path):
        """
        Carrega um arquivo de metadados local e retorna a lista de áudios.
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Arquivo de metadados não encontrado: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        return list(metadata.audio_segmentado)

    @staticmethod
    def load_huggingface_dataset(dataset_name, split="test", audio_column="audio"):
        """
        Carrega um dataset do Hugging Face e retorna os dados de áudio.
        """
        dataset = load_dataset(dataset_name, split=split)
        if audio_column not in dataset.column_names:
            raise ValueError(f"Coluna '{audio_column}' não encontrada no dataset '{dataset_name}'.")
        return dataset

    @staticmethod
    def get_audio_data(dataset, audio_column="audio", id_column="path", reference_column="text"):
        """
        Retorna os arrays de áudio, taxas de amostragem e referências de um dataset do Hugging Face.
        """
        audio_data = []
        for sample in dataset:
            id = sample[id_column]
            audio_array = sample[audio_column]["array"]
            sampling_rate = sample[audio_column]["sampling_rate"]
            reference = sample[reference_column]
            audio_data.append((id, reference, audio_array, sampling_rate))
        return audio_data