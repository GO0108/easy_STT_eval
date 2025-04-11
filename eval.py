import os
import pandas as pd
from jiwer import wer, cer, RemovePunctuation, ToLowerCase, RemoveWhiteSpace
import glob
from jiwer import wer, cer, RemovePunctuation, ToLowerCase, RemoveWhiteSpace, Compose
import string

def normalize_text(text):
    """
    Normaliza o texto removendo pontuações, espaços extras e convertendo para minúsculas.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Converte para letras minúsculas
    text = text.lower()
    return text

def evaluate_results(output_dir):
    """
    Avalia os resultados dos arquivos CSV no diretório de saída.
    Calcula WER, CER, tempo total de processamento e RTF.
    """
    evaluation_data = []

    # Iterar sobre todos os arquivos CSV no diretório de saída
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Verificar se as colunas necessárias estão presentes
        if "transcricao" not in df.columns or "referencia" not in df.columns:
            print(f"Arquivo {csv_file} não contém as colunas necessárias para avaliação.")
            continue

        # Normalizar transcrições e referências
        transcriptions = df["transcricao"].apply(normalize_text).tolist()
        references = df["referencia"].apply(normalize_text).tolist()
        tempos = df["tempo"].tolist()

        # Calcular métricas
        total_tempo = sum(tempos)
        total_audio_duracao = len(transcriptions) * 10  # Substitua pela duração real do áudio
        rtf = total_tempo / total_audio_duracao if total_audio_duracao > 0 else 0

        wer_score = wer(references, transcriptions)
        cer_score = cer(references, transcriptions)

        # Adicionar resultados à tabela
        evaluation_data.append({
            "arquivo": os.path.basename(csv_file),
            "wer": round(wer_score, 4),
            "cer": round(cer_score, 4),
            "tempo_total_s": round(total_tempo, 2),
            "rtf": round(rtf, 4)
        })

    # Criar DataFrame de avaliação
    eval_df = pd.DataFrame(evaluation_data)
    eval_df.to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False)
    print("Avaliação concluída! Resultados salvos em 'evaluation_summary.csv'.")


if __name__ == "__main__":
    output_dir = "results"  # Substitua pelo diretório onde os CSVs estão armazenados
    evaluate_results(output_dir)