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


def calculate_metrics(results_dir, dataset, output_dir='final_results'):
    """
    Calcula métricas WER, CER e outras estatísticas para os resultados de transcrição.

    Args:
        results_dir (str): Diretório contendo os arquivos CSV com as transcrições.
        dataset (Dataset): Dataset do Hugging Face contendo as referências.
        output_dir (str): Diretório para salvar os resultados finais.

    Returns:
        pd.DataFrame: DataFrame contendo as métricas calculadas.
    """
    tempo_total = sum([len(audio['array']) / audio['sampling_rate'] for audio in dataset['audio']])
    print("Tempo do dataset:", tempo_total/60, "min")
    final_results = []

    for result in glob.glob(os.path.join(results_dir, "*.csv")):
        info = pd.read_csv(result)

        # Aplica WER e CER
        info['wer'] = info.apply(lambda row: wer(normalize_text(row['referencia']), normalize_text(row['transcricao'])), axis=1)
        info['cer'] = info.apply(lambda row: cer(normalize_text(row['referencia']), normalize_text(row['transcricao'])), axis=1)
        info.to_csv(result, index=False, encoding='utf-8')

        # Calcula métricas agregadas
        tempo_inferencia = sum(info['tempo'])
        rtf = tempo_inferencia / tempo_total
        wer_mean, wer_median, cer_mean, cer_median = info['wer'].mean(), info['wer'].median(), info['cer'].mean(), info['cer'].median()

        new_row = {
            'result': result.split('/')[-1].replace('.csv', ''),
            'tempo_inferencia (min)': tempo_inferencia/60,
            'rtf': rtf,
            'wer_mean': wer_mean,
            'wer_median': wer_median,
            'cer_mean': cer_mean,
            'cer_median': cer_median
        }

        final_results.append(new_row)
    
    # Salva resultados finais
    os.makedirs(os.path.join(results_dir, output_dir), exist_ok=True)
    df_final = pd.DataFrame(final_results)
    df_final.to_csv(os.path.join(results_dir, output_dir,'final_results.csv'), index=False, encoding='utf-8')

    return df_final