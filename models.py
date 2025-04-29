from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ConformerModel

import whisper_s2t
from faster_whisper import WhisperModel as FasterWhisper
# import nemo.collections.asr as nemo_asr
# from openai import OpenAI
# import speech_recognition as sr
import torch
import torchaudio
import os
import tempfile

from config import OPENAPI_TOKEN


class ASRModel:
    """
    Classe base para modelos de reconhecimento automático de fala (ASR).
    """
    def __init__(self, model_type: str, model_id: str = None, device: str = 'cpu', torch_dtype=torch.float32):
        self.model_type = model_type
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self.pipe = None

    def load_model(self):
        """
        Método genérico para carregar o modelo. Deve ser sobrescrito pelas subclasses.
        """
        raise NotImplementedError("Este método deve ser implementado pela subclasse.")

    def transcribe(self, audio_input):
        """
        Método genérico para transcrição. Deve ser sobrescrito pelas subclasses.
        O parâmetro `audio_input` pode ser um path para um arquivo de áudio ou um array de áudio.
        """
        raise NotImplementedError("Este método deve ser implementado pela subclasse.")


class WhisperModel(ASRModel):
    """
    Modelo específico para Whisper.
    """
    def __init__(self, model_id: str, device: str = 'cpu', torch_dtype=torch.float32):
        super().__init__('whisper', model_id, device, torch_dtype)
        self.language = "en"
        self.num_beams = 5
        self.generate_kwargs = {"language": self.language, "num_beams": self.num_beams}
        self.load_model()

    def load_model(self):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=self.device,
            generate_kwargs=self.generate_kwargs
        )

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            return self.pipe(audio_input)['text']
        elif isinstance(audio_input, tuple):  
            audio_array, sampling_rate = audio_input
            inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
            inputs = inputs.to(self.device)
            predicted_ids = self.model.generate(inputs, **self.generate_kwargs)
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")


class FasterWhisperModel(ASRModel):
    """
    Modelo específico para Faster Whisper.
    """
    def __init__(self, model_id: str = "large-v3", device: str = "cpu"):
        """
        Inicializa o modelo Faster Whisper.
        :param model_id: Nome do modelo (e.g., "large-v3").
        :param device: Dispositivo para execução ("cpu" ou "cuda").
        """
        super().__init__('faster_whisper', model_id, device)
        self.load_model()

    def load_model(self):
        """
        Carrega o modelo Faster Whisper.
        """
        self.model = FasterWhisper(self.model_id, device=self.device)

    def transcribe(self, audio_input):
        """
        Transcreve o áudio usando o modelo Faster Whisper.
        :param audio_input: Caminho para o arquivo de áudio ou array de áudio.
        :return: Texto transcrito.
        """
        if isinstance(audio_input, str):  
            audio_path = audio_input
        elif isinstance(audio_input, tuple):  
            audio_array, sampling_rate = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
                torchaudio.save(audio_path, torch.tensor(audio_array).unsqueeze(0), sampling_rate)
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

        try:
            segments, _ = self.model.transcribe(audio_path)
            transcription = ""
            for segment in segments:
                transcription += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
            return transcription
        finally:
            if isinstance(audio_input, tuple):  
                os.remove(audio_path)


class WhisperS2TModel(ASRModel):
    """
    Modelo específico para Whisper S2T.
    """
    def __init__(self, model_id: str = "large-v2", backend: str = "CTranslate2", device: str = "cpu"):
        """
        Inicializa o modelo Whisper S2T.
        :param model_id: Nome do modelo (e.g., "large-v2").
        :param backend: Backend a ser usado (e.g., "CTranslate2").
        :param device: Dispositivo para execução ("cpu" ou "cuda").
        """
        super().__init__('whisper_s2t', model_id, device)
        self.backend = backend
        self.load_whisper_s2t_model()

    def load_whisper_s2t_model(self):
        """
        Carrega o modelo Whisper S2T.
        """
        self.model = whisper_s2t.load_model(model_identifier=self.model_id, backend=self.backend, device='cpu', compute_type='float32')

    def transcribe(self, audio_input, lang_code: str = "en", task: str = "transcribe", initial_prompt: str = None):
        """
        Transcreve o áudio usando o modelo Whisper S2T.
        :param audio_input: Caminho para o arquivo de áudio.
        :param lang_code: Código do idioma (e.g., "en").
        :param task: Tarefa a ser realizada (e.g., "transcribe").
        :param initial_prompt: Prompt inicial para o modelo.
        :return: Texto transcrito.
        """
        if isinstance(audio_input, str):  
            audio_path = audio_input
            temp_file_created = False
        elif isinstance(audio_input, tuple):  
            audio_array, sampling_rate = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
                torchaudio.save(audio_path, torch.tensor(audio_array).unsqueeze(0), sampling_rate)
            temp_file_created = True
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

        try:
            files = [audio_path]
            lang_codes = [lang_code]
            tasks = [task]
            initial_prompts = [initial_prompt]

            out = self.model.transcribe_with_vad(
                files,
                lang_codes=lang_codes,
                tasks=tasks,
                initial_prompts=initial_prompts,
                batch_size=1
            )

            return out[0][0]["text"]
        finally:
            if temp_file_created:
                os.remove(audio_path)
    

class Wav2VecModel(ASRModel):
    """
    Modelo específico para Wav2Vec.
    """
    def __init__(self, model_id: str, device: str = 'cpu'):
        super().__init__('wav2vec', model_id, device)
        self.load_model()

    def load_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            audio, sr = torchaudio.load(audio_input)
        elif isinstance(audio_input, tuple):  
            audio, sr = audio_input
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

    
        inputs = self.processor(audio, sampling_rate=sr,return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        text = transcription[0]
        return text


        


class Wav2VecConformerModel(ASRModel):
    """
    Modelo específico para Wav2Vec Conformer
    """
    def __init__(self, device: str = 'cpu'):
        super().__init__('wav2vec_conformer',  device)
        self.load_model()

    def load_model(self):
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        self.model = Wav2Vec2ConformerModel.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            audio, sr = torchaudio.load(audio_input)
        elif isinstance(audio_input, tuple):  
            audio, sr = audio_input
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

        input_values = self.processor(audio.squeeze(),  sampling_rate=sr, return_tensors="pt").input_values
        logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.processor.decode(pred_ids[0])


class NvidiaModel(ASRModel):
    """
    Modelo específico via NVIDIA NeMo.
    """
    def __init__(self, model_id: str = "nvidia/stt_en_conformer_ctc_large", device: str = 'cpu'):
        super().__init__('nvidia', model_id)
        self.load_model()

    def load_model(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            return self.model.transcribe([audio_input])[0].text
        elif isinstance(audio_input, tuple):  
            audio_array, sampling_rate = audio_input
            temp_path = "temp_audio.wav"
            torchaudio.save(temp_path, torch.tensor(audio_array).unsqueeze(0), sampling_rate)
            return self.model.transcribe([temp_path])[0].text
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")


class GoogleSpeechRecognitionModel(ASRModel):
    """
    Modelo para biblioteca SpeechRecognition.
    """
    def __init__(self):
        super().__init__('speech_recognition')

    def transcribe(self, audio_input):
        recognizer = sr.Recognizer()
        if isinstance(audio_input, str):  # Path para arquivo de áudio
            with sr.AudioFile(audio_input) as source:
                audio = recognizer.record(source)
        elif isinstance(audio_input, tuple):  # Array de áudio e taxa de amostragem
            audio_array, sampling_rate = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_path = temp_audio_file.name
                torchaudio.save(temp_path, torch.tensor(audio_array).unsqueeze(0), sampling_rate)
            try:
                with sr.AudioFile(temp_path) as source:
                    audio = recognizer.record(source)
            finally:
                os.remove(temp_path)  
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

        try:
            return recognizer.recognize_google(audio)
        except Exception as e:
            return f"Erro no reconhecimento: {e}"
        

class APIWhisperModel(ASRModel):
    """
    Modelo para transcrição usando a API Whisper da OpenAI.
    """
    def __init__(self, model_id: str = "whisper-1"):
        super().__init__('api_whisper', model_id)
        self.api_key = OPENAPI_TOKEN  
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            audio_path = audio_input
        elif isinstance(audio_input, tuple):  # Array de áudio e taxa de amostragem
            audio_array, sampling_rate = audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
                torchaudio.save(audio_path, torch.tensor(audio_array).unsqueeze(0), sampling_rate)
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model_id,
                    file=audio_file,
                    response_format="text"
                )
            return response["text"]
        except Exception as e:
            return f"Erro na transcrição: {e}"
        finally:
            if isinstance(audio_input, tuple): 
                os.remove(audio_path)





        # ## Canary 1b
        # model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')
        # # Parakeet
        # model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")
        # model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/parakeet-ctc-1.1b")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-rnnt-0.6b")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
        # model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-1.1b")
        # model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-110m")

        # # Conformer
        # model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_small")
        # model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")

        # # FastConformer
        # model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="nvidia/stt_en_fastconformer_ctc_large")
        # model = nemo_asr.models.EncDecCTCTBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_ctc_xlarge")
        # model = nemo_asr.models.EncDecCTCTBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_ctc_xxlarge")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_transducer_large")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_transducer_xlarge")
        # model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_transducer_xxlarge")
        # model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_hybrid_large_pc")
        # model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_hybrid_large_streaming_multi")

        # # # Optional: change the default latency. Default latency is 1040ms. Supported latencies: {0: 0ms, 1: 80ms, 16: 480ms, 33: 1040ms}.
        # # # Note: These are the worst latency and average latency would be half of these numbers.
        # # model.encoder.set_default_att_context_size([70,13])
        # # #Optional: change the default decoder. Default decoder is Transducer (RNNT). Supported decoders: {ctc, rnnt}.
        # # model.change_decoding_strategy(decoder_type='rnnt')


        # # Jasper
        # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_jasper10x5dr")
        # # Quartznet
        # asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_quartznet15x5")
        # # ContextNet
        # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256")
        # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_512")
        # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_1024")
        # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256_mls")
        # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_512_mls")
        # asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_1024_mls")

        # # Citrinet
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_256_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_384_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_512_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_768_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_1024_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_citrinet_1024_gamma_0_25")

        # # Squeezeformer
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_large_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_small_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_medium_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_medium_large_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_small_medium_ls")
        # asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_squeezeformer_ctc_xsmall_ls")




