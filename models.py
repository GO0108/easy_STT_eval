from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForCTC, AutoTokenizer, AutoFeatureExtractor
import nemo.collections.asr as nemo_asr
import torch
import speech_recognition as sr
import torchaudio
import os
import tempfile
from openai import OpenAI
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


class Wav2VecModel(ASRModel):
    """
    Modelo específico para Wav2Vec.
    """
    def __init__(self, model_id: str, device: str = 'cpu'):
        super().__init__('wav2vec', model_id, device)
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCTC.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            audio, _ = torchaudio.load(audio_input)
        elif isinstance(audio_input, tuple):  
            audio, _ = audio_input
        else:
            raise ValueError("O parâmetro `audio_input` deve ser um path ou um array de áudio.")

        input_values = self.feature_extractor(audio.squeeze().numpy(), return_tensors="pt").input_values
        logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return self.tokenizer.decode(pred_ids[0])


class ConformerModel(ASRModel):
    """
    Modelo específico para Conformer (via NVIDIA NeMo).
    """
    def __init__(self, model_id: str = "stt_en_fastconformer_transducer_large"):
        super().__init__('conformer', model_id)
        self.load_model()

    def load_model(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)

    def transcribe(self, audio_input):
        if isinstance(audio_input, str):  
            return self.model.transcribe([audio_input])[0]
        elif isinstance(audio_input, tuple):  
            audio_array, sampling_rate = audio_input
            temp_path = "temp_audio.wav"
            torchaudio.save(temp_path, torch.tensor(audio_array).unsqueeze(0), sampling_rate)
            return self.model.transcribe([temp_path])[0]
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