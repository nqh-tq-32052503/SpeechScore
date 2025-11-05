from model import DNSMOS
import soundfile as sf 

class Scoring(object):
    def __init__(self):
        self.model = DNSMOS()
        self.sample_audio = "./sample.wav"
        self.warmup()

    def warmup(self):
        self.inference(self.sample_audio)

    def inference(self, audio_path, window_size=None):
        """
        window_size: None -> assess all audios
        """
        audio_test, rate_test = sf.read(audio_path, always_2d=False)
        inputs = {"audio" : [audio_test], "rate" : rate_test}
        result = self.model.scoring(inputs, window=window_size)
        return result

    @classmethod
    def cut_audio(input_path: str, output_path: str, start_sec: float, end_sec: float):
        """
        Cut an audio file between start_sec and end_sec and save the result.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to save the trimmed audio.
            start_sec (float): Start time in seconds.
            end_sec (float): End time in seconds.
        """
        # Read the whole audio
        audio, sr = sf.read(input_path)
        
        # Compute sample indices
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        
        # Clip to valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        # Slice audio
        trimmed_audio = audio[start_sample:end_sample]
        
        # Write trimmed file
        sf.write(output_path, trimmed_audio, sr)
        
        print(f"✅ Saved trimmed audio from {start_sec}s to {end_sec}s → {output_path}")
