from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import torch

# Set your model path
MODEL_PATH = "C:\\Users\\sesha\Desktop\\task_jammu\\deepfake-voice-simulation\\recipes\\dfs\\glow_tts\\glow_tts\\run-May-23-2025_10+03PM-24ec7590\\checkpoint_340000.pth"
CONFIG_PATH = "C:\\Users\\sesha\Desktop\\task_jammu\\deepfake-voice-simulation\\recipes\\dfs\\glow_tts\\glow_tts\\run-May-23-2025_10+03PM-24ec7590\\config.json"
OUTPUT_PATH = "output.wav"

# Load config
config = GlowTTSConfig()
config.load_json(CONFIG_PATH)

# Set slow speech length scale here
config.length_scale = 1.5  # slow down speech

# Load model
model = GlowTTS.init_from_config(config)
cp = torch.load(MODEL_PATH, map_location=torch.device('cuda'))  # or 'cuda' if using GPU
model.load_state_dict(cp['model'])
model.eval()

# Audio processor
ap = AudioProcessor.init_from_config(config)

# Synthesizer
synthesizer = Synthesizer(
    model=model,
    config=config,
    speakers_file=None,
    vocoder=None,        # No vocoder used
    ap=ap,
)

# Your input text
text = "Hello Mr. Green.,I wanted to contact you privately because our department noticed a possible identity discrepancy tied to your tax ID., If it’s okay with you, I’d prefer we keep this conversation off the record to protect your information, Would you like me to explain how we can fix this quietly?"

# Run inference
wav = synthesizer.tts(text)

# Save the output
ap.save_wav(wav, OUTPUT_PATH)

print(f"Saved slow speech audio to: {OUTPUT_PATH}")
