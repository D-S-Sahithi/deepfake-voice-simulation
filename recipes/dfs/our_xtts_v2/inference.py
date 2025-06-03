import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("C:\\Users\\sesha\\Desktop\\task_jammu\\deepfake-voice-simulation\\recipes\\dfs\\our_xtts_v2\\run\\training\GPT_XTTS_v2.0_LJSpeech_FT_dfs-May-30-2025_11+01PM-900d3731\\config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="C:\\Users\\sesha\\Desktop\\task_jammu\\deepfake-voice-simulation\\recipes\\dfs\\our_xtts_v2\\run\\training\\GPT_XTTS_v2.0_LJSpeech_FT_dfs-May-31-2025_10+18PM-900d3731")
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["C:\\Users\\sesha\\Desktop\\task_jammu\\deepfake-voice-simulation\\recipes\\dfs\\myttsdataset\\wavs\\audio6.wav"])

print("Inference...")
out = model.inference(
    "Sir, the situation is critical. The hackers are trying to access your personal funds right now.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)