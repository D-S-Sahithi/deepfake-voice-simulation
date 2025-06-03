import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT_dfs"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Output paths
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = False
BATCH_SIZE = 1
GRAD_ACUMM_STEPS = 64

# Dataset Configuration
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="dfs",
    path=r"C:\Users\sesha\Desktop\task_jammu\deepfake-voice-simulation\recipes\dfs\myttsdataset",
    meta_file_train=r"C:\Users\sesha\Desktop\task_jammu\deepfake-voice-simulation\recipes\dfs\myttsdataset\metadata.txt",
    language="en",
)
DATASETS_CONFIG_LIST = [config_dataset]

# Checkpoint Files
DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

# Download files if not available
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

LANGUAGE = config_dataset.language

def main():
    # XTTS model arguments
    model_args = GPTArgs(
        max_conditioning_length= 88200,
        min_conditioning_length= 44100,
        debug_loading_failures=False,
        max_wav_length=330750,
        max_text_length=250,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # Audio configuration
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000
    )

    # Training configuration
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="Fine-tuning XTTS with lower GPU usage.",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=1,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=1,
        eval_split_max_size=48,
        print_step=10,
        plot_step=50,
        log_model_step=250,
        save_step=200,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-3},
        lr=5e-6,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [100, 250, 400, 600], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[

        ],  # Disable test sentences to save memory
    )

    # Load model and dataset
    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(DATASETS_CONFIG_LIST, eval_split=True, eval_split_max_size=config.eval_split_max_size, eval_split_size=0.01)

    trainer = Trainer(
        TrainerArgs(
            restore_path= "C:\\Users\\sesha\\Desktop\\task_jammu\\deepfake-voice-simulation\\recipes\\dfs\\our_xtts_v2\\run\\training\\GPT_XTTS_v2.0_LJSpeech_FT_dfs-May-31-2025_10+18PM-900d3731\\checkpoint_10027400.pth",
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    main()
