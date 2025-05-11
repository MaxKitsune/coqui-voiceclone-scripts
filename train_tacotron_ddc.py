import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    output_path = os.path.dirname(os.path.abspath(__file__))

    # Dataset config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=os.path.join(output_path, "/data/home/fum43851/voiceclone/dataset_urte_30122024"),
    )

    # Audio config
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        do_trim_silence=False,
        trim_db=40.0,
        signal_norm=False,
        mel_fmin=0.0,
        mel_fmax=8000,
        spec_gain=1.0,
        log_func="np.log",
        ref_level_db=20,
        preemphasis=0.0,
    )

    # Tacotron2 config
    config = Tacotron2Config(
        audio=audio_config,
        run_name="uko-test-de-ddc",
        batch_size=128,
        eval_batch_size=64,
        num_loader_workers=32,
        num_eval_loader_workers=32,
        run_eval=True,
        test_delay_epochs=-1,
        r=6,
        gradual_training=[[0, 6, 256], [10000, 4, 128], [50000, 3, 128], [100000, 2, 128]],
        double_decoder_consistency=True,
        epochs=1000,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache_ddc"),
        precompute_num_workers=8,
        print_step=25,
        print_eval=True,
        output_path=output_path,
        datasets=[dataset_config],
        mixed_precision=True,
        test_sentences=[
            "It took me time to develop a voice.",
            "Be a voice, not an echo.",
            "I'm sorry, David. I'm afraid.",
        ],
        max_decoder_steps=6000,
    )

    # Initialize audio processor
    ap = AudioProcessor.init_from_config(config)

    # Initialize tokenizer
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load dataset samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
    )

    # Initialize model
    model = Tacotron2(config, ap, tokenizer, speaker_manager=None).to('cuda')

    # Initialize and run trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    main()
