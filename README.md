# Voice Model Training with VITS and Tacotron-DDC

This repository showcases my experience in training state-of-the-art voice models (VITS and Tacotron-DDC) optimized for high-performance hardware and tuned for natural-sounding output on limited data.

---

## 📝 Project Overview

* **Models:** VITS and Tacotron with Dynamic Decoder Convolution (DDC)
* **Hardware (Optimized):** NVIDIA RTX 8000 ADA Generation GPU, AMD EPYC 128-Core Processor
* **Local Training:** NVIDIA RTX 2070-based workstation
* **Dataset:** LJSpeech-format, \~1 hour of audio (noted as minimal)

The configuration files provided here were initially based on CoquiTTS scripts by Thorsten Müller and have been adapted to make efficient use of my hardware resources and dataset characteristics.

---

## 🚀 Performance Highlights

* **Workflow Optimization:** Transitioned from CPU/RTX 2070 training (multiple days) to RTX 8000 ADA GPU servers, reducing training times from days to mere hours.
* **Model Comparison:** Tacotron-DDC outperformed VITS on my dataset, producing more natural speech and faster convergence with smaller data volumes.
* **Data Insights:** One hour of LJSpeech audio is insufficient for fully natural-sounding models—additional data significantly improves quality.

---

## ⚙️ Configuration Details

### Common Settings

* Sample Rate: 22050 Hz
* Dataset Format: LJSpeech (WAV + metadata.csv)
* Batch Size: Tuned per GPU capacity
* Learning Rates and Schedulers: Configurable in `config/*.yaml`

*Note: Exact hyperparameters and layer details can be found in the config files.*

---

## 📥 Usage

Clone the repository, configure your Coqui TTS installation, and use the provided scripts instead.


---

## 📊 Results & Observations

| Model        | Training Time (RTX 2070) | Training Time (RTX 8000 ADA) | Notes                         |
| ------------ | ------------------------ | ---------------------------- | ----------------------------- |
| VITS         | \~3 days                 | \~6 hours                    | Good quality, slower converge |
| Tacotron-DDC | \~2 days                 | \~4 hours                    | More natural, faster converge |

---

## 📚 Acknowledgments

* Initial configuration and scripts from [CoquiTTS](https://github.com/coqui-ai/TTS) by Thorsten Müller.
* Thanks to the open-source community for LJSpeech dataset and tool support.

---

## 📄 License

This project is released under the MIT License.
