import sys

TTS_PATH = "../"

# add libraries into environment
sys.path.append(TTS_PATH)  # set this if TTS is not installed globally

import os
import string

try:
    from TTS.utils.audio import AudioProcessor
except:
    from TTS.utils.audio import AudioProcessor

from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

from TTS.tts.utils.speakers import SpeakerManager
import librosa

DATA_DIR = '../samples/'
OUT_PATH = '../out/'

# create output path
os.makedirs(OUT_PATH, exist_ok=True)

# model vars
MODEL_PATH = 'best_model.pth.tar'
CONFIG_PATH = 'config.json'
TTS_LANGUAGES = "language_ids.json"
TTS_SPEAKERS = "speakers.json"
USE_CUDA = torch.cuda.is_available()
CHECKPOINT_SE_PATH = './SE_checkpoint.pth.tar'
CONFIG_SE_PATH = './config_se.json'

# load the config
C = load_config(CONFIG_PATH)

# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

C.model_args['d_vector_file'] = TTS_SPEAKERS
C.model_args['use_speaker_encoder_as_loss'] = False

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
# print(model.language_manager.num_languages, model.embedded_language_dim)
# print(model.emb_l)
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
    if "speaker_encoder" in key:
        del model_weights[key]

model.load_state_dict(model_weights)

model.eval()

if USE_CUDA:
    model = model.cuda()

# synthesize voice
use_griffin_lim = False

SE_speaker_manager = SpeakerManager(encoder_model_path=CHECKPOINT_SE_PATH,
                                    encoder_config_path=CONFIG_SE_PATH,
                                    use_cuda=USE_CUDA)


def compute_spec(ref_file):
    y, sr = librosa.load(ref_file, sr=ap.sample_rate)
    spec = ap.spectrogram(y)
    spec = torch.FloatTensor(spec).unsqueeze(0)
    return spec


print("Sampling speaker reference audios files...")
reference_files = [os.path.join(DATA_DIR, name) for name in os.listdir(DATA_DIR)]
for sample in reference_files:
    os.system(rf'ffmpeg-normalize {sample} -nt rms -t=-27 -o {sample} -ar 16000 -f')

model.length_scale = 1  # scaler for the duration predictor. The larger it is, the slower the speech.
model.inference_noise_scale = 0.3 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.3 # defines the noise variance applied to the duration predictor z vector at inference.
texts = ["It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
         "Humans use a huge amount of water.",
         "My best friend speaks english and Spanish.",
         "I brush my teeth carefully in the morning."]
model.language_manager.language_id_mapping
language_id = 0
for ref in reference_files:
    name = os.path.basename(ref).rsplit('.', 1)[0]
    spr_dir = os.path.join(OUT_PATH, name)
    os.makedirs(spr_dir, exist_ok=True)
    emb = SE_speaker_manager.compute_d_vector_from_clip(ref)
    for j, text in enumerate(texts):
        print(" > text: {}".format(text))
        wav, alignment, _, _ = synthesis(
            model,
            text,
            C,
            "cuda" in str(next(model.parameters()).device),
            ap,
            speaker_id=None,
            d_vector=emb,
            style_wav=None,
            language_id=language_id,
            enable_eos_bos_chars=C.enable_eos_bos_chars,
            use_griffin_lim=True,
            do_trim_silence=False,
        ).values()
        print("Generated Audio")
        file_name = text.replace(" ", "_")
        file_name = f'{j}.' + file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
        out_path = os.path.join(spr_dir, file_name)
        print(" > Saving output to {}".format(out_path))
        ap.save_wav(wav, out_path)
