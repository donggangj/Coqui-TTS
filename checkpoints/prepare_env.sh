python -m pip install -e ../
python -m pip install pydub ffmpeg-normalize==1.21.0
python -m pip install torchaudio==0.9.0

CHECKPOINT_SE_PATH="SE_checkpoint.pth.tar"
CHECKPOINT_MODEL_PATH="best_model.pth.tar"
SPKR_EMB_PATH="speakers.json"
gdown --id 17JsW6h6TIh7-LkU2EvB_gnNrPcdBxt7X -O $CHECKPOINT_SE_PATH
gdown --id 1sgEjHt0lbPSEw9-FSbC_mBoOPwNi87YR -O $CHECKPOINT_MODEL_PATH
gdown --id 1SZ9GE0CBM-xGstiXH2-O2QWdmSXsBKdC -O $SPKR_EMB_PATH
