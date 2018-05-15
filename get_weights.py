from keras.models import load_model, Sequential, model_from_json
from Models import gaconfig
from Models import gautils

model = gautils.load_pretrain_model(is_printed=True) 
