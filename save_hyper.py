import json
import os

# Data to be written
dictionary = {
    "name": "resnet50",
    "weight": "pretrained",
    "EPOCHS": 60,
    "BATCHSIZE": 128,
    "LEARNING_RATE": 1e-1,
    "optimizer":'sgd',
    "momentum":None
}



save_dir = 'runs_log/perf_log_resnet18_1.1'
# Serializing json
json_object = json.dumps(dictionary, indent=4)
 
# Writing to sample.json
with open(os.path.join(save_dir,"hyperparams.json"), "w") as outfile:
    outfile.write(json_object)