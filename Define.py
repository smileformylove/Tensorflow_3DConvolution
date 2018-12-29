
LAYERS = [2, 2, 3, 3]
FEATURES = [16, 32, 64, 128]

#DROP_OUT = 0.7

SEQUENCE = 30
IMAGE_WIDTH = 56
IMAGE_HEIGHT = 56
IMAGE_CHANNEL = 3

BATCH_SIZE = 8
MAX_EPOCHS = 300

LABEL_DIC = {"ApplyEyeMakeup":0,
             "Archery" : 1,
             "Basketball" : 2,
             "Biking" : 3,
             "Diving" : 4}
CLASSES = len(LABEL_DIC.keys())