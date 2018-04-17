# This is AGNet configurations

props = {
    'LOG_PATH': './AGNet_logs',
    'MODEL_PATH': './AGNet_models',
    'WEIGHT_NAME': 'AGNet-weights-improvement-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
    'IMAGE_SIZE':  64,
    'IMAGE_DEPTH':  1,
    'INPUT_SHAPE': (64, 64, 1),

    # Training
    'EPOCHS': 1,
    'BATCH_SIZE': 200
}