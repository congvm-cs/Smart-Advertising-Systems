# This is AGNet configurations

props = {
    'LOG_PATH': './AGNet_logs',
    'MODEL_PATH': './AGNet_models',
    'WEIGHT_NAME': 'AGNet_weights_1-improvement-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
    'IMAGE_SIZE':  64,
    'IMAGE_DEPTH':  3,
    'INPUT_SHAPE': (64, 64, 3),

    # Training
    'EPOCHS': 100,
    'BATCH_SIZE': 200
}