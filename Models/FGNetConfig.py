# This is FGNet configurations

props = {
    'LOG_PATH': './FGNet_logs',
    'MODEL_PATH': './models/FGNet_weights-improvement-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
    'IMAGE_SIZE':  64,
    'INPUT_SHAPE': (64, 64, 1),

    # Training
    'EPOCHS': 100,
    'BATCH_SIZE': 30
}