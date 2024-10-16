class Hyperparameters:
    input_size = 40
    hidden_size = 768
    num_layers = 3
    projection_size = 256
    num_epochs = 250
    batch_size = 64
    learning_rate = 0.001
    dataset = "data_vox"
    train_eval_dataset = "data_vox"
    eval_dataset = "data_vox"
    num_speakers = 5
    dyn_num_speakers = True
    min_num_speakers = 3
    max_num_speakers = 9
    num_segments = 200
    vad_mode = 3
    noise = True
    noise_dir = "noise"
