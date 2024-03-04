# beat_tracker
Julien Guinot, Beat Tracking Assigment for ECS7006 MUSIC INFORMATICS

This project reproduces the implementation of Temporal Convolutional Networks for musical beat tracking:

[1] MatthewDavies, E. P., and Sebastian Böck. "Temporal convolutional networks for musical audio beat tracking." 2019 27th European Signal Processing Conference (EUSIPCO). IEEE, 2019.

As well as a small implementation addition consisting of a longformer sliding attention block to avoid post-processing of downbeats conditioned on beats:

[2] Beltagy, Iz, Matthew E. Peters, and Arman Cohan. "Longformer: The long-document transformer." arXiv preprint arXiv:2004.05150 (2020).

## environment setup

Two environments are available for this project, one minimal one required to run only the inference of the model. To install this environment, run

    pip install -r requirements_minimal.txt

For further experiments such as training and logging, which require wandb and pyorch lightning, a full environment can be installed by running 

    pip install -r requirements_full.txt

I recommend these be installed in a blank virtual environment due to the delicate balance between madmom and pytorch lightning requirements for numpy and numba.

## Running the beat tracker.

All required files should be in the repo to run the beat tracker, including config and model checkpoints. to call the beat tracker, simple run the following lines:

    from tracker import BeatTracker
    tracker = BeatTracker()
    beats,downbeats = tracker(your_path)


For further control and to experiment with other checkpoints, it is possible to provide config and checkpoint paths

    from tracker import BeatTracker
    tracker = BeatTracker(
        config = your_config_path,
        checkpoint_path = your_checkpoint_path
        model_class = YourModelClass
        )



