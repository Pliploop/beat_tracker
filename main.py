# an example file that takes an audio file from the path and returns the beats and downbeats of the audio file

from tracker import BeatTracker

if __name__ == "__main__":
    beat_tracker = BeatTracker()
    beat_times, downbeat_times = beat_tracker('path/to/audio/file')
    