import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import nn
import random

class AudioDataset(Dataset):
    """
    Load non-overlapping 1s-segments of audio from dataset files.
    Split files into training and testing subsets.
    Keep only segments containing at least one sample
    that reaches or exceeds clipping threshold.
    Zero-pad the final incomplete segment of each file.
    """
    def __init__(
        self,
        directory=None,
        sr=22050,
        seg_len=None,
        mono=True,
        clipping_threshold=0.1,
        filter_unsaturated=True,
        train=True,
        ds_percent=80,
        seed=42,
    ):
        super(AudioDataset, self).__init__()

        if directory is None:
            raise ValueError("Dataset directory must be provided.")
        else:
            self.directory = directory

        self.sr = sr
        self.seg_len = sr if seg_len is None else seg_len

        if self.seg_len != self.sr:
            raise ValueError(
                "For 1-second segments, seg_len must equal sr."
            )

        self.mono = mono
        self.clipping_threshold = clipping_threshold

        if self.clipping_threshold <= 0:
            raise ValueError("Clipping_threshold must be a positive number.")

        self.filter_unsaturated = filter_unsaturated
        self.train = train

        if not 0 < ds_percent < 100:
            raise ValueError("ds_percent must be between 0 and 100, exclusive.")

        self.ds_percent = ds_percent
        self.seed = seed

        files = sorted(
            os.path.join(self.directory, file)
            for file in os.listdir(self.directory)
            if file.lower().endswith(".wav")
            and os.path.isfile(os.path.join(self.directory, file))
        )

        if not files:
            raise FileNotFoundError(f"No .wav files found in '{self.directory}'.")

        random.Random(self.seed).shuffle(files)

        # Divide files for training and testing
        self.files_total = len(files)
        dataset_idx = int(self.files_total * (self.ds_percent / 100))

        if not 0 < dataset_idx < self.files_total:
            raise ValueError(
                f"Cannot create non-empty train/test subsets from "
                f"{self.files_total} files with "
                f"ds_percent={self.ds_percent}. "
                f"Calculated split: {dataset_idx} training files and "
                f"{self.files_total - dataset_idx} test files."
            )

        if self.train:
            self.files = files[:dataset_idx]  # First part of the dataset for training
        else:
            self.files = files[dataset_idx:]  # Remaining files for testing

        self.segments = []
        self.discarded_segments = 0

        # Building index of non-overlapping 1s segments (with saturation)
        for audio_file in self.files:
            audio, loaded_sr = torchaudio.load(audio_file, normalize=True)

            if loaded_sr != self.sr:
                raise ValueError(
                    f"{audio_file}: expected sample rate "
                    f"of {self.sr} but got {loaded_sr}"
                )

            if self.mono and audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)

            for start_sample in range(0, audio.size(1), self.seg_len):
                segment = audio[:, start_sample:start_sample + self.seg_len]

                # Keep segments whose clipped measurement reaches or exceeds threshold
                if self.filter_unsaturated:
                    reaches_threshold = (segment.abs() >= self.clipping_threshold).any().item()
                    if not reaches_threshold:
                        self.discarded_segments += 1
                        continue

                # With filtering disabled, retain every segment
                self.segments.append((audio_file, start_sample))

    def load_segment(self, audio_file, start_sample):
        audio, loaded_sr = torchaudio.load(
            audio_file,
            frame_offset=start_sample,
            num_frames=self.seg_len,
            normalize=True,
        )
        if loaded_sr != self.sr:
            raise ValueError(
                f"{audio_file}: expected sample rate "
                f"of {self.sr} but got {loaded_sr}"
            )
        if self.mono and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.size(1) < self.seg_len:
            audio = nn.functional.pad(audio,(0, self.seg_len - audio.size(1)))

        return audio

    def print_params(self):
        print(f"Path:                   {self.directory}")
        print(f"Sample rate:            {self.sr}")
        print(f"Segment length:         {self.seg_len}")
        print(f"Mono:                   {self.mono}")
        print(f"Clipping threshold:     {self.clipping_threshold}")
        print(f"Total files:            {self.files_total}")
        print(f"Training split (%):     {self.ds_percent}")
        print(f"Subset:                 {'train' if self.train else 'test'}")
        print(f"Files in subset:        {len(self.files)}")
        print(f"Filter unsaturated:     {self.filter_unsaturated}")
        print(f"Retained segments:      {len(self.segments)}")
        print(f"Discarded segments:     {self.discarded_segments}")


    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        audio_file, start_sample = self.segments[index]
        return self.load_segment(audio_file, start_sample)


if __name__ == "__main__":
    dataset = AudioDataset(
        directory="dataset",
        sr=22050,
        seg_len=22050,
        mono=True,
        clipping_threshold=0.1,
        train=False
    )
    dataset.print_params()
    if len(dataset) == 0:
        raise RuntimeError("No audio segments are available in the selected subset.")
    dataloader = DataLoader(dataset, batch_size=35)
    batch = next(iter(dataloader))
    print(batch.shape)