import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
import random
import soundfile as sf
from pathlib import Path

class AudioDataset(Dataset):
    """
    Load non-overlapping 1s-segments of audio from dataset files.
    Split files into training and testing subsets.
    Preload retained mono segments into self.data [N, seg_len].
    Store aligned metadata in self.df: index, file_name, startframe.
    Optionally keep only segments containing at least one sample that reaches or exceeds clipping threshold.
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
            raise ValueError("For 1-second segments, seg_len must equal sr.")

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
            os.path.join(root, file)
            for root, _, filenames in os.walk(self.directory)
            for file in filenames
            if file.lower().endswith(".wav")
        )

        if not files:
            raise FileNotFoundError(f"No .wav files found in '{self.directory}'.")

        random.Random(self.seed).shuffle(files)

        # Calculate train/test subset sizes from the complete dataset
        self.files_total = len(files)
        dataset_idx = int(self.files_total * (self.ds_percent / 100))
        test_files_count = self.files_total - dataset_idx

        if not 0 < dataset_idx < self.files_total:
            raise ValueError(
                f"Cannot create non-empty train/test subsets from "
                f"{self.files_total} files with ds_percent={self.ds_percent}."
                f"Calculated split: {dataset_idx} training files and "
                f"{test_files_count} test files."
            )

        # create train/test subset
        if self.train:
            self.files = files[:dataset_idx]  # First part of the dataset for training
        else:
            self.files = files[dataset_idx:]

        if not self.mono:
            raise ValueError("Output shape [N, seg_len] requires mono=True.")

        self.segments = []
        self.discarded_segments = 0
        segment_tensors = []
        metadata = []

        for audio_file in self.files:
            samples, loaded_sr = sf.read(audio_file, dtype="float32", always_2d=True)

            if loaded_sr != self.sr:
                raise ValueError(f"{audio_file}: expected sample rate of {self.sr} but got {loaded_sr}")

            audio = torch.from_numpy(samples.T.copy()) # [n_channels, n_samples]
            if audio.size(0) > 1:     # verify number of channels
                audio = audio.mean(dim=0, keepdim=True)   # [1, n_samples]

            file_name = os.path.relpath(audio_file, self.directory)
            for startframe in range(0, audio.size(1), self.seg_len):
                segment = audio[0, startframe:startframe + self.seg_len]   # split file into segments

                # Keep segments whose clipped measurement reaches or exceeds threshold
                if self.filter_unsaturated:
                    reaches_threshold = (segment.abs() >= self.clipping_threshold).any().item()
                    if not reaches_threshold:
                        self.discarded_segments += 1
                        continue

                # zero-pad incomplete segments
                if segment.numel() < self.seg_len:
                    segment = nn.functional.pad(
                        segment, (0, self.seg_len - segment.numel())
                    )

                index = len(segment_tensors)
                segment_tensors.append(segment.clone())  # copy segment into segment_tensors
                self.segments.append((audio_file, startframe))  # save its origin
                metadata.append({
                    "index": index,
                    "file_name": file_name,
                    "startframe": startframe,
                })

        # stack segments of shape [seg_len] into a tensor of shape [n_segments, seg_len]
        self.data = (
            torch.stack(segment_tensors, dim=0)
            if segment_tensors
            else torch.empty((0, self.seg_len), dtype=torch.float32)
        )
        self.df = pd.DataFrame(
            metadata, columns=["index", "file_name", "startframe"]
        )
        self._segment_indices = {
            key: index for index, key in enumerate(self.segments)
        }

    def load_segment(self, audio_file, start_sample):
        index = self._segment_indices[(audio_file, start_sample)]
        return self.data[index].clone()


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
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index].clone()


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    training_run = "20261023_105951_gtzan100_46dc"
    run_dir = (project_dir / "exports_train" / training_run)
    train_files_json = run_dir / "metadata" / "train_files.json"

    dataset = AudioDataset(
        directory="genres_original",
        sr=22050,
        seg_len=22050,
        mono=True,
        clipping_threshold=0.1,
        filter_unsaturated=True,
        train=True,
        ds_percent=75,

    )
    dataset.print_params()
    print(f"Data tensor shape:      {dataset.data.shape}")
    print(f"Metadata shape:         {dataset.df.shape}")
    print(dataset.df.head())
    if len(dataset) == 0:
        raise RuntimeError("No audio segments are available in the selected subset.")
    dataloader = DataLoader(dataset, batch_size=35)
    batch = next(iter(dataloader))
    print(batch.shape)