import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import nn

class AudioDataset(Dataset):
    """
    Load 1s-segments of audio from dataset files.
    Keep only segments with at least one sample
    that reaches or exceeds clipping threshold,
    discard everything else. Zero-pad final incomplete segments.
    Clip the remaining segments before returning them.
    """
    def __init__(
        self,
        directory=None,
        sr=22050,
        seg_len=None,
        mono=True,
        clipping_threshold=0.1,
        apply_clipping=True,
        filter_unsaturated=True,
    ):
        super(AudioDataset, self).__init__()
        self.apply_clipping = apply_clipping
        self.sr = sr
        self.directory = directory
        self.seg_len = sr if seg_len is None else seg_len

        if self.seg_len != self.sr:
            raise ValueError(
                "For 1-second segments, seg_len must equal sr."
            )
        self.mono = mono
        self.appy_clipping = apply_clipping
        self.clipping_threshold = clipping_threshold

        if self.clipping_threshold <= 0:
            raise ValueError("Clipping_threshold must be positive.")

        self.filter_unsaturated = filter_unsaturated
        self.files = [
            os.path.join(directory, file)
            for file in os.listdir(directory)
            if file.endswith(".wav")
        ]

        if not self.files:
            raise FileNotFoundError(f"No .wav files found in '{directory}'.")

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

                # Keep segments whose clipped measurement reaches threshold
                if self.filter_unsaturated:
                    if (segment.abs() >= self.clipping_threshold).any().item():
                        self.segments.append((audio_file, start_sample))
                    else:
                        self.discarded_segments += 1

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

        # Return (clipped) measurement
        if self.apply_clipping:
            return torch.clamp(audio,-self.clipping_threshold, self.clipping_threshold)
        else:
            return audio

    def print_params(self):
        print(f"Path:                   {self.directory}")
        print(f"Sample rate:            {self.sr}")
        print(f"Segment length:         {self.seg_len}")
        print(f"Mono:                   {self.mono}")
        print(f"Clipping threshold:     {self.clipping_threshold}")
        print(f"Number of files:        {len(self.files)}")
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
    )
    dataset.print_params()
    if len(dataset) == 0:
        raise RuntimeError("No segments with saturation were found.")
    dataloader = DataLoader(dataset, batch_size=35)
    batch = next(iter(dataloader))
    print(batch.shape)