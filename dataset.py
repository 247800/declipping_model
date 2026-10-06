import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch import nn

class AudioDataset(Dataset):
    def __init__(
        self,
        directory=None,
        sr=22050,
        stereo=False,
    ):
        super(AudioDataset, self).__init__()
        self.sr = sr
        self.directory = directory
        self.seg_len = sr  # One second of audio
        self.stereo = stereo
        self.files = sorted(os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".wav"))

        if not self.files:
            raise FileNotFoundError( f"No .wav files found in '{directory}'.")

        self.segments = []

        for audio_file in self.files:
            audio, loaded_sr = torchaudio.load(audio_file, normalize=True)
            if loaded_sr != self.sr:
                raise ValueError(
                    f"{audio_file}: expected sample rate "
                    f"{self.sr}, but got {loaded_sr}"
                )
            # Include the final incomplete segment.
            for start_sample in range(0, audio.size(1), self.seg_len):
                self.segments.append((audio_file, start_sample))

    def load_segment(self, audio_file, start_sample):
        audio, loaded_sr = torchaudio.load(audio_file, frame_offset=start_sample, num_frames=self.seg_len, normalize=True)
        if loaded_sr != self.sr:
            raise ValueError(
                f"{audio_file}: expected sample rate "
                f"{self.sr}, but got {loaded_sr}"
            )
        if self.stereo:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.size(1) < self.seg_len:
            audio = nn.functional.pad( audio,(0, self.seg_len - audio.size(1)))
        return audio

    def print_params(self):
        print(f"Path:                   {self.directory}")
        print(f"Sample rate:            {self.sr}")
        print(f"Segment length:         {self.seg_len}")
        print(f"Stereo:                 {self.stereo}")
        print(f"Number of files:        {len(self.files)}")
        print(f"Number of segments:     {len(self.segments)}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        audio_file, start_sample = self.segments[index]
        return self.load_segment(audio_file, start_sample)

if __name__ == "__main__":
    dataset = AudioDataset(directory="dataset", sr=22050, stereo=False)
    dataset.print_params()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    print(batch.shape)