import os
import math
from typing import List
from bisect import bisect_right

import torch
import torchaudio
from tqdm import tqdm
from torch.nn import functional as F


class ChuncksDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filenames: List[str],
        max_length: float = None,
        stride: float = None,
        pad: bool = True,
        target_samplig_rate: int = 16_000,
        noisy_base_dir: str = None,
        clean_base_dir: str = None,
    ):
        self.filenames = filenames
        self.max_length = max_length * target_samplig_rate
        self.stride = stride * target_samplig_rate
        self.pad = pad
        self.target_samplig_rate = target_samplig_rate
        self.noisy_base_dir = noisy_base_dir
        self.clean_base_dir = clean_base_dir

        # Initialize num_examples and cumulative_counts
        self.num_examples_per_file = []
        self.cumulative_counts = []
        cumulative = 0

        for filenames in tqdm(self.filenames, desc="Loading dataset"):
            filepath = os.path.join(noisy_base_dir, filenames)
            file_length = self._get_audio_length(filepath)

            # assert self._get_audio_length(filepath) == self._get_audio_length(os.path.join(clean_base_dir, filenames)), f"Audio lengths are not equal: {filepath} | {os.path.join(clean_base_dir, filenames)}"

            if self.max_length is None:
                examples_per_file = 1
            elif file_length < self.max_length:
                examples_per_file = 1 if pad else 0
            elif pad:
                examples_per_file = int(math.ceil((file_length - self.max_length) / self.stride) + 1)
            else:
                examples_per_file = (file_length - self.max_length) // self.stride + 1

            self.num_examples_per_file.append(examples_per_file)
            cumulative += examples_per_file
            self.cumulative_counts.append(cumulative)

    def __len__(self):
        return self.cumulative_counts[-1]

    def _get_audio_length(self, file: str) -> int:
        # Get the length of the audio file in seconds
        info = torchaudio.info(file)
        return info.num_frames // info.sample_rate

    def _load_audio(self, filepath: str, frame_offset: int, num_frames: int) -> torch.Tensor:
        # Load the specified segment from the audio file
        samples, sr = torchaudio.load(filepath)

        # Transform to mono
        if samples.shape[0] > 1:
            samples = samples.mean(dim=0, keepdim=True)

        if sr != self.target_samplig_rate:
            samples = torchaudio.transforms.Resample(sr, self.target_samplig_rate)(samples)
            sr = self.target_samplig_rate

        # Trim the audio to the desired length
        samples = samples[..., int(frame_offset): int(frame_offset + num_frames)]

        # Pad the audio to the desired length
        if num_frames and samples.shape[-1] < num_frames and self.pad:
            samples = F.pad(samples, (0, int(num_frames - samples.shape[-1])))

        return samples, sr

    def __getitem__(self, index):
        # Binary search to find the right file index
        file_idx = bisect_right(self.cumulative_counts, index)
        filename = self.filenames[file_idx]

        # Get the filepaths
        noisy_filepath = os.path.join(self.noisy_base_dir, filename)
        clean_filepath = os.path.join(self.clean_base_dir, filename)

        # Compute the local index within the file
        local_index = index - (self.cumulative_counts[file_idx - 1] if file_idx > 0 else 0)

        frame_offset = 0
        num_frames = 0

        if self.max_length is not None:
            frame_offset = self.stride * local_index
            num_frames = self.max_length

        # Load the audios
        noisy, _ = self._load_audio(noisy_filepath, frame_offset, num_frames)
        clean, _ = self._load_audio(clean_filepath, frame_offset, num_frames)

        # Pad the output if it's shorter than the desired length
        return noisy, clean
