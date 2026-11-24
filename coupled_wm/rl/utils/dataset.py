# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


class CriticDataset:
    """Mini-batches over the flattened (time-major) critic buffers.

    Unshuffled (SHAC's behaviour): each batch is a contiguous block of rollout steps, visited in
    the same order every critic iteration; the last batch may be smaller.
    """

    def __init__(self, batch_size, obs, target_values):
        self.obs = obs.view(-1, obs.shape[-1])
        self.target_values = target_values.view(-1)
        self.batch_size = batch_size
        self.length = ((self.obs.shape[0] - 1) // self.batch_size) + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.obs.shape[0])
        return {'obs': self.obs[start_idx:end_idx, :], 'target_values': self.target_values[start_idx:end_idx]}
