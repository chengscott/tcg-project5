import os
import random
import numpy as np
import torch


class SelfPlayLoader(torch.utils.data.Dataset):
    def __init__(self, device):
        super().__init__()
        self._device = device
        self._features = []
        self._features_len = []

    def load(self, names, append=False):
        assert (isinstance(names, list))
        device = self._device
        if not append:
            self._features = []
            self._features_len = []
        for name in names:
            feature = np.load(name + '_feature.npz')
            pn = np.load(name + '_label_pn.npz')
            vn = np.load(name + '_label_vn.npz')
            # feature tensors
            features = [(
                torch.Tensor(f).to(device),
                torch.Tensor(p).to(device),
                torch.Tensor(v).to(device),
            ) for f, p, v in zip(feature, pn, vn)]
            self._features.extend(features)
            self._features_len.append(len(features))
        self._len = len(self._features)
        return self._features_len

    def append(self, names):
        """circular append"""
        nlen = len(names)
        flen = sum(self._features_len[:nlen])
        self._features = self._features[flen:]
        self._features_len = self._features_len[nlen:]
        return self.load(names, append=True)

    @staticmethod
    def _isomorphism(f, i):
        # assert (0 <= i <= 7)
        f = f if i < 4 else f.transpose(1, 2)
        return torch.rot90(f, i % 4, (1, 2))

    def sample(self, batch_size=64):
        index = np.random.randint(0, self._len, batch_size)
        batch = [self._features[idx] for idx in index]

        isom = SelfPlayLoader._isomorphism
        isom_index = map(int, np.random.randint(0, 8, batch_size))
        batch = [(isom(f, i), isom(p.reshape(1, 9, 9), i).reshape(81), v)
                 for i, (f, p, v) in zip(isom_index, batch)]
        return tuple(torch.stack(b) for b in zip(*batch))

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._features[index]


class SupervisedHelper:
    def __init__(self, dataset, dataset_prefix, device='cpu'):
        self._epoch_start = 0
        with open(dataset) as f:
            self._names = [
                os.path.join(dataset_prefix, name.strip()) for name in f
            ]
        self._loader = SelfPlayLoader(device=device)

    def init_epoch(self, epoch):
        self._epoch_start = epoch

    def __iter__(self):
        epoch_total = len(self._names) // 25
        for epoch in range(self._epoch_start, epoch_total):
            # load 25 timesteps (5000 games)
            names = self._names[25 * epoch:25 * (epoch + 1)]
            self._loader.load(names)
            num_samples = 8 * len(self._loader)  # repeat 8 times
            yield epoch, names, num_samples, self._loader


class SupervisedShiftHelper(SupervisedHelper):
    """Shift Training"""
    def __iter__(self):
        # first 20 * 25 timesteps (20 * 5000 games)
        base = 20 * 25
        names = self._names[:base]
        flens = self._loader.load(names)

        epoch_total = (len(self._names) - base) // 25
        for epoch in range(epoch_total):
            repeat = 1 if epoch < 80 else (3 if epoch < 140 else 5)
            num_samples = repeat * sum(flens[-25:])  # repeat x times
            yield epoch, names, num_samples, self._loader
            # circular append 25 timesteps (5000 games)
            names = self._names[25 * epoch + base:25 * (epoch + 1) + base]
            flens = self._loader.append(names)

        # last epoch
        num_samples = 20 * sum(flens[-25:])  # repeat 20 times
        yield epoch_total, names, num_samples, self._loader


class SupervisedEndHelper(SupervisedHelper):
    """End Training"""
    def __iter__(self):
        # load the last 25 * 25 timesteps (25 * 5000 games)
        names = self._names[-25 * 25:]
        flens = self._loader.load(names)
        num_samples = 1 * sum(flens[-25:])  # repeat 1 time
        yield 0, names, num_samples, self._loader

        for epoch in range(self._epoch_start + 1, 100):
            yield epoch, [], num_samples, self._loader
