import torch
import numpy as np
import random


class SelfPlayLoader(torch.utils.data.Dataset):
    def __init__(self, root, device='cpu'):
        from _board import Board
        import sgf
        import pathlib

        def move_to_index(move):
            (p0, p1), = move
            MOVE = 'abcdefghi'
            return MOVE.find(p0) + MOVE.find(p1) * 9

        self._features = []
        names = pathlib.Path(root).rglob('*.sgf')
        for name in names:
            f = open(name)
            collection = sgf.parse(f.read())
            f.close()

            for game in collection:
                b = Board()
                winner = 3
                features = []
                for node in game:
                    prop = node.properties
                    if 'B' in prop or 'W' in prop:
                        assert (winner < 3)
                        (k, move), = prop.items()
                        index = move_to_index(move)
                        bw = 0 if k == 'B' else 1

                        feature = b.features
                        p_label = index
                        v_label = 1 if bw == winner else -1

                        b.place(bw, index)

                        features.append(
                            (torch.Tensor(feature).to(device),
                             torch.LongTensor([p_label]).to(device),
                             torch.Tensor([v_label]).to(device)))
                        # print(k, move, bw, index)
                    elif 'RE' in prop:
                        winner = 0 if 'B+' in prop['RE'][0] else 1
                self._features.append(features)
        self._len = len(self._features)

    def sample(self, batch_size=64):
        index = np.random.randint(0, self._len, batch_size)
        batch = [random.choice(self._features[idx]) for idx in index]
        return tuple(torch.stack(b) for b in zip(*batch))

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._features[index]
