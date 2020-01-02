from _board import Board
import os
import pathlib
import random
import numpy as np
import sgf
import torch


class SelfPlayLoader(torch.utils.data.Dataset):
    def load(self, names, device='cpu'):
        def move_to_index(move):
            (p0, p1), = move
            MOVE = 'abcdefghi'
            return MOVE.find(p0) + MOVE.find(p1) * 9

        self._features = []
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
                        (k, move), (_, sl_dist_str) = prop.items()
                        index = move_to_index(move)
                        bw = 0 if k == 'B' else 1

                        feature = b.features
                        v_label = 1 if bw == winner else -1
                        # p_label
                        moves = sl_dist_str[0].split(',')[:-1]
                        counts = 0
                        for mv in moves:
                            _, c = mv.split(':')
                            counts += int(c)
                        p_label = [0] * 81
                        for mv in moves:
                            m, c = mv.split(':')
                            p_label[int(m)] = int(c) / counts

                        b.place(bw, index)

                        features.append((torch.Tensor(feature).to(device),
                                         torch.Tensor(p_label).to(device),
                                         torch.Tensor([v_label]).to(device)))
                        # print(k, move, bw, index)
                    elif 'RE' in prop:
                        winner = 0 if 'B+' in prop['RE'][0] else 1
                self._features.extend(features)
        self._len = len(self._features)

    @staticmethod
    def _isomorphism(f, i):
        return torch.rot90(f, i, (1, 2)) if i < 4 else torch.rot90(
            f.transpose(1, 2), i, (1, 2))

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
