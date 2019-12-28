from _board import Board
import os
import random
import re
import numpy as np
import sgf
import torch


class SelfPlayLoader(torch.utils.data.Dataset):
    def __init__(self, root, device='cpu', load_policy=False):
        self._features = []
        self._policy = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                name = os.path.join(dirpath, filename)
                if name[-4:] == '.sgf':
                    self._parse_sgf(name, device, load_policy)
                elif load_policy and name[-4:] == '.log':
                    self._parse_policy(name, device)
        if load_policy:
            assert (len(self._policy) == len(self._features))
            self._features = [[
                (f, p, v) for (f, _, v), p in zip(features, policy)
            ] for features, policy in zip(self._features, self._policy)]
        self._len = len(self._features)

    def _parse_sgf(self, filename, device, load_policy):
        def move_to_index(move):
            (p0, p1), = move
            MOVE = 'abcdefghi'
            return MOVE.find(p0) + MOVE.find(p1) * 9

        f = open(filename)
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
                    # (feature, p_label, v_label)
                    feature = b.features
                    p_label = index
                    v_label = 1 if bw == winner else -1
                    # place on board
                    b.place(bw, index)
                    # feature tensors
                    p_tensor = None
                    if not load_policy:
                        p_tensor = torch.eye(81)[p_label].to(device)

                    features.append(
                        (torch.Tensor(feature).to(device), p_tensor,
                         torch.Tensor([v_label]).to(device)))
                elif 'RE' in prop:
                    winner = 0 if 'B+' in prop['RE'][0] else 1
            self._features.append(features)

    def _parse_policy(self, filename, device):
        GAME_RE = re.compile(r"Game \d+\n(.+?)resign\n", re.DOTALL)
        DIST_RE = re.compile(
            r"DIST=BEGIN==========\n(.+?)\n==========DIST=END", re.DOTALL)

        f = open(filename)
        content = f.read()
        f.close()

        games = list(GAME_RE.findall(content))
        for game in games:
            assert ('DIST' in game)
            dists = DIST_RE.findall(game)
            policies = []
            for dist in dists:
                dist = re.sub(r'[BW][<>].*\n', '', dist)
                dist = [d.split() for d in dist.split('\n')]
                policy = np.zeros(81, dtype=np.float)
                for _, move, count in dist:
                    policy[int(move)] = int(count)
                policy /= sum(policy)
                policies.append(torch.Tensor(policy).to(device))
            self._policy.append(policies)

    def sample(self, batch_size=64):
        index = np.random.randint(0, self._len, batch_size)
        batch = [random.choice(self._features[idx]) for idx in index]
        return tuple(torch.stack(b) for b in zip(*batch))

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._features[index]
