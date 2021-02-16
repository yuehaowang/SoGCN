import torch
import numpy as np
import pickle

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3 or len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])[..., :3]
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')

            result = torch.tensor(self.results[run])
            if result.shape[-1] >= 4:
                train_times = result[..., 3]
                print(f' Time / Epoch: {train_times.mean():.2f}')
        else:
            result = 100 * torch.tensor(self.results)[..., :3]

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            r = torch.tensor(self.results)
            if r.shape[-1] >= 4:
                r = r[..., 3].reshape(-1)
                print(f' Time / Epoch: {r.mean():.2f}  ± {r.std():.2f}')

    def save(self, filename):
        dict_save = {}
        for i, res in enumerate(self.results):
            res = np.array(res)
            train_curve = res[:, 0]
            val_curve = res[:, 1]
            test_curve = res[:, 2]
            dict_save['run_%d' % i] = dict(
                train = train_curve,
                valid = val_curve,
                test = test_curve
            )
        with open(filename, 'wb') as f:
            pickle.dump(dict_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        self.results = []
        with open(filename, 'rb') as f:
            dict_save = pickle.load(f)

            for i, k in enumerate(sorted(dict_save.keys())):
                data = dict_save[k]

                train_curve = data['train']
                val_curve = data['valid']
                test_curve = data['test']
                results_run = np.stack([train_curve, val_curve, test_curve], -1).tolist()

                self.results.append(results_run)
