import numpy as np
import torch

from transformer.utils import CONSTANTS

class Beam():
    def __init__(self, size, tgt_vocab, device=False):
        self.size = size
        self.tgt_vocab = tgt_vocab
        self._done = False

        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        self.prev_ks = []

        self.next_ys = [torch.full((size,), tgt_vocab.stoi[CONSTANTS['pad']], dtype=torch.long, device=device)]
        self.next_ys[0][0] = tgt_vocab.stoi[CONSTANTS['start']]

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        num_words = word_prob.size(1)
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        if self.next_ys[-1][0].item() == self.tgt_vocab.stoi[CONSTANTS['end']]:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.tgt_vocab.stoi[CONSTANTS['start']]] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))
