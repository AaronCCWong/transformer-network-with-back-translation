import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.transformer import Transformer
from transformer.utils import padding_mask, subsequent_mask
from translator.beam import Beam
from translator.utils import get_inst_idx_to_tensor_position_map, collect_active_part


class Translator(object):
    def __init__(self, src_vocab, tgt_vocab, src_vocab_size, tgt_vocab_size, args):
        self.max_seq_length = args.max_seq_length
        self.device = args.device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.beam_size = args.beam_size

        model = Transformer(src_vocab_size, tgt_vocab_size, args.device)
        model.load_state_dict(torch.load(args.model))
        model = model.to(args.device)
        self.model = model
        self.model.eval()

    def collate_active_info(self, src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

        active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, self.beam_size)
        active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, self.beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

    def predict_word(self, dec_seq, src_seq, enc_output, n_active_inst):
        src_mask = padding_mask(src_seq, self.src_vocab)
        tgt_mask = padding_mask(dec_seq, self.tgt_vocab) & subsequent_mask(dec_seq).to(self.device)

        dec_seq = self.model.tgt_embedding(dec_seq) * math.sqrt(self.model.d_model)
        dec_seq = self.model.positional_encoder2(dec_seq)

        dec_output = self.model.decoder(dec_seq, enc_output, src_mask, tgt_mask)
        dec_output = dec_output[:, -1, :]
        word_prob = F.log_softmax(self.model.linear(dec_output), dim=1)
        word_prob = word_prob.view(n_active_inst, self.beam_size, -1)

        return word_prob

    def beam_decode_step(self, inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map):
        n_active_inst = len(inst_idx_to_position_map)

        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
        dec_seq = dec_partial_seq.view(-1, len_dec_seq)

        word_prob = self.predict_word(dec_seq, src_seq, enc_output, n_active_inst)
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_dec_beams[inst_idx].advance(
                word_prob[inst_position])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    def translate_batch(self, src_seq):
        src_seq = src_seq.to(self.device)

        out = self.model.src_embedding(src_seq) * math.sqrt(self.model.d_model)
        out = self.model.positional_encoder1(out)

        src_mask = padding_mask(src_seq, self.src_vocab)
        src_enc = self.model.encoder(out, src_mask)

        n_inst, len_s, d_h = src_enc.size()
        src_seq = src_seq.repeat(1, self.beam_size).view(n_inst * self.beam_size, len_s)
        src_enc = src_enc.repeat(1, self.beam_size, 1).view(n_inst * self.beam_size, len_s, d_h)

        inst_dec_beams = [Beam(self.beam_size, self.tgt_vocab, device=self.device) for _ in range(n_inst)]

        active_inst_idx_list = list(range(n_inst))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        for len_dec_seq in range(1, self.max_seq_length + 1):
            active_inst_idx_list = self.beam_decode_step(inst_dec_beams,
                                                            len_dec_seq,
                                                            src_seq,
                                                            src_enc,
                                                            inst_idx_to_position_map)

            if not active_inst_idx_list:
                break

            src_seq, src_enc, inst_idx_to_position_map = self.collate_active_info(src_seq,
                                                                                    src_enc,
                                                                                    inst_idx_to_position_map,
                                                                                    active_inst_idx_list)

        batch_hyp, batch_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            batch_scores += [scores[:1]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:1]]
            batch_hyp += [hyps]
        return batch_hyp, batch_scores
