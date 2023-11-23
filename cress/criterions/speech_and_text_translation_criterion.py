# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss,
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class SpeechAndTextTranslationCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_finetune: bool = field(
        default=False,
        metadata={"help": "st + mt multi-task finetune"},
    )

@register_criterion(
    "speech_and_text_translation", dataclass=SpeechAndTextTranslationCriterionConfig
)
class SpeechAndTextTranslationCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        mt_finetune=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.mt_finetune = mt_finetune

    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target

    def compute_jsd_loss(self, st_lprobs, mt_lprobs, st_target, mt_target, ignore_index):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        pad_mask = st_target.eq(ignore_index)
        kl_loss_st.masked_fill_(pad_mask, 0.0)
        pad_mask = mt_target.eq(ignore_index)
        kl_loss_mt.masked_fill_(pad_mask, 0.0)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

    def compute_jsd_loss_without_pad(self, st_lprobs, mt_lprobs):
        kl_loss_st = F.kl_div(mt_lprobs, st_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_mt = F.kl_div(st_lprobs, mt_lprobs, log_target=True, reduction="none").sum(-1)
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mt = kl_loss_mt.sum()
        kl_loss = (kl_loss_st + kl_loss_mt) / 2.0
        return kl_loss

    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]

        encoder_out = model.encoder(**audio_input)
        audio_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        with torch.no_grad():
            all_tokens, st_correct_tokens, st_correct_matrix, st_probs = self.collect_result(model, audio_output, sample)

        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, audio_output, sample, reduce=reduce)
        return loss, lprobs, target, all_tokens, st_correct_tokens, st_correct_matrix, st_probs, encoder_out
    
    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]

        encoder_out = model.encoder(**text_input)
        text_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )

        with torch.no_grad():
            all_tokens, mt_correct_tokens, mt_correct_matrix, mt_probs = self.collect_result(model, text_output, sample)

        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, text_output, sample, reduce=reduce)
        return loss, lprobs, target, all_tokens, mt_correct_tokens, mt_correct_matrix, mt_probs, encoder_out

    def forward_x_cross_s(self, model, sample, reduce):
        text_input = {
            "audio": sample["net_input"]["audio"],
            "audio_lengths": sample["net_input"]["audio_lengths"],
            "source": sample["net_input"]["source"],
        }
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        encoder_out = model.encoder.forward_x_cross_s(**text_input)
        decoder_out = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        loss, _, lprobs, target = self.compute_loss_with_lprobs(model, decoder_out, sample, reduce=reduce)
        return loss, lprobs, target

    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss

    def collect_result(self, model, decoder_output, sample):
        probs = model.get_normalized_probs(decoder_output, log_probs=False)
        lprobs = model.get_normalized_probs(decoder_output, log_probs=True)
        target = model.get_targets(sample, decoder_output)
        pred = torch.argmax(lprobs, dim=-1)
        pad_mask = target.eq(self.padding_idx)
        batch_tokens = int(torch.sum(~pad_mask).item())
        correct_matrix = pred.eq(target) & (~pad_mask)
        correct_tokens = int(torch.sum(correct_matrix).item())
        return batch_tokens, correct_tokens, correct_matrix.detach(), probs.detach()

    def forward_hard_words(self, model, sample, audio_enc, text_enc, st_probs, mt_probs, st_mt_incorrect_matrix):
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]
        target = sample["target"]
        # tokens, dim
        st_selected = st_probs[st_mt_incorrect_matrix]
        # tokens, dim
        mt_selected = mt_probs[st_mt_incorrect_matrix]
        # tokens, 1
        target_selected = target[st_mt_incorrect_matrix].unsqueeze(-1)

        # get hypo of mt, tokens(after softmax)
        hypo_mt = mt_selected.gather(dim=-1, index=target_selected).squeeze(-1).sum()
        # get hypo of st, tokens(after softmax)
        hypo_st = st_selected.gather(dim=-1, index=target_selected).squeeze(-1).sum()
        # get importance of st and mt for hard words
        importance_st = hypo_st / (hypo_st + hypo_mt)
        importance_mt = hypo_mt / (hypo_st + hypo_mt)

        # currently we only concat[s, x] and let y to select from them automatically
        mix_encoder_out = {
            "encoder_out": [torch.concat([audio_enc["encoder_out"][0], text_enc["encoder_out"][0]], dim=0)],  # T x B x C
            "encoder_padding_mask": [torch.concat([audio_enc["encoder_padding_mask"][0], text_enc["encoder_padding_mask"][0]], dim=1)],  # B x T
            "encoder_embedding": [None],  # B x T x C
            "encoder_states": None,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        mix_output = model.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=mix_encoder_out
        )

        lprobs = model.get_normalized_probs(mix_output, log_probs=True)
        lprobs_selected = lprobs[st_mt_incorrect_matrix]

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs_selected,
            target_selected,
            self.eps,
            ignore_index=self.padding_idx,
        )
        return loss, lprobs_selected

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])
        jsd_loss = torch.Tensor([0])
        hard_tokens_loss = torch.Tensor([0])
        st_size, mt_size, ext_mt_size = 0, 0, 0
        common_tokens_correct = st_tokens_correct = mt_tokens_correct = 0
        st_tokens_correct_except_mt = mt_tokens_correct_except_st = 0
        hard_tokens = 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            # st + mt
            if self.mt_finetune and self.training:
                target_pading_mask = sample["target"].eq(self.padding_idx)
                bsz, seq_len = sample["target"].shape
                st_loss, st_lprobs, st_target, st_all_tokens, st_tokens_correct, st_correct_matrix, st_probs, audio_enc_out = self.forward_st(model, sample, reduce)
                mt_loss, mt_lprbs, mt_target, mt_all_tokens, mt_tokens_correct, mt_correct_matrix, mt_probs, text_enc_out = self.forward_mt(model, sample, reduce)
                assert st_all_tokens == mt_all_tokens == sample["ntokens"]
                # 都能预测正确的单词
                common_correct_matrix = st_correct_matrix & mt_correct_matrix
                st_selected = st_lprobs.view(bsz, seq_len, -1)[common_correct_matrix]
                mt_selected = mt_lprbs.view(bsz, seq_len, -1)[common_correct_matrix]
                part1_loss = self.compute_jsd_loss_without_pad(st_selected, mt_selected)
                common_tokens_correct = common_correct_matrix.sum().item()
                # st和mt中, 仅st能预测正确的单词(st擅长的)
                st_correct_of_mt_matrix = st_correct_matrix & (~mt_correct_matrix)
                st_selected = st_lprobs.view(bsz, seq_len, -1)[st_correct_matrix]
                mt_selected = mt_lprbs.view(bsz, seq_len, -1)[st_correct_matrix]
                part2_loss = F.kl_div(mt_selected, st_selected.detach(), log_target=True, reduction="none").sum(-1).sum()
                st_tokens_correct_except_mt = st_correct_of_mt_matrix.sum().item()
                assert st_tokens_correct_except_mt == st_tokens_correct - common_tokens_correct
                # st和mt中，仅mt能预测正确的单词(mt擅长的)
                mt_correct_of_st_matrix = mt_correct_matrix & (~st_correct_matrix)
                st_selected = st_lprobs.view(bsz, seq_len, -1)[mt_correct_matrix]
                mt_selected = mt_lprbs.view(bsz, seq_len, -1)[mt_correct_matrix]
                part3_loss = F.kl_div(st_selected, mt_selected.detach(), log_target=True, reduction="none").sum(-1).sum()
                mt_tokens_correct_except_st = mt_correct_of_st_matrix.sum().item()
                assert mt_tokens_correct_except_st == mt_tokens_correct - common_tokens_correct
                # st和mt都不能预测正确的单词, 需要排除padding
                st_mt_incorrect_matrix = (~st_correct_matrix) & (~mt_correct_matrix) & (~target_pading_mask)
                st_selected = st_lprobs.view(bsz, seq_len, -1)[st_mt_incorrect_matrix]
                mt_selected = mt_lprbs.view(bsz, seq_len, -1)[st_mt_incorrect_matrix]
                hard_tokens_loss, mix_selected = self.forward_hard_words(model,
                                                     sample,
                                                     audio_enc_out,
                                                     text_enc_out,
                                                     st_probs,
                                                     mt_probs,
                                                     st_mt_incorrect_matrix
                                                     )
                part4_loss = (F.kl_div(mt_selected, mix_selected.detach(), log_target=True, reduction="none").sum(-1).sum() + \
                                F.kl_div(st_selected, mix_selected.detach(), log_target=True, reduction="none").sum(-1).sum()) / 2.0

                # part4_loss = self.compute_jsd_loss_without_pad(st_lprobs, mt_lprbs)
                hard_tokens = st_mt_incorrect_matrix.sum().item()
                assert hard_tokens == sample["ntokens"] - st_tokens_correct - mt_tokens_correct + common_tokens_correct
                jsd_loss = part1_loss + part2_loss + part3_loss + part4_loss
                loss = st_loss + mt_loss + hard_tokens_loss + jsd_loss
                st_size = mt_size = sample_size = sample["ntokens"]
            # st(dev or train only)
            else:
                st_loss, _, _, st_all_tokens, st_tokens_correct, st_correct_matrix, _, _ = self.forward_st(model, sample, reduce)
                loss = st_loss
                st_size = sample_size = sample["ntokens"]
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "hard_tokens_loss": hard_tokens_loss.data,
            "jsd_loss": jsd_loss.data,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "common_tokens_correct": common_tokens_correct,
            "st_tokens_correct": st_tokens_correct,
            "mt_tokens_correct": mt_tokens_correct,
            "st_tokens_correct_except_mt": st_tokens_correct_except_mt,
            "mt_tokens_correct_except_st": mt_tokens_correct_except_st,
            "hard_tokens": hard_tokens,
        }
        
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        hard_tokens_loss_sum = sum(log.get("hard_tokens_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)
        jsd_loss_sum = sum(log.get("jsd_loss", 0) for log in logging_outputs)

        common_tokens_correct_sum = sum(log.get("common_tokens_correct", 0) for log in logging_outputs)
        st_tokens_correct_sum = sum(log.get("st_tokens_correct", 0) for log in logging_outputs)
        mt_tokens_correct_sum = sum(log.get("mt_tokens_correct", 0) for log in logging_outputs)
        st_tokens_correct_except_mt_sum = sum(log.get("st_tokens_correct_except_mt", 0) for log in logging_outputs)
        mt_tokens_correct_except_st_sum = sum(log.get("mt_tokens_correct_except_st", 0) for log in logging_outputs)
        hard_tokens_sum = sum(log.get("hard_tokens", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "hard_tokens_loss", hard_tokens_loss_sum / hard_tokens_sum / math.log(2) if hard_tokens_sum != 0 else 0, hard_tokens_sum, round=3
        )
        metrics.log_scalar(
            "jsd_loss", jsd_loss_sum / sample_size / math.log(2) if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )
        metrics.log_scalar(
            "common_accuracy", common_tokens_correct_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "st_accuracy", st_tokens_correct_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "mt_accuracy", mt_tokens_correct_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
        )
        # 只有st会，而mt不会翻译的单词占比
        metrics.log_scalar(
            "st_accuracy_except_mt", st_tokens_correct_except_mt_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "mt_accuracy_except_st", mt_tokens_correct_except_st_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
        )
        metrics.log_scalar(
            "hard_word_percentage", hard_tokens_sum / sample_size if sample_size != 0 else 0, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True