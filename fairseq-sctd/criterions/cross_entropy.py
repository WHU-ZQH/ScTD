# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def get_kl_loss(self, x, y, i,j):
        size1, size2=x[1]["inner_states"][i].shape[0], y[1]["inner_states"][j].shape[0]
        kl_dis_avg=F.kl_div(F.log_softmax(torch.nn.AvgPool1d(size1)(x[1]["inner_states"][i].transpose(0,1).transpose(1,2)).squeeze(), dim=-1), 
        F.softmax(torch.nn.AvgPool1d(size2)(y[1]["inner_states"][j].transpose(0,1).transpose(1,2)).squeeze(), dim=-1), reduction="sum")

        kl_dis_cls=F.kl_div(F.log_softmax(x[1]["inner_states"][i].transpose(0,1)[:, 0, :], dim=-1), 
            F.softmax(y[1]["inner_states"][j].transpose(0,1)[:, 0, :], dim=-1), reduction="sum")
        return kl_dis_cls, kl_dis_avg

    def forward(self, model, sample, reduce=True, record_mlm_loss=False, rank=None, align=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"], rank=rank, return_all_hiddens=True)
        loss, loss_all = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        if align:
            net_output_ref = model(**sample["net_input"], rank=None, return_all_hiddens=True)
            loss_ref, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
            kl_loss_cls, kl_loss_avg=self.get_kl_loss(net_output, net_output_ref, -2, -2)
            kl_loss_cls_final, kl_loss_avg_final=self.get_kl_loss(net_output, net_output_ref, -1, -1)

            loss=0.5*loss+0.5*loss_ref+0.05*kl_loss_cls+0.05*kl_loss_cls_final
            # loss=loss+0.05*kl_loss_cls+0.05*kl_loss_cls_final
            self.kl_loss_cls=kl_loss_cls
            self.kl_loss_cls_final=kl_loss_cls_final

        else:
            kl_loss_cls=self.kl_loss_cls
            kl_loss_cls_final=self.kl_loss_cls_final

        logging_output = {
            "loss": loss.data,
            "kl_loss": kl_loss_cls.data,
            "kl_loss_final": kl_loss_cls_final.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if record_mlm_loss:
            targets = model.get_targets(sample, net_output)
            masked_tokens = sample["target"].ne(1)
            masked_tokens = torch.where(masked_tokens.any(),masked_tokens,masked_tokens.new([True]))
            if masked_tokens is not None:
                targets = targets[masked_tokens]
            return loss, sample_size, logging_output, loss_all[masked_tokens.view(-1)], targets
        else:
            return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="none",
        )
        return loss.sum(), loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        kl_loss_final_sum = sum(log.get("kl_loss_final", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("kl_loss", kl_loss_sum, round=3)
        metrics.log_scalar("kl_loss_final", kl_loss_final_sum, round=3)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
