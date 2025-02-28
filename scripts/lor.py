import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    is_torch_xla_available
)
from trl import ORPOConfig, ORPOTrainer
from typing import Optional, Union, Tuple, Callable, List, Dict, Literal
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


class LORTrainer(ORPOTrainer):
    def __init__(
            self,
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[ORPOConfig] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            processing_class: Optional[
                Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
            ] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            peft_config: Optional[Dict] = None,
            compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            compute_metrics=compute_metrics,
        )

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):

        """Compute the ORPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss,
        ) = forward_output[:5]
        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        losses, chosen_rewards, rejected_rewards, log_odds_ratio, log_odds_chosen = self.odds_ratio_loss(
            policy_chosen_logps, policy_rejected_logps
        )
        # Odds Ratio loss
        loss = -losses.mean()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean()
        metrics[f"{prefix}rewards/margins"] = self.accelerator.gather_for_metrics(
            chosen_rewards - rejected_rewards
        ).mean()
        metrics[f"{prefix}logps/rejected"] = self.accelerator.gather_for_metrics(policy_rejected_logps).detach().mean()
        metrics[f"{prefix}logps/chosen"] = self.accelerator.gather_for_metrics(policy_chosen_logps).detach().mean()
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(policy_rejected_logits).detach().mean()
        )
        metrics[f"{prefix}logits/chosen"] = self.accelerator.gather_for_metrics(policy_chosen_logits).detach().mean()
        metrics[f"{prefix}nll_loss"] = self.accelerator.gather_for_metrics(policy_nll_loss).detach().mean()
        metrics[f"{prefix}log_odds_ratio"] = self.accelerator.gather_for_metrics(log_odds_ratio).mean()
        metrics[f"{prefix}log_odds_chosen"] = self.accelerator.gather_for_metrics(log_odds_chosen).mean()
        if is_torch_xla_available():
            xm.mark_step()  # needed because .item() calls
        for k, v in metrics.items():
            metrics[k] = v.item()
        if self.aux_loss_enabled:
            loss += self.aux_loss_coef * aux_loss

        return loss, metrics