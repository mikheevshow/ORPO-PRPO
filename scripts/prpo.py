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
)
from trl import ORPOConfig, ORPOTrainer
from typing import Optional, Union, Tuple, Callable, List, Dict
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput


class PRPOTrainer(ORPOTrainer):
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

    def odds_ratio_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        log_probs_ratio = (policy_chosen_logps - policy_rejected_logps)
        ratio = F.logsigmoid(log_probs_ratio)
        losses = self.beta * ratio

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards, torch.mean(ratio), torch.mean(log_probs_ratio)
