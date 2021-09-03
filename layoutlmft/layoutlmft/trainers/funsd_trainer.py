from typing import Any, Dict, Union

import torch

from transformers import Trainer


class FunsdTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        #breakpoint()
        inputs["bbox"] = torch.clamp(inputs["bbox"], min=0, max=1000)
        #inputs["input_ids"] = torch.clamp(inputs["input_ids"], min=0, max=30000)
        for k, v in inputs.items():
            if hasattr(v, "to") and hasattr(v, "device"):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs
