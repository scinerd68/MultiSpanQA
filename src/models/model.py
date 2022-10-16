import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoModel


class MultiSpanQATagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained('roberta-base', num_labels=3)
        # self.model = AutoModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        return outputs