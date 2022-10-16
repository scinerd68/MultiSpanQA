import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from data.make_dataset import create_dataset
from models.model import MultiSpanQATagger


def convert_to_tensor(batch):
    new_batch = {}
    new_batch['example_id'] = [example['example_id'] for example in batch]
    new_batch['num_span'] = torch.Tensor([example['num_span'] for example in batch])
    new_batch['structure'] = torch.LongTensor([example['structure'] for example in batch])
    new_batch['input_ids'] = torch.stack([torch.LongTensor(example['input_ids']) for example in batch])
    new_batch['attention_mask'] = torch.stack([torch.LongTensor(example['attention_mask']) for example in batch])
    new_batch['labels'] = torch.stack([torch.LongTensor(example['labels']) for example in batch])
    return new_batch


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize model
    model = MultiSpanQATagger()
    model = model.to(device)
    # Create dataset
    train_dataset = create_dataset("D:/ComputerScience/BachKhoa/NLPLab/MultiSpanQA/data/MultiSpanQA_data/train.json",
                                   "D:/ComputerScience/BachKhoa/NLPLab/MultiSpanQA/data/MultiSpanQA_data/valid.json",
                                   "roberta-base")
    dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=convert_to_tensor, shuffle=True)
    # Train
    torch.manual_seed(0)
    epochs = 3
    print_every = 10
    optim = AdamW(model.parameters(), lr=3e-5)
    loss_func = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(epochs):
        # Set model in train mode
        model.train()
        loss_of_epoch = 0

        print("############Train############")
        for batch_idx, batch in enumerate(dataloader):
            optim.zero_grad()
            out = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            labels = batch['labels'].reshape(-1).to(device)
            outputs = out['logits'].reshape(out['logits'].shape[0] * out['logits'].shape[1], -1)
            loss = loss_func(outputs, labels)
                
            loss.backward()
            optim.step()
            loss_of_epoch += loss.item()
            if (batch_idx + 1) % print_every == 0:
                print("Batch {:} / {:}".format(batch_idx + 1, len(dataloader)))
                print("Loss:", round(loss.item(), 2))
                torch.save(model, "/models/test.pth")
            # break
        loss_of_epoch /= len(dataloader)
        print(f"\n------- Epoch {epoch + 1} -------")
        print(f"Training Loss: {loss_of_epoch}")
        print("-----------------------")
        print()
        # break