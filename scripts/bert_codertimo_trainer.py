import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from .models.bert import Bert
from .models.language_model import BertLanguageModel
from .models.lr_scheduler import LearningRateScheduler


class BertTrainer:
    """
    Description:
        training BERT model

    Arguments:
        bert: BERT model to train
        n_vocab: total size of vocabulary
        train_dataloader: train dataset dataloader
        test_dataloader: test dataset dataloader
        lr: initial learning rate of adam optimizer
        betas: betas of adam optimizer
        weight_decay: adam optimizer weight decay parameter
        log_freq: logging frequency of the batch iteration
    """

    def __init__(
        self,
        bert: Bert,
        n_vocab: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        warmup_step=10000,
        cuda_devices=None,
        log_freq=10,
    ):
        # select device to compute between GPU and CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert = bert
        # assign model as copied log-softmax tensor of BERT language model to selected device
        self.model = BertLanguageModel(bert=self.bert, n_vocab=n_vocab).to(self.device)

        # distributed GPU training if there are more than one GPU
        if torch.cuda.device_count() > 1:
            # PyTorch basically uses a GPU, so it needs to be distributed in parallel
            # set cuda_device to None as default, and it equals all possible GPUs
            self.model = nn.DataParallel(module=self.model, device_ids=cuda_devices)

        # set train and test dataset
        # DataLoader delivers samples to mini batches, and shuffles each epoch, uses multi-processing
        self.train_dataset = train_dataloader
        self.test_dataset = test_dataloader

        # set adam optimizer
        self.optim = Adam(
            params=self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        # set learning rate scheduler of optimizer
        self.lr_scheduler = LearningRateScheduler(
            optim=self.optim, d_model=self.bert.d_hidden, warmup_step=warmup_step
        )

        # set negative log likelihood loss function to predict
        # nn.CrossEntropyLoss = nn.LogSoftmax + nn.NLLLoss
        # NSP and MLM returns nn.LogSoftmax value, so nn.NLLLoss can be used
        self.loss_func = nn.NLLLoss(ignore_index=0)
        # set iteration frequency for logging
        self.log_freq = log_freq

        params_list = []
        for param in self.model.parameters():
            # nn.Module.parameters() returns torch.Tensor dtype
            # torch.Tensor.nelement returns the number of parameters
            params_list.append(param.nelement())
            print("Total params:", sum(params_list))

    def get_iteration(self, epoch: int, data_loader: DataLoader, mode="train"):
        """
        Description:
            execute iterations for train or test mode
            backpropagation is activated if only on train mode
            save the model for each epoch

        Arguments:
            epoch: current epoch value
            data_loader: torch.utils.data.DataLoader for iteration
            mode: select "train" or "test" with string type
        """

        # set iteration progress bar with tqdm library
        data_iter = tqdm(
            enumerate(data_loader),
            desc=f"{mode} mode_epoch: {epoch}",
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        # initialize metrics
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        # execute the following operations for every iterations
        for i, data in data_iter:
            for key, value in data.items():
                # data in a mini batch is put to predefined device
                data = {key: value.to(self.device)}
            # get predicted results of next sentence prediction and masked language model
            nsp_pred, mlm_pred = self.model.forward(
                data["bert_input"], data["seg_label"]
            )
            # loss for next sentence prediction and masked language model
            nsp_loss = self.loss_func(nsp_pred, data["is_next"])
            mlm_loss = self.loss_func(mlm_pred.transpose(1, 2), data["bert_label"])
            # set overall loss
            loss = nsp_loss + mlm_loss

            if mode == "train":
                # initialize gradient tensor in train mode
                self.lr_scheduler.initialize_optim()
                # compute gradient with loss function
                loss.backward()
                # update the learning rate and the parameters with computed gradient
                self.lr_scheduler.update_lr_and_step_optim()

            # get next sentence prediction accuracy
            # Tensor.argmax(dim=-1) returns the index of element with maximum probability
            # Tensor.eq(data["is_next"]) computes element-wise equality
            # Tensor.item() returns the value of this tensor
            correct = nsp_pred.argmax(dim=-1).eq(data["is_next"]).sum().item()
            # set scalar avg_loss with Tensor.item() due to loss dtype is torch.Tensor
            avg_loss += loss.item()
            # set the total number of correct and element
            total_correct += correct
            total_element += data["is_next"].nelement()

            # get the learining status of i-th iteration
            # get average loss because each iteration cannot represent the loss of the model
            learning_status = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": round(avg_loss / (i + 1), 4),
                "avg_acc": round(total_correct / total_element * 100, 4),
                "loss": loss.item(),
            }

            if i % self.log_freq == 0:
                # get logging for every ten iterations
                data_iter.write(str(learning_status))

        # get learning status after finishing an epoch
        print(
            f"{mode} mode_epoch: {epoch}, avg_loss: {avg_loss / len(data_iter):.4f}, total_acc: {total_correct * 100 / total_element:.4f}"
        )

    def save_model(self, epoch: int, file_path: str = "./checkpoint/bert_trained"):
        """
        Description:
            save the current BERT model

        Arguments:
            epoch: current epoch value
            file_path: model output path

        Returns:
            final model output path with epoch information
        """

        # set output path with current epoch information
        output_path = file_path + f"_{epoch}epoch.pt"
        # save overall model with GPU
        # if you load the model with GPU, use model.to(torch.device("cuda")) and my_tensor = my_tensor.to(torch.device('cuda')) for tensor overwrite
        torch.save(self.bert, output_path)
        print(f"Save model at {epoch}epoch")

        return output_path

    def train_model(self, epoch: int):
        # execute iteration function with train mode
        self.get_iteration(epoch=epoch, data_loader=self.train_dataset, mode="train")

    def test_model(self, epoch: int):
        # execute iteration function with test mode
        self.get_iteration(epoch=epoch, data_loader=self.test_dataset, mode="test")
