import os
import cv2
import tqdm
import torch
import numpy as np
import torch.optim as optim

from time import time
from mean_average_precision import MeanAveragePrecision
from torchvision.utils import make_grid


class Trainer:
    def __init__(self, config, train_dl, val_dl, model, summary_writer, device):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.summary_writer = summary_writer
        self.device = device
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_optimizer(self):
        optimizer_config = self.config["optimizer"]

        lr_list = optimizer_config["lr"]
        if isinstance(lr_list, list):
            param_groups = self.model.params_groups
            if not len(param_groups) == len(lr_list):
                raise ValueError("Length of lr list must match number of parameter groups")
            param_lr = [{"params": group, "lr": lr_value} for group, lr_value in zip(param_groups, lr_list)]
        else:
            param_lr = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr_list}]

        if optimizer_config["name"] == "adam":
            return optim.Adam(param_lr,
                              weight_decay=optimizer_config.get("weight_decay", 0))

        raise TypeError(f"Unknown optimizer name: {optimizer_config['name']}")

    def _get_scheduler(self):
        scheduler_config = self.config["scheduler"]

        if scheduler_config["name"] == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                        mode="max",
                                                        factor=scheduler_config["factor"],
                                                        patience=scheduler_config["patience"])
        elif scheduler_config["name"] == "step":
            return optim.lr_scheduler.StepLR(self.optimizer,
                                             step_size=scheduler_config["step_size"],
                                             gamma=scheduler_config["gamma"])

        raise TypeError(f"Unknown scheduler type: {scheduler_config['name']}")

    def train(self):
        self.model.to(self.device)
        best_metric = -float("inf")

        for epoch in range(1, self.config["num_epochs"] + 1):
            losses = self._run_epoch(epoch)
            map_metric, batch = self._validate()

            self._write_to_tensor_board(epoch, losses, map_metric, batch)

            if map_metric > best_metric:
                self._save_checkpoints()
                best_metric = map_metric

            if self.config["scheduler"] == "plateau":
                self.scheduler.step(map_metric)
            else:
                self.scheduler.step()

            print(f"\nEpoch: {epoch}; train loss = {losses['train_loss']}; validation mAP = {map_metric}")

    def _run_epoch(self, epoch):
        self.model.train()

        lr = self.optimizer.param_groups[0]["lr"]
        status_bar = tqdm.tqdm(total=len(self.train_dl))
        status_bar.set_description(f"Epoch {epoch}, lr {lr}")

        losses = {"train_loss": []}
        for images, targets in self.train_dl:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for loss_name in loss_dict:
                if loss_name not in losses:
                    losses[loss_name] = []
                losses[loss_name].append(loss_dict[loss_name].item())
            losses["train_loss"].append(loss.item())

            status_bar.update()
            status_bar.set_postfix(loss=loss.item())

        status_bar.close()
        return {loss_name: np.array(losses[loss_name]).mean() for loss_name in losses}

    def _validate(self):
        self.model.eval()

        status_bar = tqdm.tqdm(total=len(self.val_dl))

        # pred_boxes, true_boxes = [], []
        # maps = []
        metric_fn = MeanAveragePrecision(num_classes=1)
        with torch.no_grad():
            # batch_number = 0
            for images, targets in self.val_dl:
                # pred_boxes, true_boxes = [], []

                images = list(image.to(self.device) for image in images)
                cur_pred_boxes = self.model(images)

                for i in range(len(cur_pred_boxes)):
                    gt = targets[i]["boxes"].numpy()
                    gt = np.hstack((gt, np.zeros((gt.shape[0], 3))))

                    preds = cur_pred_boxes[i]["boxes"].cpu().numpy()
                    scores = np.expand_dims(cur_pred_boxes[i]["scores"].cpu().numpy(), axis=1)
                    preds = np.hstack((preds, np.zeros((preds.shape[0], 1)), scores))
                    if preds.shape[0] != 0 and gt.shape[0] != 0:
                        metric_fn.add(preds, gt)
                status_bar.update()

        status_bar.close()
        batch = {"images": list(image.cpu() for image in images),
                 "targets": targets,
                 "pred_boxes": [{k: v.cpu() for k, v in pred.items()} for pred in cur_pred_boxes]}

        mAP = metric_fn.value(iou_thresholds=0.5)['mAP']
        return mAP, batch

    def _save_checkpoints(self):
        torch.save(self.model.state_dict(), os.path.join(self.config["log_dir"], "model.pth"))

    def _write_to_tensor_board(self, epoch, losses, map_metric, batch):
        for loss_name, loss_value in losses.items():
            self.summary_writer.add_scalar(tag=loss_name, scalar_value=loss_value, global_step=epoch)
        self.summary_writer.add_scalar(tag="Validation mAP", scalar_value=map_metric, global_step=epoch)
        images_grid = self.make_tensorboard_grid(batch)
        self.summary_writer.add_image("Images", images_grid, epoch)

    def make_tensorboard_grid(self, batch_sample):
        images, targets, pred_boxes = batch_sample["images"], batch_sample["targets"], batch_sample["pred_boxes"]

        images_plus_boxes = []

        for image, target in zip(images, targets):
            image = image.numpy().transpose((1, 2, 0))
            images_plus_boxes.append(torch.FloatTensor(self.draw_bboxes(image, target["boxes"])).permute(2, 0, 1))

        for image, pred_box in zip(images, pred_boxes):
            image = image.numpy().transpose((1, 2, 0))
            images_plus_boxes.append(torch.FloatTensor(self.draw_bboxes(image, pred_box["boxes"], gt=False)).permute(2, 0, 1))

        return make_grid(images_plus_boxes, nrow=len(images))

    @staticmethod
    def draw_bboxes(img, bboxes, gt=True):
        img = img.copy()
        color = (0, 0, 1) if gt else (1, 0, 0)
        for bbox in bboxes:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        return img
