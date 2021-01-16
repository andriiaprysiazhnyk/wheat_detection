import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(pretrained=True, trainable_backbone_layers=3, pretrained_backbone=True):
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                     trainable_backbone_layers=trainable_backbone_layers)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        return model
    else:
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=2,
                                                                    trainable_backbone_layers=trainable_backbone_layers,
                                                                    pretrained_backbone=pretrained_backbone)


def get_retina_net_model(pretrained=True, pretrained_backbone=True):
    if pretrained:
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        model.head.classification_head.cls_logits = torch.nn.Conv2d(256, 18, kernel_size=(3, 3), stride=(1, 1),
                                                                    padding=(1, 1))
        model.head.classification_head.num_classes = 2
        return model
    else:
        return torchvision.models.detection.retinanet_resnet50_fpn(num_classes=2,
                                                                   pretrained_backbone=pretrained_backbone)


def get_network(model_config):
    if model_config["arch"] == "faster_rcnn":
        model = get_faster_rcnn_model(model_config["pretrained"],
                                      model_config["trainable_backbone_layers"],
                                      model_config.get("pretrained_backbone", True))
        model.params_groups = ([p for p in model.backbone.parameters() if p.requires_grad],
                               [p for p in model.rpn.parameters() if p.requires_grad] + [p for p in
                                                                                         model.roi_heads.parameters() if
                                                                                         p.requires_grad])
    elif model_config["arch"] == "retina_net":
        model = get_retina_net_model(model_config["pretrained"],
                                     model_config.get("pretrained_backbone", True))
        model.params_groups = ([p for p in model.backbone.parameters() if p.requires_grad],
                               [p for p in model.head.parameters() if p.requires_grad])
    else:
        raise ValueError(f"Model architecture {model_config['arch']} not recognized.")

    model.transform.min_size = (512,)
    return model
