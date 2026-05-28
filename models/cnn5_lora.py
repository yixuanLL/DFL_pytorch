# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=8, bias=True):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, lora_rank=4, lora_alpha=8, **kwargs):
        super(Model, self).__init__()
        self.name = "CNN_5layer_LoRA"
        self.in_channels = 3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.features = None
        self.classifier = None
        self.nn_layer = nn.ModuleList()
        self.build(**kwargs)
        freeze_non_lora(self)

    def build(self, num_groups=None, bn_stats=None, size=None):
        if size == "small":
            cfg = [16, 16, "M", 32, 32, "M", 64, "M"]
        else:
            cfg = [32, 32, "M", 64, 64, "M", 128, 128, "M"]

        layers = []
        act = nn.Tanh
        c = self.in_channels
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)
        self.nn_layer.append(self.features)

        hidden = 128
        self.classifier = nn.Sequential(
            nn.Linear(c * 4 * 4, hidden),
            act(),
            LoRALinear(hidden, self.output_dim, rank=self.lora_rank, alpha=self.lora_alpha),
        )
        self.nn_layer.append(self.classifier)

    def forward(self, x):
        for idx, layer in enumerate(self.nn_layer):
            if idx == len(self.nn_layer) - 1:
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x


def freeze_non_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def lora_trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def lora_trainable_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained_cnn5_lora(model, checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model_state_dict", checkpoint)
    current = model.state_dict()
    compatible = {}

    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value

    final_weight_keys = [
        "classifier.2.weight",
        "nn_layer.1.2.weight",
    ]
    final_bias_keys = [
        "classifier.2.bias",
        "nn_layer.1.2.bias",
    ]
    for source_key in final_weight_keys:
        if source_key in state:
            for target_key in ["classifier.2.base.weight", "nn_layer.1.2.base.weight"]:
                if target_key in current and current[target_key].shape == state[source_key].shape:
                    compatible[target_key] = state[source_key]
    for source_key in final_bias_keys:
        if source_key in state:
            for target_key in ["classifier.2.base.bias", "nn_layer.1.2.base.bias"]:
                if target_key in current and current[target_key].shape == state[source_key].shape:
                    compatible[target_key] = state[source_key]

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    freeze_non_lora(model)
    print("Loaded pretrained CNN5 weights for LoRA: %d tensors" % len(compatible))
    print("LoRA trainable parameters: %d" % lora_trainable_parameter_count(model))
    return missing, unexpected
