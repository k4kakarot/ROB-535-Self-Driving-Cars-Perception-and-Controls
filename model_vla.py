import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ====== Actions and command parsing (CoVLA version) ======

ACTIONS = [
    "MAINTAIN_SPEED",  # normal driving
    "SLOW_DOWN",   # more caution needed
    "TURN_LEFT",
    "TURN_RIGHT",    
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

SLOW_DOWN_PATTERNS = [
    "pedestrian",
    "pedestrians",
    "crosswalk",
    "intersection",
    "traffic light",
    "red light",
    "yellow light",
    "brake",
    "stop",
    "curve",
    "sharp turn",
    "narrow road",
    "keep a safe distance",
    "be prepared to stop",
    "heavy traffic",
    "congestion",
]

TURN_LEFT_PATTERNS = [
    "left turn",
    "turn left",
    "veers left",
    "left-hand turn",
]

TURN_RIGHT_PATTERNS = [
    "right turn",
    "turn right",
    "veers right",
    "right-hand turn",
]

def parse_command_to_action(command_text: str) -> str:
    text = command_text.lower().strip()

    for pattern in TURN_LEFT_PATTERNS:
        if pattern in text:
            return "TURN_LEFT"

    for pattern in TURN_RIGHT_PATTERNS:
        if pattern in text:
            return "TURN_RIGHT"

    for pattern in SLOW_DOWN_PATTERNS:
        if pattern in text:
            return "SLOW_DOWN"

    return "MAINTAIN_SPEED"



def action_to_one_hot(action: str) -> torch.Tensor:
    vec = torch.zeros(len(ACTIONS), dtype=torch.float32)
    idx = ACTION_TO_IDX[action]
    vec[idx] = 1.0
    return vec


# ====== Image preprocessing ======

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_image(path: str) -> torch.Tensor:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return IMG_TRANSFORM(img)



# ====== Vision encoder (frozen MobileNetV2) ======

class VisionEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        if pretrained:
            backbone = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )
        else:
            backbone = models.mobilenet_v2(weights=None)

        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.output_dim = backbone.last_channel

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


# ====== VLA policy model ======

class VLAPolicy(nn.Module):
    """
    Simple Vision-Language-Action model for CoVLA:
    - Image -> vision encoder -> feature vector
    - Caption -> parsed into one-hot over ACTIONS (language feature)
    - Concatenate -> MLP -> logits over ACTIONS
    """

    def __init__(self, vision_encoder: VisionEncoder, num_actions: int):
        super().__init__()
        self.vision_encoder = vision_encoder
        vision_dim = vision_encoder.output_dim
        text_dim = num_actions

        self.mlp = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, images, command_onehots):
        img_feat = self.vision_encoder(images)
        x = torch.cat([img_feat, command_onehots], dim=1)
        logits = self.mlp(x)
        return logits


# ====== Explanation generator ======

def generate_explanation(pred_action: str, command_text: str) -> str:
    base = command_text.strip()

    if pred_action == "SLOW_DOWN":
        return (
            "The system decides to slow down because the scene description "
            f"('{base}') indicates potential hazards or situations that require extra caution."
        )
    if pred_action == "MAINTAIN_SPEED":
        return (
            "The system maintains its current speed because the description "
            f"('{base}') does not mention immediate hazards, suggesting normal driving conditions."
        )
    if pred_action == "TURN_LEFT":
        return (
            "The system prepares for a left turn, as the description "
            f"('{base}') suggests an upcoming left-hand maneuver."
        )
    if pred_action == "TURN_RIGHT":
        return (
            "The system prepares for a right turn, as the description "
            f"('{base}') suggests an upcoming right-hand maneuver."
        )

    return f"The system chooses the action '{pred_action}' based on the interpreted description: '{base}'."
