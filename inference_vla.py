import argparse
import torch

from model_vla import (
    ACTIONS,
    parse_command_to_action,
    action_to_one_hot,
    load_image,
    VisionEncoder,
    VLAPolicy,
    generate_explanation,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained .pt checkpoint (e.g., checkpoints/vla_covla_best.pt)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a road scene image",
    )
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Natural-language description of the scene",
    )
    args = parser.parse_args()

    # Build model
    vision_encoder = VisionEncoder(pretrained=True, freeze=True).to(DEVICE)
    model = VLAPolicy(vision_encoder, num_actions=len(ACTIONS)).to(DEVICE)

    # Load weights
    state_dict = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare inputs
    img_tensor = load_image(args.image).unsqueeze(0).to(DEVICE)

    parsed_action_from_text = parse_command_to_action(args.description)
    cmd_vec = action_to_one_hot(parsed_action_from_text).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(img_tensor, cmd_vec)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_action = ACTIONS[pred_idx]
        confidence = probs[pred_idx].item()

    explanation = generate_explanation(pred_action, args.description)

    print(f"Description: {args.description}")
    print(f"Parsed text-based action: {parsed_action_from_text}")
    print(f"Predicted driving action: {pred_action} (confidence: {confidence:.2f})")
    print(f"Explanation: {explanation}")


if __name__ == "__main__":
    main()
