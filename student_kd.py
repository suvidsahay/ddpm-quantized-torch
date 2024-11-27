import os
import json
import torch
from ddpm_torch.models.unet import QuantizedUNet
from train import train

def load_config_and_checkpoint(fp_config_path, fp_checkpoint_path):
    # Load the FP model's configuration
    with open(fp_config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize the student model (e.g., a quantized UNet)
    student_model_config = config['model']
    student_model = QuantizedUNet(**student_model_config)

    # Load the FP model's weights into the student model
    checkpoint = torch.load(fp_checkpoint_path, map_location='cpu')
    student_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    return student_model, config

def train_student_model(args):
    # Set paths for FP config and checkpoint
    fp_config_path = args.fp_config_path
    fp_checkpoint_path = args.fp_checkpoint_path
    
    # Load the FP model configuration and initialize the student model
    student_model, config = load_config_and_checkpoint(fp_config_path, fp_checkpoint_path)
    
    # Modify training configuration for the student model
    config['train']['lr'] = args.student_lr  # Adjust learning rate for fine-tuning
    config['train']['epochs'] = args.student_epochs  # Number of epochs for student training
    
    # Train the student model
    train(rank=0, args=args, temp_dir="")
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Add arguments specific to student training
    parser.add_argument("--fp-config-path", type=str, help="Path to the FP model's configuration file")
    parser.add_argument("--fp-checkpoint-path", type=str, help="Path to the FP model's checkpoint file")
    parser.add_argument("--student-lr", default=0.0001, type=float, help="Learning rate for the student model")
    parser.add_argument("--student-epochs", default=30, type=int, help="Number of epochs for student training")
    
    # Add any other arguments required for `train`
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size for student training")
    parser.add_argument("--config-dir", default="./configs", type=str, help="Directory to save student model configs")
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str, help="Directory to save student model checkpoints")
    args = parser.parse_args()

    train_student_model(args)
