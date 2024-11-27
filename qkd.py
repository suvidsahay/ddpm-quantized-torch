import os
import torch
import torch.multiprocessing as mp
from argparse import Namespace
from train import train 


def main():
    args = Namespace(
        config_path=None,
        exp_name="experiment_name",
        dataset="cifar10",
        root="/datasets",
        epochs=50,
        lr=0.0002,
        beta1=0.9,
        beta2=0.999,
        batch_size=128,
        num_accum=1,
        block_size=1,
        timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        model_mean_type="eps",
        model_var_type="fixed-large",
        loss_type="mse",
        num_workers=4,
        train_device="cpu",
        eval_device="cpu",
        image_dir="./images",
        image_intv=10,
        num_samples=64,
        config_dir="./configs",
        chkpt_dir="./chkpts",
        chkpt_name="",
        chkpt_intv=120,
        seed=1234,
        resume=False,
        chkpt_path="",
        eval=False,
        eval_total_size=50000,
        eval_batch_size=256,
        use_ema=False,
        use_ddim=False,
        skip_schedule="linear",
        subseq_size=50,
        ema_decay=0.9999,
        distributed=False,
        rigid_launch=False,
        num_gpus=1,
        dry_run=False,
        use_quantized_unet=True
    )

    # Initial training phase with standard cross-entropy loss
    args.loss_type = "cross_entropy"
    args.epochs = 10  # Number of initial epochs for self-studying phase
    train(args=args)

    # Knowledge distillation phase
    args.loss_type = "kd"  # Assuming 'kd' is the loss type for knowledge distillation
    args.epochs = 40  # Number of epochs for knowledge distillation
    train(args=args)

if __name__ == "__main__":
    main()
