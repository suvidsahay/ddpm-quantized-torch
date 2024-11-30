import os
import json
import torch
from ddpm_torch.models.unet import QuantizedUNet
from train import train
from argparse import Namespace
import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from ddim import *
from ddpm_torch import *
from functools import partial
from torch.distributed.elastic.multiprocessing import errors
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.optim import Adam, lr_scheduler
from ddpm_torch.models.unet import UNet, QuantizedUNet

def train_student_model():

    args = Namespace(
        config_path=None,
        exp_name="experiment_name",
        dataset="cifar10",
        root="./datasets",
        epochs=5,
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
        chkpt_name="lp_student",
        chkpt_intv=120,
        seed=1234,
        resume=True,
        chkpt_path="./chkpts/cifar10/baseline",
        eval=False,
        eval_total_size=50000,
        eval_batch_size=256,
        use_ema=False,
        use_ddim=True,
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

def co_studying():
    args = Namespace(
        config_path=None,
        exp_name="experiment_name",
        dataset="cifar10",
        root="/datasets",
        epochs=5,
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
        chkpt_intv=120,
        seed=1234,
        resume=True,
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
    )
    args_student = args
    args_student.chkpt_name = "lp_student"
    args_student.chkpt_path = "./chkpts/lp_student.pth"
    args_student.use_quantized_unet = True

    args_teacher = args
    args_teacher.chkpt_name = "fp_teacher"
    args_teacher.chkpt_path = "./chkpts/fp_teacher.pth"


    distributed = args.distributed

    def logger(msg, **kwargs):
        if not distributed or dist.get_rank() == 0:
            print(msg, **kwargs)

    root = os.path.expanduser(args.root)
    if args.config_path is None:
        args.config_path = os.path.join(args.config_dir, args.dataset + ".json")
    with open(args.config_path, "r") as f:
        meta_config = json.load(f)
    exp_name = os.path.basename(args.config_path)[:-5]

    # dataset basic info
    dataset = meta_config.get("dataset", args.dataset)
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"]
    image_shape = (in_channels, ) + image_res

    # set seed for RNGs
    seed = meta_config.get("seed", args.seed)
    seed_all(seed)

    # extract training-specific hyperparameters
    gettr = partial(get_param, obj_1=meta_config.get("train", {}), obj_2=args)
    train_config = ConfigDict(**{
        k: gettr(k) for k in (
            "batch_size", "beta1", "beta2", "lr", "epochs", "grad_norm", "warmup",
            "chkpt_intv", "image_intv", "num_samples", "use_ema", "ema_decay")})
    train_config.batch_size //= args.num_accum
    train_device = torch.device(args.train_device)
    eval_device = torch.device(args.eval_device)

    # extract diffusion-specific hyperparameters
    getdif = partial(get_param, obj_1=meta_config.get("diffusion", {}), obj_2=args)
    diffusion_config = ConfigDict(**{
        k: getdif(k) for k in (
            "beta_schedule", "beta_start", "beta_end", "timesteps",
            "model_mean_type", "model_var_type", "loss_type")})

    betas = get_beta_schedule(
        diffusion_config.beta_schedule, beta_start=diffusion_config.beta_start,
        beta_end=diffusion_config.beta_end, timesteps=diffusion_config.timesteps)
    diffusion = GaussianDiffusion(betas=betas, **diffusion_config)

    # extract model-specific hyperparameters
    out_channels = 2 * in_channels if diffusion_config.model_var_type == "learned" else in_channels
    model_config = meta_config["model"]
    block_size = model_config.pop("block_size", args.block_size)
    model_config["in_channels"] = in_channels * block_size ** 2
    model_config["out_channels"] = out_channels * block_size ** 2
    _model_student = QuantizedUNet(**model_config)
    _model_teacher = UNet(**model_config)

    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        _model_student = ModelWrapper(_model_student, pre_transform, post_transform)
        _model_teacher = ModelWrapper(_model_teacher, pre_transform, post_transform)

    if distributed:
        # check whether torch.distributed is available
        # CUDA devices are required to run with NCCL backend
        assert dist.is_available() and torch.cuda.is_available()

        if args.rigid_launch:
            # launched by torch.multiprocessing.spawn
            # share information and initialize the distributed package via shared file-system (FileStore)
            # adapted from https://github.com/NVlabs/stylegan2-ada-pytorch
            # currently, this only supports single-node training
            assert temp_dir, "Temporary directory cannot be empty!"
            init_method = f"file://{os.path.join(os.path.abspath(temp_dir), '.torch_distributed_init')}"
            dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=args.num_gpus)
            local_rank = rank
            os.environ["WORLD_SIZE"] = str(args.num_gpus)
            os.environ["LOCAL_RANK"] = str(rank)
        else:
            # launched by either torch.distributed.elastic (single-node) or Slurm srun command (multi-node)
            # elastic launch with C10d rendezvous backend by default uses TCPStore
            # initialize with environment variables for maximum customizability
            world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
            # global process id across all node(s)
            local_world_size = (int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or
                                int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) or
                                torch.cuda.device_count())
            # local device id on a single node
            local_rank = int(os.environ.get("LOCAL_RANK", "0")) or rank % local_world_size
            args.num_gpus = world_size or local_world_size
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))

        logger(f"Using distributed training with {args.num_gpus} GPU(s).")
        torch.cuda.set_device(local_rank)
        _model_student.cuda()
        _model_teacher.cuda()
        model_student = DDP(_model_student, device_ids=[local_rank, ])
        model_teacher = DDP(_model_teacher, device_ids=[local_rank, ])
        train_device = torch.device(f"cuda:{local_rank}")

    else:
        rank = local_rank = 0
        model_student = _model_student.to(train_device)
        model_teacher = _model_teacher.to(train_device)

    is_leader = rank == 0  # rank 0: leader in the process group

    logger(f"Dataset: {dataset}")
    logger(
        f"Effective batch-size is {train_config.batch_size} * {args.num_accum}"
        f" = {train_config.batch_size * args.num_accum}.")

    # PyTorch's implementation of Adam differs slightly from TensorFlow in that
    # the former follows Algorithm 1 as described in the paper by Kingma & Ba (2015) [1]
    # while the latter adopts an alternative approach mentioned just before Section 2.1
    # see also https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64-L69
    optimizer_student = Adam(model_student.parameters(), lr=train_config.lr, betas=(train_config.beta1, train_config.beta2))
    optimizer_teacher = Adam(model_teacher.parameters(), lr=train_config.lr, betas=(train_config.beta1, train_config.beta2))
    # lr_lambda is used to calculate the learning rate multiplicative factor at timestep t (starting from 0)
    scheduler_student = lr_scheduler.LambdaLR(
        optimizer_student, lr_lambda=lambda t: min((t + 1) / train_config.warmup, 1.0)
    ) if train_config.warmup > 0 else None
    scheduler_teacher = lr_scheduler.LambdaLR(
        optimizer_teacher, lr_lambda=lambda t: min((t + 1) / train_config.warmup, 1.0)
    ) if train_config.warmup > 0 else None

    split = "all" if dataset == "celeba" else "train"
    num_workers = args.num_workers
    trainloader, sampler = get_dataloader(
        dataset, batch_size=train_config.batch_size, split=split, val_size=0., random_seed=seed,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
    )  # drop_last to have a static input shape; num_workers > 0 to enable asynchronous data loading

    if args.dry_run:
        logger("This is a dry run.")
        args.chkpt_intv = 1
        args.image_intv = 1

    chkpt_student_dir = os.path.join(args_student.chkpt_dir, exp_name)
    chkpt_teacher_dir = os.path.join(args_teacher.chkpt_dir, exp_name)
    chkpt_student_path = os.path.join(chkpt_student_dir, args.chkpt_name or f"{exp_name}.pt")
    chkpt_teacher_path = os.path.join(chkpt_teacher_dir, args.chkpt_name or f"{exp_name}.pt")
    chkpt_intv = args.chkpt_intv
    logger(f"Checkpoint will be saved to {os.path.abspath(chkpt_student_path)} and {os.path.abspath(chkpt_teacher_path)}", end=" ")
    logger(f"every {chkpt_intv} and epoch(s)")

    image_dir = os.path.join(args.image_dir, "train", exp_name)
    logger(f"Generated images (x{train_config.num_samples}) will be saved to {os.path.abspath(image_dir)}", end=" ")
    logger(f"every {train_config.image_intv} epoch(s)")

    if is_leader:
        model_config["block_size"] = block_size
        hps = {
            "dataset": dataset,
            "seed": seed,
            "diffusion": diffusion_config,
            "model": model_config,
            "train": train_config
        }
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")

        if not os.path.exists(chkpt_student_dir):
            os.makedirs(chkpt_student_dir)
        if not os.path.exists(chkpt_teacher_dir):
            os.makedirs(chkpt_teacher_dir)
        # keep a record of hyperparameter settings used for this experiment run
        with open(os.path.join(chkpt_student_dir, f"exp_{timestamp}.info"), "w") as f:
            json.dump(hps, f, indent=2)
        with open(os.path.join(chkpt_teacher_dir, f"exp_{timestamp}.info"), "w") as f:
            json.dump(hps, f, indent=2)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

    trainer = CoStudyTrainer(
        model_student=model_student,
        model_teacher=model_teacher,
        optimizer_student=optimizer_student,
        optimizer_teacher=optimizer_teacher,
        diffusion=diffusion,
        epochs=train_config.epochs,
        trainloader=trainloader,
        sampler=sampler,
        scheduler_student=scheduler_student,
        scheduler_teacher=scheduler_teacher,
        num_accum=args.num_accum,
        use_ema=train_config.use_ema,
        grad_norm=train_config.grad_norm,
        shape=image_shape,
        device=train_device,
        chkpt_intv=chkpt_intv,
        image_intv=train_config.image_intv,
        num_samples=train_config.num_samples,
        ema_decay=args.ema_decay,
        rank=rank,
        distributed=distributed,
        dry_run=args.dry_run
    )

    if args.use_ddim:
        subsequence = get_selection_schedule(
            args.skip_schedule, size=args.subseq_size, timesteps=diffusion_config.timesteps)
        diffusion_eval = DDIM.from_ddpm(diffusion, eta=0., subsequence=subsequence)
    else:
        diffusion_eval = diffusion

    if args.eval:
        evaluator = Evaluator(
            dataset=dataset,
            diffusion=diffusion_eval,
            eval_batch_size=args.eval_batch_size,
            eval_total_size=args.eval_total_size,
            device=eval_device
        )
    else:
        evaluator = None

    # in the case of distributed training, resume should always be turned on
    resume = args.resume or distributed
    if resume:
        try:
            map_location = {"cuda:0": f"cuda:{local_rank}"} if distributed else train_device
            _chkpt_student_path = args_student.chkpt_path or chkpt_student_path
            _chkpt_teacher_path = args_teacher.chkpt_path or chkpt_teacher_path
            trainer.load_checkpoint(_chkpt_student_path, _chkpt_teacher_path, map_location=map_location)
        except FileNotFoundError:
            logger("Checkpoint file does not exist!")
            logger("Starting from scratch...")

    # use cudnn benchmarking algorithm to select the best conv algorithm
    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa
        logger(f"cuDNN benchmark: ON")

    logger("Training starts...", flush=True)
    trainer.train(evaluator, student_chkpt_path=chkpt_student_path, teacher_chkpt_path=chkpt_teacher_path, image_dir=image_dir)
    
if __name__ == "__main__":
    
    #Phase 1: Train the student model
    train_student_model()

    #Phase 2: 
    co_studying()

    #Phase 3:

