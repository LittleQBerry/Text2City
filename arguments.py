import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "--prompt", type=str, help="The prompt for the desired editing"
    )
    parser.add_argument(
        "--init_image", type=str, 
        default="/51.4116, -0.22398, 51.43039, -0.18878_8.png",
        help="The path to the source image input"
    )
    parser.add_argument(
    "--mask", 
    type=str,
    default="test.png",
    help="The path to the mask to edit with")

    parser.add_argument(
        '--image_size',
        type =int,
        default=256
    )
    parser.add_argument(
        "--prompt_h", type=str, 
        default="A map of commercial region with secondary roads and footway roads and commercial building footprint and office building footprint and pub building footprint.",
        help="The prompt for the desired editing"
    )
    parser.add_argument(
        "--prompt_b", type=str, 
        default="A map of commercial region with secondary roads and footway roads and commercial building footprint and office building footprint and pub building footprint.",
        help="The prompt for the desired editing"
    )
    parser.add_argument(
        "--init_image_r", type=str, 
        default='test.png',
        help="The path to the source image input"
    )
    parser.add_argument(
        "--init_image_b", type=str, 
        default='test.png',
        help="The path to the source image input"
    )

    


    # Diffusion
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=25,
    )
    parser.add_argument(
        "--local_clip_guided_diffusion",
        help="Indicator for using local CLIP guided diffusion (for baseline comparison)",
        action="store_true",
        dest="local_clip_guided_diffusion",
    )
    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true",
    )

    # For more details read guided-diffusion/guided_diffusion/respace.py
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default="100",
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    #attributes net
    parser.add_argument(
        '--lr',
        type=float,
        help="learning rate",
        default=1e-3
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=5000,
        help='epoch'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default="./log/"
    )
    parser.add_argument(
        '--start_epoch',
        type =int,
        default=0,
        help= 'start epoch, resume'
    )
    parser.add_argument(\
        '--device',
        type=str,
        default='cuda:0'
        )

    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=8)

    # Loss
    parser.add_argument(
        "--clip_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=100,
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
    )
    parser.add_argument(
        "--lpips_sim_lambda",
        type=float,
        help="The LPIPS similarity to the input image",
        default=100,
    )
    parser.add_argument(
        "--l2_sim_lambda", type=float, help="The L2 similarity to the input image", default=10000,
    )
    parser.add_argument(
        "--background_preservation_loss",
        help="Indicator for using the background preservation loss",
        action="store_true",
    )
    parser.add_argument(
        '--reconstruct_lambda',
        type=float,
        default=100
    )
    parser.add_argument(
        '--target_factor',
        type=float,
        default=9,
    )
    parser.add_argument(
        '--foorprint_lambda',
        type=float,
        default=100,
    )

    # Mask
    parser.add_argument(
        "--invert_mask",
        help="Indicator for mask inversion",
        action="store_true",
        dest="invert_mask",
    )
    parser.add_argument(
        "--no_enforce_background",
        help="Indicator disabling the last background enforcement",
        action="store_false",
        dest="enforce_background",
    )

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=404)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=2)
    parser.add_argument("--output_path", type=str, default="valid/")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=5)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=1,
    )
    parser.add_argument(
        "--vid",
        help="Indicator for saving the video of the diffusion process",
        action="store_true",
        dest="save_video",
    )
    parser.add_argument(
        "--export_assets",
        help="Indicator for saving raw assets of the prediction",
        action="store_true",
        dest="export_assets",
    )

    args = parser.parse_args()
    return args
