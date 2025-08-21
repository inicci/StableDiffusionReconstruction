import h5py
from PIL import Image
import scipy.io
import argparse, os
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys
sys.path.append("../utils/")
from nsd_access.nsda import NSDAccess
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
def _ensure_1x77x768(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dim() == 1:
        x = x.view(1, 77, 768)
    elif x.dim() == 2:
        x = x.unsqueeze(0)
    return x

def apply_perdim_affine(c, args, device, eps=1e-6):
    # 没开校准直接返回（保持原行为）
    if not hasattr(args, "calib") or args.calib == "none":
        return c

    if args.calib == "ridge":
        data = np.load(args.calib_npz)
        w = data["w"]; b = data["b"]  # 59136 或 (77,768)
        w = torch.from_numpy(w).float().to(device)
        b = torch.from_numpy(b).float().to(device)
        if w.dim() == 1: w = w.view(77,768)
        if b.dim() == 1: b = b.view(77,768)
        return c * w.unsqueeze(0) + b.unsqueeze(0)

    if args.calib == "stats":
        sp = np.load(args.stats_pred)  # mu,std
        st = np.load(args.stats_tgt)
        mu_p = torch.from_numpy(sp["mu"]).float().to(device)
        sd_p = torch.from_numpy(sp["std"]).float().to(device)
        mu_t = torch.from_numpy(st["mu"]).float().to(device)
        sd_t = torch.from_numpy(st["std"]).float().to(device)
        if mu_p.dim() == 1: mu_p = mu_p.view(77,768)
        if sd_p.dim() == 1: sd_p = sd_p.view(77,768)
        if mu_t.dim() == 1: mu_t = mu_t.view(77,768)
        if sd_t.dim() == 1: sd_t = sd_t.view(77,768)
        w = (sd_t / (sd_p + eps)).unsqueeze(0)
        b = (mu_t - w[0] * mu_p).unsqueeze(0)
        return c * w + b

    return c

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidx",
        required=True,
        type=int,
        help="img idx"
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        help="cvpr or text or gan",
    )
    parser.add_argument(
    "--calib",
    type=str,
    default="none",
    choices=["none", "stats", "ridge"],
    help="Per-dimension diagonal affine: none | stats | ridge",
    )
    parser.add_argument(
    "--calib_npz",
    type=str,
    default=None,
    help="Path to npz containing 'w' and 'b' for ridge calibration.",
    )
    parser.add_argument(
    "--stats_pred",
    type=str,
    default=None,
    help="Path to npz containing 'mu' and 'std' for predicted c (stats mode).",
    )
    parser.add_argument(
    "--stats_tgt",
    type=str,
    default=None,
    help="Path to npz containing 'mu' and 'std' for target text-encoder c (stats mode).",
    )
    # Set parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    imgidx = opt.imgidx
    gpu = opt.gpu
    method = opt.method
    subject=opt.subject
    gandir = f'../../decoded/gan_recon_img/all_layers/{subject}/streams/'
    captdir = f'../../decoded/{subject}/captions/'

    # Load NSD information
    nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that mos of them are 1-base index!
    # This is why I subtract 1
    sharedix = nsd_expdesign['sharedix'] -1 

    nsda = NSDAccess('../../nsd/')
    sf = h5py.File(nsda.stimuli_file, 'r')
    sdataset = sf.get('imgBrick')

    stims_ave = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')


    tr_idx = np.zeros_like(stims_ave)
    for idx, s in enumerate(stims_ave):
        if s in sharedix:
            tr_idx[idx] = 0
        else:
            tr_idx[idx] = 1

    # Load Stable Diffusion Model
    config = './stable-diffusion/configs/stable-diffusion/v1-inference.yaml'
    ckpt = './stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    model = load_model_from_config(config, f"{ckpt}", gpu)

    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    n_iter = 5
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    outdir = f'../../decoded/image-{method}/{subject}/'
    os.makedirs(outdir, exist_ok=True)

    sample_path = os.path.join(outdir, f"samples1")
    os.makedirs(sample_path, exist_ok=True)
    precision = 'autocast'
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # Load z (Image)
    imgidx_te = np.where(tr_idx==0)[0][imgidx] # Extract test image index
    idx73k= stims_ave[imgidx_te]
    Image.fromarray(np.squeeze(sdataset[idx73k,:,:,:]).astype(np.uint8)).save(
        os.path.join(sample_path, f"{imgidx:05}_org.png"))    
    
    if method in ['cvpr','text']:
        roi_latent = 'early'
        scores_latent = np.load(f'../../decoded/{subject}/{subject}_{roi_latent}_scores_init_latent.npy')
        imgarr = torch.Tensor(scores_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')

        # -------- Generate preview image from z0 (optional, just to form init_image) --------
    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                x_samples = model.decode_first_stage(imgarr)  # (B,3,H,W) in [-1,1]
                preview = 255.*rearrange(x_samples[0].cpu().numpy(),'c h w -> h w c')
                Image.fromarray(preview.astype(np.uint8)).save(os.path.join(sample_path, f"{imgidx:05}_zonly.png"))#保存Z only
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)  # [0,1]
                # 取第 0 张构造 init image（你原来循环后只用最后一张，这里显式取第 0 张）
                x0 = 255. * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
    im = Image.fromarray(x0.astype(np.uint8)).resize((512, 512))
    im = np.array(im)

    # 如果 method=='gan'，从文件读入初始图像覆盖
    if method == 'gan':
        ganpath = f'{gandir}/recon_image_normalized-VGG19-fc8-{subject}-streams-{imgidx:06}.tiff'
        im = Image.open(ganpath).resize((512,512))
        im = np.array(im)

    # 得到 init_latent（image-to-image 的起点）
    init_image = load_img_from_arr(im).to(device)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # (B,4,40,40)

    # -------- Load c (Semantics) --------
    if method == 'cvpr':
        roi_c = 'ventral'
        scores_c = np.load(f'../../decoded/{subject}/{subject}_{roi_c}_scores_c.npy')
        carr = scores_c[imgidx, :].reshape(77, 768)
        c = torch.tensor(carr, dtype=torch.float32, device=device).unsqueeze(0)  # (1,77,768)
    elif method in ['text', 'gan']:
        captions = pd.read_csv(f'{captdir}/captions_brain.csv', sep='\t', header=None)
        c = model.get_learned_conditioning(captions.iloc[imgidx][0]).to(device)  # (1,77,768)

    # ---- (NEW) 逐维对角仿射校准（如果你已加了函数/参数，不想校准就会是 no-op） ----
    c = _ensure_1x77x768(c)                 # 统一成 (1,77,768)
    c = apply_perdim_affine(c, opt, device) # opt.calib: "none"/"stats"/"ridge"

    # ------- Sampling from z_enc with semantic conditioning c -------
    base_count = 0
    uc = model.get_learned_conditioning(batch_size * [""]).to(device)  # 放循环外更高效
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for _ in trange(n_iter, desc="Sampling"):
                    # encode (scaled latent)
                    z_enc = sampler.stochastic_encode(
                        init_latent, torch.tensor([t_enc] * batch_size, device=device)
                    )
                    # decode it
                    samples = sampler.decode(
                        z_enc, c, t_enc,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                    )
                    x_samples = model.decode_first_stage(samples)  # [-1,1]
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)  # [0,1]

                    # 保存 batch 内所有图；通常 batch_size=1
                    for b in range(x_samples.shape[0]):
                        x_np = 255. * rearrange(x_samples[b].cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_np.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{imgidx:05}_{base_count:03}.png")
                        )
                        base_count += 1
   

if __name__ == "__main__":
    main()
