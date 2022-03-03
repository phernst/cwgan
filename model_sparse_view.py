import json
import os
from os.path import join as pjoin
import time

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import RandomAffine
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse

from data import CTDataset, load_h5
from preprocessing import rescale
from unet import UNet

# Sparse-View CT Reconstruction Using Wasserstein GANs
# 10.1007/978-3-030-00129-2_9


class GeneratorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_channels=2)

    def forward(self, tensor):
        return torch.relu(tensor[:, 0:1] + self.unet(tensor))


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.LeakyReLU(.2),
            nn.LayerNorm([128, 30, 30]),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.LeakyReLU(.2),
            nn.LayerNorm([256, 14, 14]),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1),
            nn.LeakyReLU(.2),
        )

        self.last_conv = nn.Conv2d(512, 1, kernel_size=4)

    def forward(self, tensor):
        x = self.conv1(tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.last_conv(x)
        output = torch.mean(x, dim=(-1, -2))

        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class WGAN():
    def __init__(self, level):
        # parameters
        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 5e-5

        self.d_iter = 5
        self.lambda_gp = 10

        self.lambda_d = 1e-3
        self.lambda_l1 = 1

        self.level = level

        self.loss_dir = "loss"
        self.model_name = f"sparse_{self.level}"
        self.save_dir = pjoin("model", self.model_name)
        self.device = 'cpu'

        self.generator = GeneratorNet()
        self.discriminator = DiscriminatorNet()

        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
        )

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()
            self.device = 'cuda'

        initialize_weights(self.generator)
        initialize_weights(self.discriminator)

        self.dataset = CTDataset(self.level, 'train')
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
        )

        self.valid_dataset = CTDataset(self.level, 'valid')
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
        )

        self.running_best_loss = None

    def train(self):
        # repeat for number of epochs
        for epoch in range(self.epochs):
            self.validate(epoch)

            # train on batches
            progress = tqdm(enumerate(self.dataloader), total=len(self.dataset)//self.batch_size)
            for batch_index, batch in progress:
                free_img = batch["free_img"]
                noised_img = batch["noised_img"]
                prior_img = batch["prior_img"]

                # prior_img = self._misalign_prior(prior_img).to(self.device)

                # train discriminator
                for _ in range(self.d_iter):
                    loss = self._train_discriminator(free_img, noised_img, prior_img)

                # train generator
                loss = self._train_generator(free_img, noised_img, prior_img)
                progress.set_postfix(loss=loss)

                # save model after 100 batches
                if batch_index % 100 == 0:
                    self.save_model()

    @staticmethod
    def _misalign_prior(prior_batch: torch.Tensor) -> torch.Tensor:
        return RandomAffine(
            degrees=20,
            translate=(.1, .1),
            scale=(.9, 1.2)
        )(prior_batch)

    def _train_discriminator(self, free_img, noised_img, prior_img, train=True):
        self.d_optimizer.zero_grad()

        z = noised_img.clone().detach().requires_grad_(True).to(self.device)
        prior_img = prior_img.clone().detach().requires_grad_(True).to(self.device)
        real_img = free_img.clone().detach().requires_grad_(True).to(self.device)

        fake_img = self.generator(torch.cat((z, prior_img), dim=1))
        real_validity = self.discriminator(torch.cat((real_img, prior_img), dim=1))
        fake_validity = self.discriminator(torch.cat((fake_img.data, prior_img), dim=1))
        gradient_penalty = self._calc_gradient_penalty(
            real_img.data, fake_img.data, prior_img.data)

        d_loss = torch.mean(-real_validity) + torch.mean(fake_validity) + \
            self.lambda_gp * gradient_penalty
        if train:
            d_loss.backward()
            self.d_optimizer.step()

        return d_loss.data.item(), torch.mean(-real_validity).cpu().item(), \
            torch.mean(fake_validity).cpu().item(), \
            self.lambda_gp * gradient_penalty.cpu().item()

    def _train_generator(self, free_img, noised_img, prior_img, train=True):
        z = noised_img.clone().detach().requires_grad_(True).to(self.device)
        prior_img = prior_img.clone().detach().requires_grad_(True).to(self.device)
        real_img = free_img.clone().detach().to(self.device)

        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

        criterion_l1 = nn.L1Loss()

        fake_img = self.generator(torch.cat((z, prior_img), dim=1))
        l1_loss = criterion_l1(fake_img, real_img)
        if train:
            (self.lambda_l1 * l1_loss).backward(retain_graph=True)

        fake_validity = self.discriminator(torch.cat((fake_img, prior_img), dim=1))
        g_loss = self.lambda_d * torch.mean(-fake_validity)

        if train:
            g_loss.backward()
            self.g_optimizer.step()
        return g_loss.data.item(), l1_loss.data.item(), \
            torch.mean(-fake_validity).data.item()

    def _calc_gradient_penalty(self, free_img, gen_img, prior_img):
        batch_size = free_img.shape[0]
        alpha = torch.rand(batch_size, 1, requires_grad=True, device=self.device)
        alpha = alpha.expand(batch_size, free_img.numel()//batch_size)
        alpha = alpha.contiguous().view(free_img.shape).float()

        interpolates = (alpha*free_img + (1 - alpha)*gen_img).requires_grad_(True)
        disc_interpolates = self.discriminator(torch.cat((interpolates, prior_img), dim=1))
        fake = torch.ones(batch_size, 1, device=self.device)

        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def validate(self, epoch):
        total_l1_loss = 0
        total_g_loss = 0
        total_d_loss = 0
        batch_num = 0

        with torch.inference_mode():
            for batch in tqdm(self.valid_dataloader):
                free_img = batch["free_img"]
                noised_img = batch["noised_img"]
                prior_img = batch["prior_img"].to(self.device)

                loss = self._train_generator(free_img, noised_img, prior_img, train=False)
                total_g_loss += loss[0]
                total_l1_loss += loss[1]
                total_d_loss += loss[2]
                batch_num += 1

        l1_loss = total_l1_loss / batch_num
        g_loss = total_g_loss / batch_num
        d_loss = total_d_loss / batch_num
        print(
            f"{self.model_name} Epoch: {epoch} lr:{self.g_optimizer.defaults['lr']:.10}"
            f" Test Loss: g-loss: {g_loss:.4f} "
            f"l1-loss: {l1_loss:.4f} d_loss: {d_loss:.4f}"
        )
        print('compute quality')
        self.validation_metrics(epoch)
        print('save loss')
        self.save_loss((l1_loss, g_loss, d_loss))
        self.save_model()

        # save best model
        if self.running_best_loss is None or l1_loss < self.running_best_loss:
            self.running_best_loss = l1_loss
            self.save_model(best=True)

    def validation_metrics(self, epoch):
        psnr1 = []
        psnr2 = []
        nrmse1 = []
        nrmse2 = []
        ssim1 = []
        ssim2 = []

        with open('train_valid_test.json', 'r', encoding='utf-8') as file:
            subjects = json.load(file)['valid_subjects']
            filenames = sorted([
                f for f in os.listdir('data/dataset/Free')
                if f.endswith('.h5') and f.split('_', 1)[0] in subjects
            ])

        for filename in filenames:
            free_img = load_h5(pjoin('data/dataset/Free', filename))[()]
            noised_img = load_h5(f"data/dataset/noise_{self.level}/{filename}")[()]
            prior_img = load_h5(pjoin('data/dataset/priors', f"{filename.split('_', 1)[0]}.h5"))[()]

            free_img, noised_img, prior_img = rescale(
                torch.from_numpy(free_img.transpose()).float().cuda()[:, None],
                torch.from_numpy(noised_img.transpose()).float().cuda()[:, None],
                torch.from_numpy(prior_img.transpose()).float().cuda()[:, None],
                inplane_shape=(128, 128),
            )
            free_img = free_img[:, 0].cpu().numpy().transpose()
            noised_img = noised_img[:, 0].cpu().numpy().transpose()
            prior_img = prior_img[:, 0].cpu().numpy().transpose()

            denoised_img = self.denoising(noised_img, prior_img)

            max_val = np.max(free_img)
            psnr1.append(compare_psnr(free_img, noised_img, data_range=max_val))
            psnr2.append(compare_psnr(free_img, denoised_img, data_range=max_val))

            nrmse1.append(compare_nrmse(free_img, noised_img))
            nrmse2.append(compare_nrmse(free_img, denoised_img))

            ssim1.append(compare_ssim(
                free_img, noised_img,
                data_range=max_val, channel_axis=-1))
            ssim2.append(compare_ssim(
                free_img, denoised_img,
                data_range=max_val, channel_axis=-1))

            denoised_image = nib.Nifti1Image(denoised_img, np.eye(4))
            os.makedirs(pjoin("valid", self.model_name, f"{epoch}"), exist_ok=True)
            nib.save(denoised_image, pjoin("valid", self.model_name, f"{epoch}", f"{filename.split('.')[0]}_denoised.nii.gz"))
            free_img = nib.Nifti1Image(free_img, np.eye(4))
            nib.save(free_img, pjoin("valid", self.model_name, f"{epoch}", f"{filename.split('.')[0]}_free.nii.gz"))
            prior_img = nib.Nifti1Image(prior_img, np.eye(4))
            nib.save(prior_img, pjoin("valid", self.model_name, f"{epoch}", f"{filename.split('.')[0]}_prior.nii.gz"))

        psnr1 = np.mean(psnr1)
        psnr2 = np.mean(psnr2)
        nrmse1 = np.mean(nrmse1)
        nrmse2 = np.mean(nrmse2)
        ssim1 = np.mean(ssim1)
        ssim2 = np.mean(ssim2)

        timestr = time.strftime("%H:%M:%S", time.localtime())
        os.makedirs(pjoin('loss', self.model_name), exist_ok=True)
        with open(pjoin("loss", self.model_name, "psnr.csv"), "a+", encoding='utf-8') as f:
            f.write(
                f"{timestr}: {self.learning_rate:.10f}, {psnr1}, {psnr2}, {ssim1}, {ssim2}"
                f", {nrmse1}, {nrmse2}"
            )
        print(
            f"psnr: {psnr1}, {psnr2}, "
            f"ssim: {ssim1}, {ssim2}, "
            f"nrmse: {nrmse1}, {nrmse2}\n"
        )

    def denoising(self, noised_vol, prior_vol):
        denoised_vol = []
        with torch.inference_mode():
            for z_idx in range(200):
                noised_img = torch.from_numpy(noised_vol[..., z_idx]).float().cuda()
                noised_img = noised_img[None, None]
                prior_img = torch.from_numpy(prior_vol[..., z_idx]).float().cuda()
                prior_img = prior_img[None, None]
                denoised_img = self.generator(torch.cat((noised_img, prior_img), dim=1))
                denoised_vol.append(denoised_img[0, 0])

        return torch.stack(denoised_vol, dim=-1).cpu().numpy()

    def save_model(self, best: bool = False):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(
            self.generator.state_dict(),
            pjoin(self.save_dir, f"G_{self.model_name}{'_best' if best else '_latest'}.pkl"),
        )
        torch.save(
            self.discriminator.state_dict(),
            pjoin(self.save_dir, f"D_{self.model_name}{'_best' if best else '_latest'}.pkl"),
        )

    def load_model(self, best: bool = False):
        if not os.path.exists(pjoin(self.save_dir, f"G_{self.model_name}{'_best' if best else '_latest'}.pkl")) or \
                not os.path.exists(pjoin(self.save_dir, f"D_{self.model_name}{'_best' if best else '_latest'}.pkl")):
            return False

        self.generator.load_state_dict(torch.load(
            pjoin(self.save_dir, f"G_{self.model_name}{'_best' if best else '_latest'}.pkl"),
            map_location={'cuda:1': 'cuda:0'},
        ))
        self.discriminator.load_state_dict(torch.load(
            pjoin(self.save_dir, f"D_{self.model_name}{'_best' if best else '_latest'}.pkl"),
            map_location={'cuda:1': 'cuda:0'},
        ))
        return True

    def save_loss(self, loss):
        value = ""
        for item in loss:
            value += f'{item},'
        value += "\n"
        os.makedirs(self.loss_dir, exist_ok=True)
        with open(pjoin(self.loss_dir, f"{self.model_name}.csv"), "a+", encoding='utf-8') as f:
            f.write(value)


def summarize_models():
    from torchinfo import summary
    generator = GeneratorNet().cuda()
    discriminator = DiscriminatorNet().cuda()

    summary(generator, input_size=(1, 2, 128, 128), device='cuda')
    summary(discriminator, input_size=(1, 2, 128, 128), device='cuda')


def main():
    level = 13
    wgan = WGAN(level)

    # training
    wgan.train()
    # print(wgan.load_model())

    # testing
    wgan.load_model(best=True)

    with open('train_valid_test.json', 'r', encoding='utf-8') as file:
        subjects = json.load(file)['test_subjects']
        filenames = sorted([
            f for f in os.listdir('data/dataset/Free')
            if f.endswith('.h5') and f.split('_', 1)[0] in subjects
        ])
    for filename in filenames:
        x = load_h5(pjoin(f'data/dataset/noise_{level}', filename))[()]
        x = rescale(
            torch.from_numpy(x.transpose()).float().cuda()[None],
            inplane_shape=(128, 128),
        )[0][0].cpu().numpy().transpose()

        prior = load_h5(pjoin('data/dataset/priors', f"{filename.split('_')[0]}.h5"))[()]
        prior = rescale(
            torch.from_numpy(prior.transpose()).float().cuda()[None],
            inplane_shape=(128, 128),
        )[0][0].cpu().numpy().transpose()

        denoised_img = wgan.denoising(x, prior)
        print(denoised_img.shape)

        denoised_image = nib.Nifti1Image(denoised_img, np.eye(4))
        nib.save(denoised_image, f"result/{filename.split('.', 1)[0]}_{level}.nii.gz")


if __name__ == "__main__":
    main()
