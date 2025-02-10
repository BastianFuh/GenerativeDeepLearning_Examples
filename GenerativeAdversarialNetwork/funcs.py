from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def calc_loss(loss_fn, prediction, expectation, z_mean, z_log_var):
    if z_mean is None or z_log_var is None:
        loss = loss_fn(prediction)
    else:
        # 500 represent the beta loss of a beta-VAE
        reconstruction_loss = 500 * loss_fn(prediction, expectation)
        # kl = Lullback-Leibler
        kl_loss = torch.mean(
            -0.5
            * torch.sum(
                1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), axis=1
            )
        )
        loss = reconstruction_loss + kl_loss
    return loss


def train(
    model,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    progress_bar: tqdm,
    is_variational=False,
    device="cuda",
):
    """Training function"""

    num_batches = len(dataloader)
    reporting_interval = num_batches / 5
    d_test_loss, g_test_loss = 0.0, 0.0

    # Prepare data

    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["image"].to(device)
        random_latent_vectors = torch.randn(
            size=(data.size(0), 100, 1, 1), device=device
        )

        ##########
        # 1. Update Discriminator
        ##########

        model.discriminator.zero_grad()

        generated_images = model.generator(random_latent_vectors)

        real_predictions = model.discriminator(data)
        fake_predictions = model.discriminator(generated_images)

        real_labels = torch.ones_like(real_predictions, device=device)
        fake_labels = torch.zeros_like(fake_predictions, device=device)

        real_noisy_labels = real_labels - 0.1 * torch.rand_like(real_labels)
        fake_noisy_labels = fake_labels + 0.1 * torch.rand_like(fake_labels)

        d_real_loss = loss_fn(real_predictions, real_noisy_labels)
        d_fake_loss = loss_fn(fake_predictions, fake_noisy_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2.0

        d_loss.backward(retain_graph=True)

        optimizer["discriminator"].step()

        ##########
        # 2. Update Generator
        ##########

        model.generator.zero_grad()

        # Again, because the model was just updated
        fake_predictions = model.discriminator(generated_images)

        # How many fake images were guesses real
        g_loss = loss_fn(fake_predictions, real_labels)

        g_loss.backward()
        optimizer["generator"].step()

        progress_bar.update(1)

        d_test_loss += d_loss.item()
        g_test_loss += g_loss.item()

        if i % reporting_interval == 0 and i != 0:
            d_print_loss = d_test_loss / reporting_interval
            g_print_loss = g_test_loss / reporting_interval
            tqdm.write(
                f"{i}/{num_batches} Current avg training loss descriminator {d_print_loss}"
            )
            tqdm.write(
                f"{i}/{num_batches} Current avg training loss generator     {g_print_loss}"
            )
            d_test_loss = 0.0
            g_test_loss = 0.0


def eval(model, dataloader: DataLoader, loss_fn, is_variational=False, device="cuda"):
    """Evaluation function."""
    num_batches = len(dataloader)

    d_test_loss, g_test_loss = 0.0, 0.0

    progress_bar = tqdm(range(num_batches), position=1)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch["image"].to(device)
            random_latent_vectors = torch.randn(
                size=(data.size(0), 100, 1, 1), device=device
            )

            ##########
            # 1. Check Discriminator
            ##########

            generated_images = model.generator(random_latent_vectors)

            real_predictions = model.discriminator(data)
            fake_predictions = model.discriminator(generated_images)

            real_labels = torch.ones_like(real_predictions, device=device)
            fake_labels = torch.zeros_like(fake_predictions, device=device)

            d_real_loss = loss_fn(real_predictions, real_labels)
            d_fake_loss = loss_fn(fake_predictions, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2.0

            ##########
            # 2. Check Generator
            ##########

            # How many fake images were guesses real
            g_loss = loss_fn(fake_predictions, real_labels)

            progress_bar.update(1)
            d_test_loss += d_loss.item()
            g_test_loss += g_loss.item()

    d_print_loss = d_test_loss / num_batches
    g_print_loss = g_test_loss / num_batches

    tqdm.write(f"Avg training loss descriminator {d_print_loss}")
    tqdm.write(f"Avg training loss generator     {g_print_loss}")

    return g_print_loss


def collate_fn(batch):
    result = dict()

    result["image"] = torch.stack([x["image"] for x in batch])

    return result
