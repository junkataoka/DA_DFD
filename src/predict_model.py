import click
from models.reflownet import EncoderDecoderConvLSTM, da_cos_loss
from data.dataloader import generate_dataloader
from visualization.visualize import visualize
import numpy as np
import os
import torch


def load_model(arch, log, pretrained=True):
    if pretrained:
        checkpoint = torch.load(f"./models/{log}/pretrained.ckpt", map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(f"./models/{log}/trained.ckpt", map_location=torch.device('cpu'))
    arch.load_state_dict(checkpoint)

def predict(tar_x_test, model, num_areas, domain):
    with torch.no_grad():
        tar_output, _, _ = model(tar_x_test, domain=domain, future_step=num_areas)

        return tar_output

@click.command()
@click.option('--n_hidden_dim', nargs=1, type=int, default=3)
@click.option('--channel', nargs=1, type=int, default=2)
@click.option('--seq_len', nargs=1, type=int, default=15)
@click.option('--num_areas', nargs=1, type=int, default=7)
@click.argument('data_path', nargs=1, type=click.Path(exists=True))
@click.option('--log', nargs=1, type=str, default="test0_notar_all")
def main(data_path, log, n_hidden_dim, channel, seq_len, num_areas):
    test_tar_dataloader = generate_dataloader(
        geom_path = os.path.join(data_path, "test-tar-GEOM"),
        heatmap_path = os.path.join(data_path, "test-tar-HEATMAP"),
        recipe_path = os.path.join(data_path, "test-tar-RECIPE"),
        batch_size=1,
        train=False,
    )
    model_trained = EncoderDecoderConvLSTM(nf=n_hidden_dim, in_chan=channel, seq_len=seq_len).double()
    load_model(model_trained, log, pretrained=False)
    model_pretrained = EncoderDecoderConvLSTM(nf=n_hidden_dim, in_chan=channel, seq_len=seq_len).double()
    load_model(model_pretrained, log, pretrained=True)

    for i in range(len(test_tar_dataloader)):
        tar_x, tar_y = next(iter(test_tar_dataloader))
        tar_x = tar_x.double()
        tar_y = tar_y.double()
        tar_output_trained = predict(tar_x, model=model_trained, num_areas=num_areas, domain="tar")
        tar_output_pretrained = predict(tar_x, model=model_pretrained, num_areas=num_areas, domain="src")

        tar_output_trained = np.exp(tar_output_trained.view(-1).numpy())
        tar_output_pretrained = np.exp(tar_output_pretrained.view(-1).numpy())
        tar_y = tar_y.view(-1).numpy()
        
        visualize(tar_y, tar_output_trained, tar_output_pretrained, log, i)

    print("Saved prediction")

if __name__ == '__main__':
    main()