import torch
from src.avatar import WAVATAR
from src.dataloader import generate_dataset
import pytest
from src.validation import validate

# @pytest.mark.skip(reason="no way of currently testing this")
def test_validate():
    # Test validate function

    num_classes = 4
    model = WAVATAR(1, num_classes).cuda()
    src_dataset, tar_dataset = generate_dataset("/data/home/jkataok1/DA_DFD/data/processed", 
                     "CWRU", "CWRU", 0, 1)
    src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=24, shuffle=True, )
    tar_dataloader = torch.utils.data.DataLoader(tar_dataset, batch_size=24)
    logger=None
    out_dict = validate(model, src_dataloader, tar_dataloader, num_classes, logger)

