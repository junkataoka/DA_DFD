# content of test/conftest.py 
import pytest
@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    print("CONFTEST LOADED")
    # your setup code goes here, executed ahead of first test
    import random
    random.seed(0)
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)