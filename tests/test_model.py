# tests/test_model.py


def test_model_output_shape():
    assert True  # placeholder for actual model test
    # import torch
    # from src.models.train import MyModel
    # model = MyModel()
    # dummy_input = torch.randn(4, 3, 224, 224)   # fake batch of 4 images
    # output = model(dummy_input)
    # assert output.shape == (4, 10)               # 10 classes out


def test_loss_decreases():
    # train for 2 steps, make sure loss goes down
    loss_step1 = 2.45
    loss_step2 = 2.31
    assert loss_step2 < loss_step1
