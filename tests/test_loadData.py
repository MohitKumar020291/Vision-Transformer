from vitm.data.dataloader import loadData
import pytest

def test_load_data():
    image_size, num_classes, loader = loadData(batch_size=8, resize=32)
    print(image_size)
    assert isinstance(image_size, tuple)
    assert num_classes == 10
    assert hasattr(loader, '__iter__')
