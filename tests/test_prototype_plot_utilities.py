import pytest

from interpretable_mnist import params
from interpretable_mnist import prototype_plot_utilities
from interpretable_mnist.data import load_mnist
from interpretable_mnist.proto_pnet import ProtoPNetMNIST


@pytest.mark.skip(reason="should be run manually")
def test_plotting_detailed_evidence_runs():
    train_config = params.Training()
    train_config.n_classes, train_config.n_protos_per_class = 10, 2
    train_config.constrain_prototypes_to_class = False

    model = ProtoPNetMNIST(train_config, prototype_depth=64, n_trainig_batches=1)
    mnist_batch = next(iter(load_mnist(relative_size_split_dataset=0.)))
    model.update_projected_prototypes(mnist_batch[0], mnist_batch[1])
    model.push_projected_prototypes()
    class_evidence = model.get_evidence_for_class(mnist_batch[0][0, ...])
    prototype_plot_utilities.plot_class_evidence(class_evidence)

