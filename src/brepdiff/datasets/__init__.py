from brepdiff.datasets.abc_dataset import (
    ABCDataset,
    DeepCadDataset,
    ABC50,
    DeepCad30Dataset,
)
from brepdiff.datasets.custom_step_dataset import CustomStepDataset

DATASETS = {
    ABCDataset.name: ABCDataset,
    DeepCadDataset.name: DeepCadDataset,
    ABC50.name: ABC50,
    DeepCad30Dataset.name: DeepCad30Dataset,
    CustomStepDataset.name: CustomStepDataset,
}
