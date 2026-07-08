"""
전처리된 커스텀 STEP 데이터를 로드하는 Dataset 클래스.
preprocess_step_files.py로 생성된 H5 데이터를 BrepDiff 학습에 사용.
"""

import os
from typing import List

from brepdiff.config import Config
from brepdiff.datasets.abc_dataset import ABCDataset


class CustomStepDataset(ABCDataset):
    name = "custom_step"

    def __init__(self, config: Config, split: str = "train"):
        # custom_dataset_name이 설정에 없으면 기본값 사용
        if not hasattr(config, "custom_dataset_name"):
            config.custom_dataset_name = "custom"
        if not hasattr(config, "custom_max_n_prims"):
            config.custom_max_n_prims = config.max_n_prims
        super().__init__(config, split)

    def get_data_list_path(self) -> str:
        dataset_name = self.config.custom_dataset_name
        max_prims = self.config.custom_max_n_prims
        data_list_path = os.path.join(
            os.path.dirname(self.config.h5_path),
            f"{dataset_name}_{max_prims}_{self.split}.txt",
        )
        return data_list_path

    def get_invalid_step_list_path(self) -> str:
        dataset_name = self.config.custom_dataset_name
        max_prims = self.config.custom_max_n_prims
        invalid_path = os.path.join(
            os.path.dirname(self.config.h5_path),
            f"{dataset_name}_{max_prims}_pkl_absence.txt",
        )
        return invalid_path

    def get_data_list(self) -> List:
        with open(self.data_list_path, "r") as f:
            data_list = f.readlines()
        data_list = [x.replace("\n", "").encode("utf-8") for x in data_list]

        # invalid list 제거
        if os.path.exists(self.invalid_step_list_path):
            with open(self.invalid_step_list_path, "r") as f:
                invalid_step_list = f.readlines()
            invalid_step_list = [
                x.replace("\n", "").encode("utf-8") for x in invalid_step_list
            ]
            data_list = sorted(list(set(data_list) - set(invalid_step_list)))
        else:
            data_list = sorted(data_list)

        # max_n_prims 필터링
        data_list_new = []
        for uid in data_list:
            if uid not in self.h5["data"]:
                continue
            uid_data = self.h5["data"][uid]
            # use coords (always present) to get face count
            n_faces = uid_data["coords"].shape[0]
            if n_faces > self.config.max_n_prims:
                continue
            data_list_new.append(uid)

            if self.config.overfit:
                if len(data_list_new) >= self.config.overfit_data_size:
                    break
            if self.config.debug:
                if len(data_list_new) == self.config.debug_data_size:
                    break

        if self.config.overfit:
            data_list_new = data_list_new * self.config.overfit_data_repetition

        return data_list_new
