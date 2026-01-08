import os.path as osp
import json
import pickle
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from datasets.utils.smiles2graph import smile2graph4drugood


class DrugOODDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        version="chembl30",
        type="lbap",
        root="data",
        drugood_root="drugood-data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self.root = root
        self.drugood_root = drugood_root
        self.version = version
        self.type = type
        super(DrugOODDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self.data_cfg = pickle.load(open(self.processed_paths[1], "rb"))
        self.data_statistics = pickle.load(open(self.processed_paths[2], "rb"))
        self.train_index, self.valid_index, self.test_index = pickle.load(
            open(self.processed_paths[3], "rb")
        )
        self.num_tasks = 1

    @property
    def raw_dir(self):
        return self.drugood_root + "-" + self.version

    @property
    def raw_file_names(self):
        return f"{self.type}_core_{self.name}.json"

    @property
    def processed_dir(self):
        return osp.join(self.root, f"{self.type}-{self.name}-{self.version}")

    @property
    def processed_file_names(self):
        return "data.pt", "cfg.pt", "statistics.pt", "split.pt"

    def __subprocess(self, datalist):
        processed_data = []
        for datapoint in tqdm(datalist):
            smiles = datapoint["smiles"]
            x, edge_index, edge_attr = smile2graph4drugood(smiles)
            y = torch.tensor([datapoint["cls_label"]], dtype=torch.long).view(1, 1)
            data_kwargs = {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "y": y,
                "smiles": smiles,
                "reg_label": torch.tensor([datapoint["reg_label"]]),
                "assay_id": datapoint["assay_id"],
                "domain_id": datapoint["domain_id"],
            }

            if self.type != "lbap":
                data_kwargs["protein"] = datapoint["protein"]

            data = Data(**data_kwargs)

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            processed_data.append(data)
        return processed_data, len(processed_data)

    def process(self):
        json_data = json.load(open(self.raw_paths[0], "r", encoding="utf-8"))
        data_cfg, data_statistics = json_data["cfg"], json_data["statistics"]
        train_data = json_data["split"]["train"]
        valid_data = json_data["split"]["ood_val"]
        test_data = json_data["split"]["ood_test"]
        train_data_list, train_num = self.__subprocess(train_data)
        valid_data_list, valid_num = self.__subprocess(valid_data)
        test_data_list, test_num = self.__subprocess(test_data)
        data_list = train_data_list + valid_data_list + test_data_list
        train_index = list(range(train_num))
        valid_index = list(range(train_num, train_num + valid_num))
        test_index = list(
            range(train_num + valid_num, train_num + valid_num + test_num)
        )
        torch.save(self.collate(data_list), self.processed_paths[0])
        pickle.dump(data_cfg, open(self.processed_paths[1], "wb"))
        pickle.dump(data_statistics, open(self.processed_paths[2], "wb"))
        pickle.dump(
            [train_index, valid_index, test_index], open(self.processed_paths[3], "wb")
        )

    def __repr__(self):
        return "{}({})".format(self.name, len(self))


if __name__ == "__main__":
    dataset = DrugOODDataset(
        name="ic50_assay", root="../data", drugood_root="../drugood-data"
    )
    train_set = dataset[dataset.train_index]
    test_set = dataset[dataset.test_index]
    loader = DataLoader(train_set, batch_size=4, shuffle=True)
    for data in loader:
        import pdb

        pdb.set_trace()
