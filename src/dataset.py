import json
import os
from datetime import datetime

# pyrefly: ignore [missing-import]
import geopandas as gpd
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.utils.data as tdata


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        # Select Fold samples
        if folds is not None:
            self.meta_patch = pd.concat(
                [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
            )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                with open(
                    os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                ) as file:
                    normvals = json.loads(file.read())
                selected_folds = folds if folds is not None else range(1, 6)
                means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                self.norm[s] = (
                    torch.from_numpy(self.norm[s][0]).float(),
                    torch.from_numpy(self.norm[s][1]).float(),
                )
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.folder,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]

        return (data, dates), target


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


def compute_norm_vals(folder, sat):
    norm_vals = {}
    for fold in range(1, 6):
        dt = PASTIS_Dataset(folder=folder, norm=False, folds=[fold], sats=[sat])
        means = []
        stds = []
        for i, b in enumerate(dt):
            print("{}/{}".format(i, len(dt)), end="\r")
            data = b[0][0][sat]  # T x C x H x W
            data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
            means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
            stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

        mean = np.stack(means).mean(axis=0).astype(float)
        std = np.stack(stds).mean(axis=0).astype(float)

        norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

    with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
        file.write(json.dumps(norm_vals, indent=4))


class AgricultureVisionDataset(tdata.Dataset):
    def __init__(self, folder, norm=True, folds=None, reference_date="2018-09-01", target="semantic", sats=None, mono_date=None):
        super(AgricultureVisionDataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.target = target
        
        # List files and sort them
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
        
        # Partition 800 files into 5 folds based on index
        # Fold 1: 0-159, Fold 2: 160-319, Fold 3: 320-479, Fold 4: 480-639, Fold 5: 640-799
        self.fold_assignments = {}
        for idx, f in enumerate(self.files):
            fold = (idx // 160) + 1
            if fold > 5:
                fold = 5
            self.fold_assignments[f] = fold
            
        # Select files belonging to selected folds
        if folds is not None:
            self.selected_files = [f for f in self.files if self.fold_assignments[f] in folds]
        else:
            self.selected_files = self.files
            
        self.len = len(self.selected_files)
        print("AgricultureVisionDataset ready with {} samples from folds {}.".format(self.len, folds))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        filename = self.selected_files[item]
        path = os.path.join(self.folder, filename)
        
        # Load image
        img = Image.open(path)
        img_arr = np.array(img).astype(np.float32) / 255.0  # scale to [0, 1]
        
        # Standard normalization
        if self.norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_arr = (img_arr - mean) / std
            
        # Shape: C x H x W
        img_arr = img_arr.transpose(2, 0, 1)
        
        # Convert to PyTorch tensor and add temporal dimension: T x C x H x W (T = 1)
        data = torch.from_numpy(img_arr).unsqueeze(0)  # shape (1, 3, 256, 256)
        
        # Dummy dates: shape (1,)
        dates = torch.zeros(1, dtype=torch.float32)
        
        # Dummy mask target
        if self.target == "semantic":
            # shape (256, 256)
            target = torch.zeros((img_arr.shape[1], img_arr.shape[2]), dtype=torch.long)
        else:
            # For panoptic/instance targets: shape (256, 256, 7)
            # 7 channels represent: heatmap, instances, zones, size (2 channels), sem_obj, sem_pix
            target = torch.zeros((img_arr.shape[1], img_arr.shape[2], 7), dtype=torch.float32)
        
        return (data, dates), target


class KomatsunaDataset(tdata.Dataset):
    def __init__(self, folder, norm=True, folds=None, reference_date="2018-09-01", target="semantic", sats=None, mono_date=None):
        super(KomatsunaDataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.target = target
        
        self.images_dir = os.path.join(folder, "train", "images")
        self.masks_dir = os.path.join(folder, "train", "masks")
        
        # List and sort files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])
        self.mask_files = sorted([f for f in os.listdir(self.masks_dir) if f.endswith(".png")])
        
        # Partition 300 files into 5 folds based on index
        # 60 images per fold
        self.fold_assignments = {}
        for idx, f in enumerate(self.image_files):
            fold = (idx // 60) + 1
            if fold > 5:
                fold = 5
            self.fold_assignments[f] = fold
            
        # Select files belonging to selected folds
        if folds is not None:
            self.selected_files = [f for f in self.image_files if self.fold_assignments[f] in folds]
        else:
            self.selected_files = self.image_files
            
        self.len = len(self.selected_files)
        print("KomatsunaDataset ready with {} samples from folds {}.".format(self.len, folds))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_filename = self.selected_files[item]
        mask_filename = img_filename.replace("rgb_", "label_")
        
        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)
        
        # Load image and resize to 256x256
        img = Image.open(img_path).resize((256, 256), Image.BILINEAR)
        img_arr = np.array(img).astype(np.float32) / 255.0  # scale to [0, 1]
        
        # Standard normalization
        if self.norm:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_arr = (img_arr - mean) / std
            
        # Shape: C x H x W
        img_arr = img_arr.transpose(2, 0, 1)
        
        # Convert to PyTorch tensor and add temporal dimension: T x C x H x W (T = 1)
        data = torch.from_numpy(img_arr).unsqueeze(0)  # shape (1, 3, 256, 256)
        
        # Dummy dates: shape (1,)
        dates = torch.zeros(1, dtype=torch.float32)
        
        # Load mask and resize to 256x256 using Nearest Neighbor
        mask = Image.open(mask_path).resize((256, 256), Image.NEAREST)
        mask_arr = np.array(mask)  # shape (256, 256, 3)
        
        # Convert RGB to class labels:
        # Green channel > 100 -> class 1
        # Blue channel > 100 -> class 2
        # Otherwise -> class 0
        h, w, _ = mask_arr.shape
        label_mask = np.zeros((h, w), dtype=np.int64)
        label_mask[mask_arr[:, :, 1] > 100] = 1
        label_mask[mask_arr[:, :, 2] > 100] = 2
        
        if self.target == "semantic":
            target = torch.from_numpy(label_mask).long()
        else:
            # For panoptic/instance targets: shape (256, 256, 7)
            # We map target channels accordingly
            target_tensor = np.zeros((h, w, 7), dtype=np.float32)
            
            # heatmap (class 1 and 2 as objects)
            target_tensor[:, :, 0] = (label_mask > 0).astype(np.float32)
            
            # instance labels
            target_tensor[:, :, 1] = label_mask.astype(np.float32)
            
            # zones
            target_tensor[:, :, 2] = label_mask.astype(np.float32)
            
            # size (dummy size 10x10)
            target_tensor[:, :, 3] = 10.0
            target_tensor[:, :, 4] = 10.0
            
            # sem_obj
            target_tensor[:, :, 5] = label_mask.astype(np.float32)
            
            # sem_pix
            target_tensor[:, :, 6] = label_mask.astype(np.float32)
            
            target = torch.from_numpy(target_tensor).float()
            
        return (data, dates), target
