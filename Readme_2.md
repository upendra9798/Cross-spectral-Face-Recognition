".\.venv311\Scripts\python.exe" ".\sftl54_cross_spectral_pipeline.py":-

Your output indicates that the preprocessing pipeline completed successfully:

RGB and thermal files were found.
Pairs were matched correctly.
No duplicate keys were detected.
Incomplete pairs were removed.
Faces were detected and cropped.
All processed pairs have both RGB and thermal images.
Train, validation, and test subjects are separated correctly.
The final dataset is ready for PyTorch model training.
The usual next step is to use sftl54_pytorch_dataset.py inside a training script that feeds the RGB and thermal tensors into a cross-spectral face-recognition model.

RGB without thermal: 43
Thermal without RGB: 43

Final usable dataset:
train: 1772
val:    176
test:   565

### Subject separation:
train: 100 subjects
val:    10 subjects
test:   32 subjects

test_vs_train: 0
test_vs_val:   0
train_vs_val: 0  //This means the same person does not appear in multiple splits. This is called a subject-disjoint split.

# SFTL54 Cross-Spectral Dataset Pipeline

## Purpose

`sftl54_cross_spectral_pipeline.py` prepares the SFTL54 cross-spectral
face-recognition dataset for PyTorch. It matches each visible-light RGB face
image with its corresponding thermal/infrared image, validates the pairs,
uses the dataset's existing face annotations, crops both modalities, and
writes a model-ready manifest.

This script is a **preprocessing pipeline**. It does not train a neural
network and it does not create face embeddings. Embeddings are generated
later when the processed images are passed through a face-recognition model.

## Overall flow

```text
SFTL54 dataset ZIP
        |
        v
Read RGB and thermal image paths
        |
        v
Create a pair_key from each filename
        |
        v
Match one RGB image with one thermal image
        |
        v
Remove incomplete or invalid pairs
        |
        v
Verify subject-disjoint train/validation/test splits
        |
        v
Read precomputed face annotations from labels.jsonl
        |
        v
Create a square face crop with 35% margin
        |
        v
Resize RGB and thermal crops to 112 x 112
        |
        v
Save processed images and manifest.csv
        |
        v
Load the processed pairs with PyTorch
        |
        v
Train/evaluate a cross-spectral model
        |
        v
Generate RGB and thermal face embeddings
```

## Step-by-step working

### 1. Download and inspect the dataset

The script uses:

```python
DATASET_REPO = "issai/SFTL54"
DATASET_FILE = "sftl.zip"
```

It opens the ZIP archive and searches these directories:

```text
sftl/rgb/<split>/                  RGB images
sftl/iron/<split>/images/         thermal images
sftl/iron/<split>/labels.jsonl    face annotations
sftl/metadata.xlsx                subject metadata
```

The splits are `train`, `val`, and `test`.

### 2. Build a common pairing key

The image filename contains a final modality/variant number. The pipeline
removes that final number to create a common `pair_key`.

For example:

```text
RGB:     111_1_2_1_120_100_3.png
Thermal: 111_1_2_1_120_100_1.png

Pair key for both:
         111_1_2_1_120_100
```

The first filename component is also used as the `subject_id`:

```text
111_1_2_1_120_100_3.png  -> subject_id 111
```

The pair key allows the script to identify the RGB and thermal images that
belong together.

### 3. Validate image pairing

The script checks for:

- duplicate pairing keys;
- pairs containing exactly one RGB and one thermal image;
- RGB images without a thermal counterpart;
- thermal images without an RGB counterpart;
- invalid filename formats.

Only complete pairs are retained. A complete pair represents:

```text
one subject
    + one RGB image
    + one matching thermal image
```

The output from this dataset was:

```text
Complete pairs:             2513
RGB without thermal:          43
Thermal without RGB:          43
Duplicate pairing keys:        0
```

The incomplete images are not used for model training or evaluation.

### 4. Add subject metadata

The script reads `metadata.xlsx` and joins the following information to each
subject:

- `subject_id`;
- `gender`;
- `Trial_1`;
- `Trial_2`.

This metadata is saved in the paired CSV where available.

### 5. Verify subject-disjoint splits

The pipeline checks whether a subject occurs in more than one split. The
result for this dataset was:

```text
train: 100 subjects
val:    10 subjects
test:   32 subjects
```

The overlaps were:

```text
test_vs_train: 0
test_vs_val:   0
train_vs_val:  0
```

This is called a **subject-disjoint split**. It prevents data leakage: the
model is not trained on one image of a person and tested on another image of
the same person.

### 6. Read the face landmarks

The landmarks are already supplied by the SFTL54 dataset. This pipeline does
not run a new face detector such as dlib, MediaPipe, or RetinaFace.

For each split, the script reads:

```text
sftl/iron/<split>/labels.jsonl
```

A record can contain facial feature points such as:

- `chin`;
- `left_eyebrow` and `right_eyebrow`;
- `nose_bridge` and `nose_tip`;
- `left_eye` and `right_eye`;
- `lip_p1` through `lip_p6`;
- `face`.

The detailed points describe the locations of facial features in the original
image. A point is represented as `[x, y]`, where `x` is the horizontal pixel
coordinate and `y` is the vertical pixel coordinate.

For cropping, the pipeline uses only the `face` field. It contains two
corners of a face bounding box:

```json
"face": [
  [142.29, 76.43],
  [305.74, 265.25]
]
```

These coordinates mean:

```text
top-left:     x=142.29, y=76.43
bottom-right: x=305.74, y=265.25
```

The coordinates are annotations, not embeddings. An embedding is a learned
vector, such as a tensor with shape `(1, 512)`, produced later by a neural
network.

### 7. Create the face crop

The `make_square_box()` function:

1. Calculates the bounding-box width and height.
2. Finds the bounding-box center.
3. Uses the larger dimension.
4. Adds a 35% margin.
5. Makes the crop square.
6. Clips the crop to the image boundaries.

Conceptually:

```python
size = max(face_width, face_height) * 1.35
```

The margin keeps some surrounding facial context and avoids cutting off
parts of the face.

### 8. Crop both modalities and resize

The same thermal face box is used for the paired RGB and thermal images
because the dataset defines them in the same coordinate system:

```python
rgb_face = crop_face(rgb, face_box)
thermal_face = crop_face(thermal, face_box)
```

Each crop is resized to:

```text
112 x 112 pixels
```

The resulting image formats are:

```text
RGB crop:     112 x 112, RGB
Thermal crop: 112 x 112, grayscale (L)
```

The PyTorch dataset later converts the grayscale thermal image to three
channels because the selected model expects three-channel input.

### 9. Save the processed dataset

The pipeline creates a directory similar to:

```text
sftl54_processed/
├── train/
│   ├── rgb/
│   └── thermal/
├── val/
│   ├── rgb/
│   └── thermal/
├── test/
│   ├── rgb/
│   └── thermal/
└── manifest.csv
```

The manifest contains one row per complete pair:

```text
subject_id
split
pair_key
rgb_path
thermal_path
```

The final processed counts were:

```text
train: 1772 pairs
val:    176 pairs
test:   565 pairs
total: 2513 pairs
```

### 10. Verify the processed output

The script checks that:

- the processed manifest can be read;
- every split contains the expected number of rows;
- RGB paths are present;
- thermal paths are present;
- a processed RGB image can be opened;
- a processed thermal image can be opened;
- the images have the expected size and mode.

It also creates diagnostic files:

```text
sftl54_paired_dataset.csv
sftl54_pair_visualization.png
sftl54_face_crop_test.png
```

The visualization files are for human inspection. They help confirm that
the RGB/thermal images are paired correctly and that the face crop is
reasonable.

## What `sftl54_pytorch_dataset.py` does

The preprocessing pipeline creates the files. The PyTorch dataset loader
opens those files one sample at a time and supplies them to a model in a
standard PyTorch format.

The main class is `SFTL54PairDataset`, which inherits from
`torch.utils.data.Dataset`. It performs the following operations:

1. Reads `sftl54_processed/manifest.csv` with pandas.
2. Optionally selects one split: `train`, `val`, or `test`.
3. Checks that the manifest contains RGB paths, thermal paths, and subject
   IDs.
4. Resolves relative image paths from the project directory.
5. Opens the matching RGB and thermal files.
6. Converts both images to three-channel RGB format.
7. Converts the images to PyTorch tensors.
8. Normalizes the tensor values.
9. Returns the two images together with their identity and pair information.

The dataset does not load every image into memory when it is created.
`__getitem__()` loads the required pair when a particular index is requested.
This makes it possible to work with larger datasets without storing all image
pixels in RAM.

### Selecting a split

The loader can read all manifest rows or only one split:

```python
train_dataset = SFTL54PairDataset(split="train")
val_dataset = SFTL54PairDataset(split="val")
test_dataset = SFTL54PairDataset(split="test")
```

Only these split names are accepted:

```text
train, val, test
```

For this dataset, the split sizes are:

```text
train: 1772 pairs
val:    176 pairs
test:   565 pairs
```

### Loading an individual pair

For a requested index, the loader reads the paths from the selected manifest
row:

```python
rgb_path = self._resolve_path(record["rgb_path"])
thermal_path = self._resolve_path(record["thermal_path"])
```

It then opens both files. The RGB image is loaded as RGB. The thermal image
is first loaded as grayscale (`L`) and then converted to RGB:

```python
rgb_image = Image.open(rgb_path).convert("RGB")
thermal_image = Image.open(thermal_path).convert("L").convert("RGB")
```

The thermal image is still grayscale in content. Converting it to RGB
duplicates its single intensity channel into three channels so it has the
same input shape expected by a three-channel face model.

### Transforming the images into tensors

The default transform is:

```python
transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)
```

`ToTensor()` changes a PIL image from height/width/channel format into the
PyTorch channel/height/width format and scales ordinary 8-bit pixel values
from `[0, 255]` to `[0.0, 1.0]`.

Normalization then applies:

```text
normalized_value = (pixel_value - 0.5) / 0.5
```

Therefore, the usual input range becomes approximately `[-1, 1]`. The
transformation is applied independently to both the RGB and thermal image.

### Returned sample

Each call to the dataset returns a dictionary:

```python
{
    "rgb": rgb_tensor,
    "thermal": thermal_tensor,
    "label": subject_id_tensor,
    "subject_id": subject_id_integer,
    "pair_key": pair_identifier,
}
```

The fields have these purposes:

| Field | Purpose |
|---|---|
| `rgb` | Processed visible-light face tensor |
| `thermal` | Processed thermal face tensor |
| `label` | PyTorch `long` tensor containing the subject identity |
| `subject_id` | Python integer for logging or evaluation |
| `pair_key` | Identifier proving which RGB and thermal files belong together |

The `label` is the identity target for a classification loss. It is not an
embedding. The RGB and thermal tensors are also not embeddings; they are
normalized image inputs. A neural network creates embeddings after receiving
these tensors.

### Creating batches with `DataLoader`

When the script is run directly, it creates a PyTorch `DataLoader`:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)
```

The `DataLoader`:

- groups eight individual samples into one batch;
- stacks the RGB tensors into one tensor;
- stacks the thermal tensors into one tensor;
- stacks the labels into one label tensor;
- shuffles the training order;
- uses the main process for loading because `num_workers=0`.

The script retrieves one batch with:

```python
batch = next(iter(train_loader))
```

It then prints the batch shapes and metadata as a sanity check. This confirms
that the files can be read and that their dimensions are suitable for model
input.

## Relationship with the PyTorch dataset loader

After preprocessing, `sftl54_pytorch_dataset.py` reads:

```text
sftl54_processed/manifest.csv
```

When it is run with `split="train"`, it loads the 1,772 training pairs.
It returns records containing:

```python
{
    "rgb": RGB tensor,
    "thermal": thermal tensor,
    "label": subject ID,
    "subject_id": subject ID,
    "pair_key": pair identifier,
}
```

For a batch size of 8, the tensors have these shapes:

```text
RGB:     (8, 3, 112, 112)
Thermal: (8, 3, 112, 112)
Label:   (8,)
```

## What this pipeline does and does not do

### It does

- find RGB and thermal images;
- match corresponding images;
- remove incomplete pairs;
- check duplicate and invalid keys;
- preserve subject-disjoint splits;
- read precomputed face annotations;
- crop and resize faces;
- save a model-ready manifest and image directory;
- verify the saved files.

### It does not

- train a face-recognition model;
- calculate new facial landmarks;
- calculate face embeddings;
- report recognition accuracy;
- perform RGB-to-thermal feature matching by itself.

The next stage is to connect the PyTorch dataset to a cross-spectral
face-recognition model. That model will consume the processed RGB and thermal
face tensors and generate embeddings for identity classification or
cross-modal similarity comparison.
