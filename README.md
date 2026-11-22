# Traffic Sign Visibility Audit

An individual Python project using the labeled training split of the German Traffic Sign Recognition Benchmark (GTSRB). It measures image properties and applies a transparent rule to prioritize images for human review. It does not train a classifier. A review flag is a screening aid, not a verdict that a sign is unsafe or that its reference label is wrong.

## Requirements

Use Python 3.10–3.12 and an internet connection to install packages. Commands below are run from the project root. The `requirements.txt` file specifies compatible package ranges for NumPy, pandas, Matplotlib, OpenCV and Jupyter Notebook.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell's local execution policy blocks activation, use the environment's Python directly in place of `python`, for example `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.\.venv\Scripts\python.exe src/audit.py ...`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Obtain the data

The full project package includes the original training images and annotation files in `data/GTSRB/Final_Training/Images/`. If that folder is present, skip the download steps below. Otherwise:

1. Visit the [official GTSRB benchmark page](https://benchmark.ini.rub.de/gtsrb_dataset.html) and follow its GTSRB download link to the [official archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).
2. Download **GTSRB_Final_Training_Images.zip**, the official training images and annotations, and extract it locally. Keep the original image and CSV files unchanged.
3. Place the extracted `GTSRB` directory under this project's `data/` directory so the structure is:

```text
data/
└── GTSRB/
    └── Final_Training/
        └── Images/
            ├── 00000/
            │   ├── GT-00000.csv
            │   └── *.ppm
            ├── 00001/
            └── ...
```

The complete training annotation files are needed even when the expensive image measurements use only selected classes. The benchmark's per-class annotation files are semicolon-separated. A [Zenodo dataset record](https://zenodo.org/records/13741936) is also listed in the assignment; its archive may use a different layout, so use the official training archive above for this script. The included image directory is not excluded by `.gitignore`; verify the dataset's applicable terms before publishing or redistributing it.

## Run the audit

From the project root with the virtual environment activated, start Jupyter Notebook:

```bash
python -m notebook
```

Open `notebooks/traffic_sign_audit.ipynb` and run its cells in order (for example, use **Run All**). The notebook loads the included dataset, executes the audit and displays the summaries and figures. It saves the derived tables and figures under `outputs/`.

Alternatively, run the same analysis directly as a script:

```bash
python src/audit.py --data-dir data/GTSRB/Final_Training/Images --output-dir outputs --classes 0 1 14 17 25 --sample-per-class 500 --seed 42
```

These are the default settings. Class IDs may be changed, but select at least five for the assignment. `--sample-per-class` specifies the maximum number of image records sampled per selected class; the seed makes that selection reproducible. Annotation counts use all available annotation rows and are distinct from measurements of the sampled images. To see all available options, run `python src/audit.py --help`.

To run the small validation checks:

```bash
python src/audit.py --validate-only
```

The validation includes an image under an Arabic path. If OpenCV cannot read a Unicode filename directly, the audit opens the file with Python and decodes it with OpenCV, including when creating the contact sheet.

The script writes derived tables and diagnostics under `outputs/processed/` and figures under `outputs/figures/`. It does not modify raw data. Rerunning with the same inputs, environment and arguments reproduces the analysis; outputs in the chosen output directory can be replaced. Inspect the diagnostic summary for missing paths, invalid or unreadable images, and other exclusions. Use a different output directory to preserve an earlier run.

The included `report/Traffic_Sign_Visibility_Audit_Report.pdf` documents the supplied run on the official training data.  The provided results were run with Python 3.12, NumPy 2.5.5, pandas 3.0.1, Matplotlib 3.11.2 and OpenCV 5.1.0. The tests used the headless OpenCV build, which provides the same `cv2` image-processing API used here.

## Interpretation

The audit uses supplied class IDs as reference labels. Brightness is mean grayscale intensity, contrast is grayscale intensity standard deviation, and sharpness is the variance of the grayscale Laplacian response. Sharpness is a proxy for image detail, not a direct measurement of human readability. A project-specific percentile rule and alternative cutoffs illustrate how review workload and class rankings can change. No cutoff is presented as an official road-safety standard, and relationships between measurements do not establish causation.

GTSRB consists of small image crops and multiple images can come from the same physical sign. Images and classes are not independent samples of every road environment. Consequently, sampled flag rates describe this analysis, not the prevalence of unsafe signs in a local road network. Review examples manually and consider local, labeled images before transferring any decision rule.

## Dataset citation

German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum. [Benchmark information](https://benchmark.ini.rub.de/gtsrb_dataset.html); [official training archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).

Stallkamp, J., Schlipsing, M., Salmen, J., and Igel, C. (2011). *The German Traffic Sign Recognition Benchmark: A Multi-Class Classification Competition*. Proceedings of the IEEE International Joint Conference on Neural Networks, 1453–1460.
