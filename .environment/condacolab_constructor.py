# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="-aPD4P-Qn_3j"
# # Build a `constructor` installer on Colab
#
# You can pack a `conda` installation along with your usual packages to reduce install times! This tutorial will help you create one directly on Colab.
#
# First, we need to download the metadata files from the `condacolab` example.
#

# %% id="BCWj-KcmXeBK"
# !wget -q https://raw.githubusercontent.com/restlessronin/lila-reports/main/.environment/construct.yaml
# !wget -q https://raw.githubusercontent.com/restlessronin/lila-reports/main/.environment/pip-dependencies.sh

# %% [markdown] id="RlIMzQABoone"
# Now, open the sidebar and explore your files. You should see two files:
#
# - `construct.yaml`: the main metadata file. Add your `conda` packages list in the `specs` field. If you want, you can also customize the metadata (name and version), but it's not required.
# - `pip-dependencies.sh`: a helper script to add pip packages to your install. Ideally you don't need to use this because all your packages are coming from `conda`. This should only be a last resort. If you do use pip dependencies, remember to comment out the `post_install` line in `construct.yaml`.
#
# If you double click on the the files, you will be able to edit them directly on Colab. Save with <kbd>Ctrl</kbd>+<kbd>S</kbd> or <kbd>Cmd</kbd>+<kbd>S</kbd>. Once you are ready, you can continue with the rest of the (now automatic) process.

# %% id="YiKIYmleV7Uf"
# !pip install -q condacolab
import condacolab
condacolab.install()

# %% [markdown] id="gQz3rypUtEKZ"
# After the kernel restart, run the following cell to build the installer.

# %% id="-RDpyAYfXk9L"
# !mamba install -q constructor
# !constructor .

# %% [markdown] id="v-CmIYpbqB_7"
# When you are done, you can download the generated `.sh` installer with this cell below. Alternatively, use the sidebar menu to download it.

# %% id="e8sUzk5tXmPD"
from google.colab import files
# installer = !ls *-Linux-x86_64.sh
files.download(installer[0])
