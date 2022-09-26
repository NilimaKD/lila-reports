# Reporting Module and Notebooks for AVC LiLa

## JupyText

Currently we are using jupytext to persist the notebooks as `.py` files rather than as the JSON `.ipynb` files. This was done to enable usable git interop.

## Mount Google Drive on Colab

1. Go to "Shared with me" on your google drive
2. Right Click on the shared folder name (in this case "LiLa_Nagapattinam), select "Add shortcut to Drive". Select "My Drive" and click "Add Shortcut".
3. The file will now be visible in the "drive" subfolder of the data pane.
   
## Copy Data from GDrive to Kaggle

1. Download root dir from GDrive. The directory will be zipped and downloaded.
2. Create a Kaggle Dataset by uploading the zip file. Kaggle will automatically unzip the file and re-create the GDrive directory structure.
3. Share the dataset with all collaborators.
4. Attach the dataset to every notebook that needs it.