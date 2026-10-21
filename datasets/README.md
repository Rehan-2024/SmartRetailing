# Supplied dataset archive

All six original CSV files are stored here at the user's request. Their original bytes, encodings and line endings are preserved. This directory is outside `public/` and is not copied into the Vercel website.

Only `digital/online_shoppers_intention.csv` and `Wholesale customers data.csv` are verified UCI model inputs (CC BY 4.0). The four campaign, store and geographic files are archived for access, **excluded from all model training and analytical results**, and have unverified real-world provenance and licenses. Their inclusion is not certification that they are real, open-licensed datasets. No license is asserted for those four files.

Credits: Sakar, C. & Kastro, Y. (2018), [Online Shoppers Purchasing Intention](https://doi.org/10.24432/C5F88Q); Cardoso, M. (2013), [Wholesale Customers](https://doi.org/10.24432/C5030X). The manifest records SHA-256, file sizes and verification status.

All seven originals, including the main Excel workbook, are available in the connected user's [SmartRetailData Drive folder](https://drive.google.com/drive/folders/1cp5PzhIxOnkND4waySrPrOR3vw9fC-L7). The folder is not made public. Mount that same Google account in Colab:

```python
from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')
DATA_ROOT = Path('/content/drive/MyDrive/SmartRetailData')
OUTPUT_ROOT = Path('/content/drive/MyDrive/SmartRetailResults')
```

Expected layout:

```text
SmartRetailData/
  transactions/online_retail_II.xlsx
  digital/online_shoppers_intention.csv
  Wholesale customers data.csv
  marketing_campaign.csv
  store/store_sales_data (2).csv
  geo/geolocations-indian-cities.csv
  geo/Pincode_Dataset.csv
```

The workbook is stored in Drive rather than GitHub. Its public original is [Chen, D. (2012), Online Retail II, UCI, CC BY 4.0](https://doi.org/10.24432/C5CG6D). The notebook requires only the three verified datasets for training, regardless of how many other files are archived in the folder.
