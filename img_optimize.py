from pathlib import Path

from ultralytics.data.utils import compress_one_image
from ultralytics.utils.downloads import zip_directory

# 图片优化并打包zip

# Define dataset directory
path = Path("datasets")

# Optimize images in dataset (optional)
for f in path.rglob("*.jpg"):
    compress_one_image(f)

# Zip dataset into 'path/to/dataset.zip'
zip_directory(path)
