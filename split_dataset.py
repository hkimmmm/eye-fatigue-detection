import os
import shutil
from sklearn.model_selection import train_test_split

# Path ke folder dataset mentah
raw_data_dir = "data/raw/"
output_dirs = {
    "train": "data/train/",
    "validation": "data/validation/",
    "test": "data/test/"
}

# Membuat folder output jika belum ada
for key, path in output_dirs.items():
    os.makedirs(os.path.join(path, "open"), exist_ok=True)
    os.makedirs(os.path.join(path, "closed"), exist_ok=True)

# Fungsi untuk membagi dataset dan memindahkan file
def split_and_move(category, train_dir, val_dir, test_dir):
    # Path ke kategori (open/closed)
    category_path = os.path.join(raw_data_dir, category)
    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
    
    # Membagi dataset: 70% train, 15% validation, 15% test
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    # Pindahkan file ke folder masing-masing
    for file_name in train_files:
        shutil.copy(os.path.join(category_path, file_name), os.path.join(train_dir, category, file_name))
    for file_name in val_files:
        shutil.copy(os.path.join(category_path, file_name), os.path.join(val_dir, category, file_name))
    for file_name in test_files:
        shutil.copy(os.path.join(category_path, file_name), os.path.join(test_dir, category, file_name))

# Membagi kategori "open" dan "closed"
split_and_move("open", output_dirs["train"], output_dirs["validation"], output_dirs["test"])
split_and_move("closed", output_dirs["train"], output_dirs["validation"], output_dirs["test"])

print("Dataset successfully split into train, validation, and test sets!")
