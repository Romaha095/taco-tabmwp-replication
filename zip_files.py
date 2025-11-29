import os
import zipfile

BASE_DIR = "checkpoints"

# ---- 1. ZIP всех подпапок внутри checkpoints ----
subdirs = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
]

print("Найдены папки в checkpoints:", subdirs)

for folder in subdirs:
    folder_path = os.path.join(BASE_DIR, folder)
    zip_path = os.path.join(BASE_DIR, f"{folder}_root_files.zip")

    print(f"\nСоздаю ZIP для: {folder_path}")
    print(f"ZIP файл: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for name in os.listdir(folder_path):
            full_path = os.path.join(folder_path, name)

            if os.path.isfile(full_path):
                zipf.write(full_path, arcname=os.path.join(folder, name))
                print(f"  + добавлен файл: {name}")

# ---- 2. ZIP папки logs в корне проекта ----
LOGS_DIR = "logs"

if os.path.exists(LOGS_DIR) and os.path.isdir(LOGS_DIR):
    logs_zip = "logs.zip"

    print(f"\nСоздаю ZIP для папки logs: {LOGS_DIR}")
    print(f"ZIP файл: {logs_zip}")

    with zipfile.ZipFile(logs_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(LOGS_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                # arcname сохраняет структуру logs/
                arcname = os.path.relpath(full_path, LOGS_DIR)
                zipf.write(full_path, arcname=os.path.join("logs", arcname))
                print(f"  + добавлен файл: {arcname}")

    print("\nZIP logs готов!")
else:
    print("\nПапка logs не найдена, пропускаю zip.")
    
print("\nГотово!")
