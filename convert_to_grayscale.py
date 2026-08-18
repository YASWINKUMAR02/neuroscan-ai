import os
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "dataset", "guardrail_dataset", "random_images")
TARGET_DIR = os.path.join(BASE_DIR, "dataset", "guardrail_dataset", "random_images_gray")

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    print(f"Found {len(files)} images to convert.")

    converted_count = 0
    for file_name in tqdm(files, desc="Converting to Grayscale"):
        src_path = os.path.join(SOURCE_DIR, file_name)
        target_path = os.path.join(TARGET_DIR, file_name)
        
        try:
            with Image.open(src_path) as img:
                # Convert image to grayscale ("L" mode)
                gray_img = img.convert("L")
                # Save the grayscale image
                gray_img.save(target_path)
                converted_count += 1
        except Exception as e:
            print(f"\nFailed to process {file_name}: {e}")

    print(f"Successfully converted {converted_count} of {len(files)} images to grayscale.")
    print(f"Saved to: {TARGET_DIR}")

if __name__ == "__main__":
    main()
