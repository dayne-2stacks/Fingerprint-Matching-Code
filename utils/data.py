from pathlib import Path
import os
import shutil


def train_test_split(root_dir, img_path):
    os.makedirs(root_dir)
    for p in ['train', 'test', 'val']:
        os.mkdir(f'{root_dir}/{p}')
    image_dir = Path(img_path)
    images = list(image_dir.glob('*'))
    for image in images:
        identity = image.stem.split('_')[0]
        identity = int(identity)
        print(identity)
        if (identity % 5) < 2:
            dest = f'{root_dir}/train'
        elif (identity % 5) ==3:
            dest = f'{root_dir}/val'
        else:
            dest = f'{root_dir}/test'
        
        shutil.copy(image, dest)

if __name__ == "__main__":
    import shutil
    shutil.rmtree('hello')
    train_test_split('hello', './dataset/polyU/DBII')
    
