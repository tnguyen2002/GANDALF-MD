import os
import shutil
import click
from PIL import Image

@click.command()
@click.option('--source_dir1', type=click.Path(exists=True))
@click.option('--source_dir2', type=click.Path(exists=True))
@click.option('--target_dir', type=click.Path())
@click.option('--prefix', default='image', help='Prefix for the renamed files')
@click.option('--keep_name', default=False)
def combine_and_rename_directories(source_dir1, source_dir2, target_dir, prefix, keep_name):
    """Combine files from two source directories into a target directory and rename them."""
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    i = 1
    for source_dir in [source_dir1, source_dir2]:
        for filename in os.listdir(source_dir):
            file_extension = os.path.splitext(filename)[1]
            if(file_extension == ".jpg" or file_extension == ".png"):
                source_file = os.path.join(source_dir, filename)
                print("Adding and converting image: " + str(i))
                if os.path.isfile(source_file):
                    # Downscale image
                    with Image.open(source_file) as img:
                        img = img.resize((32, 32))
                        if(keep_name):
                            new_filename = filename
                        else:
                            new_filename = f"{prefix}_{i}.jpg"  # Saving as PNG
                        img.save(os.path.join(target_dir, new_filename))
                    i += 1

if __name__ == '__main__':
    combine_and_rename_directories()
