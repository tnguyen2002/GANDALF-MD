from PIL import Image
import pandas as pd


# df_path = '../../236_data/ham/HAM/HAM10000_metadata_train.csv'
# df = pd.read_csv(df_path)
# for i, r in df.iterrows():
#     image_id = r['image_id']
#     im = Image.open('../../236_data/ham/HAM/images/' + image_id + '.jpg').resize((128,128))
#     im.save('../../236_data/ham/HAM/images_128/' + image_id + '.jpg')

df_val_path = '../../236_data/ham/HAM/HAM10000_metadata_val.csv'
df_val = pd.read_csv(df_val_path)
for i, r in df_val.iterrows():
    image_id = r['image_id']
    im = Image.open('../../236_data/ham/HAM/val_images/' + image_id + '.jpg').resize((128,128))
    im.save('../../236_data/ham/HAM/val_images_128/' + image_id + '.jpg')