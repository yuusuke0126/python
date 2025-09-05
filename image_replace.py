from PIL import Image
import os

# 画像ファイルのパス
path_v4 = os.path.expanduser("~/maps/default/map_v4.pgm")
path_final = os.path.expanduser("~/Downloads/map_20250613_final.pgm")

# 画像を読み込み
img_v4 = Image.open(path_v4)
img_final = Image.open(path_final)

# 矩形領域の座標
top_left = (2640, 2212)
bottom_right = (2767, 2239)

# 切り取り範囲を取得
crop_box = (*top_left, *bottom_right)
region_from_final = img_final.crop(crop_box)

# map_v4.pgm に貼り付け
img_v4.paste(region_from_final, top_left)

# 保存
output_path = os.path.expanduser("~/maps/default/map_v5.pgm")
img_v4.save(output_path)

output_path
