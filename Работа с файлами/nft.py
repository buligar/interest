from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
# Пустой желтый фон.
im = Image.new('RGB', (10, 10), (0, 0, 0))
draw = ImageDraw.Draw(im)
for i in range(100):
    for a in range(10):
        for b in a:
            draw.point(xy=(b,a), fill='white')
            # im.save(f"C:\\Users\\bulig\\Desktop\\NFT\\nft{b:04}.bmp")
            plt.imsave(f"C:\\Users\\bulig\\Desktop\\NFT\\nft{i:04}.jpg", im)
# draw.point(xy=(i,1), fill='white')
# im.save(f"C:\\Users\\bulig\\Desktop\\NFT\\nft{i:04}.bmp")
