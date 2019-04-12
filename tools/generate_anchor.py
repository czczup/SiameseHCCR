import os
import pygame
import pandas as pd
from PIL import Image

# table = pd.read_csv("../database/gb2312_level1.csv", encoding="utf-8")
# characters = [item[2] for item in table.values]
# print(characters)
# chinese_dir = 'chinese'
# if not os.path.exists(chinese_dir):
#     os.mkdir(chinese_dir)
#
# pygame.init()
#
# for index, character in enumerate(characters):
#     font = pygame.font.Font("simkai.ttf", 64)
#     rtext = font.render(character, True, (255, 255, 255), (0, 0, 0))
#     print(type(rtext))
#     pygame.image.save(rtext, os.path.join(chinese_dir, character + ".png"))

for file in os.listdir("chinese"):
    image = Image.open("./chinese/"+file)
    image = image.crop((0,0,64,64))
    image.save("./chinese/"+file)
