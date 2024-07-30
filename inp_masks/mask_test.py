# from PIL import Image, ImageDraw, ImageFont
# # 加载图片
# # img = Image.open("path/to/your/image.jpg")  # 替换为你的图片路径
#
# from PIL import Image
# import numpy as np
#
# # 创建一个随机的 128x128 图片
# img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
#
# # 显示图片
# img.show()
#
# # 创建遮盖矩阵
# mask = Image.new("L", img.size, 255)  # 创建一个白色背景的图像作为遮盖层
# draw = ImageDraw.Draw(mask)
#
# # 字体设置
# font_path = '/home/shendi_mcj/fonts/times.ttf'  # 替换为你的字体文件路径
# # Set font
# # fpath = '/home/shendi_mcj/fonts/times.ttf
#
# font_size = 50
# font = ImageFont.truetype(font_path, font_size)
#
# # 在遮盖层上绘制文本
# text = "EAGE"
# draw.text((50, 50), text, font=font, fill=0)  # 在指定位置绘制文本
#
# # 应用遮盖
# result = Image.composite(img, Image.new("RGB", img.size, "white"), mask)
#
# # 显示或保存结果
# result.show()  # 显示结果图像
# # result.save("path/to/save/result_image.jpg")  # 保存结果图像，替换为保存路径


from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 创建一个灰度图像（单通道）
width, height = 128, 128
img = Image.new('L', (width, height), color=255)  # 创建一个灰度图像，初始颜色为白色

# 添加文字到图像上
text = "EAGE\n2024"  # 要添加的文字

# 指定文字尺寸
text_width, text_height = 128, 128

# 使用默认字体和默认字体大小
font_size = 10
# font_path = "arial.ttf"  # 替换为你自己的字体文件路径
# font = ImageFont.load_default()
fpath = '/home/shendi_mcj/fonts/times.ttf'
while True:
    # 逐步增加字体大小
    font = ImageFont.truetype(fpath, font_size)
    draw = ImageDraw.Draw(img)
    text_size = draw.textsize(text, font=font)

    # 检查文字尺寸是否达到指定尺寸
    if text_size[0] >= text_width or text_size[1] >= text_height:
        break

    font_size += 1

# 创建一个与原始图像尺寸相同的空白图像
text_img = Image.new('L', (width, height), color=255)

# 计算文字的位置
text_position = ((width - text_width) // 2, (height - text_height) // 2)

# 在空白图像上添加文字
draw = ImageDraw.Draw(text_img)
draw.text(text_position, text, font=font, fill=0)  # 将文字填充为黑色

# 显示或保存图像
text_img.show()  # 显示图像

# 转换为文字遮罩矩阵
text_mask = np.array(text_img)
# 保存为 .npy 文件
np.save('text_mask_2.npy', text_mask)
print("文字遮罩矩阵：")
print(text_mask)