from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)


s='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# for j in  s:
for i in range(0, 401):
# Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/"+"space"+"/sp"+"_" + str(i) + '.png')


