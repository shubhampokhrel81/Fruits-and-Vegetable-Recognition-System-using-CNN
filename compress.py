import os
from datetime import datetime
from PIL import Image

for image_file_name in os.listdir('potato/'):
    if image_file_name.endswith(".jpg"):
        now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')

        foo = Image.open('potato/'+image_file_name)
        foo = foo.resize((100, 100), Image.ANTIALIAS)
        foo.save('resized/potato/' + now + '.jpg', optimize=True, quality=100)
