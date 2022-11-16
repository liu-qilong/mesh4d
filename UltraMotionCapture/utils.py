import os
import imageio

def images_to_gif(path=None, remove=False):
    """ convert images in a folder into a gif  """
    files = os.listdir(path)
    files.sort()
    images = []

    for file in files:
        if ('png' in file or 'jpg' in file) and ('gif-' in file):
            images.append(imageio.imread(path + file))
            if remove:
                os.remove(path + file)

    if len(images) == 0:
        print("No images in folder")
    else:
        imageio.mimsave(path + 'output.gif', images)


if __name__ == "__main__":
    images_to_gif()