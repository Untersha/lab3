from PIL import Image
import numpy as np

classes_imgs = [
    ['data/0/{}.png'.format(i + 1) for i in range(5)],
    ['data/1/{}.png'.format(i + 1) for i in range(4)],
    ['data/2/{}.png'.format(i + 1) for i in range(4)],
    ['data/3/{}.png'.format(i + 1) for i in range(4)],
]

samples_img = ['data/samples/{}.png'.format(i) for i in range(4)] + [
    'data/samples/5.png',
    'data/samples/8.png',
    'data/samples/man.png'
]


def toimg(arr, num):
    ln = 10
    a = np.repeat(arr.astype('uint8').reshape((ln, ln, 1)), repeats=3, axis=2)
    img = Image.fromarray(a, 'RGB')
    img.save('kernel/ker{}.png'.format(num))
    img.show()


def load_classes():
    classes_RGBA = [[np.asarray(Image.open(x), dtype='uint8').reshape((100, 4)) for x in i] for i in classes_imgs]
    return [((np.sum(cl, axis=2) - 255) // 3) for cl in classes_RGBA]


def load_samples():
    samples_RGBA = [np.asarray(Image.open(x), dtype='uint8').reshape((100, 4)) for x in samples_img]
    return [(np.sum(cl, axis=1) - 255) // 3 for cl in samples_RGBA]
