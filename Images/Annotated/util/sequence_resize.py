import imgaug as ia
from imgaug import augmenters as iaa


def get():
    def sometimes(aug): return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            iaa.PadToSquare(position='center'),
            # iaa.Fliplr(0.5),  # horizontal flips
            
            # sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2),),
            # sometimes(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            # sometimes(iaa.AddToHueAndSaturation((-20, 20))), # change hue and saturation
            
            sometimes(iaa.AddToBrightness((-30, 30))),
            sometimes(iaa.LinearContrast((0.75, 1.5))), # Strengthen or weaken the contrast in each image.
            
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-15, 15)
            )),
            iaa.Resize({"height": 1024, "width": 1024})
        ],
        random_order=False
    )
