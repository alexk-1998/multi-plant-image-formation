import cv2
import numpy as np
import torch

from json import dump
from models.cut_model import CUTModel
from options import Options
from pathlib import Path
from re import match
from shutil import copy
from time import time


def main():

    start_time = time()

    print('Getting commmand line arguments and writing to file...')
    opt = Options()
    args = opt.parse()
    Path(args.root).mkdir(exist_ok=True)
    opt.write_to_file(args, Path(args.root, 'options.txt').as_posix())

    # create the model
    print('Creating the CUT model...')
    model = CUTModel(args)

    # create folders for file saving
    folders = {
        'backgrounds': Path(args.root, 'backgrounds'),
        'json_files': Path(args.root, 'json_files'),
        'multi_comp': Path(args.root, 'multi_plants_comp'),
        'multi_comp_bbox': Path(args.root, 'multi_plants_comp_bbox'),
        'multi_trans': Path(args.root, 'multi_plants_trans'),
        'multi_trans_bbox': Path(args.root, 'multi_plants_trans_bbox'),
        'singles': Path(args.root, 'singles_lab'),
        'singles_masked': Path(args.root, 'singles_lab_masked'),
        'singles_comp': Path(args.root, 'singles_comp'),
        'singles_trans': Path(args.root, 'singles_trans'),
        'singles_final': Path(args.root, 'singles_final')
    }
    for key, folder in folders.items():
        if not args.no_save_all or key == 'multi_trans' or key == 'json_files':
            folder.mkdir(exist_ok=True)

    # get list of all available backgrounds and single images
    print('Getting single and masked single datasets...')
    backgrounds = list(Path('datasets', 'backgrounds').glob('**/background*'))
    if args.plant_type == 'All':
        singles = sorted(list(Path('datasets', 'singles_lab').glob('**/*.jpg')))
    else:
        singles = sorted(list(Path('datasets', 'singles_lab').glob(f'**/{args.plant_type}*')))

    # get all plant types and position in sorted list
    valid_plants = {}
    for i, path in enumerate(singles):
        try:
            plant = match(r'[a-zA-Z]+', path.name).group(0).lower()
        except AttributeError:
            continue
        if plant not in valid_plants:
            valid_plants[plant] = [i, i]
        else:
            valid_plants[plant][1] = i
    plants = tuple(valid_plants.keys())

    print('Creating multi-plant images...')

    for i in range(args.num_images):

        lap_start = time()

        # get background for forming composite image
        idx = np.random.randint(0, len(backgrounds))
        composite_path = backgrounds[idx]
        composite = cv2.imread(composite_path.as_posix())
        translated = composite.copy()
        

        # copy blank background to folder
        if not args.no_save_all and not (folders['backgrounds']/composite_path.name).exists():
            copy(composite_path.as_posix(),
                 (folders['backgrounds']/composite_path.name).as_posix())

        # create dict for saving to json file
        data = {'bounding_boxes': []}
        if not args.no_save_all:
            data['background'] = composite_path.name

        # choose number of plants or rows in image
        if not args.make_rows:
            num_plants = np.random.randint(args.min_plants, args.max_plants+1)
        else:
            num_rows = np.random.randint(args.min_rows, args.max_rows + 1)
            num_plants = num_rows * args.n_per_row

        for k in range(num_plants):

            # create dict for storing bounding box data
            bbox = {}

            # get random single plant image
            plant_idx = np.random.randint(0, len(plants))
            label = plants[plant_idx]
            idx = np.random.randint(valid_plants[label][0], valid_plants[label][1]+1)
            single_path = singles[idx]

            # get new name for single image
            if not args.no_save_all:
                j = 1
                single_filename = get_filename(single_path.stem, i, j, single_path.suffix)
                while (folders['singles_comp']/single_filename).exists():
                    single_filename = get_filename(single_path.stem, i, j,
                                                   single_path.suffix)
                    j += 1

            # add plant type and filename to bbox
            bbox['label'] = label.capitalize()
            if not args.no_save_all:
                bbox['filename'] = single_filename

            # read single image and mask
            single = cv2.imread(single_path.as_posix())
            mask_path = single_path.as_posix().replace('_lab', '_lab_masked')
            single_masked = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if single_masked is None:
                continue
            _, thresh = cv2.threshold(single_masked, 127, 255, cv2.THRESH_BINARY)

            # copy single and masked single to new folder
            if not args.no_save_all:
                if not Path(folders['singles'], single_path.name).exists():
                    try:
                        copy(single_path.as_posix(),
                             (folders['singles']/single_path.name).as_posix())
                    except FileNotFoundError:
                        pass
                if not Path(folders['singles_masked'], single_path.name).exists():
                    try:
                        copy(mask_path,
                             (folders['singles_masked']/single_path.name).as_posix())
                    except FileNotFoundError:
                        pass

            # colour correct image
            single = colour_correct(single, thresh, args)

            # remove background from single image
            single[thresh == 0] = 0

            # resize single image so it has proper scale relative to background
            scale = np.random.uniform(args.min_scale, args.max_scale)
            scale *= max(*composite.shape) / max(*single.shape)
            single = cv2.resize(single, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)

            if not args.make_rows:
                # random pad image with black so it has size equal to full background
                # make 10 attempts to find non-overlapping single image positions
                orig = single.copy()
                attempts = 0
                while True:
                    overlapping = False
                    single = random_pad(single, composite.shape, bbox, args)
                    for bbox2 in data['bounding_boxes']:
                        if bbox_overlapping(bbox, bbox2, args):
                            overlapping = True
                            attempts += 1
                    if not overlapping or attempts >= 10:
                        break
                    else:
                        single = orig.copy()
                if attempts >= 10:
                    continue
            else:
                set_plant_location(single.shape, composite.shape, k, num_rows, bbox, args)
                single = pad_single(single, composite.shape, bbox)

            # join single and background with new mask
            grey = cv2.cvtColor(single, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(grey, 25, 255, cv2.THRESH_BINARY)
            composite[thresh == 255] = single[thresh == 255]

            # create new single image for passing to network
            bbox['x_min2'] = bbox['x_min'] - args.plant_pad // 2
            bbox['x_max2'] = bbox['x_max'] + args.plant_pad // 2
            bbox['y_min2'] = bbox['y_min'] - args.plant_pad // 2
            bbox['y_max2'] = bbox['y_max'] + args.plant_pad // 2
            if bbox['x_min2'] < 0:
                bbox['x_min2'] = 0
            if bbox['x_max2'] > composite.shape[1]:
                bbox['x_max2'] = composite.shape[1]
            if bbox['y_min2'] < 0:
                bbox['y_min2'] = 0
            if bbox['y_max2'] > composite.shape[0]:
                bbox['y_max2'] = composite.shape[0]
            single = composite[bbox['y_min2']:bbox['y_max2'], bbox['x_min2']:bbox['x_max2']]
            if not args.no_save_all:
                cv2.imwrite((folders['singles_comp']/bbox['filename']).as_posix(), single)

            # transform single for model input
            single = cv2.resize(single, (256, 256), interpolation=cv2.INTER_CUBIC)
            single = ((single / 255.0) - 0.5) / 0.5  # normalize to [-1, 1]
            single = single[:, :, ::-1]  # BGR to RGB
            single = torch.from_numpy(single.copy()).float()
            single = single.permute(2, 0, 1).unsqueeze(0) # [H,W,C] to [1,C,H,W]
            if len(args.gpu_ids) > 0:
                single = single.to(torch.device(f'cuda:{args.gpu_ids[0]}'))

            # transform image and save result
            with torch.no_grad():
                result = model.netG(single)
            result = result[0].permute(1, 2, 0).clamp(-1.0, 1.0).cpu().numpy()
            result = result[:, :, ::-1]
            result = (result + 1.0) * 127.5
            result = result.astype(np.uint8)
            result = cv2.resize(result, (bbox['x_max2']-bbox['x_min2'], bbox['y_max2']-bbox['y_min2']), interpolation=cv2.INTER_CUBIC)
            if not args.no_save_all:
                cv2.imwrite((folders['singles_trans']/bbox['filename']).as_posix(),
                            result)

            # get original composite image + border
            single_comp = composite[bbox['y_min2']:bbox['y_max2'], bbox['x_min2']:bbox['x_max2']].copy()

            # get image of border with bbox contents set to white for masking
            translated[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = 255
            removed = translated[bbox['y_min2']:bbox['y_max2'], bbox['x_min2']:bbox['x_max2']]
            grey = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(grey, 254, 255, cv2.THRESH_BINARY)

            # compute means and adjust transformed image
            comp_mean = single_comp[thresh == 0].mean(axis=(0))
            res_mean = result[thresh == 0].mean(axis=(0))
            result = result + comp_mean - res_mean
            result = np.clip(result, 0.0, 255.0)
            result = result.astype(np.uint8)
            if not args.no_save_all:
                cv2.imwrite((folders['singles_final']/bbox['filename']).as_posix(),
                            result)

            # add transformed image to background copy
            if args.replace_all:
                translated[bbox['y_min2']:bbox['y_max2'], bbox['x_min2']:bbox['x_max2']] = result
            else:
                img_x_min = bbox['x_min'] - bbox['x_min2']
                img_x_max = img_x_min + bbox['x_max'] - bbox['x_min']
                img_y_min = bbox['y_min'] - bbox['y_min2']
                img_y_max = img_y_min + bbox['y_max'] - bbox['y_min']
                translated[bbox['y_min']:bbox['y_max'], bbox['x_min']:bbox['x_max']] = result[img_y_min:img_y_max, img_x_min:img_x_max]

            # add new bounding box data
            data['bounding_boxes'].append(bbox)

        # write translated multi-plant image to disc
        filename = 'image' + str(i).zfill(3) + composite_path.suffix
        cv2.imwrite((folders['multi_trans']/filename).as_posix(), translated)

        # write original image and bounding box images to disc
        if not args.no_save_all:
            cv2.imwrite((folders['multi_comp']/filename).as_posix(), composite)
            # write bounding box images to disc
            for bbox in data['bounding_boxes']:
                cv2.rectangle(composite, (bbox['x_min'], bbox['y_min']),
                              (bbox['x_max'], bbox['y_max']), (0, 255, 0), 2)
                cv2.rectangle(translated, (bbox['x_min'], bbox['y_min']),
                              (bbox['x_max'], bbox['y_max']), (0, 255, 0), 2)
            cv2.imwrite((folders['multi_comp_bbox']/filename).as_posix(), composite)
            cv2.imwrite((folders['multi_trans_bbox']/filename).as_posix(),
                        translated)

        # write json file to disc
        json_filename = filename.replace(composite_path.suffix, '.json')
        with open((folders['json_files']/json_filename).as_posix(), 'w') as f:
            dump(data, f)

        elapsed = round(time() - lap_start, 2)
        print(f'Image "{filename}" created in {elapsed} s')

    elapsed = round(time() - start_time, 2)
    average = round(elapsed / args.num_images, 2)
    print('Image creation complete')
    print(f'Total time elapsed: {elapsed} s, {average} s/image')
    
    
def set_plant_location(image_size, final_size, i, n_rows, bbox, args):
    """
    Get location of plant when aligning into rows

    Parameters:
        image_size (tuple): size of plant image being placed
        final_size (tuple): size multi-plant image being constructed
        i (int): index of plant in image
        n_rows (int): number of rows in multi-plant image
        bbox (dict): for setting the plant location
        args (argparse.Namespace): for additional parameters
    """
    n_row = i // args.n_per_row
    n_col = i % args.n_per_row
    x_c = int(final_size[1] * (n_row + 1) / (n_rows + 1))
    x_shift_max = int(0.1 * image_size[1])
    x_c += np.random.randint(-x_shift_max, x_shift_max + 1)
    bbox['x_min'] = x_c - image_size[1]//2
    bbox['x_max'] = bbox['x_min'] + image_size[1]
    y_c = int(final_size[0] * (n_col + 1) / (args.n_per_row + 1))
    y_shift_max = int(0.1 * image_size[0])
    y_c += np.random.randint(-y_shift_max, y_shift_max + 1)
    bbox['y_min'] = y_c - image_size[0]//2
    bbox['y_max'] = bbox['y_min'] + image_size[0]


def pad_single(image, size, bbox):
    """
    Pad image so it has the correct final size

    Parameters:
        image (np.ndarray): image to be padded
        size (tuple): size that image is padded to
        bbox (dict): for passing location of the plant in the padded image
    """
    pad_top = bbox['y_min']
    pad_bottom = size[0] - image.shape[0] - pad_top
    pad_left = bbox['x_min']
    pad_right = size[1] - image.shape[1] - pad_left
    pad = (pad_top, pad_bottom, pad_left, pad_right)
    #print(bbox, size, image.shape)
    return cv2.copyMakeBorder(image, *pad, cv2.BORDER_CONSTANT)


def random_pad(image, size, bbox, args):
    """
    Pad image so it has the correct final size
    Padding is random to 'shift' the image on the background

    Parameters:
        image (np.ndarray): image to be padded
        size (tuple): size that image is padded to
        bbox (dict): for storing location of the plant in the padded image
        args (argparse.Namespace): for passing the border padding value
    """
    pad_top = np.random.randint(args.border_pad, size[0]-image.shape[0]-args.border_pad+1)
    pad_bottom = size[0] - image.shape[0] - pad_top
    pad_left = np.random.randint(args.border_pad, size[1]-image.shape[1]-args.border_pad+1)
    pad_right = size[1] - image.shape[1] - pad_left
    pad = (pad_top, pad_bottom, pad_left, pad_right)
    bbox['x_min'] = pad_left
    bbox['x_max'] = pad_left + image.shape[1]
    bbox['y_min'] = pad_top
    bbox['y_max'] = pad_top + image.shape[0]
    return cv2.copyMakeBorder(image, *pad, cv2.BORDER_CONSTANT)


def bbox_overlapping(bbox1, bbox2, args):
    """
    Return a boolean indicating whether two bounding boxes are overlapping

    Parameters:
        bbox1 (dict): first bounding box
        bbox2 (dict): second bounding box
        args (argparse.Namespace): for passing the plant padding value
    """
    if bbox1['x_min'] > bbox2['x_max'] + args.plant_pad or \
        bbox1['x_max'] < bbox2['x_min'] - args.plant_pad or \
        bbox1['y_min'] > bbox2['y_max'] + args.plant_pad or \
        bbox1['y_max'] < bbox2['y_min'] - args.plant_pad:
        return False
    return True


def get_filename(stem, i, j, suffix):
    """
    For formatting single filenames in the format "Soybean202007021339353_00i_j.jpg"

    Parameters:
        stem (str): filename, not including extension, eg. Soybean202007021339353
        i (int): counter indicating which multi-plant image the single belongs to
        j (int): counter if another image is saved with the same stem
        suffix (str): the file extension
    """
    return f'{stem}_{str(i).zfill(3)}_{j}{suffix}'


def colour_correct(image, mask, args):
    """
    Correct the colour of a single plant image to match the field data
    Colour correction only occurs where the mask is true
    Colour correction is done by shifting the mean values of a CIELAB image to desired values

    Parameters:
        image (np.ndarray): image to be colour corrected
        mask (np.ndarray): mask that indicates which pixels belong to the plant
        args (argparse.Namespace): for passing the desired means of the LAB channels
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    mean = np.array([args.l_avg, args.a_avg, args.b_avg]) - lab[mask != 0].mean(axis=(0))
    lab[mask != 0] = lab[mask != 0] + mean
    lab = np.clip(lab, 0.0, 255.0)
    lab = lab.astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


if __name__ == '__main__':
    main()
