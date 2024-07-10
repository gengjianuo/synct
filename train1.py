import time
import torch
import torch.nn as nn
from options.train_options import TrainOptions
from models import create_model
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    NormalizeIntensityd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    LoadImaged,
    RandSpatialCropd,
    RandAdjustContrastd,
    CropForegroundd,
    RandZoomd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandBiasFieldd,
    RandShiftIntensityd
)
from util.visualizer import Visualizer
from glob import glob
import numpy as np
import os
import glob
import nibabel as nib
from PIL import Image
def validate(): #num_patients=20):

    model.eval()
    val_loss_avg = CumulativeAverager()

    # add_to_log("Performing validation test on %d samples"%len(val_dataset))
    #
    # if args.iters is not None:
    #     val_data_loader.dataset.segmentation_pairs = val_data_loader.dataset.segmentation_pairs[:args.iters]

    with torch.no_grad():

        for i, data in enumerate(val_loader):
            # value1 = data["A"]
            # # print(value1.shape)
            # value2 = data["B"]
            # # print(value2.shape)
            # inputs = [value1, value2]
            # input_shapes = []
            # inputs1 = []
            # for input in inputs:
            #     # input = sample_3d_data(input, 5)
            #     input = torch.tensor(input)
            #     input_shape = list(input.shape)
            #     input_shapes.append(input_shape)
            #     inputs1.append(input)
            # output_list = process_tensor_list_by_8(input_shapes, inputs1)
            # for tensor in inputs1:
            #     print(tensor.shape)
            # data['A'] = output_list[0]
            # data['B'] = output_list[1]
            # value1, x = sample_3d_data(value1, 5)
            # value2, x = sample_3d_data(value2, 5)
            # for i in range(x):
            #     value1[i] = torch.tensor(value1[i])
            #     value2[i] = torch.tensor(value2[i])
            #     indata = {'A': value1[i], 'B': value2[i]}
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            val_loss = model.val_forward()
            val_loss_avg.update(val_loss)
        val_loss = val_loss_avg.get_average()
    #log_str = add_to_log('Validation Loss=%0.6f'%(val_loss))
    return val_loss
def convert_nifti_to_image(nifti_file, output_file):
    # 加载NIfTI图像
    nifti_img = nib.load(nifti_file)
    data = nifti_img.get_fdata()

    # 将数据缩放到0-255的范围，并转换为无符号整型
    data_scaled = (data - data.min()) / (data.max() - data.min()) * 255
    data_scaled = data_scaled.astype('uint8')

    # 创建图像对象
    image = Image.fromarray(data_scaled)

    # 保存为图像文件
    image.save(output_file)

# 示例用法
import os
import gzip
def process_tensor_list_size(tensor_list, input):
    processed_list = []
    i = 0
    while i <= 1:
        #print(input[i])
        if tensor_list[i][2] - 160 >= 0:
            processed_tensor = input[i]
        else:
            processed_tensor = torch.cat((input[i], torch.zeros(tensor_list[i][0], tensor_list[i][1], 160 - (tensor_list[i][2] ), tensor_list[i][3])), dim=2)
        if tensor_list[i][3] -160 >= 0:
            pass
        else:
            processed_tensor = torch.cat((processed_tensor, torch.zeros(tensor_list[i][0], tensor_list[i][1], list(processed_tensor.shape)[2], 160 - (tensor_list[i][3]))), dim=3)
        processed_list.append(processed_tensor)
        i += 1
    return processed_list
def decompress_nifti_gz_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
            os.makedirs(output_folder) # 确保输出文件夹存在
    for filename in os.listdir(input_folder):
        if filename.endswith('.nii.gz'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename[:-3])
            decompress_nifti_gz(input_file, output_file)
def decompress_nifti_gz(nifti_gz_file, output_nifti_file):
    with gzip.open(nifti_gz_file, 'rb') as gz_file:
        data = gz_file.read()
    with open(output_nifti_file, 'wb') as nifti_file:
        nifti_file.write(data)
def process_tensor_list_by_8(tensor_list, input):
    processed_list = []
    i = 0
    while i <= 1:
        #print(input[i])
        if tensor_list[i][2] % 8 == 0:
            processed_tensor = input[i]
        else:
            processed_tensor = torch.cat((input[i], torch.zeros(tensor_list[i][0], tensor_list[i][1], 8 - (tensor_list[i][2] % 8), tensor_list[i][3],tensor_list[i][4])), dim=2)
        if tensor_list[i][3] % 8 == 0:
            pass
        else:
            processed_tensor = torch.cat((processed_tensor, torch.zeros(tensor_list[i][0], tensor_list[i][1], list(processed_tensor.shape)[2], 8 - (tensor_list[i][3] % 8),tensor_list[i][4])), dim=3)
        processed_list.append(processed_tensor)
        i += 1
    return processed_list
# def swap_dimensions(data):
#     data["A"] = data["A"].permute(0, 3, 1, 2)
#     data["B"] = data["B"].permute(0, 3, 1, 2)
#     return data
def sample_3d_data(data,slice_interval):
    batch, depth, height, width = data.shape
    num_slices = depth //slice_interval
    data = data.numpy()
    sample_data = np.zeros((batch, slice_interval, height, width), dtype=data.dtype)
    for i in range(5):
        sample_data[:, i, :, :] = data[:, i*num_slices, :, :]
    return sample_data
class CumulativeAverager:

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.hist = []

    def update(self, newval):
        self.count += 1
        self.sum += newval
        self.hist.append(newval)

    def get_average(self):
        return self.sum / float(self.count)

    def get_full_history(self):
        return self.hist
if __name__ == '__main__':
    # 指定文件夹路径

    # mr_folder_path = 'E:/zfc/shujuji/176 32 pixel+GAN 2001'  # 替换为实际的MR文件夹路径
    # ct_folder_path = 'E:/zfc/shujuji/zzc mr'  # 替换为实际的CT文件夹路径
    # mr_input_folder = mr_folder_path
    # ct_input_folder = ct_folder_path
    # mr_output_folder = 'E:/zfc/shujuji/176 32 pixel+GAN 2001_nii'
    # ct_output_folder = 'E:/zfc/shujuji/zzc ct_nii'
    # decompress_nifti_gz_folder(mr_input_folder, mr_output_folder)
    # decompress_nifti_gz_folder(ct_input_folder, ct_output_folder)

    mr_paths = sorted(glob.glob('/home/u202231903026/all/cbct_nii' + '/*.nii'))
    ct_paths = sorted(glob.glob('/home/u202231903026/all/ct_nii' + '/*.nii'))
    # mask_paths = sorted(glob.glob('/media/wuyi/D6DA272ADA2705F9/gjn/SynCT_TcMRgFUS-main/src/datasets/ct_nii' + '/*.nii'))
    data_dicts = [{
        "A": mr_path,
        "B": ct_path,
        'A_paths': mr_path,
        'B_paths': ct_path
    } for mr_path, ct_path in zip(mr_paths, ct_paths, )]
    #  data_dicts.append(data_dict)
    train_files = data_dicts[-135:]

    val_mr_paths = sorted(glob.glob('/home/u202231903026/all/cbctval_nii' + '/*.nii'))
    val_ct_paths = sorted(glob.glob('/home/u202231903026/all/ctval_nii' + '/*.nii'))
    val_data_dicts = [{
        "A": mr_path,
        "B": ct_path,
        'A_paths': mr_path,
        'B_paths': ct_path
    } for mr_path, ct_path in zip(val_mr_paths, val_ct_paths, )]
    val_files = val_data_dicts[-30:]

    # 进行数据转换
    trainTransform = Compose([
        LoadImaged(keys=["A", "B"]),
        AddChanneld(keys=["A", "B"]),
        # 其他的转换操作...
        # MRI pre-processing
        NormalizeIntensityd(keys="A", nonzero=True),  # z-score normalization
        ScaleIntensityRangePercentilesd(keys="A", lower=0.00, upper=99.5, b_min=-1.0, b_max=1.0, clip=True,
                                        relative=False),  # normalize the intensity to [-1, 1]

        # CT pre-processing
        ScaleIntensityRanged(keys=["B"], a_min=-1024, a_max=3071, b_min=-1.0, b_max=1.0, clip=True),

        # Spatial augmentation
        # RandAffined(keys=["A", "B"],
        #             prob=0.2,
        #             rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
        #             mode=('bilinear', 'bilinear')
        #             ),

        # crop 256 x 256 x 32 volume
        RandSpatialCropd(keys=["A", "B"], roi_size=(160, 160, 32), random_size=False),
        # randomly crop patches of 256 x 256 x 32

        # Intensity augmentation
        RandShiftIntensityd(keys="A", offsets=(-0.1, 0.1), prob=0.2),
        RandAdjustContrastd(keys="A", prob=0.2, gamma=(0.8, 1.2)),

        ToTensord(keys=["A", "B"])])

    train_ds = Dataset(data=train_files, transform=trainTransform) ##mr_output_folder
    train_loader = DataLoader(train_ds,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0)
    val_ds = Dataset(data=val_files, transform=trainTransform)  ##mr_output_folder
    val_loader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0)
    opt = TrainOptions().parse()  # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_loader)  # get the number of images in the dataset.
    # print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    print(model.netG)
    print(model.netD)
    best_epoch = 0
    best_metric = float('inf')
    best_metric1 = float('inf')
    is_best = torch.tensor(float(np.inf))
    vloss = torch.tensor([1.0e100])
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        train_lossG = CumulativeAverager()
        train_lossD = CumulativeAverager()
        # train_lossD_B = CumulativeAverager()
        print("Dataloader Length:", len(train_loader))
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # value1 = data["A"]
            # # print(value1.shape)
            # value2 = data["B"]
            # # print(value2.shape)
            # inputs = [value1, value2]
            # input_shapes = []
            # inputs1=[]
            # for input in inputs:
            #     # input = sample_3d_data(input, 5)
            #     input = torch.tensor(input)
            #     input_shape = list(input.shape)
            #     input_shapes.append(input_shape)
            #     inputs1.append(input)
            # output_list = process_tensor_list_by_8(input_shapes, inputs1)
            # for tensor in inputs1:
            #     print(tensor.shape)
            # data['A'] = output_list[0]
            # data['B'] = output_list[1]

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(train_lossG, train_lossD) # calculate loss functions, get gradients, update network weights



            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        D_loss = train_lossD.get_average()
        G_loss = train_lossG.get_average()
        loss = os.path.join(opt.checkpoints_dir, opt.name, 'A_loss.txt')
        message = '(epoch: %d) ' % (epoch)
        message += '%s: %.3f ' % ('G', G_loss)
        message += '%s: %.3f ' % ('D', D_loss)
        metric1 = G_loss
        if metric1 < best_metric1:
            best_metric1 = metric1
            best_epoch = epoch
            model.save_networks('trainbest')
            with open(os.path.join(opt.checkpoints_dir, opt.name, 'train_epoch.txt'), "a") as file:
                file.write(f'{best_epoch}\n')
        val_loss = validate().cpu()
        model.update_learning_rate()
        metric = val_loss
        message += '%s: %.3f ' % ('val', val_loss)
        with open(loss, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            is_best = epoch
            model.save_networks('valbest')
            with open(os.path.join(opt.checkpoints_dir, opt.name, 'val_epoch.txt'), "a") as file:
                file.write(f'{is_best}\n')



        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

