from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops, types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import os
import math


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=True):
        super(HybridTrainPipe, self).__init__(batch_size,
                                              num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(
            file_root=data_dir, random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(
                device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(
                device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512)

        self.rrc = ops.RandomResizedCrop(device=dali_device, size=(crop, crop))
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, dali_cpu=True):
        super(HybridValPipe, self).__init__(batch_size,
                                            num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(
            file_root=data_dir, random_shuffle=False)
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(
                device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            self.decode = ops.nvJPEGDecoder(
                device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device=dali_device, resize_shorter=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu())
        return [output, self.labels]


class ImageNetIterator(DALIClassificationIterator):

    def __len__(self):
        return math.ceil(self._size / self.batch_size)

    def __next__(self):
        data = super().__next__()
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        return input, target

    def next(self):
        return super().__next__()


def get_loaders(dataroot, val_batch_size, train_batch_size, input_size, workers):
    train_pipe = HybridTrainPipe(batch_size=train_batch_size, num_threads=workers,
                                 device_id=0, data_dir=os.path.join(dataroot, 'train'), crop=224)
    val_pipe = HybridValPipe(batch_size=val_batch_size, num_threads=workers,
                             device_id=0, data_dir=os.path.join(dataroot, 'val'), crop=224, size=256)

    train_pipe.build()
    val_pipe.build()

    train_loader = ImageNetIterator(
        train_pipe, train_pipe.epoch_size("Reader"))
    val_loader = ImageNetIterator(
        val_pipe, val_pipe.epoch_size("Reader"))
    return train_loader, val_loader
