# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import glob
import os

import numpy as np
# import debug_init_paths
import torch
import torchvision.transforms as T
from PIL import Image
import h5py
import pandas as pd
import io
from torch.utils.data import Dataset, DataLoader

from utils.data_utils import imagenet_preprocess, Resize

def tell(list, array):
    id = True
    for k in list:
        if np.all(k==array):
            id = False
            break
    return id


def raw_to_agent_particles(particles, Vmat):
    '''
    Map raw particle coordinates to agent-relative coordinates

    #### INPUTS ####
    raw_particles: [B,T,N,3] tensor of N particle (x,y,z) world coordinates
    Vmat: [4,4] matrix that transforms raw to agent-relative homogeneous coordinates

    #### RETURNS ####
    agent_relative_particles: [B,T,N,3] agent-relative coordinates of particles
    '''
    shape = list(np.shape(particles))
    if len(shape) != 4:
        particles = np.reshape(particles, (1,1,-1,3))
    shape = list(np.shape(particles))
    Vmat = np.zeros([shape[0],shape[1],4,4], np.float32) + np.reshape(Vmat, [1,1,4,4])
    raw_particles = np.concatenate([particles, np.ones(shape[0:3]+[1], dtype=np.float32)],
                              axis=3) # [B,T,N,4]
    raw_particles = np.transpose(raw_particles, (0,1,3,2)) # [B,T,4,N]
    agent_relative_particles = np.matmul(Vmat, raw_particles)
    agent_relative_particles = np.transpose(agent_relative_particles, (0,1,3,2))
    agent_relative_particles = agent_relative_particles[:,:,:,0:3] # remove homogeneous coordinate

    return agent_relative_particles


def agent_particles_to_image_coordinates(particles, Pmat=None, H_out=256, W_out=256, to_integers=False, eps=1.0e-8):
    '''
    Map agent-relative particles in 3-space to image coordinates in 2-space
    via a perspective projection, encoded in Pmat

    #### INPUTS ####
    ps: [B,T,N,3] agent-relative particle (x,y,z) coordinates
    Pmat: [4,4] matrix that performs perspective projection on homogeneous coordinates
    H_out, W_out: the dimensions of the image to project onto
    to_integers: if True, returned coordinates are typed as integers and can serve as indices

    #### RETURNS ####
    particle_image_coordinates: [B,T,N,2] coordinates in 2-space between [0,H_out] x [0,W_out]
                                By convention the first coordinate indexes the vertical dimension.
    '''
    ### modified from tf into numpy arrays
    shape = list(np.shape(particles))
    if len(shape) != 4:
        particles = np.reshape(particles, (1,1,-1,3))
    shape = list(np.shape(particles))
    if len(list(np.shape(Pmat))) != 4:
        Pmat = np.zeros([shape[0], shape[1], 4, 4], dtype=np.float32) + np.reshape(Pmat, (1,1,4,4))
    ps = np.concatenate([particles, np.ones(shape[0:3]+[1], dtype=np.float32)], axis=3)
    ps = np.transpose(ps, (0,1,3,2))
    ps = np.matmul(Pmat, ps)
    ps = np.transpose(ps, (0,1,3,2))
    # normalize by homogeneous coordinate, then remove it
    ps = np.divide(ps, ps[:,:,:,3:4]+eps)[:,:,:,0:3]

    # map to image coordinates; y axis is inverted by convention
    xims = float(W_out) * 0.5 * (ps[:,:,:,0:1] + 1.0)
    yims = float(H_out) * (1.0 - 0.5*(ps[:,:,:,1:2] + 1.0))

    if to_integers:
        xims = np.floor(xims).astype(dtype=int)
        yims = np.floor(yims).astype(dtype=int)

        xims = np.minimum(np.maximum(xims, 0), W_out-1)
        yims = np.minimum(np.maximum(yims, 0), H_out-1)

    particle_image_coordinates = np.concatenate([yims, xims], axis=3)
    return particle_image_coordinates

def project_and_occlude_particles(particles,
                                  projection_matrix,
                                  im_size=[256,256],
                                  particles_mask=None,
                                  p_radius=0.0,
                                  xyz_dims=[0,3],
                                  particles_agent_centered=False,
                                  camera_matrix=None,
                                  **kwargs
):
    '''
    particles: [B,T,N,D] where the last axis contains both xyz_dims and color_dims (slices)
    im_size: [H,W] ints
    projection_matrix: [B,T,4,4] the matrix for projecting from euclidean space into canonical image coordinates
    xyz_dims: [xdim, zdim+1] must have a difference of 3

    p_radius: float that determines how particles will occlude one another.
    particles_agent_centered: if false, use camera_matrix ("V") to put particles in camera-relative coordinates
    camera_matrix: [B,T,4,4] the matrix "V" for converting global to camera-relative coordinates

    returns
    particles_im_indices: [B,T,N,2] of int32 indices into H and W dimensions of an image of size H,W
    not_occluded_mask: [B,T,N,1] float32 where 1.0 indicates the particle wasn't occluded at radius p_radius, 0.0 otherwise
    '''
    shape = list(np.shape(particles))
    if len(shape) != 4:
        particles = np.reshape(particles, (1,1,-1,3))
    B,T,N,D = list(np.shape(particles))
    H,W = im_size
    if p_radius is None:
        p_radius = 3.0 * (np.minimum(H,W).astype(float) / 256.0)
        print("p radius", p_radius)
    # get indices into image
    particles_xyz = particles[...,xyz_dims[0]:xyz_dims[1]]
    if not particles_agent_centered:
        assert camera_matrix is not None, "need a camera matrix to put into agent coordinates"
        particles_agent = raw_to_agent_particles(particles_xyz, Vmat=camera_matrix)
    else:
        particles_agent = particles_xyz

    # [B,T,N,2] <tf.float32> values in range [0,H]x[0,W]
    particles_depths = particles_agent[...,-1:]
    particles_im_coordinates = agent_particles_to_image_coordinates(particles_agent,
                                                                    Pmat=projection_matrix,
                                                                    H_out=H, W_out=W,
                                                                    to_integers=False)


    # clip coordinates
    #particles_im_coordinates = np.where(np.logical_or(np.isnan(particles_im_coordinates), np.isinf(particles_im_coordinates)),
                                        #coord_max * np.ones_like(particles_im_coordinates), # true
                                        #particles_im_coordinates # false
    #)
    #particles_im_coordinates = np.maximum(np.minimum(particles_im_coordinates, coord_max), -coord_max)

    ## particles_depths = tf.Print(particles_depths, [particles_depths.shape[2], tf.reduce_max(particles_depths), tf.reduce_min(particles_im_coordinates), tf.reduce_max(particles_im_coordinates)], message='p_depth/coords')

    # resolve occlusions
    #not_occluded_mask = occlude_particles_in_camera_volume(particles_im_coordinates,
                                                          # particles_depths,
                                                           #p_radius=p_radius,
                                                           #particles_mask=particles_mask
                                                           # particles_mask=None
   # )
    #particles_im_indices = np.round(particles_im_coordinates).astype(dtype=np.int32)
    #particles_im_indices = np.minimum(np.maximum(particles_im_indices, 0),
                                      #np.reshape(np.constant(np.array([H-1,W-1]), dtype=np.int32), [1,1,1,2]))

    return particles_im_coordinates
           #not_occluded_mask, \
           #particles_im_coordinates # last is float


class ShapeStacks(Dataset):
    def __init__(self, list_path, radius, mod, sequence_len,
                 normalize_images=True, max_samples=None, training=False):
        super().__init__()

        self.RW = self.RH = self.W = self.H = 256
        self.orig_W = self.orig_H = 256
        self.box_rad = radius
        self.list_path = list_path  ###note now that this is a path that points to the list of all hdf5 files
        #self.image_dir = image_dir ###note now that this is a list of paths
        self.ext = '.hdf5'
        self.max_samples = max_samples
        #self.dt = dt  #each trial has differing time lengths
        self.num_obj = 0
        self.training = training
        self.modality = mod
        self.sequence_len = sequence_len
        #self.sources = ['images', 'depths', 'normals', 'objects']


        transform = [Resize((self.H, self.W)), T.ToTensor()]
        obj_transform = [Resize((self.RH, self.RW)), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
            obj_transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.obj_transform = T.Compose(obj_transform)

        #with open(list_path) as fp:
            #self.index_list = [line.split()[0] for line in fp]
        #self.roidb = self.parse_gt_roidb()
        eg_path = glob.glob(os.path.join(self.list_path, '[0-9][0-9][0-9][0-9]' + self.ext)) #a list of hdf5 training files
        self.index_list = [int(line.split('\\')[-1].split('.')[0]) for line in eg_path]
        self.eg_path = eg_path
        #self.image_pref = '-'.join(os.path.basename(eg_path).split('-')[0:-1])

    def get_dt(self, index):
        '''
        :param index:
        :return: get number of images contained in index
        '''
        gt_path = get_index_after(self, index)
        h5 = h5py.File(gt_path, 'r')

        return len(h5['frames'].keys())

    def get_matrices(self, index):
        gt_path = get_index_after(self, index)
        h5 = h5py.File(gt_path, 'r')
        proj_matrix = np.asarray(h5['static']['projection_matrix'][:])
        cam_matrix = np.asarray(h5['static']['camera_matrix'][:])

        return proj_matrix, cam_matrix

    def get_images(self, index, dt):
        gt_path = get_index_after(self, index)
        h5 = h5py.File(gt_path, 'r')
        str = '%04d' % dt
        images = Image.open(io.BytesIO(h5['frames'][str]['images']['_img'][:]))
        images = np.asarray(images)
        ids = Image.open(io.BytesIO(h5['frames'][str]['images']['_id'][:]))
        ids = np.asarray(ids)
        return images, ids

    def get_all_colors(self, index):
        '''
        :param index:
        :return: number of colors used to label the objects in ids
        '''
        t = int(self.get_dt(index))
        k = self.get_num_obj(index)
        col = []
        while (int(len(col)) != int(k)):
            t = int(t/2)
            if t==0:
                print('mismatch between color numbers and object numbers')
            else:
                _, ids = self.get_images(index, t)
                b = ids.copy()
                for i in range(256):
                    for j in range(256):
                        if (np.all(b[i, j, :] == 0) == False) and tell(col, b[i, j, :]):
                            col.append(b[i, j, :])
        return col

    def get_positions(self, index, dt):
        '''
        :param index: points to the desired hdf5 file
        :param dt: points to the location in the sequence of images
        :return: get positions of objects contained in the image
        '''
        gt_path = get_index_after(self, index)
        h5 = h5py.File(gt_path, 'r')
        str = '%04d' % dt
        positions = np.asarray(h5['frames'][str]['objects']['positions'][:])
        return positions

    def get_projected_positions(self, index, dt, get_int = False):
        '''
        :param index: same as above
        :param dt: same as above
        :param get_int: whether to return a integer-values output
        :return:
        '''
        pos = self.get_positions(index, dt)
        proj, cam = self.get_matrices(index)
        pos = project_and_occlude_particles(pos,
                                      projection_matrix=proj,
                                      im_size=[256, 256],
                                      particles_mask=None,
                                      p_radius=0.0,
                                      xyz_dims=[0, 3],
                                      particles_agent_centered=False,
                                      camera_matrix=cam)
        if get_int:
            pos = np.round(pos)

        return pos

    def get_segments(self, index, ids, pos):
        '''
        index: as usual
        :param ids: numpy array giving ids as images consisting of N colors, where each N corresponds to an object
        :param pos: numpy array position of each object's part
        :return: get segments of individual objects and concatenate them based on the sequence given in pos of size (O, C, RH, RW)
        '''

        # get all colors and their crops
        colors = self.get_all_colors(index)
        crops = []
        W, H = self.W, self.H
        for i in range(len(colors)):
            img = np.zeros([W,H])
            for j in range(W):
                for k in range(H):
                    if np.all(ids[j,k,:] == colors[i]):
                        img[j, k, :] = 1
                    else:
                        img[j, k, :] = 0
            crops.append(img)

       # then we need to do something more.


        return segments


    def get_num_obj(self, index): # return the number of objects
        gt_path = get_index_after(self, index)
        h5 = h5py.File(gt_path, 'r')

        return np.shape(np.asarray(h5['frames']['0000']['objects']['positions'][:]))[0]

    def __len__(self):
        num = len(self.index_list)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        return num


    def __getitem__(self, index): #build_one_dataset
        """
        :return: src, dst. each is a list of object
        - 'image': FloatTensor of shape (dt, C, H, W). resize and normalize to faster-rcnn
        - 'crop': (O, C, RH, RW) RH >= 224
        - 'bbox': (O, 4) in xyxy (0-1) / xy logw logh
        - 'trip': (T, 3)
        - 'index': (dt,)
        """
        # the following is based on tfrecords data
        #vid_point = []
        #for dt in range(self.dt):
        #this_index = self.list_path[index]
        #curr_data_path, _, extra_tensors = this_index ###curr_data_path has subfiles images, depths, normals, objects etc.
        # the following is based on the tfrecords version of data
        #source_paths = {source: os.path.join(curr_data_path, source, 'trial', '-%04d'% index, '.tfrecords' ) for source in self.sources}

        ###read tfrecords data

        #file_datasets = {source: tf.data.Dataset.list_files(curr_files, shuffle=False) for source, curr_files in
                         #source_lists.items()}

        #if self.is_training or self.shuffle_val:
            # Shuffle file names using the same shuffle_seed
            #file_datasets = {
                #source: curr_dataset.shuffle(
                    #buffer_size=len(source_lists.values()[0]),
                    #seed=self.shuffle_seed).repeat() \
                #for source, curr_dataset in file_datasets.items()}

        #each_dataset = {
           # source: curr_dataset.apply(
                #tf.contrib.data.parallel_interleave(
                   # _fetch_dataset,
                    #cycle_length=1,
                    #sloppy=False)) \
                #for source,curr_dataset in file_datasets.items()
        #}

        # the following is based on the hdf5 hierarchy
        vid_point = []
        #curr_data_path = os.path.join(self.list_path, '%04d' % index + self.ext)
        for i in range(self.sequence_len):
            vid_obj = {}
            images,ids = self.get_images(index, i)
            images = torch.from_numpy(images)
            images = self.transform(images)
            vid['image'] = images
            pos = self.get_positions(index,i)
            crop = self.get_segments(index,ids,pos)
            crop = torch.from_numpy(crop)
            vid_obj['crop'] = crop
            trip = self._build_graph(index)

            vid_obj['bbox'] = torch.FloatTensor(norm_bbox)   ##get bboxes, must end with 4 -> decoder
            vid_obj['trip'] = trip
            valid = np.arange(3, dtype=np.int64)
            vid_obj['info'] = (self.orig_W, self.orig_H, valid)
            vid_obj['valid'] = torch.LongTensor(valid)





            norm_bbox = self.roidb[self.index_list[index]][dt] # (O, 2)
            bboxes = np.vstack((norm_bbox[:, 0] * self.orig_W, norm_bbox[:, 1] * self.orig_H)).T


        vid_point.append(vid_obj)
        return vid_point

    def get_index_after(self, index):
        return os.path.join(self.list_path, '%04d' % index + self.ext)

    def _crop_image(self, index, image, box_center):
        crop_obj = []
        x1 = box_center[:, 0] - self.box_rad
        y1 = box_center[:, 1] - self.box_rad
        x2 = box_center[:, 0] + self.box_rad
        y2 = box_center[:, 1] + self.box_rad
        bbox = np.vstack((x1, y1, x2, y2)).transpose()
        for d in range(len(box_center)):
            crp = image.crop(bbox[d]).convert('RGB')
            crp = self.transform(crp)
            crop_obj.append(crp)
        crop_obj = torch.stack(crop_obj)
        return crop_obj

    def _build_graph(self, index):
        num = self.get_num_obj(index)
        all_trip = np.zeros([0, num], dtype=np.float32)
        for i in range(num):
            for j in range(num):
                trip = [i, 0, j]
                all_trip = np.vstack((all_trip, trip))
        return torch.FloatTensor(all_trip)


def dt_collate_fn(batch):
    """
    :return: src dst. each is a list with dict element
    - 'index': list of str with length N
    - 'image': list of FloatTensor in shape (Dt, V, 1, C, H, W)
    - 'crop': list of FloatTensor in shape (Dt, V, o, C, RH, RW)
    - 'bbox': (Dt, V, o, 4)
    - 'trip': (Dt, V, t, 3)
    """
    key_set = ['index', 'image', 'crop', 'bbox', 'trip', 'valid']
    all_batch = {}
    dt = len(batch[0])
    V = len(batch)
    for key in key_set:
        all_batch[key] = []

    for f in range(dt):
        for v in range(len(batch)):
            frame = batch[v][f]
            for key in key_set:
                all_batch[key].append(frame[key])

    for key in all_batch:
        if key == 'index':
            continue
        if key in ['image', 'crop']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, 3, tensor.size(-2), tensor.size(-1))
        elif key in ['bbox', 'trip', 'valid']:
            tensor = torch.stack(all_batch[key])
            all_batch[key] = tensor.view(dt, V, -1, tensor.size(-1))
        else:
            print('key not exist', key)
            raise KeyError

    return all_batch


def build_vid_loaders(args):
    '''args need: n_tr_per_dataset, n_val_per_dataset, dataset_names
    image_dir needs filling in
    '''
    dset_kwargs = {
        'max_samples': None,
        'sequence_len': args.sequence_len,
        #'radius': args.radius,
        'training': args.is_train,
        'mod': 'rgb' # args.modality
    }
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'collate_fn': dt_collate_fn,
    }
    list_paths = args.list_paths
    #ns_train_examples = [args.n_tr_per_dataset] * len(dataset_names)
    #ns_val_examples = [args.n_val_per_dataset] * len(dataset_names)


    if args.is_train:
        data_path = os.path.join(list_paths, 'new_tfdata')
        dset_kwargs['list_path'] = data_path
        loader_kwargs['shuffle'] = True
    else:
        print('val')
        data_path = os.path.join(list_paths, 'new_tfvaldata')
        dset_kwargs['list_path'] = data_path
        dset_kwargs['max_samples'] = args.num_val_samples
        loader_kwargs['shuffle'] = args.shuffle_val
    dset_kwargs['training'] = args.is_train
    #list = dset_kwargs['list_path']
    #img_list = []
    #for i in range(len(list)):
        #img_list.append(os.path.join(list[i], 'images'))


    #dset_kwargs['image_dir'] = img_list
    train_dset = ShapeStacks(**dset_kwargs)

    ##want to also return boxnumber here
    bbox_num = train_dset.get_bbox_nums()

    loader = DataLoader(train_dset, **loader_kwargs)

    return loader, bbox_num


if __name__ == '__main__':
    # from  import parser_helper
    # from cfgs.train_cfgs import TrainOptions
    from cfgs.test_cfgs import TestOptions
    args = TestOptions().parse()
    torch.manual_seed(123)
    # args.batch_size=4
    args.dt=16
    train_loader = build_vid_loaders(args)
    key_set = ['index', 'image', 'crop', 'bbox', 'trip']

    for batch in train_loader:
        for key in key_set:
            if key == 'index':
                print(batch[key])
            else:
                pass
                print('size', key, batch[key].size())
        break
