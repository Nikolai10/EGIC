# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/Nikolai10/SwinT-ChARM/blob/master/config.py
# Copyright 2022 Nikolai Körber.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

class ConfigEGIC:

    # Google Colab
    global_path = "/content/EGIC/src"
    lpips_path = '/content/EGIC/res/models/lpips_weights'
    disc_ckp_path = '/content/EGIC/res/models/oasis_n+1_256x256_coco_weightnorm/ckpt-1000000'
    proto_config = '/content/EGIC/src/deeplab2/configs/custom/resnet50_os32_semseg_coco.textproto'

    # Docker
    #global_path = "/tf/notebooks/EGIC/src"
    #lpips_path = '/tf/notebooks/EGIC/res/models/lpips_weights'
    #disc_ckp_path = '/tf/notebooks/EGIC/res/models/oasis_n+1_256x256_coco_weightnorm/ckpt-1000000'
    #proto_config = '/tf/notebooks/EGIC/src/deeplab2/configs/custom/resnet50_os32_semseg_coco.textproto'


# SwinT-ChARM Config
class ConfigGa:
    embed_dim = [128, 192, 256, 320]
    embed_out_dim = [192, 256, 320, None]
    depths = [2, 2, 6, 2]
    head_dim = [32, 32, 32, 32]
    window_size = [8, 8, 8, 8]
    num_layers = len(depths)


class ConfigHa:
    embed_dim = [192, 192]
    embed_out_dim = [192, None]
    depths = [5, 1]
    head_dim = [32, 32]
    window_size = [4, 4]
    num_layers = len(depths)


class ConfigHs:
    embed_dim = [192, 192]
    embed_out_dim = [192, int(2 * 320)]
    depths = [1, 5]
    head_dim = [32, 32]
    window_size = [4, 4]
    num_layers = len(depths)


class ConfigGs:
    embed_dim = [320, 256, 192, 128]
    embed_out_dim = [256, 192, 128, 3]
    depths = [2, 6, 2, 2]
    head_dim = [32, 32, 32, 32]
    window_size = [8, 8, 8, 8]
    num_layers = len(depths)


class ConfigChARM:
    depths_conv0 = [64, 64, 85, 106, 128, 149, 170, 192, 213, 234]
    depths_conv1 = [32, 32, 42, 53, 64, 74, 85, 96, 106, 117]
