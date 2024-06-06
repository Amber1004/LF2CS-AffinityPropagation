import os
import sys
sys.path.append("./lf2cs")
from lf2cs.net.C4Net import EncoderC4
from lf2cs.tool.util_tools import Tools
from lf2cs.data.my_dataset import MyDataset
from lf2cs.tool.runner_tool import RunnerTool
from lf2cs.data.my_transforms import MyTransforms


class Config(object):

    def __init__(self, gpu_id=1, dataset_name=MyDataset.dataset_name_omniglot):
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.dataset_name = dataset_name
        self.is_conv_4 = True

        ###############################################################################################
        self.num_workers = 16
        self.batch_size = 64
        self.num_way = 5
        self.num_shot = 1
        self.val_freq = 5
        self.episode_size = 15
        self.test_episode = 1000
        self.knn_k = 20
        self.fsl_out_dim = 1024
        # self.lf2cs_out_dim = 1024
        self.learning_rate = 0.01
        self.train_epoch = 250
        self.first_epoch, self.t_epoch = 100, 50
        self.adjust_learning_rate = RunnerTool.adjust_learning_rate1
         ###############################################################################################

        ###############################################################################################
        self.data_root = MyDataset.get_data_root(dataset_name=self.dataset_name)
        self.transform_train_lf2cs, self.transform_train_fsl, self.transform_test = MyTransforms.get_transform(
            dataset_name=self.dataset_name, has_lf2cs=True, is_fsl_simple=True, is_css=False)

        self.encoders, self.net_name = EncoderC4(), "Conv4"
        ###############################################################################################

        self.model_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.gpu_id, self.net_name, self.train_epoch, self.batch_size,
            self.num_way, self.num_shot, self.first_epoch, self.t_epoch)

        self.time = Tools.get_format_time()
        _root_path = "./models/{}/our".format(self.dataset_name)
        self.net_dir = "{}/{}_{}.pkl".format(_root_path, self.time, self.model_name)
        self.log_file = Tools.new_dir(self.net_dir.replace(".pkl", ".txt"))
        pass

    pass

    def print_config(self, log_file=None):
        for attribute, value in self.__dict__.items():
            Tools.print("{}: {}".format(attribute, value), txt_path=log_file)
        pass