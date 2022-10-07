
class Model_Config:
    def __init__(self):
        self.use_gpu = True
        self.dirpath_train = "/media/gaurav/DATA/labs/ESPCN/dataset/train"
        self.dirpath_val = "/media/gaurav/DATA/labs/ESPCN/dataset/val"
        self.scaling_factor = 3
        self.log_interval = 10
        self.patch_size = 17
        self.stride = 13
        self.epochs = 200
        self.learning_rate = 1e-3
        self.seed = 100
        self.dirpath_out = "./Model_output/"


        self.fpath_weights ="/media/gaurav/DATA/labs/ESPCN/Model_output/epoch_2.pth"
        self.fpath_image = "/media/gaurav/DATA/labs/ESPCN/dataset/test/3063.jpg"
        self.results_output = "./results"
