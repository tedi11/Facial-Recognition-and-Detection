import os

class Parameters:
    def __init__(self):
        self.use_flip_images = True
        self.base_dir = '../data'
        self.antrenare = 'antrenare'
        self.dir_pos_examples = os.path.join(self.antrenare, 'dad')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_test_examples = 'testare/testare'
        self.path_annotations = 'validare/task1_gt_validare.txt'
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 64  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 5831  # numarul exemplelor pozitive
        self.number_negative_examples = 20000  # numarul exemplelor negative
        self.has_annotations = True
        self.threshold = 0
