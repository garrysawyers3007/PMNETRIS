class config_USC_pmnetV3_V2:
    def __init__(self,):
        # basics
        self.batch_size = 8
        self.exp_name = 'augmented_config_USC_pmnetV3_V2'
        self.num_epochs = 100
        self.val_freq = 1
        self.num_workers = 0

        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.dataset_settings()
        self.optim_settings()
        return

    def dataset_settings(self,):
        self.dataset = 'USC'
        self.cityMap = 'complete'        # complete, height
        self.sampling = 'exclusive' # random, exclusive



    def optim_settings(self,):
        self.lr = 1e-4
        self.lr_decay = 0.45
        self.step = 10

    def get_train_parameters(self,):
      return {'exp_name':self.exp_name,
        'batch_size':self.batch_size,
        'num_epochs':self.num_epochs,
        'lr':self.lr,
        'lr_decay':self.lr_decay,
        'step':self.step,
        'sampling':self.sampling}

class config_USC_RISMapNet_V1:
    def __init__(self):
        # ------------------
        # Basics
        # ------------------
        self.batch_size = 6          # FiLM + GN → stable even with small batch
        self.exp_name = 'USC_RISMapNet_RXcentered_V1'
        self.num_epochs = 150        # small dataset + smooth fields → train longer
        self.val_freq = 1
        self.num_workers = 0

        # ------------------
        # Dataset split
        # ------------------
        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.dataset_settings()
        self.optim_settings()
        self.regularization_settings()
        return

    # ------------------
    # Dataset
    # ------------------
    def dataset_settings(self):
        self.dataset = 'USC'
        self.cityMap = 'complete'        # buildings + heights
        self.sampling = 'exclusive'     # RX-centered → avoid leakage

        # Task-specific
        self.output_grid = (10, 10)
        self.rx_centered = True
        self.input_channels = ['building', 'TX', 'RX']
        self.conditioning = ['RIS_x', 'RIS_y', 'RIS_orientation']

    # ------------------
    # Optimizer & LR
    # ------------------
    def optim_settings(self):
        self.optimizer = 'AdamW'
        self.lr = 5e-5                  # lower than PMNet (FiLM is sensitive)
        self.weight_decay = 1e-4

        # Step LR works well for radio maps
        self.lr_decay = 0.5
        self.step = 25                  # slower decay than PMNet

    # ------------------
    # Regularization
    # ------------------
    def regularization_settings(self):
        self.dropout = 0.05             # light, but important for 6k samples
        self.grad_clip = 1.0            # stabilizes FiLM training
        self.use_groupnorm = True

        # Optional but recommended
        self.label_smoothing = 0.0      # usually not needed for regression
        self.use_gradient_loss = True   # encourages smooth coverage maps
        self.gradient_loss_weight = 0.1

    # ------------------
    # Trainer interface
    # ------------------
    def get_train_parameters(self):
        return {
            'exp_name': self.exp_name,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'lr': self.lr,
            'lr_decay': self.lr_decay,
            'step': self.step,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay,
            'sampling': self.sampling,
            'rx_centered': self.rx_centered,
        }
    

class config_USC_pmnetRIS_V1:
    def __init__(self):

        self.batch_size = 16
        self.exp_name = 'augmented_config_USC_pmnetRIS_V1'

        self.num_epochs = 100
        self.val_freq = 1
        self.num_workers = 4

        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.dataset_settings()
        self.optim_settings()

    def dataset_settings(self):
        self.dataset = 'USC'
        self.cityMap = 'complete'
        self.sampling = 'exclusive'

    def optim_settings(self):
        self.lr = 3e-4
        self.lr_decay = 0.5
        self.step = 25
        self.weight_decay = 1e-4

    def get_train_parameters(self):
        return {
            'exp_name': self.exp_name,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'lr': self.lr,
            'lr_decay': self.lr_decay,
            'weight_decay': self.weight_decay,
            'step': self.step,
            'sampling': self.sampling
        }

class config_USC_pmnetCrop_FT:
    def __init__(self):
        self.batch_size = 16
        self.exp_name = 'pmnet_film_crop_ft'
        self.num_epochs = 100
        self.val_freq = 1
        self.num_workers = 0

        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.dataset_settings()
        self.optim_settings()

    def dataset_settings(self):
        self.dataset = 'USC'
        self.cityMap = 'complete'
        self.sampling = 'exclusive'

    def optim_settings(self):
        self.base_lr = 1e-5
        self.film_lr = 5e-5
        self.lr_decay = 0.5
        self.step = 15
        self.weight_decay = 1e-4

    def get_train_parameters(self):
        return {
            'exp_name': self.exp_name,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'base_lr': self.base_lr,
            'film_lr': self.film_lr,
            'lr_decay': self.lr_decay,
            'weight_decay': self.weight_decay,
            'step': self.step,
            'sampling': self.sampling
        }
    
class config_USC_pmnetGeom_FT:
    def __init__(self):
        self.batch_size = 16
        self.exp_name = 'pmnet_geom_v4_ft'
        self.num_epochs = 100
        self.val_freq = 1
        self.num_workers = 0

        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.dataset_settings()
        self.optim_settings()

    def dataset_settings(self):
        self.dataset = 'USC'
        self.cityMap = 'complete'
        self.sampling = 'exclusive'

    def optim_settings(self):
        self.base_lr = 5e-5
        self.film_lr = 1e-4   # reuse as "new branch lr" for now
        self.lr_decay = 0.5
        self.step = 15
        self.weight_decay = 1e-4

    def get_train_parameters(self):
        return {
            'exp_name': self.exp_name,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'base_lr': self.base_lr,
            'film_lr': self.film_lr,
            'lr_decay': self.lr_decay,
            'weight_decay': self.weight_decay,
            'step': self.step,
            'sampling': self.sampling
        }