This is a mathematical expression recognition (MER) system which uses improved SSD300.

---- How to use -------------
In this system, there are two main files: train.py and masstest.py

------------------------------------
-- train.py ------------------------
------------------------------------
This file is responsible for training the network.
This file accept following parameter:

  -h, --help            show this help message and exit
  --lr LR               learning rate
  --momentum MOMENTUM   momentum
  --decay DECAY         decay
  --use_cuda USE_CUDA   Use CUDA for training
  --epoch_count EPOCH_COUNT
                        Number of training epoch
  --batch_size BATCH_SIZE
                        Batch size
  --resume_mode RESUME_MODE
                        Continue training mode: 
                        'none': From nothing,
                        'pretrain': From pretrain model, 
                        'continue': Continue from SSD Model
  --using_python_2 USING_PYTHON_2
                        Current python version
  --class_count CLASS_COUNT
                        Number of classes
  --network NETWORK     network type: 'SSD300': use original SSD300, 'SSD500':
                        Improved version
  --resuming_model RESUMING_MODEL
                        Model to load (Only valid for resume_mode: pretrain
                        and continue)
  --train_dir TRAIN_DIR
                        training set directory
  --train_meta TRAIN_META
                        training set metafile location
  --validate_dir VALIDATE_DIR
                        validation set directory
  --validate_meta VALIDATE_META
                        validateion set metafile location
  --output_directory OUTPUT_DIRECTORY
                        Output model directory
  --output_format OUTPUT_FORMAT
                        Format of output model's name, this file must contain
                        symbol %d for indexing purpose [For example:
                        ckpt_%d.pth]
  --epoch_cycle EPOCH_CYCLE
                        For output model name format
  --upload_model UPLOAD_MODEL
                        Upload trained model after training process

--------------------------------------------------------------------------------
When this file is run, it will first check for dataset availability, if no dataset is detected, it will automatically download required dataset.
Then, this file checks for model availability, there is 3 cases (specifies by argument: resume_mode)
                        'none'    : model is not require (however, this method is deprecated)
                        'pretrain': system will assume that specified model is a pretrained mode (pretrain from VGG16), if no model is found, this file will download a
                                    raw-pretrained-model and convert it to SSD model
                        'continue': Model must exist.

Network Parameter (and hyper-parameter) can be easily modified by modifying NetworkConfig.py in # NETWORK ACHITECHTURE # section.
However, to modify Network Achitechture, we have to modify ssd.py.

------------------------------------
-- masstest.py ---------------------
------------------------------------
This file is responsible for testing the model, it will load the specified model and detect symbol in test dataset.
If there is no test-set detected, this file will download a dataset.
This file require a model.
