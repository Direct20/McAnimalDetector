from ultralytics import YOLO  
import argparse  
  
def parse_args():  
    parser = argparse.ArgumentParser(description='Train script.')  
    parser.add_argument('data', default='mob.yaml', help='Path to the dataset configuration file (e.g., coco8.yaml).')  ##
    parser.add_argument('--model', default='weights/yolo11n.pt', help='Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file.')  
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs.')  
    parser.add_argument('--time', type=float, default=None, help='Maximum training time in hours. Overrides the epochs argument if set.')  
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to wait without improvement in validation metrics before early stopping.')  
    parser.add_argument('--batch', type=int, default=16, help='Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).')  
    parser.add_argument('--imgsz', type=int, default=640, help='Target image size for training.')  ##
    parser.add_argument('--save', type=bool, default=True, help='Enables saving of training checkpoints and final model weights.')  
    parser.add_argument('--save-period', type=int, default=10, help='Frequency of saving model checkpoints, specified in epochs. -1 disables this feature.')  
    parser.add_argument('--cache', type=bool, default=False, help='Enables caching of dataset images in memory (True/ram), on disk (disk), or disables it (False).')  
    parser.add_argument('--device', default=0, help='Specifies the computational device(s) for training.')  #
    parser.add_argument('--workers', type=int, default=2, help='Number of worker threads for data loading.')  
    parser.add_argument('--project', default=None, help='Name of the project directory where training outputs are saved.')  
    parser.add_argument('--name', default=None, help='Name of the training run.') ## 
    parser.add_argument('--exist-ok', type=bool, default=False, help='Allows overwriting of an existing project/name directory if True.')  
    parser.add_argument('--pretrained', type=bool, default=True, help='Determines whether to start training from a pretrained model.')  
    parser.add_argument('--optimizer', default='SGD', help='Choice of optimizer for training.')  
    parser.add_argument('--verbose', type=bool, default=False, help='Enables verbose output during training.')  
    parser.add_argument('--seed', type=int, default=0, help='Sets the random seed for training.')  
    parser.add_argument('--deterministic', type=bool, default=True, help='Forces deterministic algorithm use.')  
    parser.add_argument('--single-cls', type=bool, default=False, help='Treats all classes as a single class during training.')  
    parser.add_argument('--rect', type=bool, default=False, help='Enables rectangular training.')  
    parser.add_argument('--cos-lr', type=bool, default=False, help='Utilizes a cosine learning rate scheduler.')  
    parser.add_argument('--close-mosaic', type=int, default=10, help='Disables mosaic data augmentation in the last N epochs.')  
    parser.add_argument('--resume', type=bool, default=False, help='Resumes training from the last saved checkpoint.')  
    parser.add_argument('--amp', type=bool, default=True, help='Enables Automatic Mixed Precision (AMP) training.')  
    parser.add_argument('--fraction', type=float, default=1.0, help='Specifies the fraction of the dataset to use for training.')  
    parser.add_argument('--profile', type=bool, default=False, help='Enables profiling of ONNX and TensorRT speeds.')  
    parser.add_argument('--freeze', default=None, help='Freezes the first N layers of the model or specified layers by index.')  
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate.')  
    parser.add_argument('--lrf', type=float, default=0.1, help='Final learning rate as a fraction of lr0.')  
    parser.add_argument('--momentum', type=float, default=0.937, help='Momentum factor for SGD or beta1 for Adam.')  
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='L2 regularization term.')  
    parser.add_argument('--warmup-epochs', type=float, default=3.0, help='Number of epochs for learning rate warmup.')  
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='Initial momentum for warmup phase.')  
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1, help='Learning rate for bias parameters during warmup.')  
    parser.add_argument('--box', type=float, default=7.5, help='Weight of the box loss.')  
    parser.add_argument('--cls', type=float, default=0.5, help='Weight of the classification loss.')  
    parser.add_argument('--dfl', type=float, default=1.5, help='Weight of the distribution focal loss.')  
    parser.add_argument('--pose', type=float, default=12.0, help='Weight of the pose loss.')  
    parser.add_argument('--kobj', type=float, default=2.0, help='Weight of the keypoint objectness loss.')  
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Applies label smoothing.')  
    parser.add_argument('--nbs', type=int, default=64, help='Nominal batch size for normalization of loss.')  
    parser.add_argument('--overlap-mask', type=bool, default=True, help='Determines whether segmentation masks should overlap.')  
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample ratio for segmentation masks.')  
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for regularization.')  
    parser.add_argument('--val', type=bool, default=True, help='Enables validation during training.')  
    parser.add_argument('--plots', type=bool, default=False, help='Generates and saves plots of training metrics and examples.')  
  
    args = parser.parse_args()  
      
    return args  
  
def main():  
    args = parse_args()  
  
    model = YOLO(args.model)
    # if args.model is not None:
        # model.load(args.model)  # build from YAML and transfer weights  
  
    results = model.train(  
        data=args.data,  
        model=args.model,  
        epochs=args.epochs,  
        time=args.time,  
        patience=args.patience,  
        batch=args.batch,  
        imgsz=args.imgsz,  
        save=args.save,  
        save_period=args.save_period,  
        cache=args.cache,  
        device=args.device,  
        workers=args.workers,  
        project=args.project,  
        name=args.name,  
        exist_ok=args.exist_ok,  
        pretrained=args.pretrained,  
        optimizer=args.optimizer,  
        verbose=args.verbose,  
        seed=args.seed,  
        deterministic=args.deterministic,  
        single_cls=args.single_cls,  
        rect=args.rect,  
        cos_lr=args.cos_lr,  
        close_mosaic=args.close_mosaic,  
        resume=args.resume,  
        amp=args.amp,  
        fraction=args.fraction,  
        profile=args.profile,  
        freeze=args.freeze,  
        lr0=args.lr0,  
        lrf=args.lrf,  
        momentum=args.momentum,  
        weight_decay=args.weight_decay,  
        warmup_epochs=args.warmup_epochs,  
        warmup_momentum=args.warmup_momentum,  
        warmup_bias_lr=args.warmup_bias_lr,  
        box=args.box,  
        cls=args.cls,  
        dfl=args.dfl,  
        pose=args.pose,  
        kobj=args.kobj,  
        label_smoothing=args.label_smoothing,
        nbs=args.nbs,
        overlap_mask=args.overlap_mask,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        val=args.val,
        plots=args.plots
        )

    print('Done.')
    # print(results)

if __name__ == '__main__':
    main()