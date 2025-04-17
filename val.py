from ultralytics import YOLO
import argparse
def parse_args():  
    parser = argparse.ArgumentParser()  
      
    parser.add_argument('data', default='ship_v2_well.yaml', help='Path to the dataset configuration file (e.g., ship_v2_well.yaml).')  
    parser.add_argument('pt', default='yolov11n.pt', help='Path to the YOLO model file (e.g., yolov11n.pt).')  
    parser.add_argument('--imgsz', type=int, default=640, help='Size of input images (e.g., 640).')  
    parser.add_argument('--batch', type=int, default=16, help='Number of images per batch. Use -1 for AutoBatch.')  
    parser.add_argument('--save-json', action='store_true', help='Save results to a JSON file.')  
    parser.add_argument('--save-hybrid', action='store_true', help='Save a hybrid version of labels.')  
    parser.add_argument('--conf', type=float, default=0.001, help='Minimum confidence threshold for detections.')  
    parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold for Non-Maximum Suppression.')  
    parser.add_argument('--iou-correct', type=float, default=0.5, help='IoU over this threshold means correct detection.')  
    parser.add_argument('--max-det', type=int, default=300, help='Maximum number of detections per image.')  
    parser.add_argument('--half', action='store_true', help='Enable half-precision (FP16) computation.')  
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for validation (cpu, cuda:0, etc.).')  
    parser.add_argument('--dnn', action='store_true', help='Use OpenCV DNN module for ONNX model inference.')  
    parser.add_argument('--plots', action='store_true', help='Generate and save plots of predictions vs ground truth.')  
    parser.add_argument('--rect', action='store_true', help='Use rectangular inference for batching.')  
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use for validation (val, test, or train).')  
      
    return parser.parse_args()
    
def main():
    args=parse_args()
    model = YOLO(args.pt)

    metrics = model.val(  
        data=args.data,  
        imgsz=args.imgsz,  
        batch=args.batch,  
        save_json=args.save_json,  
        save_hybrid=args.save_hybrid,  
        conf=args.conf,  
        iou=args.iou,  
        max_det=args.max_det,  
        half=args.half,  
        device=args.device,  
        dnn=args.dnn,  
        plots=args.plots,  
        rect=args.rect,  
        split=args.split,
        iou_correct=args.iou_correct
    )  
    print('Done.')
    # print(metrics)
        
if __name__ == '__main__':
    main()