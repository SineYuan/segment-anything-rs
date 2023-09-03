use std::path::Path;

use ndarray::prelude::*;
use ndarray::{Array, ArrayD, IxDyn};

use opencv::core::*;
use opencv::imgcodecs::*;
use opencv::imgproc::*;

use cv_convert::*;

use segment_anything_rs::*;

fn main() {
    let prep_onnx =
        "/home/stargi/workspace/segment-anything-cpp-wrapper/models/mobile_sam_preprocess.onnx";
    let sam_onnx = "/home/stargi/workspace/segment-anything-cpp-wrapper/models/mobile_sam.onnx";

    let prep_onnx = "/data/models/sammodels/out_quant/sam_preprocess_vitb_quant.onnx";
    let sam_onnx = "/data/models/sammodels/out_quant/sam_vitb.onnx";

    let dev = segment_anything_rs::Device::CPU;
    let dev = segment_anything_rs::Device::CUDA;
    let model = SegmentAnythingModel::new(prep_onnx, sam_onnx, dev).unwrap();

    let image_path = "/home/stargi/workspace/segment-anything/notebooks/images/dog.jpg";

    let src_image = imread(image_path, ImreadModes::IMREAD_COLOR as i32).unwrap();
    println!("{:?}", src_image);

    let sz = Size {
        width: 1024,
        height: 720,
    };
    let mut resized = Mat::default();
    resize(&src_image, &mut resized, sz, 0., 0., INTER_NEAREST).unwrap();

    let mut resized_rgb = Mat::default();
    cvt_color(&resized, &mut resized_rgb, COLOR_BGR2RGB, 0).unwrap();

    let img_tensor: ArrayD<u8> = (&resized_rgb).try_into_cv().unwrap();

    let embedding = model.encode(img_tensor).unwrap();

    let mut prompts = vec![];
    prompts.push(Prompt::Point {
        x: 120,
        y: 121,
        label: 1,
    });
    prompts.push(Prompt::BBox {
        left: 152,
        top: 151,
        right: 353,
        bottom: 354,
    });
    prompts.push(Prompt::Point {
        x: 251,
        y: 250,
        label: 0,
    });
    prompts.push(Prompt::BBox {
        left: 102,
        top: 101,
        right: 303,
        bottom: 304,
    });

    let mask = model.get_mask(&embedding, &prompts).unwrap();
    println!("mask {:?}", mask);

    println!("Farewell!");
}
