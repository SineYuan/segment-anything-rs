use std::sync::Arc;

use ndarray::prelude::*;
use ndarray::{ArrayD, IxDyn};

use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, TensorElementDataType},
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, OrtError, OrtResult,
    Session, SessionBuilder,
};

pub struct ImageEmbedding {
    embedding: ArrayD<f32>,
}

pub struct SegmentAnythingModel {
    encoder_session: Session,
    decoder_session: Session,
}

pub enum Device {
    CPU,
    CUDA,
}

pub enum Prompt {
    Point {
        x: i32,
        y: i32,
        label: i32,
    },
    BBox {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    },
}

impl SegmentAnythingModel {
    pub fn new(encoder_onnx: &str, decoder_onnx: &str, device: Device) -> Result<Self, i32> {
        let providers = match device {
            Device::CPU => [ExecutionProvider::cpu()],
            Device::CUDA => {
                let pvd = ExecutionProvider::cuda();
                //.with("gpu_mem_limit", "2147483648"). // 2G
                //.with("gpu_mem_limit", "3221225472"). // 3G
                //.with_device_id(0);
                [pvd]
            }
        };
        println!("providers {:?}", providers);
        println!("cuda ok? {}", ExecutionProvider::cuda().is_available());
        let enc_env = Arc::new(
            Environment::builder()
                .with_name("encoder")
                .with_execution_providers(providers.clone())
                //.with_log_level(LoggingLevel::Verbose)
                .build()
                .unwrap(),
        );

        let enc_session = SessionBuilder::new(&enc_env)
            .unwrap()
            //.with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
            //.with_intra_threads(1).unwrap()
            .with_model_from_file(encoder_onnx)
            .unwrap();

        //let meta = session.metadata()?;
        //println!("name {}", meta.name()?);
        //println!("producer {}", meta.producer()?);
        let dec_env = Arc::new(
            Environment::builder()
                .with_name("decoder")
                .with_execution_providers(providers)
                .build()
                .unwrap(),
        );

        let dec_session = SessionBuilder::new(&dec_env)
            .unwrap()
            //.with_optimization_level(GraphOptimizationLevel::Level1).unwrap()
            //.with_intra_threads(1).unwrap()
            .with_model_from_file(decoder_onnx)
            .unwrap();

        Ok(Self {
            encoder_session: enc_session,
            decoder_session: dec_session,
        })
    }

    pub fn encode(&self, mut img_tensor: ArrayD<u8>) -> Result<ImageEmbedding, String> {
        // hwc => chw
        img_tensor.swap_axes(0, 1);
        img_tensor.swap_axes(0, 2);

        img_tensor.insert_axis_inplace(Axis(0));
        println!("img_tensor {:?}", img_tensor);

        let input = InputTensor::from_array(img_tensor);
        let outputs = self.encoder_session.run([input]).unwrap();
        //let outputs : Vec<OrtOwnedTensor<f32, _>> = session.run([input]).unwrap();
        //let embedding = &outputs[0];
        let embedding = outputs.into_iter().nth(0).unwrap();

        let embedding = try_extract_to_f32(embedding).unwrap();
        println!("embedding {:?}", embedding);

        Ok(ImageEmbedding {
            //embedding: InputTensor::from_array(embedding),
            embedding: embedding,
        })
    }
    pub fn get_mask(
        &self,
        img_embedding: &ImageEmbedding,
        prompts: &Vec<Prompt>,
    ) -> Result<ArrayD<u8>, String> {
        let embedding = img_embedding.embedding.clone();
        let embedding = InputTensor::from_array(embedding);

        let mask_input = InputTensor::from_array(ArrayD::<f32>::zeros(IxDyn(&[1, 1, 256, 256])));
        let has_mask_input = InputTensor::from_array(ArrayD::<f32>::zeros(IxDyn(&[1])));
        let mut ori_im_size = ArrayD::<f32>::zeros(IxDyn(&[2]));
        // TODO
        ori_im_size[0] = 720.;
        ori_im_size[1] = 1024.;
        println!("ori_im_size {:?}", ori_im_size);
        let ori_im_size = InputTensor::from_array(ori_im_size);

        let (mut num_points, mut num_bboxs) = (0, 0);
        let mut coords: Vec<[f32; 2]> = vec![];
        let mut labels: Vec<f32> = vec![];

        for p in prompts {
            match p {
                Prompt::Point { x, y, label } => {
                    coords.push([x.clone() as f32, y.clone() as f32]);
                    labels.push(label.clone() as f32);
                    num_points += 1;
                }
                Prompt::BBox {
                    top: _top,
                    left: _left,
                    right: _right,
                    bottom: _bottom,
                } => {
                    num_bboxs += 1;
                }
            }
        }

        let mut point_coords = ArrayD::<f32>::zeros(IxDyn(&[1, num_points + 2 * num_bboxs, 2]));
        let mut point_labels = ArrayD::<f32>::zeros(IxDyn(&[1, num_points + 2 * num_bboxs]));

        let mut i = 0;
        for p in prompts {
            match p {
                Prompt::Point { x, y, label } => {
                    //let (p1, p2, l) = x.clone(), y.clone(), label.clone();
                    println!("i: {}", i);
                    point_coords[[0, i, 0]] = x.clone() as f32;
                    point_coords[[0, i, 1]] = y.clone() as f32;
                    point_labels[[0, i]] = label.clone() as f32;
                    i += 1;
                }
                _ => {}
            }
        }

        //println!("coords: {:?}", point_coords);
        //println!("labels: {:?}", point_labels);

        let mut i = 0;
        for p in prompts {
            let offset = num_points;
            match p {
                Prompt::BBox {
                    top,
                    left,
                    right,
                    bottom,
                } => {
                    //let ii = &mut i;
                    point_coords[[0, offset + 2 * i, 0]] = left.clone() as f32;
                    point_coords[[0, offset + 2 * i, 1]] = top.clone() as f32;
                    point_coords[[0, offset + 2 * i + 1, 0]] = right.clone() as f32;
                    point_coords[[0, offset + 2 * i + 1, 1]] = bottom.clone() as f32;
                    println!("offset {} {} {}", i, offset, offset + 2 * i);
                    point_labels[[0, offset + 2 * i]] = 2.;
                    point_labels[[0, offset + 2 * i + 1]] = 3.;
                    i += 1;
                }
                _ => {}
            }
        }

        //println!("coords: {:?}", point_coords);
        //println!("labels: {:?}", point_labels);

        let point_coords = InputTensor::from_array(point_coords);
        let point_labels = InputTensor::from_array(point_labels);

        let inputs = vec![
            embedding,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
            ori_im_size,
        ];
        let mut outputs = self.decoder_session.run(inputs).unwrap();

        let low_res_mask = outputs.pop(); // pop low_res_masks
        let ious = outputs.pop().unwrap();
        let masks = outputs.pop().unwrap();

        let masks = try_extract_to_f32(masks).unwrap();
        let ious = try_extract_to_f32(ious).unwrap();
        //println!("masks: {:?}", masks);
        //println!("iou : {:?}", ious);

        // TODO better way to implement argmax
        let mut max_idx: usize = 0;
        let mut max_iou: f32 = -1.0;
        for (i, iou) in ious.iter().cloned().enumerate() {
            if iou > max_iou {
                max_idx = i;
                max_iou = iou;
            }
        }
        //println!("max iou {}, max idx : {:?}", max_iou, max_idx);

        let mask: ArrayD<u8> = masks
            .slice(s![0, max_idx, .., ..])
            .to_owned()
            .mapv(|v| if v > 0. { 1 } else { 0 })
            .into_dyn();
        //println!("mask: {:?}", mask);
        Ok(mask)
    }
}

pub fn try_extract_to_f32(tensor: DynOrtTensor<IxDyn>) -> OrtResult<ArrayD<f32>> {
    Ok(match tensor.data_type() {
        /*
        TensorElementDataType::Float16 => tensor
            .try_extract::<f16>()?
            .view()
            .to_owned()
            .mapv(|v| v.to_f32()),
        */
        TensorElementDataType::Float32 => tensor.try_extract::<f32>()?.view().to_owned(),
        TensorElementDataType::Float64 => tensor
            .try_extract::<f64>()?
            .view()
            .to_owned()
            .mapv(|v| v as f32),
        /*
        TensorElementDataType::Bfloat16 => tensor
            .try_extract::<bf16>()?
            .view()
            .to_owned()
            .mapv(|v| v.to_f32()),
        */
        //_ => return Err(format!("Unsupported output data type {:?}", tensor.data_type()).into()),
        _ => {
            return Err(OrtError::DataTypeMismatch {
                actual: tensor.data_type(),
                requested: TensorElementDataType::Float32,
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        //let result = add(2, 2);
        //assert_eq!(result, 5);
    }
}
