use anyhow::Result;
use opencv::{
    core::{self, Mat, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};
use std::ops::Div;
use tch::{Device, Kind, Tensor};

use crate::common::cvt_color_wrapper;

#[derive(Debug, Clone)]
pub struct Detection {
    pub rect: Rect,
    #[allow(dead_code)]
    pub score: f32,
    pub class_id: i32,
}

#[derive(Debug, Clone)]
pub struct PadInfo {
    pub ratio: f64,
    pub dw: f64,
    pub dh: f64,
}

pub struct YoloDetector {
    model: tch::CModule,
    device: Device,
    input_size: i32,
}

impl YoloDetector {
    pub fn new(model_path: &str, input_size: i32) -> Result<Self> {
        let device = if tch::utils::has_mps() {
            println!("Detector running on MPS");
            Device::Mps
        } else if tch::Cuda::is_available() {
            println!("Detector running on CUDA");
            Device::Cuda(0)
        } else {
            println!("Detector running on CPU");
            Device::Cpu
        };

        let mut model = tch::CModule::load(model_path)?;
        model.set_eval();
        model.to(device, Kind::Float, false);

        Ok(Self {
            model,
            device,
            input_size,
        })
    }

    pub fn detect(
        &self,
        frame: &Mat,
        conf_threshold: f64,
    ) -> Result<Vec<Detection>> {
        let (input_tensor, pad_info) = self.preprocess(frame)?;

        let output = self.model.forward_ts(&[&input_tensor])?;
        let output = if output.size()[1] != 6 {
            output.shallow_clone()
        } else {
            output.transpose(1, 2)
        };

        let detections_2d = output.squeeze_dim(0).to(self.device);
        let num_dets = detections_2d.size()[0] as usize;

        let flat_tensor = detections_2d.reshape(&[-1]);
        let flat_data: Vec<f32> = Vec::<f32>::try_from(flat_tensor)?;

        let mut results = Vec::with_capacity(num_dets);

        for i in 0..num_dets {
            let offset = i * 6;
            if offset + 5 >= flat_data.len() {
                break;
            }

            let score = flat_data[offset + 4];

            if score as f64 > conf_threshold {
                let class_id = flat_data[offset + 5] as i32;

                let x1_model = flat_data[offset] as f64;
                let y1_model = flat_data[offset + 1] as f64;
                let x2_model = flat_data[offset + 2] as f64;
                let y2_model = flat_data[offset + 3] as f64;

                // 1. Remove Padding (Letterbox shift)
                let x1_unpad = x1_model - pad_info.dw;
                let y1_unpad = y1_model - pad_info.dh;
                let x2_unpad = x2_model - pad_info.dw;
                let y2_unpad = y2_model - pad_info.dh;

                // 2. Scale to Original Resolution
                let x1_orig = x1_unpad / pad_info.ratio;
                let y1_orig = y1_unpad / pad_info.ratio;
                let x2_orig = x2_unpad / pad_info.ratio;
                let y2_orig = y2_unpad / pad_info.ratio;

                // 3. Convert to Rect (x, y, w, h)
                let w_final = (x2_orig - x1_orig).round() as i32;
                let h_final = (y2_orig - y1_orig).round() as i32;
                let x_final = x1_orig.round() as i32;
                let y_final = y1_orig.round() as i32;

                // Sanity check to ignore invalid boxes
                if w_final > 0 && h_final > 0 {
                    results.push(Detection {
                        rect: Rect::new(x_final, y_final, w_final, h_final),
                        score,
                        class_id,
                    });
                }
            }
        }

        Ok(results)
    }

    fn preprocess(&self, frame: &Mat) -> Result<(Tensor, PadInfo)> {
        let w = frame.cols();
        let h = frame.rows();
        let target = self.input_size;

        let ratio = (target as f64 / w as f64).min(target as f64 / h as f64);
        let new_w = (w as f64 * ratio).round() as i32;
        let new_h = (h as f64 * ratio).round() as i32;

        let mut resized = Mat::default();
        imgproc::resize(
            frame,
            &mut resized,
            Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let dw = (target - new_w) / 2;
        let dh = (target - new_h) / 2;

        let mut padded = Mat::default();
        core::copy_make_border(
            &resized,
            &mut padded,
            dh,
            target - new_h - dh,
            dw,
            target - new_w - dw,
            core::BORDER_CONSTANT,
            Scalar::new(114.0, 114.0, 114.0, 0.0),
        )?;

        let mut rgb = Mat::default();
        cvt_color_wrapper(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

        let size = rgb.size()?;
        let h = size.height;
        let w = size.width;
        let tensor = Tensor::from_slice(rgb.data_bytes()?)
            .reshape([h as i64, w as i64, 3])
            .permute([2, 0, 1])
            .to_kind(Kind::Float)
            .div(255.0)
            .unsqueeze(0)
            .to(self.device);

        Ok((
            tensor,
            PadInfo {
                ratio,
                dw: dw as f64,
                dh: dh as f64,
            },
        ))
    }
}
