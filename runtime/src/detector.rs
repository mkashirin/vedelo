use anyhow::Result;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast};
use tch::{Device, Kind, Tensor};

use crate::common::Rect;

#[derive(Debug, Copy, Clone)]
pub struct Detection {
    pub rect: Rect,
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
    input_size: usize,
}

impl YoloDetector {
    pub fn new(model_path: &str, input_size: i32) -> Result<Self> {
        let device = if tch::Cuda::is_available() {
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
            input_size: input_size as usize,
        })
    }

    pub fn detect(
        &self,
        image: &Image<u8, 3>,
        conf_threshold: f64,
    ) -> Result<Vec<Detection>> {
        let (input_tensor, pad_info) = self.preprocess(image)?;

        let output = self.model.forward_ts(&[&input_tensor])?;
        let output = if output.size()[1] == 6 {
            output.transpose(1, 2)
        } else {
            output.shallow_clone()
        };

        let detections_2d = output.squeeze_dim(0).to(self.device);
        let num_dets = detections_2d.size()[0] as usize;

        let flat_tensor = detections_2d.reshape([-1]);
        let flat_data: Vec<f32> = Vec::<f32>::try_from(flat_tensor)?;

        let mut results = Vec::with_capacity(num_dets);

        for i in 0..num_dets {
            let offset = i * 6;
            if offset + 5 >= flat_data.len() {
                break;
            }

            let score = flat_data[offset + 4];
            if score as f64 <= conf_threshold {
                continue;
            }

            let class_id = flat_data[offset + 5] as i32;

            let x1 = flat_data[offset] as f64;
            let y1 = flat_data[offset + 1] as f64;
            let x2 = flat_data[offset + 2] as f64;
            let y2 = flat_data[offset + 3] as f64;

            let x1_unpad = x1 - pad_info.dw;
            let y1_unpad = y1 - pad_info.dh;
            let x2_unpad = x2 - pad_info.dw;
            let y2_unpad = y2 - pad_info.dh;

            let x1_orig = x1_unpad / pad_info.ratio;
            let y1_orig = y1_unpad / pad_info.ratio;
            let x2_orig = x2_unpad / pad_info.ratio;
            let y2_orig = y2_unpad / pad_info.ratio;

            let x_final = x1_orig.round() as i32;
            let y_final = y1_orig.round() as i32;
            let w_final = (x2_orig - x1_orig).round() as i32;
            let h_final = (y2_orig - y1_orig).round() as i32;

            if w_final > 0 && h_final > 0 {
                results.push(Detection {
                    rect: Rect::new(x_final, y_final, w_final, h_final),
                    score,
                    class_id,
                });
            }
        }

        Ok(results)
    }

    fn preprocess(&self, image: &Image<u8, 3>) -> Result<(Tensor, PadInfo)> {
        let size = image.size();
        let target = self.input_size;

        let ratio = (target as f64 / size.width as f64)
            .min(target as f64 / size.height as f64);

        let new_w = (size.width as f64 * ratio).round() as usize;
        let new_h = (size.height as f64 * ratio).round() as usize;
        let mut resized = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: new_w,
                height: new_h,
            },
            0,
        )?;
        resize_fast(image, &mut resized, InterpolationMode::Nearest)?;
        let mut padded = vec![114u8; target * target * 3];

        let dw = (target - new_w) / 2;
        let dh = (target - new_h) / 2;
        let resized_data = resized.storage.as_slice();
        for y in 0..new_h {
            let dst_row = (y + dh) * target;
            let src_row = y * new_w;

            for x in 0..new_w {
                let dst_idx = (dst_row + (x + dw)) * 3;
                let src_idx = (src_row + x) * 3;

                padded[dst_idx] = resized_data[src_idx];
                padded[dst_idx + 1] = resized_data[src_idx + 1];
                padded[dst_idx + 2] = resized_data[src_idx + 2];
            }
        }

        let tensor = Tensor::from_slice(&padded)
            .reshape([target as i64, target as i64, 3])
            .permute([2, 0, 1])
            .to_kind(Kind::Float)
            / 255.0;

        let tensor = tensor.unsqueeze(0).to(self.device);

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
