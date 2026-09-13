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
    input_kind: Kind,
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
        // The exported CUDA models are FP16.  Loading them as FP32 silently
        // doubles their activation footprint and prevents Tensor Core use.
        let input_kind = if matches!(device, Device::Cuda(_)) {
            Kind::Half
        } else {
            Kind::Float
        };
        model.to(device, input_kind, false);

        Ok(Self {
            model,
            device,
            input_kind,
            input_size: input_size as usize,
        })
    }

    pub fn detect(
        &self,
        image: &Image<u8, 3>,
        conf_threshold: f64,
    ) -> Result<Vec<Detection>> {
        let images = [image];
        let batches = self.detect_batch(&images, conf_threshold)?;
        Ok(batches.into_iter().next().unwrap_or_default())
    }

    /// Runs one inference for a group of video frames.  Batching is the main
    /// throughput lever for an offline source: it amortizes launches and lets
    /// cuDNN select Tensor-Core kernels with useful occupancy.
    pub fn detect_batch(
        &self,
        images: &[&Image<u8, 3>],
        conf_threshold: f64,
    ) -> Result<Vec<Vec<Detection>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let (input_tensor, pad_infos) = self.preprocess_batch(images)?;
        let _no_grad = tch::no_grad_guard();
        let output = self.model.forward_ts(&[&input_tensor])?;
        let output = if output.size().len() == 3 && output.size()[1] == 6 {
            output.transpose(1, 2)
        } else {
            output
        };

        let sizes = output.size();
        anyhow::ensure!(
            sizes.len() == 3
                && sizes[0] == images.len() as i64
                && sizes[2] == 6,
            "expected detector output [batch, detections, 6], got {sizes:?}"
        );
        let num_dets = sizes[1] as usize;

        // This is the only mandatory GPU synchronization in detection.  Move
        // compact post-NMS records, never full image tensors, back to the CPU.
        let flat_data: Vec<f32> = Vec::<f32>::try_from(
            output
                .to(Device::Cpu)
                .to_kind(Kind::Float)
                .contiguous()
                .reshape([-1]),
        )?;

        let mut all_results = Vec::with_capacity(images.len());
        for (batch_index, pad_info) in pad_infos.iter().enumerate() {
            let start = batch_index * num_dets * 6;
            let end = start + num_dets * 6;
            all_results.push(Self::postprocess(
                &flat_data[start..end],
                num_dets,
                pad_info,
                conf_threshold,
            ));
        }
        Ok(all_results)
    }

    fn postprocess(
        flat_data: &[f32],
        num_dets: usize,
        pad_info: &PadInfo,
        conf_threshold: f64,
    ) -> Vec<Detection> {
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

        results
    }

    fn preprocess_batch(
        &self,
        images: &[&Image<u8, 3>],
    ) -> Result<(Tensor, Vec<PadInfo>)> {
        let target = self.input_size;
        let mut padded = vec![114u8; images.len() * target * target * 3];
        let mut pad_infos = Vec::with_capacity(images.len());

        for (batch_index, image) in images.iter().enumerate() {
            let size = image.size();

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

            let dw = (target - new_w) / 2;
            let dh = (target - new_h) / 2;
            let resized_data = resized.storage.as_slice();
            let batch_offset = batch_index * target * target * 3;
            for y in 0..new_h {
                let dst_row = (y + dh) * target;
                let src_row = y * new_w;

                for x in 0..new_w {
                    let dst_idx = batch_offset + (dst_row + (x + dw)) * 3;
                    let src_idx = (src_row + x) * 3;

                    padded[dst_idx] = resized_data[src_idx];
                    padded[dst_idx + 1] = resized_data[src_idx + 1];
                    padded[dst_idx + 2] = resized_data[src_idx + 2];
                }
            }
            pad_infos.push(PadInfo {
                ratio,
                dw: dw as f64,
                dh: dh as f64,
            });
        }

        let tensor = Tensor::from_slice(&padded)
            .reshape([images.len() as i64, target as i64, target as i64, 3])
            .permute([0, 3, 1, 2])
            .to_kind(self.input_kind)
            / 255.0;

        let tensor = tensor.to(self.device);
        Ok((tensor, pad_infos))
    }
}
