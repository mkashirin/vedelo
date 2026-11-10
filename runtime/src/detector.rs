use anyhow::Result;
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast};
use tch::{Device, Kind, Tensor};

use crate::common::Rect;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum ModelDevice {
    Auto,
    Cpu,
    Cuda,
    Mps,
}

impl ModelDevice {
    fn resolve(self, cuda: bool, mps: bool) -> Result<Device> {
        match self {
            Self::Auto if cuda => Ok(Device::Cuda(0)),
            Self::Auto if mps => Ok(Device::Mps),
            Self::Auto | Self::Cpu => Ok(Device::Cpu),
            Self::Cuda if cuda => Ok(Device::Cuda(0)),
            Self::Mps if mps => Ok(Device::Mps),
            Self::Cuda => {
                anyhow::bail!("CUDA model device requested but unavailable")
            }
            Self::Mps => {
                anyhow::bail!("MPS model device requested but unavailable")
            }
        }
    }
}

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
    pub fn with_device(
        model_path: &str,
        input_size: i32,
        requested: ModelDevice,
    ) -> Result<Self> {
        anyhow::ensure!(
            input_size > 0,
            "detector input_size must be positive"
        );
        checked_rgb_len(input_size as usize, input_size as usize)?;
        let device = requested
            .resolve(tch::Cuda::is_available(), tch::utils::has_mps())?;
        println!("Detector running on {device:?}");

        // Remap archive storage before loading (including CUDA exports on macOS).
        let mut model = tch::CModule::load_on_device(model_path, device)?;
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
        let (padded, pad_infos) = preprocess_bytes(images, self.input_size)?;
        // Transfer NHWC bytes, not host-normalized Half/Float pixels. Casting
        // and normalization now happen on the inference device (blocking upload).
        let tensor = bytes_to_tensor(
            &padded,
            images.len(),
            self.input_size,
            self.device,
            self.input_kind,
        )?;
        Ok((tensor, pad_infos))
    }
}

/// Upload NHWC uint8 before casting/normalizing on the inference device.
fn bytes_to_tensor(
    bytes: &[u8],
    batch: usize,
    size: usize,
    device: Device,
    kind: Kind,
) -> Result<Tensor> {
    Ok(Tensor::f_from_slice(bytes)?
        .f_reshape([
            i64::try_from(batch)?,
            i64::try_from(size)?,
            i64::try_from(size)?,
            3,
        ])?
        .f_to_device(device)?
        .f_permute([0, 3, 1, 2])?
        .f_to_kind(kind)?
        .f_div_scalar(255.0)?)
}

fn checked_rgb_len(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| {
            anyhow::anyhow!("RGB image size overflows addressable storage")
        })
}

fn filled_bytes(len: usize, value: u8) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(len)?;
    bytes.resize(len, value);
    Ok(bytes)
}

/// CPU-only letterboxing, kept separate so parity tests need no model or device.
fn preprocess_bytes(
    images: &[&Image<u8, 3>],
    target: usize,
) -> Result<(Vec<u8>, Vec<PadInfo>)> {
    anyhow::ensure!(
        target > 0 && target <= u32::MAX as usize,
        "invalid detector input_size"
    );
    let frame_len = checked_rgb_len(target, target)?;
    let batch_len = frame_len
        .checked_mul(images.len())
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "detector batch size overflows addressable storage"
            )
        })?;
    let mut padded = filled_bytes(batch_len, 114)?;
    let mut pad_infos = Vec::new();
    pad_infos.try_reserve_exact(images.len())?;
    // Reuse the owned contiguous destination for consecutive matching sizes.
    // Kornia has no safe strided-ROI destination API.
    let mut scratch: Option<Image<u8, 3>> = None;

    for (batch_index, image) in images.iter().enumerate() {
        let size = image.size();

        anyhow::ensure!(
            size.width > 0
                && size.height > 0
                && size.width <= u32::MAX as usize
                && size.height <= u32::MAX as usize,
            "invalid source image dimensions"
        );
        anyhow::ensure!(
            image.as_slice().len()
                == checked_rgb_len(size.width, size.height)?,
            "invalid source RGB storage length"
        );
        let ratio = (target as f64 / size.width as f64)
            .min(target as f64 / size.height as f64);

        let new_w = (size.width as f64 * ratio).round() as usize;
        let new_h = (size.height as f64 * ratio).round() as usize;
        // Keep rounding unchanged; a degenerate extreme aspect ratio is an
        // error rather than silently changing the sampling geometry.
        anyhow::ensure!(
            new_w > 0 && new_h > 0 && new_w <= target && new_h <= target,
            "rounded letterbox dimensions are invalid"
        );
        let resized_size = ImageSize {
            width: new_w,
            height: new_h,
        };
        let resized_data = if size == resized_size {
            // No resize allocation or intermediate copy for matching images.
            image.as_slice()
        } else {
            if scratch.as_ref().map(|s| s.size()) != Some(resized_size) {
                scratch = Some(Image::new(
                    resized_size,
                    filled_bytes(checked_rgb_len(new_w, new_h)?, 0)?,
                )?);
            }
            let resized =
                scratch.as_mut().expect("resize scratch initialized");
            resize_fast(image, resized, InterpolationMode::Nearest)?;
            resized.as_slice()
        };

        let dw = (target - new_w) / 2;
        let dh = (target - new_h) / 2;
        let batch_offset = batch_index * frame_len;
        let row_len = new_w * 3;
        for (y, row) in resized_data.chunks_exact(row_len).enumerate() {
            let dst = batch_offset + ((y + dh) * target + dw) * 3;
            padded[dst..dst + row_len].copy_from_slice(row);
        }
        pad_infos.push(PadInfo {
            ratio,
            dw: dw as f64,
            dh: dh as f64,
        });
    }

    Ok((padded, pad_infos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_policy() {
        for cuda in [false, true] {
            for mps in [false, true] {
                assert_eq!(
                    ModelDevice::Auto.resolve(cuda, mps).unwrap(),
                    if cuda {
                        Device::Cuda(0)
                    } else if mps {
                        Device::Mps
                    } else {
                        Device::Cpu
                    }
                );
                assert_eq!(
                    ModelDevice::Cpu.resolve(cuda, mps).unwrap(),
                    Device::Cpu
                );
                assert_eq!(ModelDevice::Cuda.resolve(cuda, mps).is_ok(), cuda);
                assert_eq!(ModelDevice::Mps.resolve(cuda, mps).is_ok(), mps);
            }
        }
    }

    #[test]
    fn cpu_tensor_preprocessing_parity() -> Result<()> {
        let frames = [image(7, 4), image(3, 7)];
        let refs: Vec<_> = frames.iter().collect();
        let (bytes, _) = preprocess_bytes(&refs, 7)?;
        let tensor = bytes_to_tensor(&bytes, 2, 7, Device::Cpu, Kind::Float)?;
        assert_eq!(tensor.size(), [2, 3, 7, 7]);
        assert_eq!(tensor.kind(), Kind::Float);
        assert_eq!(tensor.device(), Device::Cpu);
        let actual = Vec::<f32>::try_from(tensor.contiguous().reshape([-1]))?;
        let mut expected = Vec::new();
        for b in 0..2 {
            for c in 0..3 {
                for p in 0..49 {
                    expected.push(bytes[b * 147 + p * 3 + c] as f32 / 255.0);
                }
            }
        }
        // CPU FP32 numerical contract, not cross-backend bit equality.
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-7);
        }
        Ok(())
    }

    fn image(w: usize, h: usize) -> Image<u8, 3> {
        Image::new(
            ImageSize {
                width: w,
                height: h,
            },
            (0..w * h * 3).map(|i| (i % 251) as u8).collect(),
        )
        .unwrap()
    }

    #[test]
    fn letterbox_matches_pixel_copy_reference() -> Result<()> {
        // Square/no resize, odd vertical/horizontal padding, up/down sampling,
        // repeated sizes (scratch reuse), and a size change in the same batch.
        let frames: Vec<_> =
            [(7, 7), (7, 4), (4, 7), (13, 8), (13, 8), (2, 3), (1, 7)]
                .into_iter()
                .map(|(w, h)| image(w, h))
                .collect();
        let refs: Vec<_> = frames.iter().collect();
        let (actual, infos) = preprocess_bytes(&refs, 7)?;
        let mut expected = vec![114; actual.len()];
        for (b, src) in frames.iter().enumerate() {
            let size = src.size();
            let ratio =
                (7.0 / size.width as f64).min(7.0 / size.height as f64);
            let w = (size.width as f64 * ratio).round() as usize;
            let h = (size.height as f64 * ratio).round() as usize;
            let mut resized = Image::from_size_val(
                ImageSize {
                    width: w,
                    height: h,
                },
                0,
            )?;
            resize_fast(src, &mut resized, InterpolationMode::Nearest)?;
            let dw = (7 - w) / 2;
            let dh = (7 - h) / 2;
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        expected[b * 147 + ((y + dh) * 7 + x + dw) * 3 + c] =
                            resized.as_slice()[(y * w + x) * 3 + c];
                    }
                }
            }
            assert_eq!(infos[b].ratio, ratio);
            assert_eq!(infos[b].dw, dw as f64);
            assert_eq!(infos[b].dh, dh as f64);
        }
        assert_eq!(actual, expected);
        // Three missing rows: one above, two below (integer floor centering).
        assert_eq!(&actual[147..168], &[114; 21]);
        assert_eq!(&actual[252..294], &[114; 42]);
        Ok(())
    }

    #[test]
    fn invalid_sizes_and_empty_batch() {
        assert!(preprocess_bytes(&[], 0).is_err());
        assert!(preprocess_bytes(&[], usize::MAX).is_err());
        assert!(checked_rgb_len(usize::MAX, 2).is_err());
        let thin = image(100, 1);
        assert!(preprocess_bytes(&[&thin], 1).is_err());
        let (bytes, infos) = preprocess_bytes(&[], 7).unwrap();
        assert!(bytes.is_empty() && infos.is_empty());
        for input_size in [0, -1] {
            assert!(
                YoloDetector::with_device(
                    "nonexistent-model",
                    input_size,
                    ModelDevice::Cpu,
                )
                .is_err()
            );
        }
    }
}
