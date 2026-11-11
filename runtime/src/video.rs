//! Narrow, explicitly named H.264 encoder adapter. RGB remains host memory.
use anyhow::{Context, Result, anyhow, bail};
use clap::ValueEnum;
use ffmpeg_next as av;
use ndarray::Array3;
use std::path::Path;
use video_rs::{
    Decoder, DecoderBuilder, Time, hwaccel::HardwareAccelerationDeviceType,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Backend {
    Cpu,
    #[value(alias = "nvidia")]
    Cuda,
    Auto,
}

pub fn decoder(path: &Path, backend: Backend) -> Result<Decoder> {
    eprintln!("Decoder requested: {backend:?}");
    if !matches!(backend, Backend::Cpu) {
        match DecoderBuilder::new(path)
            .with_hardware_acceleration(HardwareAccelerationDeviceType::Cuda)
            .build()
        {
            Ok(decoder) => {
                eprintln!(
                    "Decoder initialized: CUDA device context requested successfully; per-frame NVDEC use is not observable (host RGB output)"
                );
                return Ok(decoder);
            }
            Err(error) if matches!(backend, Backend::Auto) => {
                eprintln!(
                    "CUDA decoder initialization failed: {error}; auto falling back to CPU"
                );
            }
            Err(error) => {
                return Err(anyhow!(
                    "CUDA decoder initialization failed: {error}"
                ));
            }
        }
    }
    let decoder = Decoder::new(path)?;
    eprintln!("Decoder initialized: CPU");
    Ok(decoder)
}

/// video-rs consumes ReadExhausted internally, then signals drained EOF here.
pub fn decoded_frame<T>(
    result: Result<T, video_rs::Error>,
) -> Result<Option<T>> {
    match result {
        Ok(frame) => Ok(Some(frame)),
        Err(video_rs::Error::DecodeExhausted) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn quality_options(nvenc: bool) -> av::Dictionary<'static> {
    let mut options = av::Dictionary::new();
    if nvenc {
        options.set("preset", "p4");
        options.set("rc", "vbr");
        options.set("cq", "23");
        // Override the codec's implicit 2M default; CQ is unconstrained.
        options.set("b", "0");
        options.set("maxrate", "0");
    } else {
        options.set("preset", "medium");
        options.set("crf", "23");
    }
    options
}

pub struct Encoder {
    output: av::format::context::Output,
    encoder: av::encoder::Video,
    width: u32,
    height: u32,
    time_base: av::Rational,
}

impl Encoder {
    pub fn new(
        path: &Path,
        width: u32,
        height: u32,
        fps: f32,
        backend: Backend,
    ) -> Result<Self> {
        let (num, den) = crate::pipeline::frame_rate_parts(fps)
            .map_err(anyhow::Error::msg)?;
        let frame_rate = av::Rational(num, den);
        let time_base = av::Rational(1, 90_000);
        let open = |name: &str| -> Result<av::encoder::Video> {
            let codec = av::encoder::find_by_name(name)
                .with_context(|| format!("encoder {name} is unavailable"))?;
            let mut encoder =
                av::codec::context::Context::new_with_codec(codec)
                    .encoder()
                    .video()?;
            encoder.set_width(width);
            encoder.set_height(height);
            encoder.set_format(av::format::Pixel::YUV420P);
            encoder.set_time_base(time_base);
            encoder.set_frame_rate(Some(frame_rate));
            encoder.set_gop(fps.round().max(1.0) as u32);
            // Global headers are safe for the supported MP4 output and avoid opening
            // a muxer (and truncating a file) before hardware initialization succeeds.
            encoder.set_flags(av::codec::Flags::GLOBAL_HEADER);
            let options = quality_options(name == "h264_nvenc");
            Ok(encoder
                .open_with(options)
                .with_context(|| format!("opening {name}"))?)
        };
        let (encoder, name) = match backend {
            Backend::Cpu => (open("libx264")?, "libx264"),
            Backend::Cuda => (open("h264_nvenc")?, "h264_nvenc"),
            Backend::Auto => match open("h264_nvenc") {
                Ok(encoder) => (encoder, "h264_nvenc"),
                Err(error) => {
                    eprintln!(
                        "NVENC initialization failed: {error:#}; auto falling back to libx264"
                    );
                    (open("libx264")?, "libx264")
                }
            },
        };
        eprintln!(
            "Encoder selected and opened: {name}; nominal fps={num}/{den}; {}",
            if name == "h264_nvenc" {
                "VBR CQ 23, bitrate/maxrate 0 (unconstrained)"
            } else {
                "CRF 23"
            }
        );
        let mut output = av::format::output(path)?;
        {
            let mut stream =
                output.add_stream(av::encoder::find_by_name(name))?;
            stream.set_time_base(time_base);
            stream.set_rate(frame_rate);
            stream.set_parameters(&encoder);
        }
        output.write_header()?;
        Ok(Self {
            output,
            encoder,
            width,
            height,
            time_base,
        })
    }

    pub fn scaler(&self) -> Result<av::software::scaling::Context> {
        Ok(av::software::scaling::Context::get(
            av::format::Pixel::RGB24,
            self.width,
            self.height,
            av::format::Pixel::YUV420P,
            self.width,
            self.height,
            av::software::scaling::Flags::BILINEAR,
        )?)
    }

    pub fn encode(
        &mut self,
        image: &Array3<u8>,
        time: Time,
        scaler: &mut av::software::scaling::Context,
    ) -> Result<()> {
        if image.dim() != (self.height as usize, self.width as usize, 3) {
            bail!("frame dimensions changed during encoding");
        }
        let data = image.as_slice().context("non-contiguous RGB output")?;
        let mut rgb = av::frame::Video::new(
            av::format::Pixel::RGB24,
            self.width,
            self.height,
        );
        let row_bytes = self.width as usize * 3;
        let stride = rgb.stride(0);
        for (row, source) in data.chunks_exact(row_bytes).enumerate() {
            rgb.data_mut(0)[row * stride..row * stride + row_bytes]
                .copy_from_slice(source);
        }
        let mut yuv = av::frame::Video::empty();
        scaler.run(&rgb, &mut yuv)?;
        yuv.set_pts(time.with_time_base(self.time_base).into_value());
        self.encoder.send_frame(&yuv)?;
        self.drain(false)
    }

    fn drain(&mut self, flushing: bool) -> Result<()> {
        loop {
            let mut packet = av::Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(0);
                    packet.rescale_ts(
                        self.time_base,
                        self.output
                            .stream(0)
                            .context("missing output stream")?
                            .time_base(),
                    );
                    packet.set_position(-1);
                    packet.write_interleaved(&mut self.output)?;
                }
                Err(av::Error::Eof) => return Ok(()),
                Err(av::Error::Other { errno })
                    if errno == av::util::error::EAGAIN && !flushing =>
                {
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    pub fn finish(mut self) -> Result<()> {
        self.encoder.send_eof()?;
        self.drain(true)?;
        self.output.write_trailer()?;
        Ok(())
    }
}
