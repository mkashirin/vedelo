mod common;
mod detector;
mod pipeline;
mod tracker;
mod video;

use anyhow::Result;
use clap::Parser;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::video::{Backend, Encoder};
use ab_glyph::FontRef;
use kornia_image::{Image, ImageSize};
use ndarray::Array3;
use video_rs;

use crate::common::{Rect, draw_filled_rect, draw_rect, draw_text};
use crate::detector::{ModelDevice, YoloDetector};
use crate::tracker::SortTracker;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    model: String,
    /// Model inference device, independent of video codecs.
    #[arg(long, value_enum, default_value = "auto")]
    device: ModelDevice,
    #[arg(short, long)]
    source: String,
    #[arg(short, long)]
    destination: String,
    #[arg(long, default_value_t = 0.2)]
    conf: f64,
    #[arg(long, default_value_t = 0.15)]
    iou_thresh: f64,
    #[arg(long, default_value_t = 1280)]
    imgsz: i32,
    #[arg(long, default_value_t = 60)]
    max_age: i32,
    /// Number of frames submitted to one model invocation.  Larger values
    /// improve GPU occupancy for offline video, at the cost of VRAM.
    #[arg(long, default_value_t = 8, value_parser = parse_batch_size)]
    batch_size: usize,
    /// Print detailed track state every N frames; zero disables hot-path logs.
    #[arg(long, default_value_t = 0)]
    log_every: usize,
    /// Print end-to-end throughput after the encoder is flushed.
    #[arg(long)]
    profile: bool,
    /// Video decode backend, independent of model device selection.
    #[arg(long, value_enum, default_value = "cpu")]
    decode_backend: Backend,
    /// H.264 encoder: CPU libx264 or CUDA h264_nvenc.
    #[arg(long, value_enum, default_value = "cpu")]
    encode_backend: Backend,
}

fn parse_batch_size(value: &str) -> std::result::Result<usize, String> {
    let batch_size = value
        .parse::<usize>()
        .map_err(|_| "must be a positive integer".to_owned())?;

    if batch_size == 0 {
        return Err("must be at least 1".to_owned());
    }

    Ok(batch_size)
}

struct PendingFrame {
    time: video_rs::Time,
    duration: video_rs::Time,
    image: Image<u8, 3>,
    height: usize,
    width: usize,
}

fn main() -> Result<()> {
    video_rs::init().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    // video-rs routes FFmpeg diagnostics to tracing, but this executable has
    // no tracing subscriber. Keep native codec/muxer errors visible on stderr.
    unsafe {
        ffmpeg_next::ffi::av_log_set_callback(Some(
            ffmpeg_next::ffi::av_log_default_callback,
        ));
    }
    ffmpeg_next::util::log::set_level(ffmpeg_next::util::log::Level::Warning);
    let args = Args::parse();

    let detector =
        YoloDetector::with_device(&args.model, args.imgsz, args.device)?;
    let mut tracker = SortTracker::new(args.max_age, args.iou_thresh);

    let font_data =
        include_bytes!("../third-party/fonts/roboto-mono-stripped.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    let source_path = Path::new(&args.source);
    let mut decoder = video::decoder(source_path, args.decode_backend)?;
    let (width, height) = decoder.size();
    let framerate = decoder.frame_rate();
    println!("Input: {}x{} @ {:.2} fps", width, height, framerate);

    let dest_path = Path::new(&args.destination);
    let mut encoder = Encoder::new(
        dest_path,
        width,
        height,
        framerate,
        args.encode_backend,
    )?;

    let mut unique_ids = HashSet::new();
    let mut frame_count = 0usize;
    let started = Instant::now();
    let batch_size = args.batch_size;
    pipeline::run(
        move |tx| {
            let mut produce = || -> Result<()> {
                while let Some(pending) =
                    pipeline::next_batch(batch_size, || {
                        let mut next = || -> Result<Option<PendingFrame>> {
                            let Some((time, duration, frame_array)) =
                                video::decoded_frame(&mut decoder)?
                            else {
                                return Ok(None);
                            };

                            let (h, w, _channels) = frame_array.dim();
                            let (raw_buffer, offset) =
                                frame_array.into_raw_vec_and_offset();
                            anyhow::ensure!(
                                offset.unwrap_or(0) == 0,
                                "decoded RGB storage has a nonzero offset"
                            );

                            let image = Image::new(
                                ImageSize {
                                    width: w,
                                    height: h,
                                },
                                raw_buffer,
                            )?;

                            Ok(Some(PendingFrame {
                                time,
                                duration,
                                image,
                                height: h,
                                width: w,
                            }))
                        };
                        next().map_err(|error| format!("{error:#}"))
                    })
                    .map_err(anyhow::Error::msg)?
                {
                    tx.send(pending).map_err(|_| {
                        anyhow::anyhow!("inference disconnected")
                    })?;
                }
                Ok(())
            };
            produce().map_err(|error| format!("decode: {error:#}"))
        },
        |mut pending| {
            process_batch(
                &mut pending,
                &detector,
                &mut tracker,
                &font,
                &mut unique_ids,
                &mut frame_count,
                &args,
            )
            .map_err(|error| format!("inference/tracking: {error:#}"))
        },
        move |rx| {
            let consume = || -> Result<()> {
                // Scaling context stays on the encoding thread and is reused.
                let mut scaler = encoder.scaler()?;
                for batch in rx {
                    for (time, duration, frame) in batch {
                        encoder.encode(&frame, time, duration, &mut scaler)?;
                    }
                }
                encoder.finish()
            };
            consume().map_err(|error| format!("encode: {error:#}"))
        },
    )
    .map_err(anyhow::Error::msg)?;
    println!("Done! Total unique IDs: {}", unique_ids.len());
    if args.profile {
        let elapsed = started.elapsed();
        let fps = frame_count as f64 / elapsed.as_secs_f64();
        println!(
            "Profile: {} frames in {:.3}s ({fps:.2} FPS)",
            frame_count,
            elapsed.as_secs_f64()
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_batch(
    pending: &mut Vec<PendingFrame>,
    detector: &YoloDetector,
    tracker: &mut SortTracker,
    font: &FontRef,
    unique_ids: &mut HashSet<usize>,
    frame_count: &mut usize,
    args: &Args,
) -> Result<Vec<(video_rs::Time, video_rs::Time, Array3<u8>)>> {
    let images = pending.iter().map(|frame| &frame.image).collect::<Vec<_>>();
    let detections_per_frame = detector.detect_batch(&images, args.conf)?;
    anyhow::ensure!(
        detections_per_frame.len() == pending.len(),
        "inference returned {} results for {} frames",
        detections_per_frame.len(),
        pending.len()
    );
    let mut output = Vec::with_capacity(pending.len());

    for (mut pending_frame, detections) in
        pending.drain(..).zip(detections_per_frame)
    {
        let mut image = &mut pending_frame.image;
        let active_tracks = tracker.update(detections);

        for track in &active_tracks {
            unique_ids.insert(track.id);

            // Generate color based on ID
            let seed = track.id as f64;
            let color = [
                ((seed * 123.0) % 255.0) as u8,
                ((seed * 456.0) % 255.0) as u8,
                ((seed * 789.0) % 255.0) as u8,
            ];

            // Draw bounding box
            draw_rect(&mut image, track.last_rect, color);

            // Draw Label
            let annotation = format!(
                "ID:{},C:{},S:{:.2}",
                track.id, track.class_id, track.det_score
            );
            let ann_rect =
                Rect::new(track.last_rect.x, track.last_rect.y - 20, 140, 20);
            draw_filled_rect(&mut image, ann_rect, color);

            draw_text(
                &mut image,
                &font,
                &annotation,
                track.last_rect.x + 2,
                track.last_rect.y - 2,
                [255, 255, 255],
            );
        }
        // Move the owned CPU allocation; codec conversion still copies.
        let modified_buffer = pending_frame.image.0.storage.into_vec();

        // Reconstruct the Array3
        let out_frame_array = Array3::from_shape_vec(
            (pending_frame.height, pending_frame.width, 3),
            modified_buffer,
        )?;

        // 4. Encode with the ORIGINAL timestamp
        output.push((
            pending_frame.time,
            pending_frame.duration,
            out_frame_array,
        ));

        *frame_count += 1;
        if args.log_every != 0 && *frame_count % args.log_every == 0 {
            println!(
                "Tracks at frame {}: {}",
                frame_count,
                active_tracks.len()
            );
        }
    }
    Ok(output)
}
