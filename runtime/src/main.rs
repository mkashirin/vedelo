mod common;
mod detector;
mod tracker;

use anyhow::Result;
use clap::Parser;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use ab_glyph::FontRef;
use kornia_image::{Image, ImageSize};
use ndarray::Array3;
use video_rs::{self, Decoder, Encoder, encode::Settings};

use crate::common::{Rect, draw_filled_rect, draw_rect, draw_text};
use crate::detector::YoloDetector;
use crate::tracker::SortTracker;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    model: String,
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
    image: Image<u8, 3>,
    height: usize,
    width: usize,
}

fn main() -> Result<()> {
    video_rs::init().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let args = Args::parse();

    let detector = YoloDetector::new(&args.model, args.imgsz)?;
    let mut tracker = SortTracker::new(args.max_age, args.iou_thresh);

    let font_data =
        include_bytes!("../third-party/fonts/roboto-mono-stripped.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    let source_path = Path::new(&args.source);
    let mut decoder = Decoder::new(source_path)?;
    let (width, height) = decoder.size();
    let framerate = decoder.frame_rate();
    println!("Input: {}x{} @ {:.2} fps", width, height, framerate);

    let dest_path = Path::new(&args.destination);
    let settings =
        Settings::preset_h264_yuv420p(width as usize, height as usize, false)
            .with_keyframe_interval(framerate.round() as u64);

    let mut encoder = Encoder::new(dest_path, settings)?;

    let mut unique_ids = HashSet::new();
    let mut frame_count = 0usize;
    let started = Instant::now();
    let mut pending = Vec::with_capacity(args.batch_size);
    for frame_res in decoder.decode_iter() {
        let (time, frame_array) = match frame_res {
            Ok(f) => f,
            Err(_) => break, // End of stream or error
        };

        let (h, w, _channels) = frame_array.dim();
        let (raw_buffer, _offset) = frame_array.into_raw_vec_and_offset();

        let image = Image::new(
            ImageSize {
                width: w,
                height: h,
            },
            raw_buffer,
        )?;

        pending.push(PendingFrame {
            time,
            image,
            height: h,
            width: w,
        });
        if pending.len() < args.batch_size {
            continue;
        }
        process_batch(
            &mut pending,
            &detector,
            &mut tracker,
            &font,
            &mut encoder,
            &mut unique_ids,
            &mut frame_count,
            &args,
        )?;
    }

    // Do not discard the tail of a source whose frame count is not divisible
    // by the selected micro-batch size.
    if !pending.is_empty() {
        process_batch(
            &mut pending,
            &detector,
            &mut tracker,
            &font,
            &mut encoder,
            &mut unique_ids,
            &mut frame_count,
            &args,
        )?;
    }

    encoder.finish()?;
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
    encoder: &mut Encoder,
    unique_ids: &mut HashSet<usize>,
    frame_count: &mut usize,
    args: &Args,
) -> Result<()> {
    let images = pending.iter().map(|frame| &frame.image).collect::<Vec<_>>();
    let detections_per_frame = detector.detect_batch(&images, args.conf)?;

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
        // video-rs currently accepts ndarray-owned host memory.  The only
        // remaining full-frame copy is isolated here for a future NVENC path.
        let modified_buffer = image.as_slice().to_vec();

        // Reconstruct the Array3
        let out_frame_array = Array3::from_shape_vec(
            (pending_frame.height, pending_frame.width, 3),
            modified_buffer,
        )?;

        // 4. Encode with the ORIGINAL timestamp
        encoder.encode(&out_frame_array, pending_frame.time)?;

        *frame_count += 1;
        if args.log_every != 0 && *frame_count % args.log_every == 0 {
            println!(
                "Tracks at frame {}: {}",
                frame_count,
                active_tracks.len()
            );
        }
    }
    Ok(())
}
