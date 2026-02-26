mod common;
mod detector;
mod tracker;

use anyhow::Result;
use clap::Parser;
use std::collections::HashSet;
use std::path::Path;

use ab_glyph::FontRef;
use kornia::image::{Image, ImageSize, allocator::CpuAllocator};
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
}

fn main() -> Result<()> {
    video_rs::init().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let args = Args::parse();

    let detector = YoloDetector::new(&args.model, args.imgsz)?;
    let mut tracker = SortTracker::new(args.max_age, args.iou_thresh);

    let font_data = include_bytes!("../third-party/fonts/Roboto-Regular.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    let source_path = Path::new(&args.source);
    let mut decoder = Decoder::new(source_path)?;
    let (width, height) = decoder.size();

    let dest_path = Path::new(&args.destination);
    let settings =
        Settings::preset_h264_yuv420p(width as usize, height as usize, false)
            .with_keyframe_interval(decoder.frame_rate().round() as u64);
    let mut encoder = Encoder::new(dest_path, settings)?;

    let mut unique_ids = HashSet::new();
    let mut frame_count = 0;

    println!("Video resolution: {}x{}", width, height);

    for frame_res in decoder.decode_raw_iter() {
        let mut frame = match frame_res {
            Ok(f) => f,
            Err(_) => break,
        };

        let width = frame.width() as usize;
        let height = frame.height() as usize;

        let data = frame.data(0);
        let stride = frame.stride(0);

        // Copy frame into contiguous RGB buffer
        let mut buffer = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            let row = &data[y * stride..y * stride + width * 3];
            buffer.extend_from_slice(row);
        }

        // Wrap into Kornia Image
        let mut image =
            Image::new(ImageSize { width, height }, buffer, CpuAllocator)?;

        // 1. Detect
        let detections = detector.detect(&image, args.conf)?;

        // 2. Track
        let active_tracks = tracker.update(detections);

        // 3. Draw
        for track in active_tracks {
            unique_ids.insert(track.id);

            let seed = track.id as f64;
            let color = [
                ((seed * 123.0) % 255.0) as u8,
                ((seed * 321.0) % 255.0) as u8,
                ((seed * 789.0) % 255.0) as u8,
            ];

            draw_rect(&mut image, track.last_rect, color);

            let label = format!("ID:{}", track.id);
            let label_rect =
                Rect::new(track.last_rect.x, track.last_rect.y - 22, 60, 22);

            draw_filled_rect(&mut image, label_rect, color);

            draw_text(
                &mut image,
                &font,
                &label,
                track.last_rect.x + 2,
                track.last_rect.y - 20,
                [255, 255, 255],
            );
        }

        // Copy modified image back into frame
        let image_data = image.storage.as_slice();
        let data_mut = frame.data_mut(0);

        for y in 0..height {
            let dest = &mut data_mut[y * stride..y * stride + width * 3];
            let src = &image_data[y * width * 3..(y + 1) * width * 3];
            dest.copy_from_slice(src);
        }

        // 4. Encode
        encoder.encode_raw(frame)?;

        frame_count += 1;
        println!("Frames: {}", frame_count);
    }

    encoder.finish()?;
    println!("Done! Unique IDs: {}", unique_ids.len());

    Ok(())
}
