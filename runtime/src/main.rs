mod common;
mod detector;
mod tracker;

use anyhow::Result;
use clap::Parser;
use std::collections::HashSet;
use std::path::Path;

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
}

fn main() -> Result<()> {
    video_rs::init().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let args = Args::parse();

    // --- Init Models ---
    let detector = YoloDetector::new(&args.model, args.imgsz)?;
    let mut tracker = SortTracker::new(args.max_age, args.iou_thresh);

    // --- Init Assets ---
    let font_data = include_bytes!("../third-party/fonts/Roboto-Regular.ttf");
    let font = FontRef::try_from_slice(font_data)?;

    // --- Init Decoder ---
    let source_path = Path::new(&args.source);
    let mut decoder = Decoder::new(source_path)?;
    let (width, height) = decoder.size();
    let framerate = decoder.frame_rate();
    println!("Input: {}x{} @ {:.2} fps", width, height, framerate);

    // --- Init Encoder ---
    let dest_path = Path::new(&args.destination);
    let settings =
        Settings::preset_h264_yuv420p(width as usize, height as usize, false)
            .with_keyframe_interval(framerate.round() as u64);
    // Note: No .with_frame_rate() needed; timestamps will dictate speed.

    let mut encoder = Encoder::new(dest_path, settings)?;

    let mut unique_ids = HashSet::new();
    let mut frame_count = 0;

    // --- Processing Loop ---
    // decode_iter() returns (Time, Array3<u8>)
    for frame_res in decoder.decode_iter() {
        let (time, frame_array) = match frame_res {
            Ok(f) => f,
            Err(_) => break, // End of stream or error
        };

        let (h, w, _channels) = frame_array.dim();
        let (raw_buffer, _offset) = frame_array.into_raw_vec_and_offset();

        let mut image = Image::new(
            ImageSize {
                width: w,
                height: h,
            },
            raw_buffer,
        )?;

        // 2. Logic (Detect / Track / Draw)
        let detections = detector.detect(&image, args.conf)?;
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
            let annotation = format!("ID:{}, C:{}", track.id, track.class_id);
            let ann_rect =
                Rect::new(track.last_rect.x, track.last_rect.y - 20, 70, 20);
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
        // 3. Zero-Copy conversion back: Kornia Image -> Vec<u8> -> ndarray
        // This is safe because Kornia didn't change the dimensions, only pixel values.
        let modified_buffer = image.as_slice().to_vec();

        // Reconstruct the Array3
        let out_frame_array =
            Array3::from_shape_vec((h, w, 3), modified_buffer)?;

        // 4. Encode with the ORIGINAL timestamp
        encoder.encode(&out_frame_array, time)?;

        frame_count += 1;
        let formatted = active_tracks
            .iter()
            .map(|t| {
                format!(
                    "(id={}, age={}, tsu={}, hits={}, class={})",
                    t.id, t.age, t.time_since_update, t.hits, t.class_id
                )
            })
            .collect::<Vec<_>>()
            .join(", ");

        println!("Tracks at frame {}: {}", frame_count, formatted);
    }

    encoder.finish()?;
    println!("Done! Total Unique IDs: {}", unique_ids.len());

    Ok(())
}
