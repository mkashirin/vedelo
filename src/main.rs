mod common;
mod detector;
mod tracker;

use anyhow::{Result, anyhow};
use clap::Parser;
use opencv::{core::Size, highgui, prelude::*, videoio};
use std::collections::HashSet;

use crate::detector::YoloDetector;
use crate::tracker::SortTracker;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long, value_hint = "PATH")]
    model: String,

    #[arg(short, long, value_hint = "PATH")]
    source: String,

    #[arg(short, long, value_hint = "PATH")]
    destination: String,

    #[arg(long, default_value_t = 0.5)]
    conf: f64,

    #[arg(long, default_value_t = 0.3)]
    iou_thresh: f64,

    #[arg(long, default_value_t = 1280)]
    imgsz: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let detector = YoloDetector::new(&args.model, args.imgsz)?;
    let mut tracker = SortTracker::new(30, args.iou_thresh);

    let mut cam =
        videoio::VideoCapture::from_file(&args.source, videoio::CAP_FFMPEG)?;
    if !cam.is_opened()? {
        return Err(anyhow!("Cannot open video: {}", args.source));
    }

    let width = cam.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cam.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cam.get(videoio::CAP_PROP_FPS)?;

    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = videoio::VideoWriter::new(
        &args.destination,
        fourcc,
        fps,
        Size::new(width, height),
        true,
    )?;

    let mut frame = Mat::default();
    let mut unique_ids = HashSet::new();
    println!("Starting processing...");

    loop {
        if !cam.read(&mut frame)? || frame.empty() {
            break;
        }
        let detections = detector.detect(&frame, args.conf)?;
        let active_tracks = tracker.update(detections);

        for track in active_tracks {
            unique_ids.insert(track.id);
            common::draw_prediction(&mut frame, track)?;
        }

        writer.write(&frame)?;
        highgui::imshow("YOLO Tracker", &frame)?;

        if highgui::wait_key(1)? == 113 {
            break;
        }
    }

    println!("Done. Saved to: {}", args.destination);
    println!("Unique IDs found: {}", unique_ids.len());
    Ok(())
}
