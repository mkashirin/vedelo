mod kalman;

use self::kalman::KalmanFilter;
use crate::common::calculate_iou;
use crate::detector::Detection;
use opencv::core::Rect;
use std::collections::HashSet;

pub struct Track {
    pub id: usize,
    pub kf: KalmanFilter,
    pub age: i32,
    pub time_since_update: i32,
    pub hits: i32,
    pub last_rect: Rect,
    pub class_id: i32,
}

pub struct SortTracker {
    tracks: Vec<Track>,
    next_id: usize,
    max_age: i32,
    iou_threshold: f64,
}

impl SortTracker {
    pub fn new(max_age: i32, iou_threshold: f64) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            iou_threshold,
        }
    }

    pub fn update(&mut self, detections: Vec<Detection>) -> Vec<&Track> {
        for track in &mut self.tracks {
            track.last_rect = track.kf.predict();
            track.time_since_update += 1;
        }

        let num_tracks = self.tracks.len();
        let num_dets = detections.len();
        let mut matched_pairs = Vec::new();
        let mut unmatched_tracks = (0..num_tracks).collect::<HashSet<_>>();
        let mut unmatched_dets = (0..num_dets).collect::<HashSet<_>>();

        if num_tracks > 0 && num_dets > 0 {
            let mut cost_matrix = Vec::with_capacity(num_tracks * num_dets);

            for track in &self.tracks {
                for det in &detections {
                    let iou = calculate_iou(track.last_rect, det.rect);
                    // Convert float cost to int for Hungarian: Cost = (1.0 - IoU) * 1000
                    let weight = ((1.0 - iou) * 1000.0) as i32;
                    cost_matrix.push(weight);
                }
            }

            // Call minimize with the flattened slice
            let assignment =
                hungarian::minimize(&cost_matrix, num_tracks, num_dets);

            for (t_index, &maybe_d_index) in assignment.iter().enumerate() {
                if let Some(d_idx) = maybe_d_index {
                    // Re-calculate cost or retrieve from matrix logic
                    // cost_matrix index = t_idx * num_dets + d_idx
                    let cost = cost_matrix[t_index * num_dets + d_idx];

                    // Convert back to IoU: iou = 1.0 - (cost / 1000.0)
                    let iou = 1.0 - (cost as f64 / 1000.0);

                    if iou > self.iou_threshold {
                        matched_pairs.push((t_index, d_idx));
                        unmatched_tracks.remove(&t_index);
                        unmatched_dets.remove(&d_idx);
                    }
                }
            }
        } else {
            // If tracks = 0, all dets are unmatched. If dets = 0, all tracks are unmatched.
        }

        for (t_index, d_index) in matched_pairs {
            let track = &mut self.tracks[t_index];
            let det = &detections[d_index];
            track.kf.update(det.rect);
            track.last_rect = det.rect;
            track.class_id = det.class_id;
            track.hits += 1;
            track.time_since_update = 0;
            track.age += 1;
        }

        for d_idx in unmatched_dets {
            let det = &detections[d_idx];
            self.tracks.push(Track {
                id: self.next_id,
                kf: KalmanFilter::new(det.rect),
                age: 1,
                time_since_update: 0,
                hits: 1,
                last_rect: det.rect,
                class_id: det.class_id,
            });
            self.next_id += 1;
        }

        self.tracks.retain(|t| t.time_since_update < self.max_age);

        self.tracks
            .iter()
            .filter(|t| t.time_since_update < 2)
            .collect()
    }
}
