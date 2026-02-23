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
                    // Convert float cost to integer for Hungarian:
                    //
                    // $$ \textrm{Cost} = (1.0 - \textrm{IoU}) * 1000 $$
                    let weight = ((1.0 - iou) * 1000.0) as i32;
                    cost_matrix.push(weight);
                }
            }

            // Call minimize with the flattened slice.
            let assignment =
                hungarian::minimize(&cost_matrix, num_tracks, num_dets);

            for (t_ind, &maybe_d_ind) in assignment.iter().enumerate() {
                if let Some(d_ind) = maybe_d_ind {
                    // Re-calculate cost or retrieve from matrix logic:
                    //
                    // $$ \textrm{CM}_i = t_i * n_d + d_i $$
                    //
                    // where:
                    //     * $\texrm{CM}$ is for `cost_matrix`;
                    //     * $t_i$, $d_i$ are `t_ind` and `d_ind`;
                    //     * $n_d$ means `num_dets`
                    let cost = cost_matrix[t_ind * num_dets + d_ind];

                    // Convert back to IoU: $\textrm{IoU} = 1.0 - (c / 1000.0)$
                    let iou = 1.0 - (cost as f64 / 1000.0);

                    if iou > self.iou_threshold {
                        matched_pairs.push((t_ind, d_ind));
                        unmatched_tracks.remove(&t_ind);
                        unmatched_dets.remove(&d_ind);
                    }
                }
            }
        } else {
            // If `tracks` is 0, all `dets` are unmatched, and the other way around.
        }

        for (t_ind, d_ind) in matched_pairs {
            let track = &mut self.tracks[t_ind];
            let det = &detections[d_ind];
            track.kf.update(det.rect);
            track.last_rect = det.rect;
            track.class_id = det.class_id;
            track.hits += 1;
            track.time_since_update = 0;
            track.age += 1;
        }

        for d_ind in unmatched_dets {
            let det = &detections[d_ind];
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
