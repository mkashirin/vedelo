mod kalman;

use self::kalman::KalmanFilter;
use crate::common::{Rect, calculate_iou};
use crate::detector::Detection;

pub struct Track {
    pub id: usize,
    pub kf: KalmanFilter,
    pub age: i32,
    pub time_since_update: i32,
    pub hits: i32,
    pub last_rect: Rect,
    pub class_id: i32,
    pub det_score: f32,
}

pub struct SortTracker {
    tracks: Vec<Track>,
    next_id: usize,
    max_age: i32,
    iou_threshold: f64,
    cost_matrix: Vec<i32>,
}

impl SortTracker {
    pub fn new(max_age: i32, iou_threshold: f64) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: 1,
            max_age,
            iou_threshold,
            cost_matrix: Vec::new(),
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
        // Dense flags are faster and allocation-free compared with HashSet for
        // the compact, index-addressed assignment problem.
        let mut unmatched_dets = vec![true; num_dets];

        if num_tracks > 0 && num_dets > 0 {
            self.cost_matrix.clear();
            self.cost_matrix.reserve(num_tracks * num_dets);

            for track in &self.tracks {
                for det in &detections {
                    // Vehicles of different classes should never consume an
                    // assignment merely because their boxes overlap.
                    let iou = if track.class_id == det.class_id {
                        calculate_iou(track.last_rect, det.rect)
                    } else {
                        0.0
                    };
                    // Convert float cost to integer for Hungarian:
                    //
                    // $$ \textrm{Cost} = (1.0 - \textrm{IoU}) * 1000 $$
                    let weight = ((1.0 - iou) * 1000.0) as i32;
                    self.cost_matrix.push(weight);
                }
            }

            // Call minimize with the flattened slice.
            let assignment =
                hungarian::minimize(&self.cost_matrix, num_tracks, num_dets);

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
                    let cost = self.cost_matrix[t_ind * num_dets + d_ind];

                    // Convert back to IoU: $\textrm{IoU} = 1.0 - (c / 1000.0)$
                    let iou = 1.0 - (cost as f64 / 1000.0);

                    if iou > self.iou_threshold {
                        matched_pairs.push((t_ind, d_ind));
                        unmatched_dets[d_ind] = false;
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
            track.det_score = det.score;
            track.hits += 1;
            track.time_since_update = 0;
            track.age += 1;
        }

        for (d_ind, is_unmatched) in unmatched_dets.into_iter().enumerate() {
            if !is_unmatched {
                continue;
            }
            let det = &detections[d_ind];
            self.tracks.push(Track {
                id: self.next_id,
                kf: KalmanFilter::new(det.rect),
                age: 1,
                time_since_update: 0,
                hits: 1,
                last_rect: det.rect,
                class_id: det.class_id,
                det_score: det.score,
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
