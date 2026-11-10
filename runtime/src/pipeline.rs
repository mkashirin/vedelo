//! Two capacity-one batch queues; workers own their codecs.
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;

type StageResult<T> = Result<T, String>;

/// Assemble directly from a fallible frame source; EOF alone releases a partial tail.
/// Errors are never converted to EOF. A zero batch size fails before reading.
pub fn next_batch<T>(
    size: usize,
    mut next: impl FnMut() -> Result<Option<T>, String>,
) -> Result<Option<Vec<T>>, String> {
    if size == 0 {
        return Err("batch size must be at least 1".to_owned().into());
    }
    let mut batch = Vec::with_capacity(size);
    while batch.len() < size {
        match next()? {
            Some(frame) => batch.push(frame),
            None => break,
        }
    }
    Ok((!batch.is_empty()).then_some(batch))
}

/// Preserve sub-integer FPS from video-rs' f32 API to micro-fps precision.
pub fn frame_rate_parts(fps: f32) -> Result<(i32, i32), String> {
    let scaled = (f64::from(fps) * 1_000_000.0).round();
    if !fps.is_finite()
        || fps <= 0.0
        || scaled < 1.0
        || scaled > i32::MAX as f64
    {
        return Err(format!(
            "invalid or unsupported source frame rate: {fps}"
        ));
    }
    let mut a = scaled as i32;
    let mut b = 1_000_000;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    Ok((scaled as i32 / a, 1_000_000 / a))
}

fn joined<T>(
    result: thread::Result<StageResult<T>>,
    stage: &str,
) -> StageResult<T> {
    result.unwrap_or_else(|_| Err(format!("{stage} panicked")))
}

pub fn run<I: Send, O: Send, R: Send>(
    produce: impl FnOnce(SyncSender<I>) -> StageResult<()> + Send,
    mut transform: impl FnMut(I) -> StageResult<O>,
    consume: impl FnOnce(Receiver<O>) -> StageResult<R> + Send,
) -> StageResult<R> {
    thread::scope(|scope| {
        let (input_tx, input_rx) = sync_channel(1);
        let (output_tx, output_rx) = sync_channel(1);
        let producer = scope.spawn(move || produce(input_tx));
        let consumer = scope.spawn(move || consume(output_rx));
        // Catch unwind so both channels are disconnected BEFORE joining workers.
        let middle =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for batch in input_rx.iter() {
                    let output = transform(batch)?;
                    output_tx
                        .send(output)
                        .map_err(|_| "encoder disconnected".to_owned())?;
                }
                Ok(())
            }));
        drop(input_rx);
        drop(output_tx);
        let produced = joined(producer.join(), "decoder");
        let consumed = joined(consumer.join(), "encoder");
        let transformed = joined(middle, "inference");
        // Retain root errors even when another stage reports disconnection.
        let mut errors = Vec::new();
        if let Err(error) = produced {
            errors.push(error);
        }
        if let Err(error) = transformed {
            errors.push(error);
        }
        match consumed {
            Ok(result) if errors.is_empty() => Ok(result),
            Ok(_) => Err(errors.join("; ")),
            Err(error) => {
                errors.push(error);
                Err(errors.join("; "))
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembly_boundaries_and_timestamps() {
        for count in [0, 1, 3, 4, 5, 12, 13] {
            let expected: Vec<_> =
                (0..count).map(|i| (i, i * i * 97)).collect();
            let mut frames = expected.clone().into_iter();
            let mut batches = Vec::new();
            while let Some(batch) =
                next_batch::<_>(4, || Ok(frames.next())).unwrap()
            {
                assert!(batch.len() <= 4);
                batches.push(batch);
            }
            assert_eq!(
                batches.into_iter().flatten().collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn assembly_errors_and_zero_size() {
        let mut calls = 0;
        let error = next_batch::<usize>(4, || {
            calls += 1;
            if calls == 1 {
                Ok(Some(7))
            } else {
                Err("real decode error".into())
            }
        })
        .unwrap_err();
        assert_eq!(error, "real decode error");
        assert!(next_batch::<usize>(0, || panic!("must not read")).is_err());
    }

    #[test]
    fn fractional_fps_and_validation() {
        for fps in [24.0, 25.0, 29.97, 30_000.0 / 1001.0, 60.0] {
            let (n, d) = frame_rate_parts(fps).unwrap();
            assert!((n as f64 / d as f64 - fps as f64).abs() < 0.000001);
        }
        assert_eq!(frame_rate_parts(25.0).unwrap(), (25, 1));
        for fps in [
            0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
        ] {
            assert!(frame_rate_parts(fps).is_err());
        }
    }

    #[test]
    fn order_and_partial_tail() {
        let result = run(
            |tx| {
                for batch in [vec![0, 1], vec![2, 3], vec![4]] {
                    tx.send(batch).map_err(|e| e.to_string())?;
                }
                Ok(())
            },
            |batch| Ok(batch),
            |rx| Ok(rx.into_iter().flatten().collect::<Vec<_>>()),
        )
        .unwrap();
        assert_eq!(result, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn failures_disconnect_blocked_workers() {
        for panic in [false, true] {
            let result = run(
                |tx| {
                    for i in 0..100 {
                        tx.send(i).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                },
                |i| Ok(i),
                move |_rx| -> StageResult<()> {
                    if panic {
                        panic!("injected");
                    }
                    Err("injected encoder error".into())
                },
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn capacity_one_applies_backpressure() {
        let (tx, rx) = sync_channel(1);
        tx.send(1).unwrap();
        assert!(matches!(
            tx.try_send(2),
            Err(std::sync::mpsc::TrySendError::Full(2))
        ));
        assert_eq!(rx.recv().unwrap(), 1);
    }

    #[test]
    fn producer_and_transform_errors_propagate() {
        let result = run::<usize, usize, ()>(
            |_tx| Err("decode error".into()),
            Ok,
            |rx| {
                for _ in rx {}
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err(), "decode error");
        let result = run(
            |tx| tx.send(1).map_err(|e| e.to_string()),
            |_| Err::<usize, _>("inference error".into()),
            |rx| {
                for _ in rx {}
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err(), "inference error");
    }
}
